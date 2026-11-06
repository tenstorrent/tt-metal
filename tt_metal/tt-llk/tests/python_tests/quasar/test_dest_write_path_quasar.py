# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
32-bit Dest write paths on Quasar: unpack-to-Dest vs Unpacker 2 -> SrcS -> SFPU -> Dest.

Paths (DestWritePath):
  A-drain  TRISC0 UNPACR_DEST per bank, then STALLWAIT + CLEARDVALID.
  A-hw     Same, with SetDatValid on the bank's last UNPACR_DEST instead of the drain.
  B-drain  TRISC3 Unpacker 2 -> SrcS -> SFPU -> Dest per bank, then _llk_math_set_dvalid_<SFPU>.
  B-done   Same, with done on the bank's last SFPSTORE (copy) / SFPNOP (SFPI ops) instead of the drain.

Kernels:
  copy    general sweep, bit-exact L1 -> Dest -> L1 (Float32 and Int32).
  square  SFPU-light op on the way into Dest.
  exp     SFPU-heavy op (accurate fp32 exp) on the way into Dest.
  matmul  FPU kernel that needs Dest, in test_dest_preload_matmul_quasar.py.

TRISC2 records a wall-clock stamp after every bank into buffer_C. Each run logs the per-bank deltas and appends
them to <artefacts>/dest_write_path_bank_stamps.jsonl; ``python -m quasar.test_dest_write_path_quasar`` from
tests/python_tests fits cycles/bank against tiles/bank (slope = per-tile cost, intercept = bank-switch cost).
"""

import json
import os
import struct
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from statistics import median

import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import quantize_mx_tensor_chunked
from helpers.llk_params import (
    DestAccumulation,
    DestSync,
    PerfRunType,
    format_dict,
)
from helpers.logger import logger
from helpers.param_config import parametrize
from helpers.perf.core import create_test_or_perf_config
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    DEST_SYNC,
    LOOP_FACTOR,
    TILE_COUNT,
    TemplateParameter,
)
from helpers.utils import passed_test

TILE_ELEMENTS = 1024
FACE_ELEMENTS = 256
NUM_BLOCKS = 16
# Both banks start free, so the first two banks never wait on the packer.
WARMUP_BANKS = 2
# A 32-bit SyncHalf bank holds up to 4 tiles of 32x32.
TILES_PER_BANK_SWEEP = [1, 2, 3, 4]
SFPU_TILES_PER_BANK_SWEEP = [1, 4]
STAMPS_FILE_NAME = "dest_write_path_bank_stamps.jsonl"
# The kernels only implement the full L1 -> L1 flow; the isolate run types would break the Dest dvalid chains.
PERF_RUN_TYPES = [[PerfRunType.L1_TO_L1]]
# 16 blocks x 4 = 64 banks per launch; inputs are reused modulo the 16 blocks.
PERF_LOOP_FACTOR = 4


class DestWritePath(Enum):
    A_DRAIN = "A-drain"
    A_HW = "A-hw"
    B_DRAIN = "B-drain"
    B_DONE = "B-done"

    @property
    def srcs_writer(self) -> bool:
        return self in (DestWritePath.B_DRAIN, DestWritePath.B_DONE)

    @property
    def hw_bank_switch(self) -> bool:
        return self in (DestWritePath.A_HW, DestWritePath.B_DONE)


class SrcsIssueOpt(Enum):
    """Path B issue options on TRISC3 (copy only), to test whether the per-tile cost is the kernel or the path.

    addrmod     the odd-column SFPSTORE steps Dest through ADDR_MOD_6, dropping the per-slice INCRWC (82 -> 66 per tile).
    bankunpack  one UNPACR2 auto-loop per bank instead of per tile, so SrcS refills while the SFPU works.
    multibank   addrmod with one UNPACR2 per group of banks (dwp::srcs_banks_per_unpack), so SrcS also refills across
                the bank hand-off.
    """

    BASE = "base"
    ADDR_MOD = "addrmod"
    BANK_UNPACK = "bankunpack"
    BOTH = "both"
    MULTI_BANK = "multibank"

    @property
    def addr_mod_step(self) -> bool:
        return self in (
            SrcsIssueOpt.ADDR_MOD,
            SrcsIssueOpt.BOTH,
            SrcsIssueOpt.MULTI_BANK,
        )

    @property
    def bank_unpack(self) -> bool:
        return self in (
            SrcsIssueOpt.BANK_UNPACK,
            SrcsIssueOpt.BOTH,
            SrcsIssueOpt.MULTI_BANK,
        )

    @property
    def multi_bank_unpack(self) -> bool:
        return self == SrcsIssueOpt.MULTI_BANK


class DestWriteKernel(Enum):
    """Value is DWP_KERNEL_OP in dest_write_path_quasar_test.cpp; Matmul has its own source."""

    Copy = 0
    Square = 1
    Exp = 2
    Matmul = 3
    Chain = 4


@dataclass
class DEST_WRITE_PATH(TemplateParameter):
    # Field names become perf CSV columns and must be unique across all parameter classes.
    dwp_path: DestWritePath
    dwp_tiles_per_bank: int
    dwp_kernel: DestWriteKernel = DestWriteKernel.Copy
    dwp_num_blocks: int = NUM_BLOCKS
    dwp_loop_factor: int = 1
    dwp_kt_dim: int = 1
    # Block on tensix_sync() before each pack-side stamp, so it marks pack completion of the bank.
    dwp_stamp_sync: bool = False
    # L1 holds a narrower float that the unpacker widens into Float32 SrcS / Dest.
    dwp_widen_to_fp32: bool = False
    # Path B issue options, see SrcsIssueOpt.
    dwp_srcs_addr_mod_step: bool = False
    dwp_srcs_bank_unpack: bool = False
    dwp_srcs_multi_bank_unpack: bool = False
    # Matmul only: add the partial with PACK0's L1 accumulation instead of preloading it into Dest.
    dwp_pack_l1_acc: bool = False
    # Pack only face 0 of each bank's first tile, so the bank stamps time the Dest writer, not PACK0.
    dwp_writer_only: bool = False

    dwp_producer_gap: int = 0
    dwp_consumer_gap: int = 0
    dwp_input_stride: int = 1
    dwp_math_depth: int = 1
    dwp_stamp_every: int = 1
    dwp_pack1: bool = False

    def convert_to_cpp(self) -> str:
        kernel_op = (
            0 if self.dwp_kernel == DestWriteKernel.Matmul else self.dwp_kernel.value
        )
        return "\n".join(
            [
                f"constexpr unsigned DWP_PRODUCER_GAP = {self.dwp_producer_gap};",
                f"constexpr unsigned DWP_CONSUMER_GAP = {self.dwp_consumer_gap};",
                f"constexpr unsigned DWP_INPUT_STRIDE = {self.dwp_input_stride};",
                f"constexpr unsigned DWP_MATH_DEPTH = {self.dwp_math_depth};",
                f"constexpr unsigned DWP_STAMP_EVERY = {self.dwp_stamp_every};",
                f"constexpr bool DWP_PACK1 = {str(self.dwp_pack1).lower()};",
                f"// Dest write path {self.dwp_path.value}, kernel {self.dwp_kernel.name}",
                f"constexpr bool DWP_SRCS_WRITER = {str(self.dwp_path.srcs_writer).lower()};",
                f"constexpr bool DWP_HW_BANK_SWITCH = {str(self.dwp_path.hw_bank_switch).lower()};",
                f"constexpr std::uint32_t DWP_KERNEL_OP = {kernel_op};",
                f"constexpr std::uint32_t DWP_TILES_PER_BANK = {self.dwp_tiles_per_bank};",
                f"constexpr std::uint32_t DWP_NUM_BLOCKS = {self.dwp_num_blocks};",
                f"constexpr std::uint32_t DWP_LOOP_FACTOR = {self.dwp_loop_factor};",
                f"constexpr std::uint32_t DWP_KT_DIM = {self.dwp_kt_dim};",
                f"constexpr bool DWP_STAMP_SYNC = {str(self.dwp_stamp_sync).lower()};",
                f"constexpr bool DWP_WIDEN_TO_FP32 = {str(self.dwp_widen_to_fp32).lower()};",
                f"constexpr bool DWP_SRCS_ADDR_MOD_STEP = {str(self.dwp_srcs_addr_mod_step).lower()};",
                f"constexpr bool DWP_SRCS_BANK_UNPACK = {str(self.dwp_srcs_bank_unpack).lower()};",
                f"constexpr bool DWP_SRCS_MULTI_BANK_UNPACK = {str(self.dwp_srcs_multi_bank_unpack).lower()};",
                f"constexpr bool DWP_PACK_L1_ACC = {str(self.dwp_pack_l1_acc).lower()};",
                f"constexpr bool DWP_WRITER_ONLY = {str(self.dwp_writer_only).lower()};",
            ]
        )


# --- per-bank wall-clock stamps -------------------------------------------------------------


def _stamps_buffer() -> torch.Tensor:
    return torch.zeros(TILE_ELEMENTS, dtype=torch.int32)


def read_bank_stamps(
    stimuli: StimuliConfig, location: str, stamp_every: int = 1
) -> dict:
    """Decode the TRISC2 stamp record (layout in dest_write_path_quasar.h) into per-bank cycle deltas."""
    raw = stimuli.collect_raw_buffer_c_bytes(location)
    words = struct.unpack(f"<{len(raw) // 4}I", raw)
    num_banks = words[0]
    assert 0 < num_banks <= 1000, f"invalid stamp count: {num_banks}"
    start = words[1]
    banks = list(words[2 : 2 + num_banks])
    end = words[2 + num_banks]

    def delta(a, b):
        return (b - a) % (1 << 32)

    sampled = [
        (i + 1, stamp)
        for i, stamp in enumerate(banks)
        if stamp_every and (i + 1) % stamp_every == 0
    ]
    previous_bank, previous_stamp = 0, start
    deltas = []
    steady = []
    for bank, stamp in sampled:
        per_bank = delta(previous_stamp, stamp) / (bank - previous_bank)
        deltas.append(per_bank)
        if previous_bank >= WARMUP_BANKS:
            steady.append(per_bank)
        previous_bank, previous_stamp = bank, stamp
    return {
        "num_banks": num_banks,
        "first_bank_cycles": deltas[0] if deltas else None,
        "steady_cycles_per_bank": median(steady) if steady else None,
        "total_cycles": delta(start, end),
        "completed_cycles_per_bank": delta(start, end) / num_banks,
        "bank_deltas": deltas,
        "hash_registers": dict(
            zip(
                ("ctrl", "mask", "match", *[f"addr{i}" for i in range(4, 12)]),
                words[1004:1015],
            )
        ),
    }


def report_bank_stamps(
    stimuli: StimuliConfig,
    kernel: DestWriteKernel,
    path: DestWritePath,
    tiles_per_bank: int,
    data_format: DataFormat,
    is_perf: bool,
    **extra,
) -> dict:
    timing = read_bank_stamps(
        stimuli, TestConfig.TENSIX_LOCATION, extra.get("stamp_every", 1)
    )
    if not extra.get("experimental", False):
        timing["hash_registers"] = None
    record = {
        "kernel": kernel.name,
        "path": path.value,
        "tiles_per_bank": tiles_per_bank,
        "format": data_format.name,
        "is_perf": is_perf,
        **extra,
        **timing,
        "steady_cycles_per_tile": (
            timing["steady_cycles_per_bank"] / tiles_per_bank
            if timing["steady_cycles_per_bank"] is not None
            else None
        ),
        "completed_cycles_per_tile": timing["completed_cycles_per_bank"]
        / tiles_per_bank,
        "recorded_at_ns": time.time_ns(),
        "requested_hash": os.environ.get("PERF_L1_HASH", "unspecified"),
        "l1_selection": os.environ.get("LLK_PERF_L1_CLIENT_SEL", "none"),
    }
    logger.info(
        "dest-write {} {} tpb={} {} {}: first bank {} cyc, steady {} cyc/bank ({} cyc/tile), total {} cyc",
        kernel.name,
        path.value,
        tiles_per_bank,
        data_format.name,
        extra,
        record["first_bank_cycles"],
        record["steady_cycles_per_bank"],
        record["steady_cycles_per_tile"],
        record["total_cycles"],
    )
    stamps_path = Path(TestConfig.ARTEFACTS_DIR) / STAMPS_FILE_NAME
    stamps_path.parent.mkdir(parents=True, exist_ok=True)
    with stamps_path.open("a") as f:
        f.write(json.dumps(record) + "\n")
    return record


def summarize_bank_stamps(stamps_path: Path) -> None:
    """Fit steady cycles/bank = slope * tiles/bank + intercept per (kernel, path, format, extra) group."""
    groups = defaultdict(lambda: defaultdict(list))
    for line in stamps_path.read_text().splitlines():
        record = json.loads(line)
        key = (
            record["kernel"],
            record["path"],
            f"{record['format']}->{record.get('out_format', record['format'])}"
            f"{'' if record.get('stamp_sync', True) else ' nosync'}"
            f"{'' if record.get('srcs_opt', 'base') == 'base' else ' ' + record['srcs_opt']}"
            f"{' writer-only' if record.get('writer_only') else ''}"
            + " "
            + json.dumps(
                {
                    k: record.get(k)
                    for k in (
                        "experimental",
                        "hash_registers",
                        "num_blocks",
                        "loop_factor",
                        "producer_gap",
                        "consumer_gap",
                        "input_stride",
                        "math_depth",
                        "stamp_every",
                        "pack1",
                        "slots",
                        "two_pass",
                    )
                },
                sort_keys=True,
            ),
            record.get("kt_dim"),
            record.get("fidelity"),
            record["is_perf"],
        )
        if record["steady_cycles_per_bank"] is not None:
            groups[key][record["tiles_per_bank"]].append(
                record["steady_cycles_per_bank"]
            )

    print(
        f"{'kernel':8} {'path':8} {'format':36} {'kt':>3} {'fidelity':8} {'perf':5} "
        f"{'cyc/bank by tpb':40} {'slope':>8} {'intercept':>10}"
    )
    for key, by_tpb in sorted(groups.items(), key=lambda kv: str(kv[0])):
        xs = sorted(by_tpb)
        ys = [median(by_tpb[x]) for x in xs]
        if len(xs) >= 2:
            x_mean = sum(xs) / len(xs)
            y_mean = sum(ys) / len(ys)
            slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / sum(
                (x - x_mean) ** 2 for x in xs
            )
            intercept = f"{y_mean - slope * x_mean:10.1f}"
            slope = f"{slope:8.1f}"
        else:
            slope, intercept = f"{'-':>8}", f"{'-':>10}"
        kernel, path, fmt, kt, fidelity, is_perf = key
        points = " ".join(f"{x}:{y:g}" for x, y in zip(xs, ys))
        print(
            f"{kernel:8} {path:8} {fmt:36} {str(kt or '-'):>3} {str(fidelity or '-'):8} {str(is_perf):5} "
            f"{points:40} {slope} {intercept}"
        )


# --- copy / SFPU-op kernels -----------------------------------------------------------------


def _eltwise_inputs(
    kernel: DestWriteKernel, data_format: DataFormat, tile_count: int
) -> torch.Tensor:
    num = tile_count * TILE_ELEMENTS
    if data_format == DataFormat.Int32:
        return torch.randint(-(2**30), 2**30, (num,), dtype=torch.int32)
    if kernel == DestWriteKernel.Exp:
        return torch.empty(num, dtype=torch.float32).uniform_(-4.0, 4.0)
    if kernel == DestWriteKernel.Square:
        return torch.empty(num, dtype=torch.float32).uniform_(-16.0, 16.0)
    return torch.randn(num, dtype=torch.float32) * 100.0


def _eltwise_golden(kernel: DestWriteKernel, src: torch.Tensor) -> torch.Tensor:
    if kernel == DestWriteKernel.Exp:
        return torch.exp(src)
    if kernel == DestWriteKernel.Square:
        return src * src
    return src


def _assert_bit_exact(src: torch.Tensor, res: torch.Tensor, label: str) -> None:
    """src must already be in res's dtype; Float16_b -> Float32 -> Float16_b round-trips exactly."""
    bits = torch.int16 if res.element_size() == 2 else torch.int32
    mask = (1 << (8 * res.element_size())) - 1
    src_bits = src.view(bits)
    res_bits = res.view(bits)
    mismatch = (src_bits != res_bits).nonzero().flatten()
    if mismatch.numel():
        first = int(mismatch[0])
        tile, in_tile = divmod(first, TILE_ELEMENTS)
        face, in_face = divmod(in_tile, 256)
        row, col = divmod(in_face, 16)
        raise AssertionError(
            f"{label}: {mismatch.numel()} of {src.numel()} datums differ; first at tile {tile} face {face} "
            f"row {row} col {col}: expected 0x{int(src_bits[first]) & mask:08X}, "
            f"got 0x{int(res_bits[first]) & mask:08X}"
        )


def run_dest_write_path_eltwise(
    path: DestWritePath,
    tiles_per_bank: int,
    data_format: DataFormat,
    kernel: DestWriteKernel,
    run_types=(PerfRunType.L1_TO_L1,),
    loop_factor: int = 1,
    is_perf: bool = False,
    perf_report=None,
    out_format: DataFormat = None,
    stamp_sync: bool = False,
    srcs_opt: SrcsIssueOpt = SrcsIssueOpt.BASE,
    writer_only: bool = False,
    num_blocks: int = NUM_BLOCKS,
    producer_gap: int = 0,
    consumer_gap: int = 0,
    input_stride: int = 1,
    math_depth: int = 1,
    stamp_every: int = 1,
    pack1: bool = False,
    experimental: bool = False,
):
    """data_format is the L1 input format. A Float16_b input is widened to Float32 in SrcS / Dest; out_format
    (default: the input format) is what the packer writes back to L1. An MX out_format is lossy, so it is checked
    against an MX-quantized golden instead of bit-exact. writer_only packs just face 0 of each bank's first tile
    (L1 face slot = block), so the stamps time the Dest writer; only those faces are checked.
    """
    assert kernel in (
        DestWriteKernel.Copy,
        DestWriteKernel.Square,
        DestWriteKernel.Exp,
        DestWriteKernel.Chain,
    )
    assert kernel == DestWriteKernel.Copy or data_format == DataFormat.Float32
    assert srcs_opt == SrcsIssueOpt.BASE or (
        path.srcs_writer
        and (kernel == DestWriteKernel.Copy or not srcs_opt.addr_mod_step)
    ), "Arithmetic supports bank unpack; addrmod is currently copy-only"
    assert (
        loop_factor > 0
        and num_blocks > 0
        and num_blocks % 2 == 0
        and num_blocks * loop_factor <= 1000
    )
    assert (
        input_stride >= 1
        and math_depth >= 1
        and min(producer_gap, consumer_gap, stamp_every) >= 0
    )
    assert input_stride == 1 or not srcs_opt.bank_unpack
    assert not pack1 or (
        path == DestWritePath.B_DONE
        and kernel == DestWriteKernel.Chain
        and input_stride == 1
        and not writer_only
        and srcs_opt == SrcsIssueOpt.BASE
    )
    widen = data_format == DataFormat.Float16_b
    assert widen or data_format in (DataFormat.Float32, DataFormat.Int32)
    out_format = out_format or data_format
    assert not out_format.is_mx_format() or data_format != DataFormat.Int32
    assert (
        not widen
        or out_format in (DataFormat.Float32, DataFormat.Float16_b)
        or out_format.is_mx_format()
    )
    assert not (
        writer_only and out_format.is_mx_format()
    ), "MX blocks span 32 datums, wider than the one-face pack"
    assert experimental or (
        num_blocks == NUM_BLOCKS
        and producer_gap == consumer_gap == 0
        and input_stride == math_depth == stamp_every == 1
        and not pack1
        and kernel != DestWriteKernel.Chain
    )
    torch.manual_seed(42)

    tile_count = num_blocks * tiles_per_bank
    input_tiles = tile_count * input_stride
    src = _eltwise_inputs(kernel, data_format, input_tiles)
    if widen:
        src = src.to(torch.bfloat16)
    torch_format = format_dict[data_format]
    torch_out_format = format_dict[out_format]

    stimuli = StimuliConfig(
        src,
        data_format,
        torch.zeros(TILE_ELEMENTS, dtype=torch_format),
        data_format,
        out_format,
        tile_count_A=input_tiles,
        tile_count_B=1,
        tile_count_res=tile_count,
        buffer_C=_stamps_buffer(),
        stimuli_C_format=DataFormat.Int32,
        tile_count_C=1,
        num_faces=4,
    )

    test_config_kwargs = {
        "test_name": (
            "sources/quasar/dest_write_pack1_quasar_test.cpp"
            if pack1
            else (
                "sources/quasar/dest_write_experiments_quasar_test.cpp"
                if experimental
                else "sources/quasar/dest_write_path_quasar_test.cpp"
            )
        ),
        "formats": InputOutputFormat(data_format, out_format),
        "templates": [
            DEST_SYNC(DestSync.Half),
            DEST_WRITE_PATH(
                dwp_path=path,
                dwp_num_blocks=num_blocks,
                dwp_producer_gap=producer_gap,
                dwp_consumer_gap=consumer_gap,
                dwp_input_stride=input_stride,
                dwp_math_depth=math_depth,
                dwp_stamp_every=stamp_every,
                dwp_pack1=pack1,
                dwp_tiles_per_bank=tiles_per_bank,
                dwp_kernel=kernel,
                dwp_loop_factor=loop_factor,
                dwp_stamp_sync=stamp_sync,
                dwp_widen_to_fp32=widen,
                dwp_srcs_addr_mod_step=srcs_opt.addr_mod_step,
                dwp_srcs_bank_unpack=srcs_opt.bank_unpack,
                dwp_srcs_multi_bank_unpack=srcs_opt.multi_bank_unpack,
                dwp_writer_only=writer_only,
            ),
        ],
        "runtimes": [LOOP_FACTOR(loop_factor), TILE_COUNT(tile_count)],
        "variant_stimuli": stimuli,
        "unpack_to_dest": not path.srcs_writer,
        "disable_format_inference": True,
        "dest_acc": DestAccumulation.Yes,
    }
    configuration = create_test_or_perf_config(
        is_perf=is_perf,
        run_types=run_types,
        test_config_kwargs=test_config_kwargs,
    )

    if is_perf:
        configuration.run(perf_report)
    else:
        outcome = configuration.run()
        res = torch.tensor(outcome.result, dtype=torch_out_format)
        src = src.view(input_tiles, TILE_ELEMENTS)[::input_stride].flatten()
        assert res.numel() == src.numel(), "result size"
        label = f"{kernel.name} {path.value} tpb={tiles_per_bank} {data_format.name}->{out_format.name}"
        if writer_only:
            src = src.view(num_blocks, tiles_per_bank, TILE_ELEMENTS)[
                :, 0, :FACE_ELEMENTS
            ]
            src = src.flatten()
            res = res[: src.numel()]
        if kernel == DestWriteKernel.Copy and out_format.is_mx_format():
            golden = quantize_mx_tensor_chunked(src.to(torch.bfloat16), out_format)
            assert passed_test(golden, res, out_format), label
        elif kernel == DestWriteKernel.Copy:
            _assert_bit_exact(src.to(torch_out_format), res, label)
        else:
            golden = _eltwise_golden(kernel, src)
            if kernel == DestWriteKernel.Chain:
                golden = src
                for _ in range(math_depth):
                    golden = golden * 0.5 + 0.125
            assert passed_test(golden, res, data_format), label

    report_bank_stamps(
        stimuli,
        kernel,
        path,
        tiles_per_bank,
        data_format,
        is_perf,
        loop_factor=loop_factor,
        out_format=out_format.name,
        stamp_sync=stamp_sync,
        srcs_opt=srcs_opt.value,
        writer_only=writer_only,
        num_blocks=num_blocks,
        producer_gap=producer_gap,
        consumer_gap=consumer_gap,
        input_stride=input_stride,
        math_depth=math_depth,
        stamp_every=stamp_every,
        pack1=pack1,
        experimental=experimental,
    )


@parametrize(
    path=list(DestWritePath),
    tiles_per_bank=TILES_PER_BANK_SWEEP,
    data_format=[DataFormat.Float32, DataFormat.Int32],
)
def test_dest_write_path_copy_quasar(path, tiles_per_bank, data_format):
    run_dest_write_path_eltwise(path, tiles_per_bank, data_format, DestWriteKernel.Copy)


@parametrize(
    kernel=[DestWriteKernel.Square, DestWriteKernel.Exp],
    path=list(DestWritePath),
    tiles_per_bank=SFPU_TILES_PER_BANK_SWEEP,
)
def test_dest_write_path_sfpu_quasar(kernel, path, tiles_per_bank):
    run_dest_write_path_eltwise(path, tiles_per_bank, DataFormat.Float32, kernel)


# (L1 input, L1 output) for the format sweep, all through a 32-bit SrcS / Dest. Float32 -> Float32 is the baseline
# without stamp sync; the Float16_b rows halve the unpacker's L1 reads, Float16_b -> Float16_b the packer's L1
# writes too, and the MxFp8R rows cut the packer's L1 writes to about a quarter of Float32.
FORMAT_SWEEP = [
    (DataFormat.Float32, DataFormat.Float32),
    (DataFormat.Float16_b, DataFormat.Float32),
    (DataFormat.Float16_b, DataFormat.Float16_b),
    (DataFormat.Float32, DataFormat.MxFp8R),
    (DataFormat.Float16_b, DataFormat.MxFp8R),
]


@parametrize(
    path=list(DestWritePath),
    tiles_per_bank=TILES_PER_BANK_SWEEP,
    formats=FORMAT_SWEEP,
)
def test_dest_write_path_format_sweep_quasar(path, tiles_per_bank, formats):
    l1_format, out_format = formats
    run_dest_write_path_eltwise(
        path,
        tiles_per_bank,
        l1_format,
        DestWriteKernel.Copy,
        out_format=out_format,
        stamp_sync=False,
    )


@parametrize(
    path=[DestWritePath.B_DRAIN, DestWritePath.B_DONE],
    tiles_per_bank=TILES_PER_BANK_SWEEP,
    formats=FORMAT_SWEEP,
    srcs_opt=[SrcsIssueOpt.BOTH],
)
def test_dest_write_path_format_srcs_opt_quasar(
    path, tiles_per_bank, formats, srcs_opt
):
    l1_format, out_format = formats
    run_dest_write_path_eltwise(
        path,
        tiles_per_bank,
        l1_format,
        DestWriteKernel.Copy,
        out_format=out_format,
        stamp_sync=False,
        srcs_opt=srcs_opt,
    )


@parametrize(
    path=[DestWritePath.B_DRAIN, DestWritePath.B_DONE],
    tiles_per_bank=TILES_PER_BANK_SWEEP,
    srcs_opt=list(SrcsIssueOpt),
)
def test_dest_write_path_srcs_opt_quasar(path, tiles_per_bank, srcs_opt):
    run_dest_write_path_eltwise(
        path,
        tiles_per_bank,
        DataFormat.Float32,
        DestWriteKernel.Copy,
        stamp_sync=False,
        srcs_opt=srcs_opt,
    )


# Writer-only sweep: Float32 and Float16_b L1 inputs into a 32-bit Dest; path B uses its fastest issue (BOTH).
WRITER_ONLY_FORMATS = [
    (DataFormat.Float32, DataFormat.Float32),
    (DataFormat.Float16_b, DataFormat.Float32),
]


def writer_only_srcs_opt(path: DestWritePath) -> SrcsIssueOpt:
    return SrcsIssueOpt.BOTH if path.srcs_writer else SrcsIssueOpt.BASE


@parametrize(
    path=list(DestWritePath),
    tiles_per_bank=TILES_PER_BANK_SWEEP,
    formats=WRITER_ONLY_FORMATS,
)
def test_dest_write_path_writer_only_quasar(path, tiles_per_bank, formats):
    l1_format, out_format = formats
    run_dest_write_path_eltwise(
        path,
        tiles_per_bank,
        l1_format,
        DestWriteKernel.Copy,
        out_format=out_format,
        stamp_sync=False,
        srcs_opt=writer_only_srcs_opt(path),
        writer_only=True,
    )


if __name__ == "__main__":
    summarize_bank_stamps(
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else TestConfig.resolve_artefacts_path() / STAMPS_FILE_NAME
    )
