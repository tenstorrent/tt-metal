# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Microbench: the combine accumulator consuming quantized expert rows (common/kernels/quantized_row.hpp) through the
consumer API - quantized_row::RowStager (reader, kernels/dataflow/quantized_row_stage.hpp) and quantized_row::RowWeigher
(compute, kernels/compute/quantized_row_weigh.hpp) - vs bf16 rows (today).

One accumulator core, rows landed in L1 (a pool of P slots, reused), each row added into its token's int32 sum with
combine_snap (q = 2^-14), as routed_experts_combine's accumulator does. Kernels: kernels/combine_consume_*.cpp.

Rows (GLM-5.3, H = 6144, 32 producer cores of 192 columns), then the 32-byte trailer (word 1: the gate's bf16 bits):
  bf16:  [H bf16]                                                            12320 B
  fp8:   [H e4m3][32 cells x 16 B: 12 e8m0 bytes, one per 16 columns]         6688 B
  int8:  [H sign-magnitude int8][32 cells: 12 Bfp8_b exponent bytes]           6688 B

The goldens model the device's arithmetic exactly (outputs within 1 bf16 ulp: the count -> bf16 readout). Each
format's quantizer is the producer's (the flat expert's down cores, y_quant.hpp): moe_block/quantized_row_golden.py.

Perf (device profiler zones CMB-RD / CMB-CMP, cycles a row):
  TT_METAL_DEVICE_PROFILER=1 VARIANT=int8-1x32 FIDELITY=HiFi4 python <this file>
"""

import importlib.util
import os
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn

HERE = Path(__file__).resolve().parent
KERNELS = "tests/ttnn/unit_tests/operations/experimental/moe_block/routed_experts_combine/kernels"
_spec = importlib.util.spec_from_file_location("quantized_row_golden", HERE.parent / "quantized_row_golden.py")
qgolden = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(qgolden)

_RowFormat = ttnn._ttnn.operations.experimental.moe_block.RoutedRowFormat
FORMATS = {"bf16": _RowFormat.Bf16.value, "fp8": _RowFormat.Fp8E4m3.value, "int8": _RowFormat.Int8.value}  # (kernel arg)
H = 6144
ROW_TILES = H // 1024
PRODUCERS = 32
CELL_EXPONENTS = H // PRODUCERS // 16
POOL = 16  # landing slots
TOKENS = 8
Q = 2.0**-14


def row_bytes(fmt):
    return 2 * H if fmt == "bf16" else H + PRODUCERS * 16


def slot_bytes(fmt):
    return -(-(row_bytes(fmt) + 32) // 64) * 64


def make_rows(seed=0):
    """POOL expert output rows y (fp32 [POOL, H], per-row magnitudes 2^-6 .. 2^3) and their gates (bf16)."""
    g = torch.Generator().manual_seed(seed)
    y = torch.randn(POOL, H, generator=g) * torch.pow(2.0, torch.randint(-6, 4, (POOL, 1), generator=g).float())
    gates = torch.rand(POOL, generator=g).to(torch.bfloat16)
    return y, gates


def quantize(y, fmt, block):
    """(value bytes, exponent bytes a 16 values, the values the consumer reads): moe_block/quantized_row_golden.py"""
    return qgolden.quantize(y, fmt, block)


def wire_rows(y, gates, fmt, block):
    rows = torch.zeros(POOL, slot_bytes(fmt), dtype=torch.uint8)
    if fmt == "bf16":
        rows[:, : 2 * H] = y.to(torch.bfloat16).view(torch.uint8).reshape(POOL, -1)
    else:
        codes, e16, _ = quantize(y, fmt, block)
        rows[:, :H] = codes
        cells = torch.zeros(POOL, PRODUCERS, 16, dtype=torch.uint8)
        cells[:, :, :CELL_EXPONENTS] = e16.reshape(POOL, PRODUCERS, CELL_EXPONENTS)
        rows[:, H : H + PRODUCERS * 16] = cells.reshape(POOL, -1)
    t = row_bytes(fmt)
    rows[:, t + 4 : t + 6] = gates.view(torch.uint8).reshape(POOL, 2)  # trailer word 1: the gate's bits
    return rows


def golden(y, gates, fmt, block, n_rows):
    """The accumulator's output (bf16 [TOKENS, H]) and the unquantized reference sum (fp64)."""
    if fmt == "fp8":
        codes, e16, _ = quantize(y, fmt, block)
        # the FP16 factor: the scale x 2^10, its exponent field clamped to 1 .. 30 (quantized_row.hpp)
        field = torch.clamp(e16.int() - 127 + 10 + 15, 1, 30)
        rows = codes.view(torch.float8_e4m3fn).float() * torch.pow(2.0, field.float() - 15).repeat_interleave(16, 1)
        factor = gates.float() * 2.0**4  # (gate x 2^-10 / q)
    else:
        rows = y.to(torch.bfloat16).float() if fmt == "bf16" else quantize(y, fmt, block)[2]
        rows = rows * gates.float()[:, None]  # (an 8-bit x 8-bit product: exact in fp32)
        factor = torch.full((POOL,), 2.0**14)
    counts = torch.zeros(TOKENS, H, dtype=torch.int64)
    ref = torch.zeros(TOKENS, H, dtype=torch.float64)
    for n in range(n_rows):
        s, t = n % POOL, n % TOKENS
        counts[t] += torch.round((rows[s] * factor[s]).float()).to(torch.int64)  # (fp32 product, ties to even)
        ref[t] += y[s].double() * gates[s].double()
    return (counts.float() * Q).to(torch.bfloat16), ref


ACCUMULATOR_COMPUTE = "ttnn/cpp/ttnn/operations/experimental/moe_block/routed_experts_combine/device/kernels/compute/accumulator_compute.cpp"


def run(device, fmt, block=32, n_rows=256, fidelity=ttnn.MathFidelity.HiFi4, seed=0, accumulator=False):
    """accumulator: the combine's own accumulator_compute.cpp (its named args, the reader speaking its meta protocol)
    instead of the bench's compute kernel."""
    y, gates = make_rows(seed)
    sb = slot_bytes(fmt)
    t_in = ttnn.from_torch(
        wire_rows(y, gates, fmt, block),
        dtype=ttnn.uint8,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    t_out = ttnn.from_torch(
        torch.zeros(TOKENS, H, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])

    def cb(idx, dt, page, pages):
        return ttnn.CBDescriptor(
            total_size=pages * page,
            core_ranges=core,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dt, page_size=page)],
        )

    # the row CB: quantized_row::row_page_bytes
    row_dt, row_page = {
        "bf16": (ttnn.bfloat16, 2048),
        "fp8": (ttnn.DataType.FP8_E4M3, 1024),
        "int8": (ttnn.bfloat8_b, 1088),
    }[fmt]
    cbs = [
        cb(0, row_dt, row_page, 2 * ROW_TILES),
        cb(1, ttnn.bfloat16, 2048, 2),
        cb(2, ttnn.uint16, 2048, 2 * ROW_TILES),  # (fp8's FP16 factors: the JIT takes no FP16 CB next to BF16 ones)
        cb(3, ttnn.uint32, 64, 4),
        cb(10, ttnn.uint32, sb, POOL + 1),  # (the pool, then the reader's cells scratch)
        cb(16, ttnn.bfloat16, 2048, 2 * ROW_TILES),
        cb(24, ttnn.int32, 4096, TOKENS * ROW_TILES),
    ]
    compute_config = ttnn.ComputeConfigDescriptor(math_fidelity=fidelity, fp32_dest_acc_en=True)
    u2d = [ttnn.UnpackToDestMode.Default] * 64
    u2d[24] = ttnn.UnpackToDestMode.UnpackToDestFp32
    compute_config.unpack_to_dest_mode = u2d
    reader_rt = ttnn.RuntimeArgs()
    reader_rt[0][0] = [t_in.buffer_address()]
    writer_rt = ttnn.RuntimeArgs()
    writer_rt[0][0] = [t_out.buffer_address()]
    f = FORMATS[fmt]
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=f"{KERNELS}/combine_consume_reader.cpp",
            core_ranges=core,
            compile_time_args=[f, n_rows, POOL, sb, H, PRODUCERS, TOKENS, int(accumulator)]
            + ttnn.TensorAccessorArgs(t_in).get_compile_time_args(),
            runtime_args=reader_rt,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=f"{KERNELS}/combine_consume_writer.cpp",
            core_ranges=core,
            compile_time_args=[ROW_TILES, TOKENS] + ttnn.TensorAccessorArgs(t_out).get_compile_time_args(),
            runtime_args=writer_rt,
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=f"{KERNELS}/combine_consume_compute.cpp",
            core_ranges=core,
            compile_time_args=[f, n_rows, H, TOKENS],
            runtime_args=[],
            config=compute_config,
        ),
    ]
    if accumulator:
        compute = ttnn.KernelDescriptor(
            kernel_source=ACCUMULATOR_COMPUTE,
            core_ranges=core,
            compile_time_args=[],
            runtime_args=[],
            config=compute_config,
        )
        compute.named_compile_time_args = [
            ("row_cb", 0),
            ("weight_cb", 1),
            ("factor_cb", 2),
            ("meta_cb", 3),
            ("output_cb", 16),
            ("accumulator_cb", 24),
            ("row_tiles", ROW_TILES),
            ("max_tokens_per_accumulator", TOKENS),
            ("snap_step_bits", 0x38800000),  # q = 2^-14
            ("row_format", f),
        ]
        kernels[2] = compute
    ttnn.generic_op([t_in, t_out], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
    got = ttnn.to_torch(t_out)
    want, ref = golden(y, gates, fmt, block, n_rows)
    return got, want, ref


def compare(got, want, ref):
    ulp = (got.view(torch.int16).int() - want.view(torch.int16).int()).abs()
    rel = ((got.double() - ref).norm() / ref.norm()).item()
    return int((ulp != 0).sum()), int(ulp.max()), rel


CASES = [
    pytest.param("bf16", 32, id="bf16"),
    pytest.param("fp8", 96, id="fp8-1x96"),
    pytest.param("int8", 16, id="int8-1x16"),
    pytest.param("int8", 32, id="int8-1x32"),
    pytest.param("int8", 96, id="int8-1x96"),
]


@pytest.mark.parametrize("accumulator", [False, True], ids=["bench", "accumulator_compute"])
@pytest.mark.parametrize("fmt, block", CASES)
def test_combine_quantized_row_consume(device, fmt, block, accumulator):
    """accumulator_compute: the combine's own accumulator_compute.cpp on the staged rows."""
    if ttnn.get_arch_name() != "blackhole":
        pytest.skip("Blackhole only")
    got, want, ref = run(device, fmt, block, accumulator=accumulator)
    diff, max_ulp, rel = compare(got, want, ref)
    logger.info(
        f"{fmt} 1x{block}: {diff} of {got.numel()} outputs differ from the golden (max {max_ulp} bf16 ulp); "
        f"relative error vs the unquantized sum {rel:.3e}"
    )
    assert max_ulp <= 1, f"max {max_ulp} ulp off the golden"


def _zone_cycles(n_rows):
    import collections
    import csv

    path = Path(os.environ.get("TT_METAL_HOME", ".")) / "generated/profiler/.logs/profile_log_device.csv"
    rows = list(csv.reader(open(path)))
    hdr = [h.strip() for h in rows[1]]
    ix = {h: i for i, h in enumerate(hdr)}
    starts, out = {}, collections.defaultdict(list)
    for r in rows[2:]:
        name = r[ix["zone name"]].strip()
        if not name.startswith("CMB-"):
            continue
        key = (r[ix["core_x"]], r[ix["core_y"]], r[ix["RISC processor type"]], name)
        t, typ = int(r[ix["time[cycles since reset]"]]), r[ix["type"]].strip()
        if typ == "ZONE_START":
            starts[key] = t
        elif key in starts:
            out[(name, key[2].strip())].append(t - starts.pop(key))
    return {k: v[-1] / n_rows for k, v in out.items()}


if __name__ == "__main__":
    n_rows = int(os.environ.get("ROWS", "512"))
    variant = os.environ.get("VARIANT", "bf16")  # bf16 | fp8-1x96 | int8-1x32 ...
    fid = getattr(ttnn.MathFidelity, os.environ.get("FIDELITY", "HiFi4"))
    fmt = variant.split("-")[0]
    block = int(variant.split("x")[-1]) if "-" in variant else 32
    dev = ttnn.open_device(device_id=0)
    try:
        got, want, ref = run(dev, fmt, block, n_rows=n_rows, fidelity=fid)
        diff, max_ulp, rel = compare(got, want, ref)
        ttnn.ReadDeviceProfiler(dev)
    finally:
        ttnn.close_device(dev)
    cyc = _zone_cycles(n_rows)
    per = ", ".join(f"{k[0]}/{k[1]} {v:.0f}" for k, v in sorted(cyc.items()))
    print(
        f"BENCH {variant} {os.environ.get('FIDELITY', 'HiFi4')} rows {n_rows}: cycles a row: {per} | "
        f"{diff} outputs off the golden (max {max_ulp} ulp), rel err vs unquantized {rel:.3e}"
    )
