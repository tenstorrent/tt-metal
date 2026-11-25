# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Microbench: the combine accumulator consuming fp8 expert rows (flat_routed_expert FP8_Y_PLAN.md) vs bf16 rows (today).

One accumulator core, rows landed in L1 (a pool of P slots, reused), each row added into its token's int32 sum with
combine_snap (q = 2^-14), as routed_experts_combine's accumulator does. Kernels: kernels/combine_consume_*.cpp.

Wire rows (GLM-5.3, H = 6144, 32 down cores of 192 columns):
  bf16: [H bf16][32 B trailer]                                         12320 B (12352 a slot)
  fp8:  [H e4m3][32 cells x 16 B: 12 e8m0 bytes (one per 16 columns) + 4 spare][32 B trailer]   6688 B (6720)

The fp8 consumer: a row is 6 flat 1024-code "tiles"; 16 consecutive codes are one face row (one DEST row). The reader
writes each face row's factor (its e8m0 scale x 2^10, FP16) in column 0 of an FP16 factor tile, and the compute does
codes x factor on the FPU, srcB's column 0 broadcast row by row (ELWMUL SRCB_BCAST_COL, as sparse_sdpa's fused
dequant: exact at LoFi), with the gate and 2^-10 folded into the snap's factor. Any scale block that is a multiple of
16 columns (1x32, 1x96) fits. FP16 because fp8 unpacks to FP16 in srcA and the FPU takes one exponent family. The
factor is clamped to FP16's normal range: exact for the snap (q = 2^-14), as a block whose factor would underflow has
|y| < 2^-15.2 (every term rounds to 0), and overflow needs |y| > 14336. The unpacker flushes e4m3 subnormal codes
(|q| < 2^-6) to 0: the golden does too (the sender need not produce them).

Run the perf comparison (device profiler zones CMB-RD / CMB-CMP, cycles a row):
  TT_METAL_DEVICE_PROFILER=1 python tests/ttnn/unit_tests/operations/experimental/moe_block/routed_experts_combine/test_combine_fp8_consume_bench.py
"""

import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn

HERE = Path(__file__).resolve().parent
KERNELS = "tests/ttnn/unit_tests/operations/experimental/moe_block/routed_experts_combine/kernels"
_spec = importlib.util.spec_from_file_location(
    "y_fp8_quant_test", HERE.parent / "flat_routed_expert" / "test_flat_y_fp8_quant.py"
)
quant = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(quant)

H = 6144
ROW_TILES = H // 1024
CORES = 32  # down cores (192 columns each)
CELLS_BYTES = CORES * 16
POOL = 16  # landing slots
TOKENS = 8
Q = 2.0**-14


def slot_bytes(fp8):
    raw = H + (CELLS_BYTES if fp8 else 0) + 32 if fp8 else 2 * H + 32
    return -(-raw // 64) * 64


def make_rows(seed=0):
    """POOL expert output rows y (fp32 [POOL, H], per-row magnitudes 2^-6 .. 2^3) and their gates (bf16)."""
    g = torch.Generator().manual_seed(seed)
    y = torch.randn(POOL, H, generator=g) * torch.pow(2.0, torch.randint(-6, 4, (POOL, 1), generator=g).float())
    gates = torch.rand(POOL, generator=g).to(torch.bfloat16)
    return y, gates


def quantize(y, block):
    """fp8 codes [POOL, H] uint8, per-16 exponents [POOL, H / 16] uint8 (the e8m0 bytes of each block's scale)."""
    q = quant.BLOCK
    quant.BLOCK = block  # (golden reads the block width)
    codes, scales = quant.golden(y, block)
    quant.BLOCK = q
    e = (scales.view(torch.int32) >> 23).to(torch.uint8)  # [POOL, H / block]
    return codes, e.repeat_interleave(block // 16, dim=1)


def wire_rows(y, gates, fp8, block):
    sb = slot_bytes(fp8)
    rows = torch.zeros(POOL, sb, dtype=torch.uint8)
    if fp8:
        codes, e16 = quantize(y, block)
        rows[:, :H] = codes
        cells = torch.zeros(POOL, CORES, 16, dtype=torch.uint8)
        cells[:, :, :12] = e16.reshape(POOL, CORES, 12)
        rows[:, H : H + CELLS_BYTES] = cells.reshape(POOL, -1)
        trailer = H + CELLS_BYTES
    else:
        rows[:, : 2 * H] = y.to(torch.bfloat16).view(torch.uint8).reshape(POOL, -1)
        trailer = 2 * H
    rows[:, trailer + 4 : trailer + 6] = gates.view(torch.uint8).reshape(POOL, 2)  # trailer word 1: the gate's bits
    return rows


def golden(y, gates, fp8, block, n_rows):
    """The accumulator's output (bf16 [TOKENS, H]) and the unquantized reference sum (fp32)."""
    if fp8:
        codes, e16 = quantize(y, block)
        # the factor tile: scale x 2^10, its FP16 exponent clamped to the normal range (1..30)
        field = torch.clamp(e16.int() - 127 + 10 + 15, 1, 30)
        scale = torch.pow(2.0, field.float() - 15).repeat_interleave(16, dim=1)
        # (the unpacker flushes e4m3 subnormals, exponent field 0, to 0)
        rows = torch.where((codes & 0x78) == 0, 0.0, codes.view(torch.float8_e4m3fn).float()) * scale  # exact
        factor = gates.float() * 2.0**4
    else:
        rows = y.to(torch.bfloat16).float() * gates.float()[:, None]  # (bf16 x bf16: exact in fp32)
        factor = torch.full((POOL,), 2.0**14)
    counts = torch.zeros(TOKENS, H, dtype=torch.int64)
    ref = torch.zeros(TOKENS, H, dtype=torch.float64)
    for n in range(n_rows):
        s, t = n % POOL, n % TOKENS
        counts[t] += torch.round((rows[s] * factor[s]).float()).to(torch.int64)  # (fp32 product, ties to even)
        ref[t] += y[s].double() * gates[s].double()
    return (counts.float() * Q).to(torch.bfloat16), ref


def run(device, fp8, block=32, n_rows=256, fidelity=ttnn.MathFidelity.HiFi4, seed=0, build_factors=True):
    y, gates = make_rows(seed)
    sb = slot_bytes(fp8)
    t_in = ttnn.from_torch(
        wire_rows(y, gates, fp8, block),
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

    def cb(idx, fmt, page, pages):
        return ttnn.CBDescriptor(
            total_size=pages * page,
            core_ranges=core,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=fmt, page_size=page)],
        )

    row_fmt, row_page = (ttnn.DataType.FP8_E4M3, 1024) if fp8 else (ttnn.bfloat16, 2048)
    cbs = [
        cb(0, row_fmt, row_page, 2 * ROW_TILES),
        cb(1, ttnn.bfloat16, 2048, 2),
        cb(2, ttnn.uint16, 2048, 2 * ROW_TILES),  # (FP16 in the kernel: the JIT takes no FP16 CB next to BF16 ones)
        cb(3, ttnn.uint32, 64, 4),
        cb(10, ttnn.uint32, sb, POOL),
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
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=f"{KERNELS}/combine_consume_reader.cpp",
            core_ranges=core,
            compile_time_args=[
                int(fp8),
                n_rows,
                POOL,
                sb,
                H if fp8 else 2 * H,
                ROW_TILES,
                TOKENS,
                CORES,
                int(build_factors),
            ]
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
            compile_time_args=[int(fp8), n_rows, ROW_TILES, TOKENS],
            runtime_args=[],
            config=compute_config,
        ),
    ]
    ttnn.generic_op([t_in, t_out], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
    got = ttnn.to_torch(t_out)
    want, ref = golden(y, gates, fp8, block, n_rows)
    return got, want, ref


def compare(got, want, ref):
    gb, wb = got.view(torch.int16).int(), want.view(torch.int16).int()
    ulp = (gb - wb).abs()
    rel = ((got.double() - ref).norm() / ref.norm()).item()
    return int((ulp != 0).sum()), int(ulp.max()), rel


CASES = [
    pytest.param(False, 32, id="bf16"),
    pytest.param(True, 32, id="fp8-1x32"),
    pytest.param(True, 96, id="fp8-1x96"),
]


@pytest.mark.parametrize("fp8, block", CASES)
def test_combine_fp8_consume(device, fp8, block):
    if ttnn.get_arch_name() != "blackhole":
        pytest.skip("Blackhole only")
    got, want, ref = run(device, fp8, block)
    diff, max_ulp, rel = compare(got, want, ref)
    logger.info(
        f"{'fp8 1x' + str(block) if fp8 else 'bf16'}: {diff} of {got.numel()} outputs differ from the golden "
        f"(max {max_ulp} bf16 ulp); relative error vs the unquantized sum {rel:.3e}"
    )
    assert max_ulp <= 1, f"max {max_ulp} ulp off the golden"


def _zone_cycles(n_rows):
    import csv
    import collections

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
    variant = os.environ.get("VARIANT", "bf16")  # bf16 | fp8-1x32 | fp8-1x96
    fid = getattr(ttnn.MathFidelity, os.environ.get("FIDELITY", "HiFi4"))
    fp8 = variant.startswith("fp8")
    block = int(variant.split("x")[-1]) if fp8 else 32
    dev = ttnn.open_device(device_id=0)
    try:
        got, want, ref = run(
            dev, fp8, block, n_rows=n_rows, fidelity=fid, build_factors=os.environ.get("NO_FACTORS") != "1"
        )
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
