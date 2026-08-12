# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated bake-off for rms_norm perf idea I4 — the reduce-scaler preparation.

WHAT IS ISOLATED. Only the reader's OPENING sequence plus the one consumer that
makes its ordering constraint real:

    reader   prepare_reduce_scaler(1/W) ; read this core's C input tiles of a
             DRAM-interleaved bf16 TILE tensor behind ONE barrier
    compute  sumsq (eltwise Mul + DEST PerRow accumulation over C tiles)
             -> reduce<SUM, REDUCE_ROW, cb_stat_sq, cb_scaler, cb_stat_partial>
    writer   store the stat tile

Everything else the real op does (gamma, ragged-tail mask, hidden chunking,
sharding, multi-block, the cross-core combine, the apply, the untilize) is REMOVED,
so the measured delta belongs to where/how the scaler tile is built.

GEOMETRY. `cores` cores (default 110 = the focus shape's group size), each owning C
hidden tiles of ONE tile-row, so total DRAM traffic and per-core transaction count
match the focus geometry ((1,1,32,7168) interleaved -> 110 cores x C=2..3 tiles).
Core i reads tiles [i*C, (i+1)*C) of a [32, cores*C*32] input and writes tile i of
a [32*cores, 32] output.

PRECISION CONTRACT (fixed, identical for every variant): input/output bf16,
TILE layout, MathFidelity.HiFi2, fp32_dest_acc_en=False, scaler CB bf16.

CORRECTNESS GATE. The reduce output's COLUMN 0 is `sum_c x[r,c]^2 / W_true` for
this core's slice — the scaler IS the divisor, so a wrong scaler is a scale error,
not a noise error. The gate therefore checks the ratio to a torch fp32 reference
(max |ratio-1|), not just PCC.
"""

from __future__ import annotations

import struct
from pathlib import Path

import torch
import ttnn

TILE = 32
KERNEL_DIR = Path(__file__).parent / "kernels"

CB_INPUT_TILES = 1
CB_SCALER = 2
CB_WMASK = 3
CB_STAT_SQ = 5
CB_STAT_PARTIAL = 7

BF16_TILE_BYTES = 2 * TILE * TILE

# variant name -> compile-time id (see kernels/bench_reader.cpp)
VARIANTS = {
    "prep_first": 0,  # BASELINE — the op's current order
    "after_issue": 1,
    "after_push": 2,
    "cheap_first": 3,
    "cheap_after_push": 4,
    "writer_prep": 5,
    "cheap_poisoned": 6,  # correctness probe (ns includes the poison fill)
}
BASELINE = "prep_first"


def _core_range_set(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in cores])


def _cb(index, core_ranges, num_pages, page_size, data_format):
    return ttnn.CBDescriptor(
        total_size=num_pages * page_size,
        core_ranges=core_ranges,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=data_format, page_size=page_size)],
    )


def core_list(n, grid_w=11):
    """`n` cores in row-major order over a grid_w-wide grid (the focus geometry's 11x10)."""
    return [(i % grid_w, i // grid_w) for i in range(n)]


def make_tensors(device, n_cores, c_tiles):
    torch.manual_seed(0)
    w = n_cores * c_tiles * TILE
    torch_in = torch.randn((1, 1, TILE, w), dtype=torch.float32)
    tt_in = ttnn.from_torch(torch_in.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, TILE * n_cores, TILE]),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
        ttnn.DRAM_MEMORY_CONFIG,
    )
    return torch_in, tt_in, tt_out


def descriptor(variant, tt_in, tt_out, n_cores, c_tiles, inv_w, mask=False):
    cores = core_list(n_cores)
    crs = _core_range_set(cores)
    vid = VARIANTS[variant]
    mid = 1 if mask else 0
    nc = 1 + mid
    inv_w_bits = struct.unpack("<I", struct.pack("<f", inv_w))[0]

    cbs = [
        _cb(CB_INPUT_TILES, crs, c_tiles, BF16_TILE_BYTES, ttnn.bfloat16),
        _cb(CB_SCALER, crs, 1, BF16_TILE_BYTES, ttnn.bfloat16),
        _cb(CB_WMASK, crs, 1, BF16_TILE_BYTES, ttnn.bfloat16),
        _cb(CB_STAT_SQ, crs, nc, BF16_TILE_BYTES, ttnn.bfloat16),
        _cb(CB_STAT_PARTIAL, crs, 1, BF16_TILE_BYTES, ttnn.bfloat16),
    ]

    reader_ct = [c_tiles, vid, mid]
    reader_ct.extend(ttnn.TensorAccessorArgs(tt_in).get_compile_time_args())
    writer_ct = [vid, mid]
    writer_ct.extend(ttnn.TensorAccessorArgs(tt_out).get_compile_time_args())
    compute_ct = [c_tiles, mid]

    reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for i, (cx, cy) in enumerate(cores):
        reader_rt[cx][cy] = [tt_in.buffer_address(), i * c_tiles, inv_w_bits]
        writer_rt[cx][cy] = [tt_out.buffer_address(), i, inv_w_bits]
        compute_rt[cx][cy] = []

    # THE user precision contract, identical for every variant.
    compute_cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False)

    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "bench_reader.cpp"),
            core_ranges=crs,
            compile_time_args=reader_ct,
            runtime_args=reader_rt,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "bench_writer.cpp"),
            core_ranges=crs,
            compile_time_args=writer_ct,
            runtime_args=writer_rt,
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "bench_compute.cpp"),
            core_ranges=crs,
            compile_time_args=compute_ct,
            runtime_args=compute_rt,
            config=compute_cfg,
        ),
    ]
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)


def reference(torch_in, n_cores, c_tiles, inv_w):
    """Per-core column-0 truth: sum over this core's C tiles of x^2, times inv_w."""
    x = torch_in[0, 0].to(torch.float32)  # [32, W]
    per_core = []
    for i in range(n_cores):
        sl = x[:, i * c_tiles * TILE : (i + 1) * c_tiles * TILE]
        per_core.append((sl * sl).sum(dim=1) * inv_w)
    return torch.stack(per_core)  # [n_cores, 32]


def check(out_tensor, torch_in, n_cores, c_tiles, inv_w):
    got = ttnn.to_torch(out_tensor)[0, 0].to(torch.float32)  # [32*n_cores, 32]
    col0 = torch.stack([got[i * TILE : (i + 1) * TILE, 0] for i in range(n_cores)])
    ref = reference(torch_in, n_cores, c_tiles, inv_w)
    ratio = (col0 / ref).flatten()
    max_dev = float((ratio - 1.0).abs().max())
    pcc = float(torch.corrcoef(torch.stack([col0.flatten(), ref.flatten()]))[0, 1])
    other = torch.stack([got[i * TILE : (i + 1) * TILE, 1:] for i in range(n_cores)])
    return {
        "max_ratio_dev": max_dev,
        "pcc": pcc,
        "cols_1_31_absmax": float(other.abs().max()),
        "cols_1_31_nonfinite": int((~torch.isfinite(other)).sum()),
    }


def device_kernel_ns(device):
    ttnn.synchronize_device(device)
    ttnn.ReadDeviceProfiler(device)
    per_chip = ttnn.get_latest_programs_perf_data() or {}
    ns = None
    for programs in per_chip.values():
        for program in programs:
            results = getattr(program, "program_analyses_results", None) or {}
            entry = results.get("DEVICE KERNEL DURATION [ns]")
            if entry is None:
                continue
            d = float(entry.duration)
            ns = d if ns is None else max(ns, d)
    return ns


def run(device, variant, n_cores=110, c_tiles=2, w_true=None, verify=True, mask=False):
    """One fresh dispatch of `variant`; returns (ns, correctness dict)."""
    torch_in, tt_in, tt_out = make_tensors(device, n_cores, c_tiles)
    w_true = w_true if w_true is not None else n_cores * c_tiles * TILE
    inv_w = 1.0 / float(w_true)
    desc = descriptor(variant, tt_in, tt_out, n_cores, c_tiles, inv_w, mask=mask)
    ttnn.ReadDeviceProfiler(device)  # flush the from_torch prep out of the log
    out = ttnn.generic_op([tt_in, tt_out], desc)
    ns = device_kernel_ns(device)
    stats = check(out, torch_in, n_cores, c_tiles, inv_w) if verify else None
    del out, tt_in, tt_out
    return ns, stats
