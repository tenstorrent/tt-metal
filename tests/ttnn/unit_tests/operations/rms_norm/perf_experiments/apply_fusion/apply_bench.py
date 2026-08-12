# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated bake-off host for rms_norm design lamp P4 / idea I5 — fuse the apply's
two block-wide FPU passes into one and delete the `cb_normed` L1 round trip.

Concept isolation: every operand is a resident L1 shard pinned zero-copy under a CB,
so there is NO DRAM traffic, no reader/writer, no reduction and no multicast — the
measured `DEVICE KERNEL DURATION [ns]` delta is the apply's compute alone.

The focus geometry's block is THREE tiles, far below a dispatch's launch floor, so
each variant is measured at two iteration counts and the per-block cost is recovered
as the SLOPE  (T(iters_hi) - T(iters_lo)) / (iters_hi - iters_lo).

The precision contract is pinned to the perf loose case and is IDENTICAL for every
variant: bf16 activations / bf16 gamma / TILE / fp32_dest_acc_en=False /
MathFidelity.HiFi2.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import ttnn

TILE = 32
TILE_BYTES_BF16 = 2048

# The precision contract is a fixed INPUT to every variant, never a lever. `dtype` picks
# WHICH pinned contract is under test: the perf loose case (bf16 + fp32_dest_acc_en=False)
# or the fp32 corner (float32 + fp32_dest_acc_en=True), which is the same kernel code path
# in the real op. Both are run identically for every option.
DTYPES = {
    # name -> (torch dtype, ttnn dtype, tile bytes, fp32_dest_acc_en, math_fidelity)
    "bf16": (torch.bfloat16, ttnn.bfloat16, 2048, False, "HiFi2"),
    "fp32": (torch.float32, ttnn.float32, 4096, True, "HiFi2"),
    # HiFi4 is what actually resolves an fp32 operand in the FPU (the src registers hold
    # ~11 mantissa bits per pass), so the fp32 precision question has to be asked here too.
    "fp32hifi4": (torch.float32, ttnn.float32, 4096, True, "HiFi4"),
}

KERNEL = str(Path(__file__).resolve().parent / "apply_kernel.cpp")

CB_IN = 0
CB_RSTD = 1
CB_GAMMA = 2
CB_EXP = 3
CB_EXP2 = 4
CB_NORMED = 14
CB_OUT = 16

VARIANTS = {
    "baseline": 0,
    "fused_rstd": 1,
    "fused_gamma": 2,
    "fused_sfpu": 3,
    "fold_gamma": 4,
}

# Junk written into the structurally-invalid positions of the two broadcast operands
# (rstd is COLUMN-0-valid, gamma is ROW-0-valid). A variant that reads the wrong
# datum picks these up and fails the correctness gate instead of passing quietly.
RSTD_JUNK = 3.0
GAMMA_JUNK = 5.0


def _cores(grid):
    gx, gy = grid
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))])


def _shard_cfg(shard_h, shard_w, grid):
    """`grid` cores, each owning a [shard_h, shard_w] HEIGHT shard."""
    return ttnn.create_sharded_memory_config(
        shape=(shard_h, shard_w),
        core_grid=_cores(grid),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _scratch_cb(index, num_pages, grid, dtype="bf16"):
    _, tt_dtype, page, _, _ = DTYPES[dtype]
    return ttnn.CBDescriptor(
        total_size=num_pages * page,
        core_ranges=_cores(grid),
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=tt_dtype, page_size=page)],
    )


def make_inputs(device, rows_t, cols, grid=(1, 1), seed=0, dtype="bf16"):
    """Per-core: an (rows_t x cols) tile block, rows_t column-0-valid rstd tiles, cols
    row-0-valid gamma tiles. Replicated across `grid` cores by height-sharding."""
    torch.manual_seed(seed)
    ncores = grid[0] * grid[1]
    h, w = rows_t * TILE, cols * TILE
    torch_dtype, tt_dtype, _, _, _ = DTYPES[dtype]

    x = torch.randn((h, w), dtype=torch.float32).to(torch_dtype)
    rstd_col = (torch.rand((h, 1), dtype=torch.float32) + 0.5).to(torch_dtype)
    gamma_row = torch.randn((1, w), dtype=torch.float32).to(torch_dtype)

    rstd_tile = torch.full((h, TILE), RSTD_JUNK, dtype=torch_dtype)
    rstd_tile[:, 0:1] = rstd_col
    gamma_tile = torch.full((TILE, w), GAMMA_JUNK, dtype=torch_dtype)
    gamma_tile[0:1, :] = gamma_row

    def to_dev(t, shard_h, shard_w):
        stacked = torch.cat([t] * ncores, dim=0) if ncores > 1 else t
        return ttnn.from_torch(
            stacked.reshape(1, 1, *stacked.shape),
            dtype=tt_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=_shard_cfg(shard_h, shard_w, grid),
        )

    tt = {
        "x": to_dev(x, h, w),
        "rstd": to_dev(rstd_tile, h, TILE),
        "gamma": to_dev(gamma_tile, TILE, w),
    }
    ref = (x.to(torch.float32) * rstd_col.to(torch.float32)) * gamma_row.to(torch.float32)
    return tt, ref, (h, w)


def create_program_descriptor(
    tt, out, *, rows_t, cols, variant, blk, iters, out_bulk, reconfig, grid, cfg, dtype="bf16"
):
    n_block = rows_t * cols
    ct = [rows_t, cols, VARIANTS[variant], blk, iters, 1 if out_bulk else 0, int(reconfig)]

    cbs = [
        ttnn.cb_descriptor_from_sharded_tensor(CB_IN, tt["x"]),
        ttnn.cb_descriptor_from_sharded_tensor(CB_RSTD, tt["rstd"]),
        ttnn.cb_descriptor_from_sharded_tensor(CB_GAMMA, tt["gamma"]),
        ttnn.cb_descriptor_from_sharded_tensor(CB_OUT, out),
    ]
    # Only the variant's own scratch is declared, so the L1 cost of each option is honest.
    if variant == "baseline":
        cbs.append(_scratch_cb(CB_NORMED, n_block, grid, dtype))
    elif variant == "fused_rstd":
        cbs.append(_scratch_cb(CB_EXP, rows_t, grid, dtype))
    elif variant == "fused_gamma":
        cbs.append(_scratch_cb(CB_EXP, cols, grid, dtype))
    elif variant == "fold_gamma":
        cbs.append(_scratch_cb(CB_EXP, cols, grid, dtype))
        cbs.append(_scratch_cb(CB_EXP2, 1, grid, dtype))

    compute = ttnn.KernelDescriptor(
        kernel_source=KERNEL,
        core_ranges=_cores(grid),
        compile_time_args=ct,
        config=cfg,
    )
    return ttnn.ProgramDescriptor(kernels=[compute], semaphores=[], cbs=cbs)


def run(device, tt, hw, *, rows_t, cols, variant, blk=1, iters=1, out_bulk=True, reconfig=1, grid=(1, 1), dtype="bf16"):
    h, w = hw
    ncores = grid[0] * grid[1]
    _, tt_dtype, _, fp32_dest, fidelity = DTYPES[dtype]
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, h * ncores, w]),
        tt_dtype,
        ttnn.TILE_LAYOUT,
        device,
        _shard_cfg(h, w, grid),
    )
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=getattr(ttnn.MathFidelity, fidelity), fp32_dest_acc_en=fp32_dest)
    desc = create_program_descriptor(
        tt,
        out,
        rows_t=rows_t,
        cols=cols,
        variant=variant,
        blk=blk,
        iters=iters,
        out_bulk=out_bulk,
        reconfig=reconfig,
        grid=grid,
        cfg=cfg,
        dtype=dtype,
    )
    return ttnn.generic_op([tt["x"], tt["rstd"], tt["gamma"], out], desc)


def _device_ns():
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


def measure(device, tt, hw, **kw):
    """One fresh dispatch; returns (ns, output_tensor)."""
    ttnn.ReadDeviceProfiler(device)
    out = run(device, tt, hw, **kw)
    ttnn.synchronize_device(device)
    ttnn.ReadDeviceProfiler(device)
    return _device_ns(), out


def _pcc(a, b):
    a = a.flatten().to(torch.float32)
    b = b.flatten().to(torch.float32)
    if torch.allclose(a, b):
        return 1.0
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def check(out_tensor, ref, hw, grid=(1, 1)):
    """PCC + median(got/true). A fused path that silently loses a mantissa shows up as
    a uniform scale shift, which PCC hides — hence the ratio median tripwire."""
    h, w = hw
    got = ttnn.to_torch(out_tensor).reshape(-1, w)[:h, :].to(torch.float32)
    ratio = got / torch.where(ref.abs() > 1e-3, ref, torch.full_like(ref, float("nan")))
    ratio = ratio[~torch.isnan(ratio)]
    return _pcc(got, ref), float(ratio.median()), float((got - ref).abs().max())
