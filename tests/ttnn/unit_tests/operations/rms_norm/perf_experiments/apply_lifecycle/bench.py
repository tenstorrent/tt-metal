# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated bake-off host for rms_norm perf round 2 / idea I11 — the APPLY pass's
output-CB LIFECYCLE (PerTile vs PerChunk vs bulk) and DEST-lane BLOCKING.

Concept isolation: every operand is a resident L1 shard pinned zero-copy under a CB,
so there is NO DRAM traffic, no reader/writer, no reduction and no multicast — the
measured `DEVICE KERNEL DURATION [ns]` delta is the apply's compute alone.

Per-chunk cost is recovered as the SLOPE (T(hi) - T(lo)) / (hi - lo) over ITERS,
because the focus geometry's chunk (3 tiles) is far below a dispatch's launch floor.

Precision contract pinned and IDENTICAL for every option: bf16 / TILE / HiFi2 /
fp32_dest_acc_en=False.
"""

from __future__ import annotations

from pathlib import Path

import torch
import ttnn

TILE = 32
TILE_BYTES_BF16 = 2048

KERNEL = str(Path(__file__).resolve().parent / "apply_lifecycle_kernel.cpp")

CB_IN = 0
CB_RSTD = 1
CB_GAMMA = 2
CB_NORMED = 14
CB_OUT = 16

P_PERTILE, P_BULK, P_PERCHUNK, P_CALLER_STRIDED = 0, 1, 2, 3

RSTD_JUNK = 3.0
GAMMA_JUNK = 5.0


def _cores(grid):
    gx, gy = grid
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))])


def _shard_cfg(shard_h, shard_w, grid):
    return ttnn.create_sharded_memory_config(
        shape=(shard_h, shard_w),
        core_grid=_cores(grid),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _scratch_cb(index, num_pages, grid):
    return ttnn.CBDescriptor(
        total_size=num_pages * TILE_BYTES_BF16,
        core_ranges=_cores(grid),
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.bfloat16, page_size=TILE_BYTES_BF16)
        ],
    )


def make_inputs(device, rows_t, cols, grid=(1, 1), seed=0):
    torch.manual_seed(seed)
    ncores = grid[0] * grid[1]
    h, w = rows_t * TILE, cols * TILE

    x = torch.randn((h, w), dtype=torch.float32).to(torch.bfloat16)
    rstd_col = (torch.rand((h, 1), dtype=torch.float32) + 0.5).to(torch.bfloat16)
    gamma_row = torch.randn((1, w), dtype=torch.float32).to(torch.bfloat16)

    rstd_tile = torch.full((h, TILE), RSTD_JUNK, dtype=torch.bfloat16)
    rstd_tile[:, 0:1] = rstd_col
    gamma_tile = torch.full((TILE, w), GAMMA_JUNK, dtype=torch.bfloat16)
    gamma_tile[0:1, :] = gamma_row

    def to_dev(t, shard_h, shard_w):
        stacked = torch.cat([t] * ncores, dim=0) if ncores > 1 else t
        return ttnn.from_torch(
            stacked.reshape(1, 1, *stacked.shape),
            dtype=ttnn.bfloat16,
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


def run(device, tt, hw, *, rows_t, cols, blk, iters, out_policy, normed_policy, grid=(1, 1)):
    h, w = hw
    ncores = grid[0] * grid[1]
    n_block = rows_t * cols
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, h * ncores, w]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, _shard_cfg(h, w, grid)
    )
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False)
    cbs = [
        ttnn.cb_descriptor_from_sharded_tensor(CB_IN, tt["x"]),
        ttnn.cb_descriptor_from_sharded_tensor(CB_RSTD, tt["rstd"]),
        ttnn.cb_descriptor_from_sharded_tensor(CB_GAMMA, tt["gamma"]),
        ttnn.cb_descriptor_from_sharded_tensor(CB_OUT, out),
        # cb_normed is sized on the whole chunk, exactly as the op sizes it (R*WC).
        _scratch_cb(CB_NORMED, n_block, grid),
    ]
    kernel = ttnn.KernelDescriptor(
        kernel_source=KERNEL,
        core_ranges=_cores(grid),
        compile_time_args=[rows_t, cols, blk, iters, out_policy, normed_policy],
        config=cfg,
    )
    desc = ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=cbs)
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
    h, w = hw
    got = ttnn.to_torch(out_tensor).reshape(-1, w)[:h, :].to(torch.float32)
    ratio = got / torch.where(ref.abs() > 1e-3, ref, torch.full_like(ref, float("nan")))
    ratio = ratio[~torch.isnan(ratio)]
    return _pcc(got, ref), float(ratio.median()), float((got - ref).abs().max())
