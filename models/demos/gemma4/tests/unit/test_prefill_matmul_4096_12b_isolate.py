# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolate M=4096 prefill matmuls for Gemma-4 12B TP=8 shapes.

31B isolates used ``2048×5376…``. 12B long-4k is a single ``4096`` chunk with
``hidden=3840``; LoFi+auto / cutoff-reshape winners may not transfer.

Shipping reference (WH tall prefill):
  * gate_up / QKV — DRAM-in0 auto + LoFi
  * down / o_proj — cutoff-reshape 2D + LoFi (``prefill_linear_above_cutoff``)

PCC is vs HiFi2 ``auto`` (program-config / fidelity changes are not bit-exact).
Timing is opt-in via ``GEMMA4_MM_SWEEP=1``.

    unset TT_METAL_DEVICE_PROFILER
    MESH_DEVICE=1x8 GEMMA4_MM_SWEEP=1 pytest \\
        models/demos/gemma4/tests/unit/test_prefill_matmul_4096_12b_isolate.py \\
        -k 1x8 -sv --timeout=1800
"""

from __future__ import annotations

import math
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.gemma4.tt.dram_sharded import (
    _PREFILL_CUTOFF,
    TILE_SIZE,
    _find_largest_divisor,
    l1_block_sharded_memcfg,
    prefill_progcfg,
)

from ..test_factory import parametrize_mesh_with_fabric

_M = 4096
_PCC = 0.99
_REPEATS = 3
_TRACE_REPS = 20
# TP=8 shards for google/gemma-4-12B-it (H=3840, I=15360, heads=16, kv=8, hd=256).
_SHAPES = (
    ("gate_up", 3840, 3840),
    ("qkv", 3840, 1024),
    ("down", 1920, 3840),
    ("o_proj", 512, 3840),
)
# Ops that already ship cutoff-reshape at M>_PREFILL_CUTOFF.
_SHIP_RESHAPE = frozenset({"down", "o_proj"})


def _lofi():
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )


def _hifi2():
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )


def _device_grid(mesh_device):
    g = mesh_device.compute_with_storage_grid_size()
    return (int(g.x), int(g.y))


def _subblock(per_core_m, per_core_n, max_hw=8):
    best = (1, 1)
    for h in range(1, min(per_core_m, max_hw) + 1):
        if per_core_m % h:
            continue
        for w in range(1, min(per_core_n, max_hw // h) + 1):
            if per_core_n % w == 0 and h * w > best[0] * best[1]:
                best = (h, w)
    return best


def _2d_progcfg(m, k, n, grid, in0_block_w=None, sharded_in0=False):
    gx, gy = grid
    per_core_m = max(1, math.ceil(m / TILE_SIZE / gy))
    per_core_n = max(1, math.ceil(n / TILE_SIZE / gx))
    kt = math.ceil(k / TILE_SIZE)
    k_for_bw = kt // gx if sharded_in0 and gx > 0 and kt % gx == 0 else kt
    cap = min(8, k_for_bw)
    if in0_block_w is None:
        in0_block_w = _find_largest_divisor(k_for_bw, max_div=cap)
    if k_for_bw % in0_block_w:
        return None
    oh, ow = _subblock(per_core_m, per_core_n)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=in0_block_w,
        out_subblock_h=oh,
        out_subblock_w=ow,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=bool(sharded_in0),
    )


def _shard_grid_xy(memcfg):
    spec = memcfg.shard_spec
    if spec is None:
        return None
    box = spec.grid.bounding_box().grid_size()
    return (int(box.x), int(box.y))


def _upload(mesh_device, torch_t, *, dtype, mapper):
    return ttnn.from_torch(
        torch_t,
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper,
    )


def _host(t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()


def _time_trace(mesh_device, capture_fn, *, reps=_TRACE_REPS):
    out = capture_fn()
    ttnn.synchronize_device(mesh_device)
    out.deallocate(True)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out = capture_fn()
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    out.deallocate(True)
    t0 = time.perf_counter()
    for _ in range(reps):
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    us = (time.perf_counter() - t0) / reps * 1e6
    ttnn.release_trace(mesh_device, tid)
    return us


def _linear(act, weight, *, pc=None, ckc=None):
    kwargs = {"memory_config": ttnn.DRAM_MEMORY_CONFIG}
    if pc is not None:
        kwargs["program_config"] = pc
    if ckc is not None:
        kwargs["compute_kernel_config"] = ckc
    return ttnn.linear(act, weight, **kwargs)


def _run_arm(name, fn):
    try:
        out = fn()
        host = _host(out)
        out.deallocate(True)
        return name, host, None
    except Exception as exc:  # noqa: BLE001 — isolate must survive L1 clash
        logger.warning(f"arm {name} failed: {type(exc).__name__}: {exc}")
        return name, None, str(exc)


def _shipping_name(label: str) -> str:
    return "reshape_cutoff_lofi" if label in _SHIP_RESHAPE else "auto_lofi"


@parametrize_mesh_with_fabric(device_params_extra={"trace_region_size": 64_000_000})
@pytest.mark.parametrize("label,k,n", _SHAPES, ids=[s[0] for s in _SHAPES])
def test_prefill_matmul_4096_12b_isolate(label, k, n, mesh_device, reset_seeds):
    """PCC every arm vs HiFi2 auto. Timing matrix behind GEMMA4_MM_SWEEP=1."""
    if os.environ.get("GEMMA4_MM_SWEEP", "0").lower() not in ("1", "true", "yes"):
        pytest.skip("set GEMMA4_MM_SWEEP=1 to run the M=4096 12B matmul isolate")

    torch.manual_seed(0)
    mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    x_torch = torch.randn(1, 1, _M, k, dtype=torch.bfloat16) * 0.02
    w_torch = torch.randn(k, n, dtype=torch.bfloat16) * 0.02
    x = _upload(mesh_device, x_torch, dtype=ttnn.bfloat16, mapper=mapper)
    w = _upload(mesh_device, w_torch, dtype=ttnn.bfloat8_b, mapper=mapper)
    ckc_hifi = _hifi2()
    ckc_lofi = _lofi()
    grid = _device_grid(mesh_device)
    cutoff = _PREFILL_CUTOFF

    arms = []

    def auto():
        return _linear(x, w, ckc=ckc_hifi)

    arms.append(("auto", auto, None, False))

    def auto_lofi():
        return _linear(x, w, ckc=ckc_lofi)

    arms.append(("auto_lofi", auto_lofi, None, False))

    if _M % cutoff == 0:
        pc_cut = prefill_progcfg(cutoff, k, n)

        def reshape_cutoff(pc=pc_cut):
            x_r = ttnn.reshape(x, (1, _M // cutoff, cutoff, k))
            out_r = _linear(x_r, w, pc=pc, ckc=ckc_hifi)
            return ttnn.reshape(out_r, (1, 1, _M, n))

        arms.append(("reshape_cutoff", reshape_cutoff, pc_cut, False))

        def reshape_cutoff_lofi(pc=pc_cut):
            x_r = ttnn.reshape(x, (1, _M // cutoff, cutoff, k))
            out_r = _linear(x_r, w, pc=pc, ckc=ckc_lofi)
            return ttnn.reshape(out_r, (1, 1, _M, n))

        arms.append(("reshape_cutoff_lofi", reshape_cutoff_lofi, pc_cut, False))

    pc_narrow = prefill_progcfg(_M, k, n)

    def dram_narrow(pc=pc_narrow):
        return _linear(x, w, pc=pc, ckc=ckc_lofi)

    arms.append(("dram_narrow_lofi", dram_narrow, pc_narrow, False))

    pc_dram = _2d_progcfg(_M, k, n, grid)
    if pc_dram is not None:

        def dram_2d(pc=pc_dram):
            return _linear(x, w, pc=pc, ckc=ckc_lofi)

        arms.append(("dram_2d_lofi", dram_2d, pc_dram, False))

    in0_mc = l1_block_sharded_memcfg(_M, k, grid=grid)
    if in0_mc.is_sharded():
        shard_grid = _shard_grid_xy(in0_mc)
        pc_block = _2d_progcfg(_M, k, n, shard_grid, sharded_in0=True) if shard_grid else None
        if pc_block is not None:

            def block_2d(pc=pc_block, mc=in0_mc):
                act = ttnn.to_memory_config(x, mc)
                out = _linear(act, w, pc=pc, ckc=ckc_lofi)
                act.deallocate(True)
                return out

            arms.append(("block_2d_lofi", block_2d, pc_block, True))

    kt = math.ceil(k / TILE_SIZE)
    if label == "down":
        for bw in (2, 4, 5, 6):
            if kt % bw:
                continue
            pc_bw = _2d_progcfg(_M, k, n, grid, in0_block_w=bw)
            if pc_bw is None:
                continue

            def dram_bw(pc=pc_bw, bw=bw):
                return _linear(x, w, pc=pc, ckc=ckc_lofi)

            arms.append((f"dram_bw{bw}_lofi", dram_bw, pc_bw, False))

    ref_host = None
    results = []
    for name, fn, pc, _i2s in arms:
        arm_name, host, err = _run_arm(name, fn)
        if err is not None:
            results.append((arm_name, None, None, err))
            continue
        if name == "auto":
            ref_host = host
            pcc = 1.0
        else:
            _, pcc = comp_pcc(ref_host, host, _PCC)
            pcc = float(pcc)
        results.append((arm_name, pcc, pc, None))
        logger.info(f"{label} {arm_name}: pcc={pcc:.6f}" + (f" {pc}" if pc is not None else " auto"))

    assert ref_host is not None, "auto arm failed"
    for name, pcc, _pc, err in results:
        if name == "auto":
            continue
        if err is not None:
            continue
        assert pcc >= _PCC, f"{label} {name} pcc={pcc:.6f} < {_PCC}"

    rows = []
    for name, fn, pc, i2s in arms:
        if any(r[0] == name and r[3] is not None for r in results):
            logger.info(f"{label} {name}: skip timing (compile/clash)")
            continue
        times = []
        for _ in range(_REPEATS):
            times.append(_time_trace(mesh_device, fn))
        best = min(times)
        bw = getattr(pc, "in0_block_w", None) if pc is not None else None
        cores = None
        if pc is not None:
            gx, gy = (
                (pc.compute_with_storage_grid_size.x, pc.compute_with_storage_grid_size.y)
                if hasattr(pc.compute_with_storage_grid_size, "x")
                else pc.compute_with_storage_grid_size
            )
            cores = int(gx) * int(gy)
        pcc = next(r[1] for r in results if r[0] == name)
        rows.append((name, best, times, bw, cores, i2s, pcc))
        logger.info(
            f"{label} {name}: min={best:.1f}us times={[round(t, 1) for t in times]} "
            f"bw={bw} cores={cores} i2s={i2s} pcc={pcc:.6f}"
        )

    ship = _shipping_name(label)
    ship_row = next((r for r in rows if r[0] == ship), None)
    assert ship_row is not None, f"shipping arm {ship} missing for {label}"
    # Prefer a faster arm that still clears PCC vs HiFi2 auto (already asserted).
    eligible = [r for r in rows if r[6] is not None and r[6] >= _PCC]
    winner = min(eligible, key=lambda r: r[1])
    logger.info(
        f"{label} shipping={ship} {ship_row[1]:.1f}us pcc={ship_row[6]:.6f}; "
        f"winner={winner[0]} {winner[1]:.1f}us "
        f"({ship_row[1] / winner[1]:.2f}x vs shipping) pcc={winner[6]:.6f}"
    )
    x.deallocate(True)
    w.deallocate(True)


def test_12b_shapes_divide_for_2d():
    """Host check: 12B TP=8 shapes admit a full-grid 2D program config at M=4096."""
    grid = (8, 8)
    for label, k, n in _SHAPES:
        pc = _2d_progcfg(_M, k, n, grid)
        assert pc is not None, label
        kt = math.ceil(k / TILE_SIZE)
        assert kt % pc.in0_block_w == 0, (label, kt, pc.in0_block_w)
