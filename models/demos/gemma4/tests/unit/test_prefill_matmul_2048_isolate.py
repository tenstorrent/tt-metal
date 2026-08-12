# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolate M=2048 prefill matmul layouts vs shipping auto DRAM-in0.

Tracy exp5 (sliding-prefill_2048-1x8): the four layer matmuls stream in0 from
DRAM interleaved. Interleaved L1 cannot hold ``[2048, 5376]`` bf16 (~21 MiB vs
the 4 MiB hoist cap). Block-shard per-core is ~336 KB and does fit.

Shipping auto at this height already uses 64 cores / HiFi2 / BFP8. Do **not**
blindly pin ``prefill_progcfg`` (it narrows to 7 columns and forces
``out_subblock_h=1``). Arms:

* ``auto`` — production (no program config)
* ``reshape_cutoff`` — DramShardedLinear pattern: ``[1, M/cutoff, cutoff, K]``
  + 2D progcfg sized to the cutoff (metadata reshape, DRAM in0)
* ``dram_2d`` — full worker-grid 2D, DRAM in0, subblock ``h*w<=8``
* ``block_2d`` — I2S to L1 block-shard matching that grid, then the same 2D
  kernel (I2S is inside the timed region — the model pays it at M=2048)
* ``dram_bw{2,4}`` — down_proj report flag: DRAM in0, ``in0_block_w>=2``

PCC vs auto (program-config changes are not bit-exact). Timing is opt-in.

    unset TT_METAL_DEVICE_PROFILER
    HF_MODEL=google/gemma-4-31B-it GEMMA4_MM_SWEEP=1 pytest \\
        models/demos/gemma4/tests/unit/test_prefill_matmul_2048_isolate.py \\
        -k "gate_up and 1x8" -sv --timeout=1800
"""

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
    prefill_grid_default,
    prefill_progcfg,
)

from ..test_factory import parametrize_mesh_with_fabric

_M = 2048
_PCC = 0.99
_REPEATS = 3
_TRACE_REPS = 20
_SHAPES = (
    ("gate_up", 5376, 5376),
    ("qkv", 5376, 2048),
    ("down", 2688, 5376),
    ("o_proj", 1024, 5376),
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
        # 2D kernel TT_FATALs unless fuse_batch is on for a sharded in0.
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


@parametrize_mesh_with_fabric(device_params_extra={"trace_region_size": 64_000_000})
@pytest.mark.parametrize("label,k,n", _SHAPES, ids=[s[0] for s in _SHAPES])
def test_prefill_matmul_2048_isolate(label, k, n, mesh_device, reset_seeds):
    """PCC every arm vs auto. Timing matrix behind GEMMA4_MM_SWEEP=1."""
    if os.environ.get("GEMMA4_MM_SWEEP", "0").lower() not in ("1", "true", "yes"):
        pytest.skip("set GEMMA4_MM_SWEEP=1 to run the M=2048 matmul isolate")

    torch.manual_seed(0)
    mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    x_torch = torch.randn(1, 1, _M, k, dtype=torch.bfloat16) * 0.02
    w_torch = torch.randn(k, n, dtype=torch.bfloat16) * 0.02
    x = _upload(mesh_device, x_torch, dtype=ttnn.bfloat16, mapper=mapper)
    w = _upload(mesh_device, w_torch, dtype=ttnn.bfloat8_b, mapper=mapper)
    ckc = _hifi2()
    grid = _device_grid(mesh_device)
    cutoff = _PREFILL_CUTOFF

    arms = []

    def auto():
        return _linear(x, w)

    arms.append(("auto", auto, None, False))

    if _M % cutoff == 0:
        pc_cut = prefill_progcfg(cutoff, k, n)

        def reshape_cutoff(pc=pc_cut):
            x_r = ttnn.reshape(x, (1, _M // cutoff, cutoff, k))
            out_r = _linear(x_r, w, pc=pc, ckc=ckc)
            return ttnn.reshape(out_r, (1, 1, _M, n))

        arms.append(("reshape_cutoff", reshape_cutoff, pc_cut, False))

    pc_narrow = prefill_progcfg(_M, k, n)

    def dram_narrow(pc=pc_narrow):
        return _linear(x, w, pc=pc, ckc=ckc)

    arms.append(("dram_narrow", dram_narrow, pc_narrow, False))

    pc_dram = _2d_progcfg(_M, k, n, grid)
    if pc_dram is not None:

        def dram_2d(pc=pc_dram):
            return _linear(x, w, pc=pc, ckc=ckc)

        arms.append(("dram_2d", dram_2d, pc_dram, False))

    in0_mc = l1_block_sharded_memcfg(_M, k, grid=grid)
    if in0_mc.is_sharded():
        shard_grid = _shard_grid_xy(in0_mc)
        pc_block = _2d_progcfg(_M, k, n, shard_grid, sharded_in0=True) if shard_grid else None
        if pc_block is not None:

            def block_2d(pc=pc_block, mc=in0_mc):
                act = ttnn.to_memory_config(x, mc)
                out = _linear(act, w, pc=pc, ckc=ckc)
                act.deallocate(True)
                return out

            arms.append(("block_2d", block_2d, pc_block, True))

    kt = math.ceil(k / TILE_SIZE)
    if label == "down":
        for bw in (2, 4, 7):
            if kt % bw:
                continue
            pc_bw = _2d_progcfg(_M, k, n, grid, in0_block_w=bw)
            if pc_bw is None:
                continue

            def dram_bw(pc=pc_bw, bw=bw):
                return _linear(x, w, pc=pc, ckc=ckc)

            arms.append((f"dram_bw{bw}", dram_bw, pc_bw, False))

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
        rows.append((name, best, times, bw, cores, i2s))
        logger.info(
            f"{label} {name}: min={best:.1f}us times={[round(t, 1) for t in times]} " f"bw={bw} cores={cores} i2s={i2s}"
        )

    winner = min(rows, key=lambda r: r[1])
    auto_us = next(r[1] for r in rows if r[0] == "auto")
    logger.info(
        f"{label} winner={winner[0]} {winner[1]:.1f}us " f"({auto_us / winner[1]:.2f}x vs auto {auto_us:.1f}us)"
    )
    x.deallocate(True)
    w.deallocate(True)


def test_2d_fullgrid_knobs_divide_kt():
    """Host check: full-grid 2D in0_block_w divides Kt (and per-core Kt when sharded)."""
    grid = prefill_grid_default()
    for label, k, n in _SHAPES:
        pc = _2d_progcfg(_M, k, n, grid)
        assert pc is not None, label
        kt = math.ceil(k / TILE_SIZE)
        assert kt % pc.in0_block_w == 0, (label, kt, pc.in0_block_w)
        gx, gy = grid
        assert pc.per_core_M == math.ceil(_M / TILE_SIZE / gy)
        assert pc.per_core_N == math.ceil(n / TILE_SIZE / gx)
        in0_mc = l1_block_sharded_memcfg(_M, k, grid=grid)
        if in0_mc.is_sharded():
            sg = _shard_grid_xy(in0_mc)
            pc_s = _2d_progcfg(_M, k, n, sg, sharded_in0=True)
            assert pc_s is not None, label
            kt_core = kt // sg[0]
            assert kt_core % pc_s.in0_block_w == 0, (label, kt_core, pc_s.in0_block_w)
