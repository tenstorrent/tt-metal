# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Trace-mode per-op speedup table: fused neighbor_pad_conv3d vs standalone (NP + conv3d).

``test_bench`` captures each path as a ttnn trace and replays it, so the wall/iter is the op's true
device latency (host dispatch removed). Standalone = two ops (NeighborPadAsync + Conv3d, both full-grid);
fused = one NpConv3d op (reserves column-0 for the NP fabric). Reports fused-vs-standalone per shape.

ACCURACY: this is the WAN op in isolation — accurate for "did my kernel change move this op's device
time" and "which op is faster for this shape alone" (blocking-verified, no cross-iter overlap). It is
NOT the e2e routing metric: the production decode is LTXVideoDecoder (different standalone path) and
per-layer cost differs in-context. For the deployment decision use the whole-decode trace
(models/tt_dit/tests/models/ltx/prof_vae_ltx.py::test_prof_vae_ltx_trace).

Coverage: every LTX VAE production shape on 2x4 (real NP topology) and 4x8mock (4x8 per-device conv
sizes on this 2x4 box — NP topology is still 2x4), plus a real 4x8 set (skipped unless a 32-chip mesh
is present). The fused op is trace-safe (all NP sems self-reset on-device, no host reset).
"""

from __future__ import annotations

import os
import time

import pytest
import torch
from loguru import logger

import ttnn

from ....models.vae.vae_wan2_1 import WanCausalConv3d, WanConv2d
from ....parallel.config import ParallelFactor, VaeHWParallelConfig
from ....parallel.manager import CCLManager
from ....utils.conv3d import ConvDims, conv_pad_height, conv_pad_in_channels
from ....utils.tensor import typed_tensor_2dshard

# Real 4x8 (32-device) shapes need a 32-chip mesh; skip on the 8-chip BH-LB so the coverage gap is
# visible in the report rather than silently dropped. 4x8mock shapes run here (they use a 2x4 mesh).
_SKIP_4X8 = pytest.mark.skip(reason="real 4x8 (32-device) shape — needs a 32-chip mesh, not 8-chip BH-LB")

# 4x8mock runs 4x8 per-device conv sizes on this 2x4 box. The 2x4 config keys miss the 4x8-tuned
# blockings (config is keyed by mesh) and fall back ~16x slower, so restore the real 4x8 blockings —
# the conv is the part the overlap scheme acts on, so this is a faithful proxy (NP topology still 2x4).
# Keyed by (C_in, C_out, H, W) of the 4x8mock shape.
_MOCK_4X8_BLK = {
    (512, 512, 34, 60): (64, 256, 1, 4, 8),  # ltx_s1_res 4x8 per-dev 17x15
    (512, 512, 68, 120): (64, 256, 1, 8, 4),  # ltx_s2_res 4x8 per-dev 34x30
    (256, 256, 68, 120): (64, 256, 1, 8, 4),  # ltx_s3_res 4x8 per-dev 34x30
    (256, 512, 68, 120): (64, 256, 1, 8, 4),  # ltx_s3_chg 4x8 per-dev 34x30
    (128, 128, 136, 240): (128, 64, 6, 4, 4),  # ltx_s4_res 4x8 per-dev 68x60 — force_spatial 6,4,4
    (128, 48, 136, 240): (128, 64, 6, 2, 16),  # ltx_s4_out 4x8 per-dev 68x60 — standalone-optimal
    (1024, 1024, 20, 32): (128, 64, 5, 4, 8),  # ltx ups_post_res 4x8 per-dev 10x8
    (1024, 128, 20, 32): (128, 64, 7, 8, 4),  # ltx ups_final 4x8 per-dev 10x8
    (128, 1024, 18, 32): (64, 128, 7, 8, 4),  # ltx_s0_conv_in 4x8 per-dev 9x8
    (1024, 4096, 18, 32): (128, 64, 5, 4, 8),  # ltx_s0_up 4x8 per-dev 9x8
    (512, 4096, 34, 60): (128, 64, 5, 4, 8),  # ltx_s1_up 4x8 per-dev 17x15
}


def _set_blk(model, ci, co, t, h, w):
    """Override the conv3d blocking (C_in, C_out, T, H, W block sizes)."""
    model.conv_config.C_in_block = ci
    model.conv_config.C_out_block = co
    model.conv_config.T_out_block = t
    model.conv_config.H_out_block = h
    model.conv_config.W_out_block = w


def _env_blk(name):
    """Parse a 'Cin,Cout,T,H,W' env override into a 5-int tuple, or None."""
    v = os.environ.get(name)
    return tuple(int(x) for x in v.split(",")) if v else None


def _build_model(
    mesh_device, B, C_in, C_out, T, H, W, kernel_size, padding, h_axis, w_axis, num_links, dtype, *, use_fused
):
    h_factor = tuple(mesh_device.shape)[h_axis]
    w_factor = tuple(mesh_device.shape)[w_axis]
    H_dev, W_dev = H // h_factor, W // w_factor
    kernel_tuple = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)

    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear, num_links=num_links)
    parallel_config = VaeHWParallelConfig(
        height_parallel=ParallelFactor(factor=h_factor, mesh_axis=h_axis),
        width_parallel=ParallelFactor(factor=w_factor, mesh_axis=w_axis),
    )

    if kernel_tuple[0] == 1:
        model = WanConv2d(
            in_channels=C_in,
            out_channels=C_out,
            kernel_size=kernel_tuple,
            mesh_device=mesh_device,
            stride=1,
            padding=padding,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            dtype=dtype,
            conv_dims=ConvDims(T=T, H=H_dev, W=W_dev),
        )
    else:
        model = WanCausalConv3d(
            in_channels=C_in,
            out_channels=C_out,
            kernel_size=kernel_size,
            mesh_device=mesh_device,
            stride=1,
            padding=padding,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            dtype=dtype,
            conv_dims=ConvDims(T=T, H=H_dev, W=W_dev),
            use_fused=use_fused,
        )

    # Random weights — this test checks device time, not correctness (see the PCC test for that).
    torch.manual_seed(0)
    weight = torch.randn(C_out, C_in, *kernel_tuple, dtype=torch.float32) * 0.01
    state = {"weight": weight, "bias": torch.zeros(C_out, dtype=torch.float32)}
    if kernel_tuple[0] == 1 and state["weight"].ndim == 5:
        state["weight"] = state["weight"].squeeze(2)
    model.load_torch_state_dict(state)

    if isinstance(model, WanCausalConv3d):
        # 4x8mock blocking restore; then optional dev sweep overrides (CONV3D_BLOCKING_SWEEP_RUNBOOK):
        #   NP_S4_BLK / NP_S4OUT_BLK pin a per-shape blocking on BOTH models; NP_BLK pins a fused-only
        #   blocking so the standalone baseline in the same run keeps its tuned entry.
        if (C_in, C_out, H, W) in _MOCK_4X8_BLK:
            _set_blk(model, *_MOCK_4X8_BLK[(C_in, C_out, H, W)])
        if (blk := _env_blk("NP_S4_BLK")) and (C_in, C_out, T) == (128, 128, 147):
            _set_blk(model, *blk)
        if (blk := _env_blk("NP_S4OUT_BLK")) and (C_in, C_out, T) == (128, 48, 147):
            _set_blk(model, *blk)
        if (blk := _env_blk("NP_BLK")) and use_fused:
            _set_blk(model, *blk)

    if use_fused and isinstance(model, WanCausalConv3d):
        # Bypass the MIN_T_FOR_FUSED hybrid-dispatch threshold so perf covers all shapes, incl. small T.
        if not model._use_fused and model._needs_halo:
            model._use_fused = True
        if os.environ.get("NP_FORCE_SPATIAL"):
            model.conv_config.force_spatial_parallel = True
        if os.environ.get("NP_HALO_LAST"):
            model.conv_config.halo_last = True

    # Standalone conv3d always runs on the FULL grid — the real deployed comparison against the fused op,
    # which reserves column-0 for the NP fabric (conv on ~102 of 110 cores). No grid reduction here.
    return model, h_factor, w_factor, parallel_config


def _build_input(mesh_device, B, C_in, T, H, W, h_axis, w_axis):
    torch.manual_seed(42)
    x = torch.randn(B, C_in, T, H, W, dtype=torch.float32).permute(0, 2, 3, 4, 1)
    x = conv_pad_in_channels(x)
    x, logical_h = conv_pad_height(x, tuple(mesh_device.shape)[h_axis])
    x = typed_tensor_2dshard(
        x, mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, shard_mapping={h_axis: 2, w_axis: 3}, dtype=ttnn.bfloat16
    )
    return x, logical_h


def _param(*row):
    """Build a parametrize entry; id is the trailing shape_id, real 4x8 (mesh (4,8)) is auto-skipped."""
    return pytest.param(*row, id=row[-1], marks=(_SKIP_4X8,) if row[8] == (4, 8) else ())


# (B, C_in, C_out, T, H, W, kernel_size, padding, mesh_device, h_axis, w_axis, num_links, shape_id)
# num_links=2 everywhere: 8 fabric cores (num_links*2 H + pad2_num_links*2 W) balance the NP+border
# work against the ~102 bulk conv cores. 2x4 H,W are chosen so the per-device key hits the tuned
# blocking; real 4x8 H,W = 4x8mock per-device sizes * (4,8) so they shard evenly on a 32-chip mesh.
_PERF_PARAMS = [
    # --- LTX VAE 2x4 production (real NP topology) ---
    _param(1, 128, 1024, 21, 34, 60, 3, 1, (2, 4), 0, 1, 2, "ltx_s0_conv_in_2x4"),  # per-dev 17x15
    _param(1, 1024, 4096, 21, 34, 60, 3, 1, (2, 4), 0, 1, 2, "ltx_s0_up_2x4"),  # per-dev 17x15
    _param(1, 512, 4096, 39, 68, 120, 3, 1, (2, 4), 0, 1, 2, "ltx_s1_up_2x4"),  # per-dev 34x30
    _param(1, 1024, 1024, 21, 34, 60, 3, 1, (2, 4), 0, 1, 2, "ltx_s0_res_2x4"),  # per-dev 17x15
    _param(1, 512, 512, 39, 68, 120, 3, 1, (2, 4), 0, 1, 2, "ltx_s1_res_2x4"),  # per-dev 34x30
    _param(1, 512, 512, 75, 136, 240, 3, 1, (2, 4), 0, 1, 2, "ltx_s2_res_2x4"),  # per-dev 68x60
    _param(1, 256, 256, 147, 136, 240, 3, 1, (2, 4), 0, 1, 2, "ltx_s3_res_2x4"),  # per-dev 68x60
    _param(1, 256, 512, 147, 136, 240, 3, 1, (2, 4), 0, 1, 2, "ltx_s3_chg_2x4"),  # per-dev 68x60
    _param(1, 128, 128, 147, 272, 480, 3, 1, (2, 4), 0, 1, 2, "ltx_s4_res_2x4"),  # per-dev 136x120
    _param(1, 128, 48, 147, 272, 480, 3, 1, (2, 4), 0, 1, 2, "ltx_s4_out_2x4"),  # per-dev 136x120
    # --- 4x8mock: 4x8 per-device conv sizes on this 2x4 box (NP topology still 2x4) ---
    _param(1, 128, 1024, 21, 18, 32, 3, 1, (2, 4), 0, 1, 2, "ltx_s0_conv_in_4x8mock"),  # per-dev 9x8
    _param(1, 1024, 4096, 21, 18, 32, 3, 1, (2, 4), 0, 1, 2, "ltx_s0_up_4x8mock"),  # per-dev 9x8
    _param(1, 512, 4096, 39, 34, 60, 3, 1, (2, 4), 0, 1, 2, "ltx_s1_up_4x8mock"),  # per-dev 17x15
    _param(1, 512, 512, 39, 34, 60, 3, 1, (2, 4), 0, 1, 2, "ltx_s1_res_4x8mock"),  # per-dev 17x15
    _param(1, 512, 512, 75, 68, 120, 3, 1, (2, 4), 0, 1, 2, "ltx_s2_res_4x8mock"),  # per-dev 34x30
    _param(1, 256, 256, 147, 68, 120, 3, 1, (2, 4), 0, 1, 2, "ltx_s3_res_4x8mock"),  # per-dev 34x30
    _param(1, 256, 512, 147, 68, 120, 3, 1, (2, 4), 0, 1, 2, "ltx_s3_chg_4x8mock"),  # per-dev 34x30
    _param(1, 128, 128, 147, 136, 240, 3, 1, (2, 4), 0, 1, 2, "ltx_s4_res_4x8mock"),  # per-dev 68x60
    _param(1, 128, 48, 147, 136, 240, 3, 1, (2, 4), 0, 1, 2, "ltx_s4_out_4x8mock"),  # per-dev 68x60
    _param(1, 1024, 1024, 21, 20, 32, 3, 1, (2, 4), 0, 1, 2, "ltx_ups_post_res_4x8mock"),  # per-dev 10x8
    _param(1, 1024, 128, 21, 20, 32, 3, 1, (2, 4), 0, 1, 2, "ltx_ups_final_4x8mock"),  # per-dev 10x8
    # --- Real 4x8 (32-chip mesh): same per-device sizes on a real (4,8) mesh → real 4x8 NP topology ---
    _param(1, 128, 1024, 21, 36, 64, 3, 1, (4, 8), 0, 1, 2, "ltx_s0_conv_in_4x8"),  # per-dev 9x8
    _param(1, 1024, 4096, 21, 36, 64, 3, 1, (4, 8), 0, 1, 2, "ltx_s0_up_4x8"),  # per-dev 9x8
    _param(1, 512, 4096, 39, 68, 120, 3, 1, (4, 8), 0, 1, 2, "ltx_s1_up_4x8"),  # per-dev 17x15
    _param(1, 512, 512, 39, 68, 120, 3, 1, (4, 8), 0, 1, 2, "ltx_s1_res_4x8"),  # per-dev 17x15
    _param(1, 512, 512, 75, 136, 240, 3, 1, (4, 8), 0, 1, 2, "ltx_s2_res_4x8"),  # per-dev 34x30
    _param(1, 256, 256, 147, 136, 240, 3, 1, (4, 8), 0, 1, 2, "ltx_s3_res_4x8"),  # per-dev 34x30
    _param(1, 256, 512, 147, 136, 240, 3, 1, (4, 8), 0, 1, 2, "ltx_s3_chg_4x8"),  # per-dev 34x30
    _param(1, 128, 128, 147, 272, 480, 3, 1, (4, 8), 0, 1, 2, "ltx_s4_res_4x8"),  # per-dev 68x60
    _param(1, 128, 48, 147, 272, 480, 3, 1, (4, 8), 0, 1, 2, "ltx_s4_out_4x8"),  # per-dev 68x60
    _param(1, 1024, 1024, 21, 40, 64, 3, 1, (4, 8), 0, 1, 2, "ltx_ups_post_res_4x8"),  # per-dev 10x8
    _param(1, 1024, 128, 21, 40, 64, 3, 1, (4, 8), 0, 1, 2, "ltx_ups_final_4x8"),  # per-dev 10x8
]


# =====================================================================================================
# Trace-mode per-op bench (ported from cglagovich/fused_rms_norm test_bench/_trace_and_time/_print_table).
# Trace replay strips per-op host dispatch, so the replay wall IS the op's device latency. Standalone
# (full-grid NP + conv3d, two ops) is captured as one trace, fused (one NpConv3d op) as another; we
# report fused vs standalone wall + speedup per shape. Blocking-verified (NP_BENCH_BLOCKING=1 matches the
# default), so the numbers are the true single-dispatch device latency, no cross-iter overlap.
#
# SCOPE: this is the WAN op in ISOLATION. It answers "is the fused OP faster than the standalone NP+conv
# OP for this shape, run alone" — accurate for kernel-change validation. It does NOT predict the LTX VAE
# decode: that uses a different model (LTXVideoDecoder) and the per-layer cost differs in-context. For
# the e2e routing decision use the whole-decode trace (prof_vae_ltx.py::test_prof_vae_ltx_trace), which
# is the production-representative metric. The fused op is trace-safe (every progress/neighbor/barrier
# sem self-resets on-device, no host reset) so it replays cleanly.
# =====================================================================================================
_BENCH_ITERS = 30
_PINGPONG = 2  # distinct resource sets alternated across replays (absorbs cross-device fabric skew)


def _trace_and_time(mesh_device, run_ops, *, num_iters):
    """Capture each run_op as its own trace, replay round-robin, return host wall us/iter (= device time).

    Exact port of cglagovich/fused_rms_norm `_trace_and_time`. A LIST ping-pongs distinct resource sets
    (each call advances the CCLManager's per-call sem/halo-buffer ping-pong) so a lagging device's late
    fabric atomic-inc isn't clobbered by an op's end-of-op sem reset under replay; a single callable is
    plain single-trace timing.
    """
    if callable(run_ops):
        run_ops = [run_ops]
    n = len(run_ops)
    for run_op in run_ops:  # warmup + cold-compile each program
        run_op()
    ttnn.synchronize_device(mesh_device)
    trace_ids = []
    for run_op in run_ops:
        tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        run_op()
        ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        trace_ids.append(tid)
    ttnn.synchronize_device(mesh_device)
    # NP_BENCH_BLOCKING=1 serializes each replay (host waits per iter) to rule out any cross-iter
    # trace-boundary overlap — the unambiguous single-dispatch latency. Default non-blocking (queue all,
    # one final sync) measures the back-to-back replay wall.
    blocking = os.environ.get("NP_BENCH_BLOCKING") == "1"
    t0 = time.perf_counter()
    for i in range(num_iters):
        ttnn.execute_trace(mesh_device, trace_ids[i % n], cq_id=0, blocking=blocking)
    ttnn.synchronize_device(mesh_device)
    elapsed_us = (time.perf_counter() - t0) * 1e6
    for tid in trace_ids:
        ttnn.release_trace(mesh_device, tid)
    return elapsed_us / num_iters


def _bench_shapes(suffix):
    """The _PERF_PARAMS rows whose shape_id ends with `suffix` (exact, so _4x8 excludes _4x8mock).

    NP_BENCH_ONLY=<substr>[,<substr>...] restricts to matching shape_ids (focused profiling).
    """
    only = [s for s in os.environ.get("NP_BENCH_ONLY", "").split(",") if s]
    out = []
    for p in _PERF_PARAMS:
        sid = p.values[-1]
        in_set = (
            (suffix == "4x8mock" and sid.endswith("_4x8mock"))
            or (suffix == "2x4" and sid.endswith("_2x4"))
            or (suffix == "4x8" and sid.endswith("_4x8") and not sid.endswith("_4x8mock"))
        )
        if in_set and (not only or any(o in sid for o in only)):
            out.append(p.values)
    return out


def _format_bench_table(rows, title):
    """Render the bench table as a single string (emitted via one logger call so it can't be fragmented
    by stdout interleaving with kernel-compile logs)."""
    cid_w = max(len("config_id"), max(len(r["cid"]) for r in rows))
    header = (
        f"{'config_id':<{cid_w}}  {'C_in':>5} {'C_out':>5} {'T':>4} {'HxW(dev)':>9} "
        f"{'standalone us':>13} {'fused us':>10} {'speedup':>8}"
    )
    box = "=" * max(len(header), len(title))
    lines = [box, title, box, header, "-" * len(header)]
    for r in rows:
        sa = f"{r['sa']:>13.1f}" if r["sa"] is not None else f"{r.get('sa_err', 'n/a'):>13}"
        fu = f"{r['f']:>10.1f}" if r["f"] is not None else f"{r.get('f_err', 'n/a'):>10}"
        sp = f"{r['sa'] / r['f']:>7.2f}x" if (r["sa"] and r["f"]) else f"{'-':>8}"
        lines.append(f"{r['cid']:<{cid_w}}  {r['ci']:>5} {r['co']:>5} {r['t']:>4} {r['hw']:>9} {sa} {fu} {sp}")
    lines += [box, "speedup = standalone/fused (>1.0 => fusion faster); trace-mode wall = device time."]
    return "\n".join(lines)


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 16777216}],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device, shape_set",
    [
        ((2, 4), "2x4"),
        ((2, 4), "4x8mock"),
        pytest.param((4, 8), "4x8", marks=_SKIP_4X8),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.timeout(1800)
def test_bench(mesh_device, device_params, shape_set):
    """Accurate trace-mode fused-vs-standalone speedup table for one shape set (rms test_bench port)."""
    rows = []
    for B, C_in, C_out, T, H, W, kernel_size, padding, _mesh, h_axis, w_axis, num_links, sid in _bench_shapes(
        shape_set
    ):
        row = {"cid": sid, "ci": C_in, "co": C_out, "t": T, "hw": f"{H // 2}x{W // 4}", "sa": None, "f": None}
        for key, use_fused in (("sa", False), ("f", True)):
            try:
                model, *_ = _build_model(
                    mesh_device,
                    B,
                    C_in,
                    C_out,
                    T,
                    H,
                    W,
                    kernel_size,
                    padding,
                    h_axis,
                    w_axis,
                    num_links,
                    ttnn.bfloat16,
                    use_fused=use_fused,
                )
                # _PINGPONG run_ops: each call advances the CCLManager's per-call ping-pong (h/w neighbor
                # sems + halo buffer), so the traces bake distinct banks and round-robin replay absorbs skew.
                run_ops = []
                for _ in range(_PINGPONG):
                    x, lh = _build_input(mesh_device, B, C_in, T, H, W, h_axis, w_axis)
                    run_ops.append(lambda m=model, xx=x, llh=lh: m(xx, logical_h=llh))
                row[key] = _trace_and_time(mesh_device, run_ops, num_iters=_BENCH_ITERS)
            except Exception as e:  # noqa: BLE001 — one shape's failure must not lose the table
                row[f"{key}_err"] = type(e).__name__
                logger.warning(f"{sid} {'fused' if use_fused else 'standalone'} FAILED: {str(e)[:160]}")
        rows.append(row)
        logger.info(f"BENCH {sid}: standalone={row['sa']} fused={row['f']}")
    logger.info("\n" + _format_bench_table(rows, f"neighbor_pad_conv3d trace-mode bench — BH {shape_set}"))
