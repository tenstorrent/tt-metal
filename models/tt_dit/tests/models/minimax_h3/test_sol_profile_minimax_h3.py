# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Device-profiler split of the Sol-Attn path on real captured activations (sol-coordination's task of 2026-10-16):
for one clip length, dense ring op and sparse (pre-pass + ring op) at one tau, record every device program with the
device profiler (TT_METAL_DEVICE_PROFILER=1) phase by phase, so the op wall can be split into attention-kernel time,
all-gathers, pre-pass programs and launch gaps. Writes <SOL_PROFILE_DIR>/phases.json (host timing + program counts
per phase) next to the profiler's own generated/profiler/.logs/profile_log_device.csv (TT_METAL_PROFILER_DIR).
Env: SOL_GEOM 5s|10s|15s, SOL_CAPTURE (s10_l12), SOL_TAU (1.0), SOL_REPS (5), SOL_MODE (compact), SOL_PROFILE_DIR,
SOL_PROFILE_SAVE (1: also write <SOL_PROFILE_DIR>/sol_outputs.pt = the pre-pass outputs of the first sparse call, for
bit-identity checks between variants; see sol_attn_work/exp/e1/compare_outputs.py).
Phases: dense_warm, dense x REPS, sparse_warm (pre-pass + op), prepass x REPS, sparse_op x REPS, sparse_full x REPS.
E1 copy: the committed driver (9bdf5634a87) has no stacked pooled all-gather (stack_kv), so the prepass_stacked phases of
the lead's version are not run here; the driver's route unit kind (SOL_ROUTE_UNITS_PER_SHARD) is logged and saved.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch
from loguru import logger

import ttnn

from ....models.transformers.minimax_h3.sol_prepass import SolGeometry
from ....models.transformers.minimax_h3.sol_prepass_fused import SolPrepassFused
from ....parallel.manager import CCLManager
from ....utils.tensor import bf16_tensor, from_torch
from .common import GALAXY_RING
from .test_sol_realdata_minimax_h3 import (
    CAPTURES,
    K_CHUNK,
    LOGICAL_N,
    Q_CHUNK,
    ROWS,
    VIDEO_START,
    _sol_mode_kwargs,
)
from .test_transformer_minimax_h3 import logical_length_tensor


def _sync(mesh_device):
    ttnn.synchronize_device(mesh_device)


def _csv_rows():
    p = Path(os.environ.get("TT_METAL_PROFILER_DIR", "generated/profiler")) / ".logs" / "profile_log_device.csv"
    if not p.exists():
        return 0
    with p.open("rb") as f:
        return sum(1 for _ in f)


@GALAXY_RING
def test_sol_profile(mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, reset_seeds):
    assert os.environ.get("TT_METAL_DEVICE_PROFILER") == "1", "run with TT_METAL_DEVICE_PROFILER=1"
    S = tuple(mesh_device.shape)[sp_axis]
    assert S == 8
    H, D = 14, 128
    scale = D**-0.5
    cap = os.environ.get("SOL_CAPTURE", "s10_l12")
    tau = float(os.environ.get("SOL_TAU", "1.0"))
    reps = int(os.environ.get("SOL_REPS", "5"))
    mode = os.environ.get("SOL_MODE", "compact")
    out_dir = Path(os.environ.get("SOL_PROFILE_DIR", os.environ.get("TT_METAL_PROFILER_DIR", "generated/profiler")))
    out_dir.mkdir(parents=True, exist_ok=True)

    blob = torch.load(CAPTURES / f"{cap}.pt", map_location="cpu", mmap=True)

    def shard(x):
        return x[:H].view(H, S, ROWS, D).permute(1, 0, 2, 3).contiguous()

    q, k, v = (shard(blob[n]).bfloat16() for n in ("q", "k", "v"))
    geo = SolGeometry(shards=S, rows=ROWS, logical_n=LOGICAL_N, q_chunk=Q_CHUNK, k_chunk=K_CHUNK)
    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)

    def upload(x):
        return from_torch(
            x, device=mesh_device, dtype=ttnn.bfloat16, mesh_axes=[sp_axis, None, None, None], layout=ttnn.TILE_LAYOUT
        )

    tq, tk, tv = upload(q), upload(k), upload(v)
    grid = mesh_device.compute_with_storage_grid_size()
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid.x - 1, grid.y),
        q_chunk_size=Q_CHUNK,
        k_chunk_size=K_CHUNK,
        exp_approx_mode=False,
    )
    ckc = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False
    )
    dummy_joint = bf16_tensor(torch.zeros((1, H, 0, D)), device=mesh_device)
    kshape = tq.shape
    ln = logical_length_tensor(mesh_device, LOGICAL_N)

    def run_ring(**sol_kwargs):
        out, _, _ = ttnn.transformer.ring_joint_scaled_dot_product_attention(
            tq,
            tk,
            tv,
            dummy_joint,
            dummy_joint,
            dummy_joint,
            persistent_output_buffer_k=ccl_manager.get_ag_ping_pong_buffer(kshape, 2, sp_axis, dtype=ttnn.bfloat16),
            persistent_output_buffer_v=ccl_manager.get_ag_ping_pong_buffer(kshape, 2, sp_axis, dtype=ttnn.bfloat16),
            joint_strategy="rear",
            logical_n=ln,
            program_config=program_config,
            compute_kernel_config=ckc,
            dim=2,
            multi_device_global_semaphore=ccl_manager.get_ag_ping_pong_semaphore(sp_axis),
            num_links=num_links,
            cluster_axis=sp_axis,
            mesh_device=mesh_device,
            topology=topology,
            subdevice_id=ccl_manager.ccl_sub_device_id,
            ccl_core_grid_offset=(grid.x - 1, 0),
            use_column_major_ccl=True,
            **sol_kwargs,
        )
        return out

    phases = []

    def phase(name, fn, n=1):
        """Run fn n times, each synchronized and host-timed, then read the device profiler: everything the device ran
        between the previous read and this one belongs to this phase (in program order)."""
        _sync(mesh_device)
        t_start = time.time()
        wall = []
        for _ in range(n):
            _sync(mesh_device)
            t0 = time.perf_counter()
            r = fn()
            _sync(mesh_device)
            wall.append((time.perf_counter() - t0) * 1e3)
            if isinstance(r, ttnn.Tensor):
                ttnn.deallocate(r)
            elif isinstance(r, dict):
                for t in r.values():
                    if isinstance(t, ttnn.Tensor):
                        ttnn.deallocate(t)
        t_end = time.time()
        ttnn.ReadDeviceProfiler(mesh_device)
        # rows [csv_rows_before, csv_rows_after) of the profiler CSV belong to this phase
        rows_after = _csv_rows()
        phases.append(
            {
                "phase": name,
                "n": n,
                "wall_ms": wall,
                "t_start": t_start,
                "t_end": t_end,
                "csv_rows": [phases[-1]["csv_rows"][1] if phases else 0, rows_after],
            }
        )
        logger.info(f"[profile] {name} x{n}: wall ms {' / '.join(f'{w:.2f}' for w in wall)}")

    # ---- dense (first, lead's rule)
    phase("dense_warm", lambda: run_ring(), 1)
    phase("dense", lambda: run_ring(), reps)

    # ---- sparse: fused pre-pass (two gathers) + ring op
    pre = SolPrepassFused(
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        sp_axis=sp_axis,
        tau=tau,
        mode=mode,
        dense_prefix_queries=False,
        emit_bias=False,
    )
    pre.prepare(geo, VIDEO_START, scale)
    sol = pre(tq, tk, tv, scale=scale, report_density=True)
    density = pre.last_density
    logger.info(f"[profile] route program units (count, per_shard) = {pre.route_units}; mask density {density:.4f}")
    if os.environ.get("SOL_PROFILE_SAVE", "1") == "1":
        # Host copies of the pre-pass outputs: mesh axis sp -> dim 0, axis tp -> dim 1 (the TP rows hold replicas, keep
        # row 0); the gathered pooled K/V are identical on every SP device, keep device (tp 0, sp 0).
        dims = [None, None]
        dims[sp_axis] = 0
        dims[tp_axis] = 1
        comp = ttnn.ConcatMesh2dToTensor(mesh_device, dims=dims, mesh_shape=tuple(mesh_device.shape))
        host = lambda t: ttnn.to_torch(t, mesh_composer=comp)
        saved = {
            "geom": os.environ.get("SOL_GEOM", "15s"),
            "capture": cap,
            "tau": tau,
            "mode": mode,
            "route_units": pre.route_units,
            "density": density,
            "block_mask": host(sol["block_mask"])[:, :H].contiguous(),  # [S, H, nqc_pad, width] uint32 (0/1)
            "pooled_k": host(sol["pooled_k"])[0, :H].contiguous(),  # [H, n_blocks_pad, D] bf16
            "pooled_v": host(sol["pooled_v"])[0, :H].contiguous(),
        }
        if sol.get("bias") is not None:
            saved["bias"] = host(sol["bias"])[:, :H].contiguous()
        torch.save(saved, out_dir / "sol_outputs.pt")
        logger.info(f"[profile] wrote {out_dir / 'sol_outputs.pt'} (mask {tuple(saved['block_mask'].shape)})")

    def sol_kwargs(d):
        kw = dict(sol_block_mask=d["block_mask"], sol_pooled_k=d["pooled_k"], **_sol_mode_kwargs(mode))
        if d.get("pooled_v") is not None:
            kw["sol_pooled_v"] = d["pooled_v"]
        if d.get("bias") is not None:
            kw["sol_bias"] = d["bias"]
        return kw

    phase("sparse_warm", lambda: run_ring(**sol_kwargs(sol)), 1)
    phase("prepass", lambda: pre(tq, tk, tv, scale=scale), reps)
    phase("sparse_op", lambda: run_ring(**sol_kwargs(sol)), reps)
    phase("sparse_full", lambda: run_ring(**sol_kwargs(pre(tq, tk, tv, scale=scale))), reps)

    for t in sol.values():
        if isinstance(t, ttnn.Tensor):
            ttnn.deallocate(t)
    rec = {
        "geom": os.environ.get("SOL_GEOM", "15s"),
        "capture": cap,
        "tau": tau,
        "mode": mode,
        "reps": reps,
        "kept_density": density,
        "route_units": pre.route_units,
        "chunks": [Q_CHUNK, K_CHUNK],
        "rows": ROWS,
        "logical_n": LOGICAL_N,
        "node": os.uname().nodename,
        "phases": phases,
    }
    (out_dir / "phases.json").write_text(json.dumps(rec, indent=1))
    logger.info(f"[profile] wrote {out_dir / 'phases.json'}")
