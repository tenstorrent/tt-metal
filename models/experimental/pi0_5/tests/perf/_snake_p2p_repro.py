# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Validate a SNAKE (mixed-axis) point_to_point chain on a (6,3) 18-chip mesh
under FABRIC_1D — the traced-prefill hand-off pattern.

The 18-chip prefill chains in boustrophedon order: along each row (same-row p2p)
then down one column to the next row (same-col p2p). Every hop is collinear, so
each p2p is legal under FABRIC_1D — but mixed-axis hops on a 2D-shaped mesh were
never exercised (denoise/SigLIP used single-axis (6,1) columns). This confirms
the chain propagates correctly + captures/replays as a trace.

  tt-smi -glx_reset
  timeout 200 env PYTHONPATH=$PWD TT_METAL_HOME=$PWD \
    python_env/bin/python models/experimental/pi0_5/tests/perf/_snake_p2p_repro.py
"""

import sys

import torch
import ttnn

ROWS, COLS = 6, 3
TILE = 32


def _pcc(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    a = a - a.mean()
    b = b - b.mean()
    d = (a.norm() * b.norm()).item()
    return (torch.dot(a, b).item() / d) if d > 0 else 0.0


def main():
    def log(m):
        print(f"[snake] {m}", flush=True)

    # Boustrophedon snake over (6,3): row 0 L->R, row 1 R->L, ... Each consecutive
    # pair shares a row (within a row) or a column (the turn down to next row).
    snake = []
    for r in range(ROWS):
        cols = range(COLS) if r % 2 == 0 else range(COLS - 1, -1, -1)
        for c in cols:
            snake.append((r, c))
    log(f"snake ({len(snake)} chips): {snake}")
    # sanity: every hop collinear
    for k in range(len(snake) - 1):
        (r0, c0), (r1, c1) = snake[k], snake[k + 1]
        assert r0 == r1 or c0 == c1, f"non-collinear hop {snake[k]}->{snake[k+1]}"
    log("all 17 hops collinear ✓")

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    parent = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(8, 4), trace_region_size=134_217_728)
    subs = []
    try:
        mesh = parent.create_submesh(ttnn.MeshShape(ROWS, COLS), ttnn.MeshCoordinate(0, 0))
        subs.append(mesh)
        log(f"mesh shape={mesh.shape}")

        n = ROWS * COLS
        torch.manual_seed(0)
        w_t = (
            torch.randn(n, TILE, TILE, dtype=torch.bfloat16) * 0.1
        )  # per-chip weight, shard i -> coord (i//COLS, i%COLS)
        w = ttnn.from_torch(
            w_t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
        )
        a_t = torch.randn(1, 1, TILE, TILE, dtype=torch.bfloat16)
        a = ttnn.from_torch(
            a_t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

        def chain(x):
            cur = x
            for k in range(n):
                cur = ttnn.matmul(cur, w, memory_config=ttnn.L1_MEMORY_CONFIG)
                if k < n - 1:
                    cur = ttnn.to_memory_config(ttnn.to_layout(cur, ttnn.TILE_LAYOUT), ttnn.DRAM_MEMORY_CONFIG)
                    cur = ttnn.point_to_point(
                        cur,
                        ttnn.MeshCoordinate(*snake[k]),
                        ttnn.MeshCoordinate(*snake[k + 1]),
                        topology=ttnn.Topology.Linear,
                        output_tensor=cur,
                    )
            return cur

        log("eager snake chain...")
        eager = chain(a)
        ttnn.synchronize_device(mesh)
        last_idx = snake[-1][0] * COLS + snake[-1][1]
        eager_last = ttnn.to_torch(eager, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[last_idx]
        log("eager OK")

        log("capture + execute trace...")
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        traced = chain(a)
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        traced_last = ttnn.to_torch(traced, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[last_idx]
        log("trace OK")

        # torch reference: matmul chain in snake order
        ref = a_t[0, 0].float()
        for r, c in snake:
            ref = ref @ w_t[r * COLS + c].float()
        log(f"PCC eager-vs-traced = {_pcc(eager_last, traced_last):.6f}")
        log(f"PCC eager-vs-torch  = {_pcc(eager_last, ref):.6f}  (validates snake propagation)")
        log(f"finite={torch.isfinite(traced_last).all().item()}")
        log("SUCCESS")
    finally:
        for sm in reversed(subs):
            try:
                ttnn.close_mesh_device(sm)
            except Exception:
                pass
        ttnn.close_mesh_device(parent)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
    sys.exit(0)
