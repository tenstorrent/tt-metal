# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Validate the per-stage single-mesh + point_to_point trace concept.

Mimics one pipeline stage on a SINGLE mesh (no 1x1 submeshes, no sockets):
  - weights are a per-chip-different sharded mesh tensor (each chip = its own layer)
  - a chain of K chips: matmul (SPMD, per-chip weights) -> point_to_point hand-off
  - the whole chain captured as ONE trace on the mesh

Key questions this answers (run with TT_TRACE_POPULATE_DEBUG=1):
  - Does a single-mesh op chain capture a NON-EMPTY trace? (vs the 64-byte empty
    trace we got capturing ops on 1x1 child submeshes)
  - Does point_to_point chain + trace replay without hanging, and match torch?

Run:
  tt-smi -glx_reset
  TT_TRACE_POPULATE_DEBUG=1 timeout 200 env PYTHONPATH=$PWD TT_METAL_HOME=$PWD \
    python_env/bin/python models/experimental/pi0_5/tests/perf/_trace_p2p_pipeline_repro.py
Exit 124 = hung. Watch [repro] markers + [populate_dbg] n_entries/total_trace_size.

Env: REPRO_K = chain length (default 6, like the denoise stage).
"""

import os
import sys

import torch
import ttnn


def main():
    def log(m):
        print(f"[repro] {m}", flush=True)

    K = int(os.environ.get("REPRO_K", "4"))
    TILE = 32  # M = Kdim = N = 32 (one tile) so the chain composes dimensionally
    n_total = 28  # (7,4) compute submesh
    axis = os.environ.get("REPRO_AXIS", "row")  # row -> (0,c); col -> (r,0)
    fabric = os.environ.get("REPRO_FABRIC", "1d").lower()
    use_matmul = os.environ.get("REPRO_NOMATMUL", "0") != "1"
    do_trace = os.environ.get("REPRO_NOTRACE", "0") != "1"

    # Chain path along one axis -> adjacent coords -> clean linear p2p hops.
    coords = [(0, c) for c in range(K)] if axis == "row" else [(r, 0) for r in range(K)]
    log(f"K={K} axis={axis} fabric={fabric} matmul={use_matmul} trace={do_trace} coords={coords}")

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D if fabric == "1d" else ttnn.FabricConfig.FABRIC_2D)
    parent = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(8, 4), trace_region_size=134_217_728)
    submeshes = []
    try:
        compute = parent.create_submesh(ttnn.MeshShape(7, 4), ttnn.MeshCoordinate(0, 0))
        submeshes.append(compute)
        log(f"compute mesh shape={compute.shape}")

        # Per-chip-different weights: (28, TILE, TILE) sharded dim0 -> each chip its own (TILE,TILE).
        torch.manual_seed(0)
        w_torch = torch.randn(n_total, TILE, TILE, dtype=torch.bfloat16) * 0.1
        weights = ttnn.from_torch(
            w_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=compute,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(compute, dim=0),
        )
        # Activation: replicated (1,1,TILE,TILE); only the active chip's shard matters per step.
        a_torch = torch.randn(1, 1, TILE, TILE, dtype=torch.bfloat16)
        act = ttnn.from_torch(
            a_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=compute,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(compute),
        )
        log("uploaded sharded weights + replicated activation")

        def chain(a):
            cur = a
            for k in range(K):
                out = ttnn.matmul(cur, weights, memory_config=ttnn.L1_MEMORY_CONFIG) if use_matmul else cur
                if k < K - 1:
                    recv = ttnn.MeshCoordinate(*coords[k + 1])
                    send = ttnn.MeshCoordinate(*coords[k])
                    cur = ttnn.point_to_point(out, recv, send, topology=ttnn.Topology.Linear)
                else:
                    cur = out
            return cur

        log("warmup chain (JIT)...")
        warm = chain(act)
        ttnn.synchronize_device(compute)
        log("warmup done (eager chain OK)")

        if do_trace:
            log("begin_trace_capture")
            tid = ttnn.begin_trace_capture(compute, cq_id=0)
            captured = chain(act)
            log("captured body; end_trace_capture (the old HANG POINT)")
            ttnn.end_trace_capture(compute, tid, cq_id=0)
            log("END_TRACE_CAPTURE OK")

            log("execute_trace")
            ttnn.execute_trace(compute, tid, cq_id=0, blocking=True)
            log("EXECUTE_TRACE OK")
        else:
            captured = warm

        # Read the last chip's shard and compare to a torch reference chain.
        out_full = ttnn.to_torch(captured, mesh_composer=ttnn.ConcatMeshToTensor(compute, dim=0))
        last_idx = coords[-1][0] * 4 + coords[-1][1]
        got = out_full[last_idx].float()
        ref = a_torch[0, 0].float()
        for k in range(K):
            ref = ref @ w_torch[coords[k][0] * 4 + coords[k][1]].float()
        err = (got - ref).abs().max().item()
        log(f"last-chip max abs err vs torch chain = {err:.4f}")
        log(f"finite={torch.isfinite(got).all().item()}")
        log("SUCCESS")
    finally:
        for sm in reversed(submeshes):
            try:
                ttnn.close_mesh_device(sm)
            except Exception:
                pass
        ttnn.close_mesh_device(parent)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
    sys.exit(0)
