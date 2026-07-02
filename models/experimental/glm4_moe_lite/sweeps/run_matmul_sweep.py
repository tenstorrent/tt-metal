# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Entry point: brute-force matmul config sweep on the GLM 2x4 mesh.

    python models/experimental/glm4_moe_lite/sweeps/run_matmul_sweep.py \
        --mesh-rows 2 --mesh-cols 4

No env vars required from you: importing the `sweeps` package sets the profiler
flags before ttnn loads.  Sweep axes/targets are edited in code (targets.py) or
via the small CLI below.
"""

from __future__ import annotations

# NOTE: this import MUST precede `import ttnn` (it sets the profiler env vars).
import models.experimental.glm4_moe_lite.sweeps as _sweeps  # noqa: F401

import argparse

import ttnn

from models.experimental.glm4_moe_lite.sweeps.harness import run_sweep, print_table
from models.experimental.glm4_moe_lite.sweeps.targets import (
    GLM_TARGETS_2D,
    GLM_TARGETS_PHASE2,
    PhaseSpec,
    SweepAxis,
)


def _set_fabric(n_devices: int) -> None:
    if n_devices <= 1:
        return
    is_galaxy = ttnn.cluster.get_cluster_type() == ttnn.cluster.ClusterType.GALAXY
    fabric = ttnn.FabricConfig.FABRIC_1D_RING if is_galaxy else ttnn.FabricConfig.FABRIC_1D
    ttnn.set_fabric_config(
        fabric,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
    )


def _dispatch_cfg():
    if ttnn.cluster.get_cluster_type() == ttnn.cluster.ClusterType.GALAXY:
        return ttnn.DispatchCoreConfig(axis=ttnn.DispatchCoreAxis.ROW)
    return ttnn.DispatchCoreConfig(ttnn.device.get_default_dispatch_core_type())


def main() -> int:
    ap = argparse.ArgumentParser(description="GLM matmul device-time sweep")
    ap.add_argument("--mesh-rows", type=int, default=2)
    ap.add_argument("--mesh-cols", type=int, default=4)
    ap.add_argument("--phase2", action="store_true", help="also sweep head-parallel + sparse experts (phase 2)")
    ap.add_argument("--targets", default="", help="comma-separated target names to filter (default: all)")
    ap.add_argument("--phase", default="decode", help="decode | prefill | both")
    ap.add_argument("--batch", default="1,32", help="comma-separated batch sizes")
    ap.add_argument("--seq-len", default="128", help="comma-separated sequence lengths (prefill only)")
    ap.add_argument("--prefill-chunk", type=int, default=128, help="cap on prefill M (0 = no chunk)")
    ap.add_argument(
        "--dp-rows", type=int, default=1, help="split attention tokens across N mesh rows (2 = realistic 2x4 attn_dp)"
    )
    args = ap.parse_args()

    n_dev = args.mesh_rows * args.mesh_cols
    targets = list(GLM_TARGETS_2D) + (list(GLM_TARGETS_PHASE2) if args.phase2 else [])
    if args.targets.strip():
        want = {s.strip() for s in args.targets.split(",") if s.strip()}
        targets = [t for t in targets if t.name in want]

    batches = [int(x) for x in args.batch.split(",") if x.strip()]
    seqs = [int(x) for x in args.seq_len.split(",") if x.strip()]
    want_phase = args.phase.lower()
    phases: list[PhaseSpec] = []
    if want_phase in ("decode", "both"):
        phases += [PhaseSpec("decode", batch=b) for b in batches]
    if want_phase in ("prefill", "both"):
        phases += [PhaseSpec("prefill", batch=b, seq_len=s) for b in batches for s in seqs]
    if not phases:
        raise SystemExit(f"--phase must be decode|prefill|both, got {args.phase!r}")

    axis = SweepAxis(phases=phases, prefill_chunk=args.prefill_chunk, dp_rows=args.dp_rows)

    _set_fabric(n_dev)
    mesh = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(args.mesh_rows, args.mesh_cols),
        dispatch_core_config=_dispatch_cfg(),
    )
    try:
        results = run_sweep(mesh, targets, axis)
        print_table(results)
    finally:
        ttnn.close_mesh_device(mesh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
