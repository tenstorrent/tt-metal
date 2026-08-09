# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""On-device collective conformance (phase 11b core).

`conform.py` (11a) checks the shim's shape rules for compute ops but skips the
collectives, because they need fabric + a `CCLManager` (semaphores, subdevices).
This runs the block's actual collectives -- all-gather and reduce-scatter, over
both the tp and sp axes -- on the real 2x4 Loudbox through the same `CCLManager`
the model uses, and diffs each output's per-device shape against what the shim's
collective rule predicts.

It is the substance of a whole-block collective-log diff (does the collective
fire with the shape/axis the shim believes) without running the entire block
forward: the *ordering* of collectives is already pinned by the dry-run oracle,
so what was left to confirm on silicon is the collectives themselves. The full
block forward (ring SDPA + matmuls end-to-end) and the 4x8 Ring finding remain
the larger, Galaxy-gated part of 11b.

    python3 models/tt_dit/tools/dit_analyzer/conform_collectives.py --mesh 2 4
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dit_analyzer.region import shard_chunk_size  # noqa: E402


def _set_fabric_1d(ttnn, topo: str = "linear") -> None:
    """Mirror conftest.set_fabric for FABRIC_1D (collectives need a fabric).

    Ring needs a ring fabric: FABRIC_1D has no wrap link, so a ring hop across the seam has no
    route and ttnn aborts with "Could not find any forwarding direction" (see conform_encoder).
    """
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D_RING if topo == "ring" else ttnn.FabricConfig.FABRIC_1D,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
    )


def _mapper(ttnn, mesh, placements: Sequence[Optional[int]]):
    p = [ttnn.PlacementShard(d) if d is not None else ttnn.PlacementReplicate() for d in placements]
    return ttnn.create_mesh_mapper(mesh, ttnn.MeshMapperConfig(p))


class CollCase:
    """A collective to run: op, the tensor to build, and which axis/dim it acts on."""

    def __init__(self, name, op, logical, placements, dim, mesh_axis):
        self.name = name
        self.op = op  # "all_gather" | "reduce_scatter"
        self.logical = tuple(logical)
        self.placements = list(placements)
        self.dim = dim
        self.mesh_axis = mesh_axis

    def predicted_out_local(self, mesh_shape) -> Tuple[int, ...]:
        """Shim rule: all_gather un-shards `dim`; reduce_scatter shards it."""
        s = list(self.logical)
        # start from the input's per-device shape
        for m, td in enumerate(self.placements):
            if td is not None:
                s[td] = shard_chunk_size(s[td], mesh_shape[m])
        if self.op == "all_gather":
            s[self.dim] = self.logical[self.dim]  # gathered axis becomes full
        else:  # reduce_scatter shards `dim` over mesh_axis
            s[self.dim] = shard_chunk_size(self.logical[self.dim], mesh_shape[self.mesh_axis])
        return tuple(s)


def coll_cases(sp_axis, tp_axis) -> List[CollCase]:
    ag_tp = [None, None]
    ag_tp[tp_axis] = 3  # feature sharded on tp
    ag_sp = [None, None]
    ag_sp[sp_axis] = 2  # sequence sharded on sp
    return [
        # all-gather the feature axis over tp (the block's pre-attention / pre-out gather)
        CollCase("ag_tp", "all_gather", [1, 1, 512, 2048], ag_tp, dim=3, mesh_axis=tp_axis),
        # all-gather the sequence axis over sp (ring-SDPA's internal K/V gather shape)
        CollCase("ag_sp", "all_gather", [1, 1, 2048, 512], ag_sp, dim=2, mesh_axis=sp_axis),
        # reduce-scatter the feature axis over tp (RowParallel end-of-block)
        CollCase("rs_tp", "reduce_scatter", [1, 1, 512, 2048], [None, None], dim=3, mesh_axis=tp_axis),
    ]


def run(mesh_shape: Sequence[int], topo: str = "linear", sp_axis: int = 0, tp_axis: int = 1) -> int:
    import torch

    import ttnn
    from models.tt_dit.parallel.manager import CCLManager

    # sd35_block preset is SP on axis0 / TP on axis1; the H3 Galaxy config is sp1tp0, so the
    # axes are selectable rather than baked in.
    _set_fabric_1d(ttnn, topo)
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*mesh_shape))
    failures = 0
    try:
        topology = ttnn.Topology.Ring if topo == "ring" else ttnn.Topology.Linear
        ccl = CCLManager(mesh_device=mesh, num_links=1, topology=topology)
        print(
            "collective conformance on a real %s mesh (fabric 1D%s, sp=axis%d tp=axis%d)\n"
            % (tuple(mesh_shape), " ring" if topo == "ring" else "", sp_axis, tp_axis)
        )
        print(
            "%-8s %-10s %-26s %-22s %-22s %s" % ("case", "op", "logical", "ttnn out (per dev)", "shim out", "verdict")
        )
        cases = coll_cases(sp_axis, tp_axis)
        for c in cases:
            try:
                host = torch.zeros(c.logical, dtype=torch.bfloat16)
                tt = ttnn.from_torch(
                    host,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat16,
                    mesh_mapper=_mapper(ttnn, mesh, c.placements),
                    device=mesh,
                )
                if c.op == "all_gather":
                    out = ccl.all_gather(tt, dim=c.dim, mesh_axis=c.mesh_axis, use_hyperparams=False)
                else:
                    out = ccl.reduce_scatter(tt, dim=c.dim, mesh_axis=c.mesh_axis)
                real = tuple(int(d) for d in ttnn.get_device_tensors(out)[0].shape)
                shim = c.predicted_out_local(mesh_shape)
                ok = real == shim
                failures += 0 if ok else 1
                print(
                    "%-8s %-10s %-26s %-22s %-22s %s" % (c.name, c.op, c.logical, real, shim, "PASS" if ok else "FAIL")
                )
                ttnn.deallocate(tt)
                ttnn.deallocate(out)
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print("%-8s %-10s %-26s %s" % (c.name, c.op, c.logical, "ERROR: " + str(exc).splitlines()[-1][:70]))
    finally:
        ttnn.close_mesh_device(mesh)
        try:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        except Exception:  # noqa: BLE001
            pass

    total = len(coll_cases(sp_axis, tp_axis))
    print("\n%d/%d collectives matched real ttnn" % (total - failures, total))
    return failures


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mesh", type=int, nargs=2, default=[2, 4], metavar=("R", "C"))
    ap.add_argument("--topology", choices=["ring", "linear"], default="linear")
    ap.add_argument("--sp-axis", type=int, choices=[0, 1], default=0)
    ap.add_argument("--tp-axis", type=int, choices=[0, 1], default=1)
    args = ap.parse_args()
    raise SystemExit(1 if run(args.mesh, args.topology, args.sp_axis, args.tp_axis) else 0)


if __name__ == "__main__":
    main()
