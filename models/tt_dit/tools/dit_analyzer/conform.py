# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""On-device shape conformance: does the shim's per-device shape match ttnn's?

The dry run computes each tensor's per-device shape on a laptop, from
``region.shard_chunk_size`` (ttnn's chunk rule) and tile padding. Until that is
checked against real ttnn, every finding is only "the shim believes" (roadmap
§ conformance, blocker 36/42). This is the minimal form of phase 11's per-op
conformance and it discharges phase 7b's acceptance criterion: *for one block,
every per-device shape the shim computes matches a recorded real run.*

It is deliberately not a dataflow capture -- it builds representative tensors on
a real mesh, distributes them exactly the way ``utils/tensor.from_torch`` does
(``create_mesh_mapper`` + ``PlacementShard``), reads back each device's shard,
and diffs its logical and tile-padded shape against the shim's prediction. A flat
per-tensor check needs no tensor identity, so none of the capture hazards apply.

    # on the device broker, 2x4 Blackhole:
    python3 models/tt_dit/tools/dit_analyzer/conform.py --mesh 2 4

Runs entirely on ``ttnn``; imports only ``region`` from the analyzer (pure
Python), so the shim is never installed in the same process as real ttnn.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dit_analyzer.region import pad_extent, shard_chunk_size  # noqa: E402

# LTX-2.3 block dimensions (see dryrun/targets.py), plus a few shapes chosen to
# exercise tile padding and an uneven shard specifically.
VIDEO_N, VIDEO_DIM, VIDEO_HEAD_DIM = 38912, 4096, 128
AUDIO_N, AUDIO_DIM = 256, 2048


class Case:
    """A tensor to distribute, and how ttnn should fracture it.

    ``placements[m]`` is the tensor dim sharded on mesh axis ``m`` (or ``None``
    to replicate on that axis), matching ``MeshMapperConfig``'s per-mesh-axis
    placement list.
    """

    def __init__(self, name: str, logical: Sequence[int], placements: Sequence[Optional[int]]):
        self.name = name
        self.logical = tuple(int(d) for d in logical)
        self.placements = list(placements)

    def predicted_local(self, mesh_shape: Sequence[int]) -> Tuple[int, ...]:
        """The shim's per-device logical shape: ttnn's chunk rule per sharded axis."""
        s = list(self.logical)
        for mesh_axis, tensor_dim in enumerate(self.placements):
            if tensor_dim is None:
                continue
            n = mesh_shape[mesh_axis]
            s[tensor_dim] = shard_chunk_size(s[tensor_dim], n)
        return tuple(s)

    def predicted_padded(self, mesh_shape: Sequence[int]) -> Tuple[int, ...]:
        """Per-device shape with the innermost two axes tiled to 32."""
        s = list(self.predicted_local(mesh_shape))
        for a in (-2, -1):
            if len(s) >= abs(a):
                s[a] = pad_extent(s[a])
        return tuple(s)


def cases(mesh_shape: Sequence[int]) -> List[Case]:
    a0, a1 = mesh_shape[0], mesh_shape[1]  # 2, 4 on the Loudbox
    return [
        # video activation: sp fractures N (dim2), tp fractures D (dim3)
        Case("video_act", [1, 1, VIDEO_N, VIDEO_DIM], placements=[3, 2]),
        # audio activation, same layout, small N
        Case("audio_act", [1, 1, AUDIO_N, AUDIO_DIM], placements=[3, 2]),
        # fused qkv weight [K, 3*D], tp-sharded on output columns (mesh axis 0)
        Case("qkv_weight", [VIDEO_DIM, 3 * VIDEO_DIM], placements=[1, None]),
        # rope table, sp-sharded on N only, head_dim replicated
        Case("rope_cos", [1, 1, VIDEO_N, VIDEO_HEAD_DIM], placements=[None, 2]),
        # tile-padding probe: last dim 48 is not a tile, must round to 64
        Case("tile_pad", [1, 1, a1 * 64, 48], placements=[None, 2]),
        # uneven shard probe: dim2 not divisible by the sp factor
        Case("uneven_n", [1, 1, VIDEO_N + 2, VIDEO_DIM], placements=[None, 2]),
    ]


def _mapper(ttnn, mesh, placements: Sequence[Optional[int]]):
    p = [ttnn.PlacementShard(d) if d is not None else ttnn.PlacementReplicate() for d in placements]
    return ttnn.create_mesh_mapper(mesh, ttnn.MeshMapperConfig(p))


def run(mesh_shape: Sequence[int]) -> int:
    import torch

    import ttnn

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(*mesh_shape))
    failures = 0
    try:
        print("shape conformance on a real %s mesh\n" % (tuple(mesh_shape),))
        print("%-12s %-26s %-20s %-20s %s" % ("case", "logical", "ttnn local", "shim local", "verdict"))
        for c in cases(mesh_shape):
            pred_local = c.predicted_local(mesh_shape)
            pred_padded = c.predicted_padded(mesh_shape)
            uneven = any(c.logical[d] % mesh_shape[m] for m, d in enumerate(c.placements) if d is not None)
            host = torch.zeros(c.logical, dtype=torch.bfloat16)
            try:
                tt = ttnn.from_torch(
                    host,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat16,
                    mesh_mapper=_mapper(ttnn, mesh, c.placements),
                    device=mesh,
                )
            except Exception as exc:  # noqa: BLE001
                # ttnn rejects a non-uniform (uneven) shard under TILE_LAYOUT; its
                # shards must be equal. tt_dit therefore pads such an axis before
                # sharding, so this never reaches a real forward -- our shim guards
                # it too. Report it, and confirm the shim's chunk size is what ttnn
                # *expected* before it refused, not a disagreement.
                verdict = (
                    "ttnn refuses uneven tile shard (shim chunk=%d)"
                    % pred_local[
                        next(d for m, d in enumerate(c.placements) if d is not None and c.logical[d] % mesh_shape[m])
                    ]
                )
                if not uneven:
                    verdict = "FAIL (unexpected ttnn error: %s)" % str(exc).splitlines()[0][:80]
                    failures += 1
                print("%-12s %-26s %-20s %-20s %s" % (c.name, c.logical, "(refused)", pred_local, verdict))
                continue
            shards = ttnn.get_device_tensors(tt)
            real_local = tuple(int(d) for d in shards[0].shape)
            real_padded = tuple(int(d) for d in shards[0].padded_shape)
            ok = real_local == pred_local and real_padded == pred_padded
            failures += 0 if ok else 1
            verdict = "PASS" if ok else "FAIL"
            if not ok:
                verdict += " (padded real %s vs shim %s)" % (real_padded, pred_padded)
            print("%-12s %-26s %-20s %-20s %s" % (c.name, c.logical, real_local, pred_local, verdict))
            ttnn.deallocate(tt)
    finally:
        ttnn.close_mesh_device(mesh)

    print("\n%d/%d cases matched real ttnn" % (len(cases(mesh_shape)) - failures, len(cases(mesh_shape))))
    return failures


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mesh", type=int, nargs=2, default=[2, 4], metavar=("R", "C"), help="mesh shape (default 2 4)")
    args = ap.parse_args()
    raise SystemExit(1 if run(args.mesh) else 0)


if __name__ == "__main__":
    main()
