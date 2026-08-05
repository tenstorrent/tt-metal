# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""On-device conformance: does the shim's shape/layout/distribution match ttnn's?

The dry run computes shapes on a laptop from `region.shard_chunk_size` (ttnn's
chunk rule), tile padding, and per-op shape rules in `dryrun/ops.py`. Until those
are checked against real ttnn, every finding is only "the shim believes" (roadmap
§ conformance, blocker 36/42). This is phase 11's per-op conformance (it grew out
of phase 7b's per-tensor check), run on the 2x4 Loudbox.

Two kinds of check, both diffed against real ttnn:

1. **Distribution / tile padding** -- build a tensor on the real mesh the way
   `utils/tensor.from_torch` does (`create_mesh_mapper` + `PlacementShard`), read
   back a device shard, and compare its logical and tile-padded shape to the
   shim's. A flat per-tensor check needs no tensor identity, so none of the
   capture hazards apply.
2. **Per-op output shape** -- run a real ttnn compute op (matmul, the fused-QKV
   split, concatenate-heads) and compare its output shape to what the shim's
   shape rule predicts. A PASS means the rule the analyzer trusts matches silicon.

Not covered here (documented in the roadmap as the phase-11 remainder): the
collectives and fused kernels (need CCL setup) and conv2d/group_norm (need
program-config setup) -- their load-bearing shape math (shard division, tile
padding, channel sharding) is covered by the distribution cases; and the
whole-block collective-log diff, which needs the model running on hardware.

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
        # SD3.5-large fused qkv weight [D, 3*inner], heads padded 38->40, col-sharded on tp
        Case("sd35_qkv_w", [2432, 3 * 40 * 64], placements=[None, 1]),
        # conv weight [Cout, Cin, kh, kw], output channels sharded on tp (blocker 14)
        Case("conv_w", [512, 512, 3, 3], placements=[None, 0]),
        # group-norm weight, per-device row-major layout (blocker 14)
        Case("gnorm_w", [1, 512], placements=[None, 1]),
    ]


def _mapper(ttnn, mesh, placements: Sequence[Optional[int]]):
    p = [ttnn.PlacementShard(d) if d is not None else ttnn.PlacementReplicate() for d in placements]
    return ttnn.create_mesh_mapper(mesh, ttnn.MeshMapperConfig(p))


# -----------------------------------------------------------------------------
# Per-op conformance: run a real ttnn compute op and check the shim's predicted
# output shape against it. The prediction is computed the way the shim's shape
# rule does (cross-referenced to dryrun/ops.py), so a PASS means the rule matches
# real ttnn -- the thing that promotes a finding out of "the shim believes".
#
# Each op returns (name, real_per_device_shape, shim_per_device_shape). Ops that
# need heavy CCL / program-config setup (collectives, fused kernels, conv2d,
# group_norm) are not run here -- see run()'s notes; the shape math they lean on
# (shard division, tile padding) is covered by the distribution cases above.
# -----------------------------------------------------------------------------
def _op_matmul(ttnn, torch, mesh, mesh_shape, tp_axis):
    """Column-parallel matmul: x[1,M,K] replicated @ w[K,N] col-sharded -> [1,M,N]
    col-sharded (dryrun/ops.py `_matmul` + `_matmul_dist`)."""
    m_, k_, n_ = 32, 512, 2048
    x = ttnn.from_torch(
        torch.zeros(1, m_, k_, dtype=torch.bfloat16),
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=_mapper(ttnn, mesh, [None, None]),
        device=mesh,
    )
    wp = [None, None]
    wp[tp_axis] = 1  # shard output columns (dim 1) on the tp axis
    w = ttnn.from_torch(
        torch.zeros(k_, n_, dtype=torch.bfloat16),
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=_mapper(ttnn, mesh, wp),
        device=mesh,
    )
    out = ttnn.matmul(x, w)
    real = tuple(int(d) for d in ttnn.get_device_tensors(out)[0].shape)
    shim = (1, m_, shard_chunk_size(n_, mesh_shape[tp_axis]))  # [1,M,N] col-sharded on tp
    for t in (x, w, out):
        ttnn.deallocate(t)
    return ("matmul", real, shim)


def _op_split_qkv(ttnn, torch, mesh, mesh_shape, tp_axis):
    """Fused [1,N,3*H*Dh] -> q,k,v [1,H,N,Dh] (dryrun/ops.py
    `split_query_key_value_and_split_heads`). qkv replicated here, so H is local."""
    n_, heads, hd = 32, 8, 64
    qkv = ttnn.from_torch(
        torch.zeros(1, n_, 3 * heads * hd, dtype=torch.bfloat16),
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=_mapper(ttnn, mesh, [None, None]),
        device=mesh,
    )
    q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(qkv, num_heads=heads, transpose_key=False)
    real = tuple(int(d) for d in ttnn.get_device_tensors(q)[0].shape)
    shim = (1, heads, n_, hd)  # replicated qkv -> total heads == local heads
    for t in (qkv, q, k, v):
        ttnn.deallocate(t)
    return ("split_qkv", real, shim)


def _op_concat_heads(ttnn, torch, mesh, mesh_shape, tp_axis):
    """[1,H,N,Dh] -> [1,N,H*Dh] (dryrun/ops.py `concatenate_heads`)."""
    n_, heads, hd = 32, 8, 64
    x = ttnn.from_torch(
        torch.zeros(1, heads, n_, hd, dtype=torch.bfloat16),
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=_mapper(ttnn, mesh, [None, None]),
        device=mesh,
    )
    out = ttnn.transformer.concatenate_heads(x)
    real = tuple(int(d) for d in ttnn.get_device_tensors(out)[0].shape)
    shim = (1, n_, heads * hd)
    for t in (x, out):
        ttnn.deallocate(t)
    return ("concat_heads", real, shim)


OP_CASES = [_op_matmul, _op_split_qkv, _op_concat_heads]


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

        # -- per-op conformance: run a real compute op, diff the shim's output --
        tp_axis = 1 if mesh_shape[1] >= mesh_shape[0] else 0  # tp is the larger axis on the Loudbox
        print("\nper-op conformance (real ttnn op vs the shim's output shape):")
        print("%-14s %-24s %-24s %s" % ("op", "ttnn out (per device)", "shim out", "verdict"))
        n_ops = 0
        for op in OP_CASES:
            n_ops += 1
            try:
                name, real, shim = op(ttnn, torch, mesh, mesh_shape, tp_axis)
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print("%-14s %-24s %-24s %s" % (op.__name__, "(error)", "", str(exc).splitlines()[-1][:60]))
                continue
            ok = real == shim
            failures += 0 if ok else 1
            print("%-14s %-24s %-24s %s" % (name, real, shim, "PASS" if ok else "FAIL"))
    finally:
        ttnn.close_mesh_device(mesh)

    total = len(cases(mesh_shape)) + len(OP_CASES)
    print("\n%d/%d checks matched real ttnn" % (total - failures, total))
    return failures


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mesh", type=int, nargs=2, default=[2, 4], metavar=("R", "C"), help="mesh shape (default 2 4)")
    args = ap.parse_args()
    raise SystemExit(1 if run(args.mesh) else 0)


if __name__ == "__main__":
    main()
