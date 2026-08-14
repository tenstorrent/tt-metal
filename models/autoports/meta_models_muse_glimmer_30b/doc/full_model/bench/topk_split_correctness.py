# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Does a two-stage split ``ttnn.topk`` return the same answer as one call?

The split that reaches ``ttnn.topk``'s multi-core factory is worth 9.2 ms/token, but
the first end-to-end run of it changed the sampled tokens, so before anything else
the *semantics* have to be pinned down rather than assumed.

The question is precisely what ``indices_tensor=`` means. The unsplit path relies on
``ttnn.topk(x, indices_tensor=I)`` returning **the corresponding entries of I**, not
positions into ``x`` -- that is what makes the returned values local shard indices
that ``tt_indices_device_offsets`` can offset into global vocab ids. A two-stage
split relies on the same contract *twice*: once per piece, and once more over the
concatenated candidates.

It checks the rejected two-stage arm, the **shipped** concat-only arm, and a single
call, against a torch reference.

One thing not to misread in the output: ``single_call_matches_torch`` and
``two_stage_matches_torch`` are both False, and only the second is a defect. bf16 has an
8-bit mantissa, so a 65536-sample Gaussian produces many *exactly tied* values near the
maximum -- six of the top eight here are 3.3125 -- and any correct top-k may order tied
values however it likes. So the torch comparison is reported for the record but the
shipped path is judged on the two properties the sampler actually needs: that the
candidate set **contains** the true top-K, and that each returned index is the position
its returned value came from. The two-stage arm fails the second, which is the whole
finding.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch

import ttnn

ROOT = pathlib.Path(__file__).resolve().parents[3]
REPO = ROOT.parents[2]
sys.path.insert(0, str(REPO))

from models.autoports.meta_models_muse_glimmer_30b.tt.multichip_decoder import (  # noqa: E402
    close_multichip_mesh,
    open_multichip_mesh,
)

ROWS = 32
K = 32


def say(*args) -> None:
    print(*args, flush=True)


def to_device(mesh, tensor, dtype):
    return ttnn.from_torch(
        tensor,
        device=mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def first_row(tensor) -> torch.Tensor:
    return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0])[0, 0, 0].float()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=65536)
    parser.add_argument("--pieces", type=int, default=2)
    parser.add_argument("--out", default="topk_split_correctness.json")
    args = parser.parse_args()

    width, pieces = args.width, args.pieces
    per_piece = width // pieces

    torch.manual_seed(5)
    values = torch.randn(1, 1, ROWS, width, dtype=torch.bfloat16)
    # An identity map, so a returned index can be checked against the value at that
    # position. Note the *values* are not distinct -- see the module docstring.
    indices = torch.arange(width, dtype=torch.int32).reshape(1, 1, 1, width).repeat(1, 1, ROWS, 1)

    reference = torch.topk(values[0, 0, 0].float(), k=K)
    mesh = open_multichip_mesh()
    out = {}
    try:
        stable = ttnn.device.is_wormhole_b0(mesh) or ttnn.device.is_blackhole(mesh)
        tt_values = to_device(mesh, values, ttnn.bfloat16)
        tt_indices = to_device(mesh, indices, ttnn.uint16)

        # Stage 0: one call over the whole width -- the contract the unsplit path uses.
        one_v, one_i = ttnn.topk(tt_values, k=K, dim=-1, indices_tensor=tt_indices, stable=stable)
        out["single_call_indices"] = [int(v) for v in first_row(one_i)[:8]]
        out["single_call_matches_torch"] = out["single_call_indices"] == [int(v) for v in reference.indices[:8]]

        # Stage 1: per-piece top-k.
        x_list = ttnn.split(tt_values, per_piece, dim=3)
        i_list = ttnn.split(tt_indices, per_piece, dim=3)
        vals, idxs = [], []
        for x_piece, i_piece in zip(x_list, i_list):
            v, i = ttnn.topk(x_piece, k=K, dim=-1, indices_tensor=i_piece, stable=stable)
            vals.append(v)
            idxs.append(i)
        out["pieces"] = len(vals)
        out["piece0_indices"] = [int(v) for v in first_row(idxs[0])[:8]]
        out["piece1_indices"] = [int(v) for v in first_row(idxs[1])[:8]]
        # Piece 1's indices must be >= per_piece if indices_tensor is honoured; if the
        # op returns positions instead they will be < per_piece.
        out["piece1_indices_are_global"] = all(v >= per_piece for v in out["piece1_indices"])

        cat_v = ttnn.concat(vals, dim=3)
        cat_i = ttnn.concat(idxs, dim=3)
        out["concat_width"] = int(cat_v.shape[-1])
        out["concat_indices_dtype"] = str(cat_i.dtype)

        # Stage 2: reduce the candidates back to K.
        two_v, two_i = ttnn.topk(cat_v, k=K, dim=-1, indices_tensor=cat_i, stable=stable)
        out["two_stage_indices"] = [int(v) for v in first_row(two_i)[:8]]
        out["two_stage_values"] = [round(float(v), 4) for v in first_row(two_v)[:8]]
        out["single_call_values"] = [round(float(v), 4) for v in first_row(one_v)[:8]]
        out["torch_indices"] = [int(v) for v in reference.indices[:8]]
        out["torch_values"] = [round(float(v), 4) for v in reference.values[:8]]
        out["two_stage_matches_single"] = out["two_stage_indices"] == out["single_call_indices"]
        out["two_stage_matches_torch"] = out["two_stage_indices"] == out["torch_indices"]

        # ---------------------------------------------------------- the shipped path
        # The two-stage arm above is the one that was *rejected*.  What ships is the
        # concat of the per-piece top-k, with no second reduction, so that is what has
        # to be asserted -- and asserted on the two properties the sampler depends on
        # rather than on an exact match against torch's ordering.
        #
        # (1) **Containment.** The gathered candidate set must contain the true top-K,
        #     because everything downstream selects from it. Order within the set does
        #     not matter: ``ttnn.sampling`` re-selects, and the tie-break pass takes an
        #     explicit min over global indices.
        # (2) **Value/index correspondence.** Each returned index must actually be the
        #     position its returned value came from. This is the property the rejected
        #     two-stage arm broke, and the only one that can silently produce a wrong
        #     token.
        shipped_values = torch.cat([first_row(v).reshape(1, -1) for v in vals], dim=1)[0]
        shipped_indices = torch.cat([first_row(i).reshape(1, -1) for i in idxs], dim=1)[0]
        reference_set = {int(v) for v in reference.indices}
        shipped_set = {int(v) for v in shipped_indices}
        out["shipped_candidate_count"] = int(shipped_indices.numel())
        out["shipped_contains_global_topk"] = reference_set.issubset(shipped_set)
        out["shipped_missing_from_topk"] = sorted(reference_set - shipped_set)
        flat = values[0, 0, 0].float()
        mismatched = [
            int(idx)
            for value, idx in zip(shipped_values.tolist(), shipped_indices.tolist())
            if abs(float(flat[int(idx)]) - float(value)) > 0
        ]
        out["shipped_value_index_mismatches"] = mismatched
        out["shipped_value_index_consistent"] = not mismatched

        # Why the two ``*_matches_torch`` rows above are False, so a reader does not have
        # to guess: bf16 has an 8-bit mantissa, so a 65536-sample Gaussian has many exact
        # ties near the maximum, and *any* correct top-k may order tied values however it
        # likes. Count them rather than assert an ordering.
        top_values = reference.values.tolist()
        out["distinct_values_in_torch_topk"] = len(set(top_values))
        out["torch_topk_has_ties"] = len(set(top_values)) < len(top_values)
        out["values_agree_single_vs_two_stage"] = out["two_stage_values"] == out["single_call_values"]
        out["values_agree_single_vs_torch"] = out["single_call_values"] == out["torch_values"]

        for key, value in out.items():
            say(f"SPLIT {key}={value}")
        (ROOT / "doc/full_model" / args.out).write_text(json.dumps(out, indent=2) + "\n")
        say("SPLIT_OK")
        return 0
    finally:
        close_multichip_mesh(mesh)


if __name__ == "__main__":
    raise SystemExit(main())
