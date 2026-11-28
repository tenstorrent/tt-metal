# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Teacher-forced DFlash acceptance of a published draft on real target hidden states (CPU).

Input is a capture written by ``tests/gen_dflash_target_capture.py``: every target layer's
output for one fixed token sequence plus the target's teacher-forced greedy token at every
position.  For each anchor position ``p`` the draft receives the captured states of
positions ``< p`` (the last 511), the known token at ``p`` as the bonus, and proposes 15
tokens for ``p + 1 .. p + 15`` exactly as one served DFlash round does.

Two numbers per anchor:

* ``first``: the first proposal equals the target's greedy token for ``p + 1``.  This is
  exactly the first acceptance decision of a served round.
* ``accepted``: proposals are accepted while each equals the target's greedy token for its
  position; the count can only continue past a proposal that also equals the forced
  sequence token, because the capture holds target predictions for the forced prefix only.
  It is therefore a lower bound on served acceptance.

Usage: ``python -m models.demos.laguna.tests.dflash_acceptance`` prints
the selected checkpoint's numbers for its published target layers and for a wrong-layer
control (the inputs of the target layers instead of their outputs).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch


def default_capture_path(model_slug: str) -> Path:
    override = os.environ.get("LAGUNA_DFLASH_TARGET_CAPTURE")
    if override:
        return Path(override)
    return Path.home() / ".cache" / "laguna_dflash" / f"{model_slug}_aime24_target_capture.pt"


def load_target_embedding_and_lm_head(model_id: str) -> tuple[torch.Tensor, torch.Tensor]:
    """BF16 ``model.embed_tokens.weight`` and ``lm_head.weight`` from the cached target snapshot."""

    import json

    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    snapshot = Path(snapshot_download(model_id, allow_patterns=["*.json"], local_files_only=True))
    with (snapshot / "model.safetensors.index.json").open() as index_file:
        weight_map = json.load(index_file)["weight_map"]
    out = {}
    for key in ("model.embed_tokens.weight", "lm_head.weight"):
        with safe_open(snapshot / weight_map[key], "pt") as weights:
            out[key] = weights.get_tensor(key).to(torch.bfloat16)
    return out["model.embed_tokens.weight"], out["lm_head.weight"]


@dataclass(frozen=True)
class AcceptanceResult:
    anchors: tuple[int, ...]
    first: tuple[bool, ...]
    accepted: tuple[int, ...]

    @property
    def first_rate(self) -> float:
        return sum(self.first) / len(self.first)

    @property
    def mean_accepted(self) -> float:
        return sum(self.accepted) / len(self.accepted)


@torch.inference_mode()
def teacher_forced_acceptance(
    reference,
    capture: dict,
    layer_ids: Sequence[int],
    *,
    target_embedding: torch.Tensor,
    target_lm_head: torch.Tensor,
    anchors: Sequence[int],
) -> AcceptanceResult:
    """Run one served-style DFlash round per anchor on captured target states."""

    from models.demos.laguna.tt.dflash_reference import build_proposal_block

    config = reference.config
    tokens = capture["token_ids"]
    greedy = capture["greedy"]
    aux = capture["layer_outputs"][list(layer_ids)].permute(1, 0, 2).contiguous()  # [S, num_aux, H]
    first, accepted = [], []
    for p in anchors:
        start = max(0, int(p) - (config.sliding_window - 1))
        block = build_proposal_block(config, bonus_token_id=int(tokens[p]), last_valid_position=int(p) - 1)
        logits = reference.proposal_logits(
            block,
            target_embedding_weight=target_embedding,
            target_lm_head_weight=target_lm_head,
            context_aux_hidden_states=aux[start:p],
            context_positions=torch.arange(start, int(p)),
        )
        drafts = torch.argmax(logits.float(), dim=-1).tolist()
        first.append(drafts[0] == int(greedy[p]))
        count = 0
        for k, draft in enumerate(drafts):
            position = int(p) + 1 + k  # the position this proposal predicts
            if position >= len(tokens) or draft != int(greedy[position - 1]):
                break
            count += 1
            if draft != int(tokens[position]):
                break  # the target's predictions beyond here are for a different prefix
        accepted.append(count)
    return AcceptanceResult(tuple(int(a) for a in anchors), tuple(first), tuple(accepted))


def main():
    from models.demos.laguna.tt.dflash_reference import LagunaDFlashCheckpoint
    from models.demos.laguna.tt.model_spec import DFLASH_SPEC, MODEL_ID, MODEL_SLUG

    capture = torch.load(default_capture_path(MODEL_SLUG))
    assert capture["model"] == MODEL_ID, capture["model"]
    reference = LagunaDFlashCheckpoint().load_reference()
    embedding, lm_head = load_target_embedding_and_lm_head(MODEL_ID)
    prompt_len = int(capture["prompt_len"])
    total = int(capture["token_ids"].numel())
    anchors = range(prompt_len, total - 1)
    print(f"{MODEL_ID}: {len(anchors)} anchors, readiness agreement {capture['readiness_top1_agreement']:.4f}")
    ids = DFLASH_SPEC.target_layer_ids
    controls = {
        "published target layers": ids,
        "layer inputs (ids - 1)": tuple(i - 1 for i in ids),
        "reversed slice order": tuple(reversed(ids)),
        "last target layer only": (ids[-1],) * len(ids),
    }
    for name, layer_ids in controls.items():
        result = teacher_forced_acceptance(
            reference, capture, layer_ids, target_embedding=embedding, target_lm_head=lm_head, anchors=anchors
        )
        print(
            f"{name:26s} {layer_ids}: first={result.first_rate:.3f} mean_accepted={result.mean_accepted:.3f} "
            f"hist={[result.accepted.count(n) for n in range(16)]}",
            flush=True,
        )


if __name__ == "__main__":
    main()
