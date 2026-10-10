# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU golden tests for the H3 adapter loader's silent wrong-answer paths: per-target alpha scaling,
file-alpha fallback, and the fused-QKV validation. No device -- the fused register is driven with a
capturing stub so the delta scaling can be asserted without a transformer or hardware."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from ....experimental.lora.h3_adapter_loader import _collect_pairs, _file_alpha, _register_fused, _scale_of

NUM_HEADS = 4
HEAD_DIM = 8
INNER_DIM = NUM_HEADS * HEAD_DIM  # 32
RANK = 2
IN_DIM = 5


class _CaptureQKV:
    """Stands in for `to_qkv`; records the (A, B, scale) register_lora would fuse."""

    def __init__(self):
        self.calls: list[tuple[torch.Tensor, torch.Tensor, float]] = []

    def register_lora(self, a, b, *, scale, name):
        self.calls.append((a, b, scale))
        return len(self.calls) - 1


def _attn():
    return SimpleNamespace(
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        rotary_dim=HEAD_DIM,
        inner_dim=INNER_DIM,
        n_local_heads=NUM_HEADS,
        parallel_config=SimpleNamespace(tensor_parallel=SimpleNamespace(factor=1)),
        to_qkv=_CaptureQKV(),
    )


def _qkv(seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        slot: {
            "A": torch.randn(RANK, IN_DIM, generator=g, dtype=torch.float32),
            "B": torch.randn(INNER_DIM, RANK, generator=g, dtype=torch.float32),
        }
        for slot in ("q", "k", "v")
    }


def _col_block(b_fused, position):
    return b_fused[:, position * RANK : (position + 1) * RANK]


def test_fused_folds_each_sources_own_alpha():
    """Per-target alphas must scale their own q/k/v columns -- not all three by to_q's alpha."""
    qkvs = _qkv()
    base = lambda slot: f"transformer_blocks.0.attn.to_{slot}"  # noqa: E731

    attn1 = _attn()
    _register_fused(attn1, qkvs, 1.0, "a", {base(s): RANK for s in ("q", "k", "v")}, None, "transformer_blocks", 0)
    b1, scale1 = attn1.to_qkv.calls[0][1], attn1.to_qkv.calls[0][2]

    attn2 = _attn()
    factors = {"q": 1, "k": 2, "v": 3}
    _register_fused(
        attn2, qkvs, 1.0, "a", {base(s): RANK * factors[s] for s in ("q", "k", "v")}, None, "transformer_blocks", 0
    )
    b2 = attn2.to_qkv.calls[0][1]

    # alpha/rank == 1 for every slot in run 1, so run 2's column blocks are run 1's times each slot's factor.
    for pos, slot in enumerate(("q", "k", "v")):
        torch.testing.assert_close(_col_block(b2, pos), _col_block(b1, pos) * factors[slot])

    # The caller strength is what reaches register_lora; the alpha is already folded into B.
    assert scale1 == 1.0


def test_fused_file_alpha_fallback_scales_every_slot():
    """With no per-target alphas, the file alpha applies uniformly (alpha/rank)."""
    qkvs = _qkv(seed=1)
    attn_unit = _attn()
    _register_fused(attn_unit, qkvs, 1.0, "a", {}, float(RANK), "transformer_blocks", 0)
    attn_2x = _attn()
    _register_fused(attn_2x, qkvs, 1.0, "a", {}, float(2 * RANK), "transformer_blocks", 0)
    torch.testing.assert_close(attn_2x.to_qkv.calls[0][1], attn_unit.to_qkv.calls[0][1] * 2.0)


def test_fused_missing_slot_raises(expect_error):
    qkvs = _qkv()
    del qkvs["v"]
    with expect_error(RuntimeError, "missing"):
        _register_fused(_attn(), qkvs, 1.0, "a", {}, None, "transformer_blocks", 0)


def test_fused_rank_disagreement_raises(expect_error):
    qkvs = _qkv()
    qkvs["k"]["A"] = torch.randn(RANK + 1, IN_DIM, dtype=torch.float32)
    with expect_error(ValueError, "ranks disagree"):
        _register_fused(_attn(), qkvs, 1.0, "a", {}, None, "transformer_blocks", 0)


def test_scale_of_prefers_per_target_then_file_then_one():
    ab = {"A": torch.zeros(8, IN_DIM)}  # rank 8
    assert _scale_of("t.0.attn.to_q", ab, {"t.0.attn.to_q": 16.0}, None) == pytest.approx(2.0)
    assert _scale_of("t.0.attn.to_q", ab, {}, 16.0) == pytest.approx(2.0)
    assert _scale_of("t.0.attn.to_q", ab, {}, None) == 1.0


def test_file_alpha_reads_known_metadata_keys():
    assert _file_alpha({"alpha": "16"}) == 16.0
    assert _file_alpha({"lora_alpha": "8"}) == 8.0
    assert _file_alpha({}) is None


def test_collect_pairs_maps_slots_and_alphas():
    raw = {
        "transformer_blocks.0.attn.to_q.lora_A.weight": torch.zeros(RANK, IN_DIM),
        "transformer_blocks.0.attn.to_q.lora_B.weight": torch.zeros(INNER_DIM, RANK),
        "transformer_blocks.0.attn.to_q.alpha": torch.tensor(16.0),
    }
    pairs, alphas = _collect_pairs(raw)
    assert set(pairs["transformer_blocks.0.attn.to_q"]) == {"A", "B"}
    assert alphas["transformer_blocks.0.attn.to_q"] == 16.0


def test_collect_pairs_rejects_non_lora_key(expect_error):
    with expect_error(RuntimeError, "no LoRA convention"):
        _collect_pairs({"transformer_blocks.0.attn.to_q.weight": torch.zeros(1)})


def test_collect_pairs_rejects_half_pair(expect_error):
    with expect_error(RuntimeError, "half pair"):
        _collect_pairs({"transformer_blocks.0.attn.to_q.lora_A.weight": torch.zeros(RANK, IN_DIM)})
