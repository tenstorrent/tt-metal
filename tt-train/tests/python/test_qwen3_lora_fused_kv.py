# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Host-side tests for Qwen3 LoRA targeting + merge/export of the fused KV adapter.

Qwen3 attention fuses K and V into a single ``kv_proj`` projection, so:

  - LoRA targets must name ``kv_proj`` (not the pre-fusion ``k_proj``/``v_proj``);
    ``k_proj``/``v_proj`` are accepted as deprecated back-compat aliases that
    normalize to ``kv_proj`` with a ``DeprecationWarning``
    (``utils.lora.normalize_lora_targets``), so the advertised "all projections"
    default still installs an adapter that trains K and V.
  - On HF export the single fused ``kv_proj`` LoRA delta (spanning
    ``[2*kv_out, hidden]``) must be split into the HF ``k_proj`` / ``v_proj``
    halves (K re-permuted for RoPE, V as-is) and each half added to its own HF
    entry -- the same split the base weight uses (``_split_fused_kv``).

These tests are pure numpy/torch (no Tenstorrent device required).
"""

import os
import sys
import warnings

import pytest

torch = pytest.importorskip("torch")

_QWEN3_EXAMPLE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "sources", "examples", "qwen3")
)
# Force the qwen3 example dir to the front of sys.path so ``import utils`` resolves
# here even if a sibling example dir is already on the path.
if _QWEN3_EXAMPLE_DIR in sys.path:
    sys.path.remove(_QWEN3_EXAMPLE_DIR)
sys.path.insert(0, _QWEN3_EXAMPLE_DIR)

from ttml.models.qwen3.weights import unpermute_proj_rows  # noqa: E402

# The qwen3 example ships a top-level ``utils`` package, and so do sibling example
# dirs (e.g. examples/grpo, imported by test_grpo_trainer). In a single pytest
# session another test module may have already imported its own ``utils`` first,
# caching it in sys.modules and shadowing ours (sys.path.insert cannot override an
# already-imported module). Temporarily evict any cached ``utils*`` so the imports
# below resolve against _QWEN3_EXAMPLE_DIR, then restore the sibling's modules so
# we do not break whichever test imported them.
_saved_utils = {k: sys.modules.pop(k) for k in list(sys.modules) if k == "utils" or k.startswith("utils.")}
try:
    from utils.lora import (  # noqa: E402
        LORA_TARGETS_ACCEPTED,
        LORA_TARGETS_ALL,
        normalize_lora_targets,
    )
    from utils.save_load import _split_fused_kv  # noqa: E402
finally:
    for _k in [k for k in list(sys.modules) if k == "utils" or k.startswith("utils.")]:
        del sys.modules[_k]
    sys.modules.update(_saved_utils)

# ---------------------------------------------------------------------------
# LoRA target retargeting / alias normalization
# ---------------------------------------------------------------------------


def test_targets_all_uses_fused_kv_not_k_v():
    """The canonical target set names the fused kv_proj and drops k_proj/v_proj."""
    assert "kv_proj" in LORA_TARGETS_ALL
    assert "k_proj" not in LORA_TARGETS_ALL
    assert "v_proj" not in LORA_TARGETS_ALL
    # Sanity: the other projections are still present.
    for proj in ("q_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
        assert proj in LORA_TARGETS_ALL


def test_aliases_accepted_and_normalized_to_kv_proj():
    """k_proj / v_proj are accepted (CLI) and normalize to the single fused kv_proj."""
    assert "k_proj" in LORA_TARGETS_ACCEPTED
    assert "v_proj" in LORA_TARGETS_ACCEPTED
    assert normalize_lora_targets(["k_proj"]) == ["kv_proj"]
    assert normalize_lora_targets(["v_proj"]) == ["kv_proj"]
    # Both aliases collapse to a single kv_proj (de-duped), order preserved.
    assert normalize_lora_targets(["q_proj", "k_proj", "v_proj"]) == ["q_proj", "kv_proj"]


def test_aliases_emit_deprecation_warning():
    """The k_proj / v_proj aliases still resolve, but warn that they are deprecated."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert normalize_lora_targets(["q_proj", "k_proj", "v_proj"]) == ["q_proj", "kv_proj"]

    messages = [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(messages) == 1, f"Expected exactly one DeprecationWarning, got {[str(w.message) for w in caught]}"
    assert "k_proj" in messages[0] and "v_proj" in messages[0]
    assert "kv_proj" in messages[0]

    # Canonical-only targets must stay warning-free.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        normalize_lora_targets(list(LORA_TARGETS_ALL))
    assert not [w for w in caught if issubclass(w.category, DeprecationWarning)]


def test_normalize_is_idempotent_and_passes_through_unknowns():
    assert normalize_lora_targets(list(LORA_TARGETS_ALL)) == list(LORA_TARGETS_ALL)
    assert normalize_lora_targets(normalize_lora_targets(["k_proj", "v_proj"])) == ["kv_proj"]
    # Unknown names pass through unchanged (caller validates separately).
    assert normalize_lora_targets(["not_a_proj"]) == ["not_a_proj"]


# ---------------------------------------------------------------------------
# Fused kv_proj LoRA delta split on export
# ---------------------------------------------------------------------------

NUM_KV_HEADS = 4
HEAD_DIM = 8  # even (RoPE)
HIDDEN = 16
KV_OUT = NUM_KV_HEADS * HEAD_DIM


def _build_fused_delta(k_delta_hf, v_delta_hf, tp_size, interleaved):
    """Build a fused kv_proj delta in ttml layout from per-projection HF deltas.

    Mirrors how the fused base weight is laid out: K is un-permuted (RoPE row
    permute) then placed with V. single-device -> [all-K ; all-V];
    TP -> per-shard interleave [K_s0,V_s0,K_s1,V_s1,...].
    """
    k = unpermute_proj_rows(k_delta_hf, num_heads=NUM_KV_HEADS)
    v = v_delta_hf
    if not interleaved:
        return torch.cat([k, v], dim=0)
    per = KV_OUT // tp_size
    k_blk = k.reshape(tp_size, per, HIDDEN)
    v_blk = v.reshape(tp_size, per, HIDDEN)
    return torch.stack([k_blk, v_blk], dim=1).reshape(2 * KV_OUT, HIDDEN)


@pytest.mark.parametrize("interleaved", [False, True], ids=["single", "tp"])
@pytest.mark.parametrize("tp_size", [1, 2, 4])
def test_fused_kv_delta_splits_into_hf_k_and_v(interleaved, tp_size):
    """A fused kv_proj LoRA delta splits+re-permutes back to the original HF K/V deltas.

    This is exactly the transform _merge_lora_inplace applies for the kv_proj
    target on export (split via _split_fused_kv, re-permute the K half, V as-is).
    """
    if not interleaved and tp_size != 1:
        pytest.skip("single-device layout is only meaningful at tp=1")

    from ttml.models.qwen3.weights import repermute_proj_rows

    gen = torch.Generator().manual_seed(100 + tp_size)
    k_delta_hf = torch.randn(KV_OUT, HIDDEN, generator=gen)
    v_delta_hf = torch.randn(KV_OUT, HIDDEN, generator=gen)

    fused_delta = _build_fused_delta(k_delta_hf, v_delta_hf, tp_size, interleaved)
    assert fused_delta.shape[0] == 2 * KV_OUT

    k_out = _split_fused_kv(fused_delta, "k", KV_OUT, tp_size, interleaved=interleaved)
    v_out = _split_fused_kv(fused_delta, "v", KV_OUT, tp_size, interleaved=interleaved)
    k_out = repermute_proj_rows(k_out, num_heads=NUM_KV_HEADS)  # K re-permuted; V untouched

    torch.testing.assert_close(k_out, k_delta_hf, rtol=0, atol=0)
    torch.testing.assert_close(v_out, v_delta_hf, rtol=0, atol=0)
