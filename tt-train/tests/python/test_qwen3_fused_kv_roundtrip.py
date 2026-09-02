# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Host-side round-trip tests for the Qwen3 fused-KV export split.

The Qwen3 loader fuses the HF ``k_proj`` and ``v_proj`` weights into a single
``kv_proj`` parameter (K rows re-permuted for RoPE, then concatenated with V):

  - single-device / FSDP (``combine_kv``):    rows are ``[all-K ; all-V]``
  - ColumnParallel TP    (``combine_kv_tp``): rows are the per-shard interleave
                                              ``[K_s0, V_s0, K_s1, V_s1, ...]``

On export / checkpoint save the reverse must reproduce the two original HF
tensors exactly: split the fused param, de-interleave (TP only), and re-permute
the K half back to HF layout (V is never permuted). This is the inverse the
export path (``utils.save_load._split_fused_kv`` /
``_build_inv_transforms`` / ``_apply_inv_transform``) is responsible for.

These tests exercise the REAL shipped split helpers against the REAL permutation
helpers and assert a bit-exact round-trip for weights and biases, single-device
and TP (tp in {1,2,4}), including when the gathered fused tensor still carries
TILE row-padding. The math is pure torch, so no Tenstorrent device is required
(the file is intentionally NOT marked ``requires_device``).
"""

import os
import sys

import pytest

torch = pytest.importorskip("torch")

# The split helpers under test live in the Qwen3 example package
# (sources/examples/qwen3/utils/save_load.py), which is not part of the installed
# ``ttml`` wheel, so add the example root to sys.path for ``import utils.save_load``.
_QWEN3_EXAMPLE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "sources", "examples", "qwen3")
)
# Force the qwen3 example dir to the front of sys.path so ``import utils`` resolves
# here even if a sibling example dir is already on the path.
if _QWEN3_EXAMPLE_DIR in sys.path:
    sys.path.remove(_QWEN3_EXAMPLE_DIR)
sys.path.insert(0, _QWEN3_EXAMPLE_DIR)

# Real permutation helpers (HF <-> ttml layout) from the shipped package.
from ttml.models.qwen3.weights import unpermute_proj_rows  # noqa: E402

# The qwen3 example ships a top-level ``utils`` package, and so do sibling example
# dirs (e.g. examples/grpo, imported by test_grpo_trainer). In a single pytest
# session another test module may have already imported its own ``utils`` first,
# caching it in sys.modules and shadowing ours (sys.path.insert cannot override an
# already-imported module). Temporarily evict any cached ``utils*`` so the import
# below resolves against _QWEN3_EXAMPLE_DIR, then restore the sibling's modules so
# we do not break whichever test imported them.
_saved_utils = {k: sys.modules.pop(k) for k in list(sys.modules) if k == "utils" or k.startswith("utils.")}
try:
    # Real export-side split helpers under test.
    from utils.save_load import (  # noqa: E402
        _apply_inv_transform,
        _build_inv_transforms,
        _split_fused_kv,
    )
finally:
    for _k in [k for k in list(sys.modules) if k == "utils" or k.startswith("utils.")]:
        del sys.modules[_k]
    sys.modules.update(_saved_utils)

# ---------------------------------------------------------------------------
# Reference load-side combine (the input generator for the round-trip)
#
# Mirrors ttml/models/qwen3/weights.py (combine_kv) and
# examples/qwen3/model_qwen3_distributed.py (combine_kv_tp). Built here as the
# "known-good" fused tensor so the assertions target the shipped SPLIT code.
# ---------------------------------------------------------------------------


def _combine_kv_single(k_hf, v_hf, num_kv_heads):
    """[all-K ; all-V] with K re-permuted for RoPE (single-device / FSDP)."""
    k_w = unpermute_proj_rows(k_hf, num_heads=num_kv_heads)
    return torch.cat([k_w, v_hf], dim=0)


def _combine_kv_tp(k_hf, v_hf, num_kv_heads, tp_size):
    """Per-shard interleave [K_s0,V_s0,K_s1,V_s1,...] with K re-permuted (TP)."""
    k_w = unpermute_proj_rows(k_hf, num_heads=num_kv_heads)
    v_w = v_hf
    kv_out = k_w.shape[0]
    per = kv_out // tp_size
    if k_w.dim() == 2:
        k_blk = k_w.reshape(tp_size, per, k_w.shape[1])
        v_blk = v_w.reshape(tp_size, per, v_w.shape[1])
        return torch.stack([k_blk, v_blk], dim=1).reshape(2 * kv_out, k_w.shape[1])
    k_blk = k_w.reshape(tp_size, per)
    v_blk = v_w.reshape(tp_size, per)
    return torch.stack([k_blk, v_blk], dim=1).reshape(2 * kv_out)


def _tile_pad_rows(t, pad):
    """Append ``pad`` garbage rows to simulate TILE row-padding on the gathered tensor."""
    if pad == 0:
        return t
    tail = torch.randn(pad, t.shape[1]) if t.dim() == 2 else torch.randn(pad)
    return torch.cat([t, tail], dim=0)


# ---------------------------------------------------------------------------
# Fixtures / config
# ---------------------------------------------------------------------------

NUM_KV_HEADS = 4
HEAD_DIM = 8  # must be even (RoPE splits each head into real/imag halves)
HIDDEN = 16
KV_OUT = NUM_KV_HEADS * HEAD_DIM  # true K (and V) output width


def _random_kv(is_weight, seed):
    g = torch.Generator().manual_seed(seed)
    if is_weight:
        return (
            torch.randn(KV_OUT, HIDDEN, generator=g),
            torch.randn(KV_OUT, HIDDEN, generator=g),
        )
    return (
        torch.randn(KV_OUT, generator=g),
        torch.randn(KV_OUT, generator=g),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("is_weight", [True, False], ids=["weight", "bias"])
@pytest.mark.parametrize("pad", [0, 17], ids=["nopad", "tilepad"])
def test_split_fused_kv_single_device(is_weight, pad):
    """Single-device combine -> split reproduces K and V exactly."""
    k_hf, v_hf = _random_kv(is_weight, seed=0)
    fused = _tile_pad_rows(_combine_kv_single(k_hf, v_hf, NUM_KV_HEADS), pad)

    # _split_fused_kv returns the raw halves BEFORE the exporter's K re-permute,
    # so the K half here is still the loader's permuted k_w (the full re-permute
    # round-trip is asserted in test_apply_inv_transform_full_roundtrip).
    k_out = _split_fused_kv(fused, "k", kv_out=KV_OUT, tp_size=1, interleaved=False)
    v_out = _split_fused_kv(fused, "v", kv_out=KV_OUT, tp_size=1, interleaved=False)

    # V is never permuted -> exact match straight out of the split.
    assert torch.equal(v_out, v_hf), "single-device V half mismatch"
    # The K half must equal exactly the loader's permuted k_w (correct boundary).
    assert torch.equal(k_out, unpermute_proj_rows(k_hf, num_heads=NUM_KV_HEADS)), "single-device K half mismatch"


@pytest.mark.parametrize("is_weight", [True, False], ids=["weight", "bias"])
@pytest.mark.parametrize("tp_size", [1, 2, 4])
@pytest.mark.parametrize("pad", [0, 17], ids=["nopad", "tilepad"])
def test_split_fused_kv_tp(is_weight, tp_size, pad):
    """TP per-shard-interleaved combine -> split reproduces K and V exactly."""
    k_hf, v_hf = _random_kv(is_weight, seed=tp_size)
    fused = _tile_pad_rows(_combine_kv_tp(k_hf, v_hf, NUM_KV_HEADS, tp_size), pad)

    k_out = _split_fused_kv(fused, "k", kv_out=KV_OUT, tp_size=tp_size, interleaved=True)
    v_out = _split_fused_kv(fused, "v", kv_out=KV_OUT, tp_size=tp_size, interleaved=True)

    assert torch.equal(v_out, v_hf), f"TP(tp={tp_size}) V half mismatch"
    assert torch.equal(k_out, unpermute_proj_rows(k_hf, num_heads=NUM_KV_HEADS)), f"TP(tp={tp_size}) K half mismatch"


@pytest.mark.parametrize("interleaved", [False, True], ids=["single", "tp"])
@pytest.mark.parametrize("is_weight", [True, False], ids=["weight", "bias"])
@pytest.mark.parametrize("tp_size", [1, 2, 4])
def test_apply_inv_transform_full_roundtrip(interleaved, is_weight, tp_size):
    """End-to-end via the public entry point: _build_inv_transforms + _apply_inv_transform
    recover the ORIGINAL HF k_proj and v_proj (K re-permuted back to HF layout)."""
    if not interleaved and tp_size != 1:
        pytest.skip("single-device layout is only meaningful at tp=1")

    hp = "model.layers.0"
    k_name = f"{hp}.self_attn.k_proj.weight" if is_weight else f"{hp}.self_attn.k_proj.bias"
    v_name = f"{hp}.self_attn.v_proj.weight" if is_weight else f"{hp}.self_attn.v_proj.bias"

    fwd_name = "combine_kv_tp" if interleaved else "combine_kv"
    fwd_transforms = {k_name: (fwd_name, NUM_KV_HEADS, v_name)}
    inv = _build_inv_transforms(fwd_transforms)

    # Both HF names must get an inverse (the core of the fix: V is emitted too).
    assert k_name in inv and v_name in inv, "both k_proj and v_proj must have an inverse"
    split_tag = "split_kv_tp" if interleaved else "split_kv"
    assert inv[k_name] == (split_tag, "k", NUM_KV_HEADS)
    assert inv[v_name] == (split_tag, "v", NUM_KV_HEADS)

    k_hf, v_hf = _random_kv(is_weight, seed=100 + tp_size)
    if interleaved:
        fused = _combine_kv_tp(k_hf, v_hf, NUM_KV_HEADS, tp_size)
    else:
        fused = _combine_kv_single(k_hf, v_hf, NUM_KV_HEADS)

    k_rt = _apply_inv_transform(fused, k_name, inv, kv_out=KV_OUT, tp_size=tp_size)
    v_rt = _apply_inv_transform(fused, v_name, inv, kv_out=KV_OUT, tp_size=tp_size)

    assert torch.equal(k_rt, k_hf), "k_proj not recovered to original HF layout"
    assert torch.equal(v_rt, v_hf), "v_proj not recovered to original HF layout"


def test_build_inv_transforms_preserves_other_transforms():
    """The fused-KV addition must not disturb the existing unpermute inverses."""
    fwd = {
        "model.layers.0.self_attn.q_proj.weight": ("unpermute_proj", 8),
        "model.layers.0.self_attn.q_norm.weight": ("unpermute_norm",),
        "model.layers.0.self_attn.k_proj.weight": (
            "combine_kv",
            4,
            "model.layers.0.self_attn.v_proj.weight",
        ),
    }
    inv = _build_inv_transforms(fwd)
    assert inv["model.layers.0.self_attn.q_proj.weight"] == ("repermute_proj", 8)
    assert inv["model.layers.0.self_attn.q_norm.weight"] == ("repermute_norm",)
    assert inv["model.layers.0.self_attn.k_proj.weight"] == ("split_kv", "k", 4)
    assert inv["model.layers.0.self_attn.v_proj.weight"] == ("split_kv", "v", 4)


def test_split_is_true_inverse_not_naive_crop():
    """Guard against regressing to the old (buggy) behaviour.

    The pre-fix export path cropped the fused tensor's first ``kv_out`` rows with
    NO re-permute and never emitted V. Prove the correct K half differs from that
    naive crop (so the re-permute is actually exercised) and that V is non-empty.
    """
    k_hf, v_hf = _random_kv(is_weight=True, seed=7)
    fused = _combine_kv_single(k_hf, v_hf, NUM_KV_HEADS)

    naive_crop_k = fused[:KV_OUT]  # old path: crop, no repermute
    inv = _build_inv_transforms({"k": ("combine_kv", NUM_KV_HEADS, "v")})
    correct_k = _apply_inv_transform(fused, "k", inv, kv_out=KV_OUT, tp_size=1)
    correct_v = _apply_inv_transform(fused, "v", inv, kv_out=KV_OUT, tp_size=1)

    assert not torch.equal(naive_crop_k, correct_k), "re-permute must change the K half vs a naive crop"
    assert torch.equal(correct_k, k_hf), "correct K must equal the original HF k_proj"
    assert correct_v.shape[0] == KV_OUT and torch.equal(correct_v, v_hf), "V must be recovered, not dropped"
