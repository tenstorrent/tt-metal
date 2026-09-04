# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Host-side round-trip tests for the public Qwen3 safetensors fused-KV path.

The public ``ttml.models.qwen3`` model fuses HF ``k_proj`` and ``v_proj`` into a
single ``kv_proj`` parameter of width ``2*kv_out`` (rows ``[all-K ; all-V]``, K
re-permuted for RoPE, V as-is). ``safetensors_loader`` must therefore:

  - on load (``load_from_safetensors``): stage K and V, then fuse into kv_proj
    (``_fuse_kv``), and
  - on export (``export_hf_model``): split kv_proj back into k_proj / v_proj and
    re-permute K (``_split_kv``).

These tests exercise the REAL numpy helpers ``_fuse_kv`` / ``_split_kv`` (and the
name matcher ``_kv_match``) and assert a bit-exact HF -> ttml -> HF round-trip
for weights and biases. The math is pure numpy, so no Tenstorrent device is
required (the file is intentionally NOT marked ``requires_device``).
"""

import pytest

np = pytest.importorskip("numpy")

from ttml.models.qwen3.safetensors_loader import (  # noqa: E402
    _fuse_kv,
    _kv_match,
    _repermute_proj_rows,
    _split_kv,
    _unpermute_proj_rows,
)

NUM_KV_HEADS = 4
HEAD_DIM = 8  # even (RoPE splits each head into real/imag halves)
HIDDEN = 16
KV_OUT = NUM_KV_HEADS * HEAD_DIM


def _random_kv(is_weight, seed):
    rng = np.random.default_rng(seed)
    if is_weight:
        return (
            rng.standard_normal((KV_OUT, HIDDEN)).astype(np.float32),
            rng.standard_normal((KV_OUT, HIDDEN)).astype(np.float32),
        )
    return (
        rng.standard_normal((KV_OUT,)).astype(np.float32),
        rng.standard_normal((KV_OUT,)).astype(np.float32),
    )


@pytest.mark.parametrize("is_weight", [True, False], ids=["weight", "bias"])
def test_fuse_then_split_roundtrip(is_weight):
    """HF k_proj/v_proj -> fused kv_proj -> split back reproduces the originals exactly."""
    k_hf, v_hf = _random_kv(is_weight, seed=0)

    fused = _fuse_kv(k_hf, v_hf, NUM_KV_HEADS)
    assert fused.shape[0] == 2 * KV_OUT, "fused kv_proj must have 2*kv_out rows"

    k_rt, v_rt = _split_kv(fused, KV_OUT, NUM_KV_HEADS)
    np.testing.assert_array_equal(k_rt, k_hf, err_msg="k_proj not recovered after fuse->split")
    np.testing.assert_array_equal(v_rt, v_hf, err_msg="v_proj not recovered after fuse->split")


@pytest.mark.parametrize("is_weight", [True, False], ids=["weight", "bias"])
def test_fused_layout_is_k_then_v_with_permuted_k(is_weight):
    """The fused tensor is [K ; V] with only K carrying the RoPE row-permute."""
    k_hf, v_hf = _random_kv(is_weight, seed=1)
    fused = _fuse_kv(k_hf, v_hf, NUM_KV_HEADS)

    # First half is the *permuted* K (unpermute_proj_rows applied), second half is V verbatim.
    np.testing.assert_array_equal(fused[:KV_OUT], _unpermute_proj_rows(k_hf, NUM_KV_HEADS))
    np.testing.assert_array_equal(fused[KV_OUT : 2 * KV_OUT], v_hf)


@pytest.mark.parametrize("is_weight", [True, False], ids=["weight", "bias"])
def test_split_tolerates_tile_padding(is_weight):
    """A TILE-padded fused tensor (extra tail rows) still splits to exact K and V."""
    k_hf, v_hf = _random_kv(is_weight, seed=2)
    fused = _fuse_kv(k_hf, v_hf, NUM_KV_HEADS)

    pad = 13  # simulate rows padded past 2*kv_out
    if fused.ndim == 2:
        fused_padded = np.concatenate([fused, np.random.default_rng(9).standard_normal((pad, HIDDEN))], axis=0)
    else:
        fused_padded = np.concatenate([fused, np.random.default_rng(9).standard_normal((pad,))], axis=0)

    k_rt, v_rt = _split_kv(fused_padded, KV_OUT, NUM_KV_HEADS)
    np.testing.assert_array_equal(k_rt, k_hf)
    np.testing.assert_array_equal(v_rt, v_hf)


def test_split_k_differs_from_naive_crop():
    """Guard against regressing to a naive crop: the correct K half must be re-permuted,
    i.e. differ from the raw first-kv_out-rows slice, and equal the original HF k_proj."""
    k_hf, v_hf = _random_kv(is_weight=True, seed=7)
    fused = _fuse_kv(k_hf, v_hf, NUM_KV_HEADS)

    naive_crop_k = fused[:KV_OUT]  # the buggy behaviour: crop, no re-permute
    k_rt, v_rt = _split_kv(fused, KV_OUT, NUM_KV_HEADS)

    assert not np.array_equal(naive_crop_k, k_rt), "re-permute must change the K half vs a naive crop"
    np.testing.assert_array_equal(k_rt, k_hf, err_msg="correct K must equal the original HF k_proj")
    np.testing.assert_array_equal(v_rt, v_hf, err_msg="V must be recovered, not dropped")


def test_kv_match_recognizes_k_and_v_only():
    """``_kv_match`` fires on k_proj/v_proj weight+bias and nothing else."""
    assert _kv_match("model.layers.0.self_attn.k_proj.weight") == (0, "k", "weight")
    assert _kv_match("model.layers.5.self_attn.v_proj.weight") == (5, "v", "weight")
    assert _kv_match("model.layers.2.self_attn.k_proj.bias") == (2, "k", "bias")
    assert _kv_match("model.layers.2.self_attn.v_proj.bias") == (2, "v", "bias")
    # Non-KV names must NOT match (they go through the generic mapping path).
    assert _kv_match("model.layers.0.self_attn.q_proj.weight") is None
    assert _kv_match("model.layers.0.self_attn.o_proj.weight") is None
    assert _kv_match("model.layers.0.self_attn.kv_proj.weight") is None
    assert _kv_match("model.embed_tokens.weight") is None
