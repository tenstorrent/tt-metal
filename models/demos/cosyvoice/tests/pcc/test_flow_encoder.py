# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The flow Conformer encoder: 6 pre-norm blocks with rel-pos attention.

Part of the flow stage. The same block structure serves the LLM's text encoder
and its 14-block AR decoder, so what passes here is reusable there too.
"""
from __future__ import annotations

import os

import pytest
import torch

from models.demos.cosyvoice.tt.common import GOLDEN_DIR, as_torch, load_golden, pcc
from models.demos.cosyvoice.tt.flow.encoder import espnet_rel_positional_encoding
from models.demos.cosyvoice.tt.weights import default_weights_path

FLOW_WEIGHTS = default_weights_path().replace("hift_", "flow_")

needs_l1_small = pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
needs_weights = pytest.mark.skipif(
    not os.path.exists(FLOW_WEIGHTS),
    reason="run scripts/export_weights.py --module flow in the CosyVoice venv first",
)
needs_golden = pytest.mark.skipif(
    not os.path.exists(os.path.join(GOLDEN_DIR, "flow.encoder.npz")),
    reason="run scripts/gen_golden.py in the CosyVoice venv first",
)


# --------------------------------------------------------------------------
# host tier
# --------------------------------------------------------------------------
@needs_golden
def test_positional_encoding_is_generated_not_stored():
    """The ESPnet rel-pos encoding is deterministic given (T, d_model), so it is
    generated rather than exported. It must reproduce what the reference produced,
    or `rel_shift` operates on the wrong window.

    Compared with a tolerance rather than `torch.equal`: this is sin/cos in
    float32, and different torch builds round the last ulp differently -- it is
    bit-identical against the torch that generated the golden and ~1.5e-05 off
    against the one in the container image. Bit-exactness across builds is not a
    property worth asserting; agreement to float32 rounding is.
    """
    want = as_torch(load_golden("flow.rel_pos_attention")["call0.in_pos_emb"])
    got = espnet_rel_positional_encoding(338, 512)
    assert got.shape == want.shape, (got.shape, want.shape)
    assert torch.allclose(got, want, atol=1e-4), (got - want).abs().max()
    assert pcc(got, want) >= 0.999999, pcc(got, want)


def test_positional_encoding_is_reversed_in_the_positive_half():
    """The positive half is FLIPPED before concatenation -- that is the shifting
    trick from arXiv:1901.02860, and sampling it forwards instead yields a
    plausible encoding with time running backwards. Pinned because nothing
    downstream would obviously break."""
    pe = espnet_rel_positional_encoding(8, 16)[0]  # [15, 16]
    mid = pe.shape[0] // 2
    # position 0 sits at the centre; moving outward in either direction the
    # encodings must differ, and the two halves must not be equal
    assert not torch.allclose(pe[mid - 1], pe[mid + 1])
    assert torch.allclose(pe[mid], espnet_rel_positional_encoding(1, 16)[0, 0])


@needs_weights
def test_flow_weight_export_has_every_encoder_tensor():
    """Cheap structural check: 6 layers, each with attention, FFN and two norms."""
    from models.demos.cosyvoice.tt.weights import WeightBag

    bag = WeightBag.load(FLOW_WEIGHTS)
    meta = bag.meta
    assert meta["module"] == "flow"
    assert meta["n_layers"] == 6 and meta["n_head"] == 8 and meta["d_k"] == 64
    assert meta["d_model"] == 512 and meta["ffn_dim"] == 2048
    assert meta["has_macaron"] is False and meta["has_conv_module"] is False
    assert meta["ff_scale"] == 1.0 and meta["normalize_before"] is True

    enc = bag.sub("encoder")
    for i in range(meta["n_layers"]):
        layer = enc.sub(f"encoders.{i}")
        for name in ("linear_q", "linear_k", "linear_v", "linear_out"):
            assert layer.sub(f"self_attn.{name}").has("weight"), (i, name)
        assert layer.sub("self_attn.linear_pos").has("weight")
        assert not layer.sub("self_attn.linear_pos").has("bias"), "linear_pos is bias-free upstream"
        assert layer.has("self_attn.pos_bias_u") and layer.has("self_attn.pos_bias_v")
        assert layer.sub("feed_forward.w_1").tensor("weight").shape == (2048, 512)
        assert layer.sub("feed_forward.w_2").tensor("weight").shape == (512, 2048)
        for norm in ("norm_mha", "norm_ff"):
            assert layer.sub(norm).has("weight") and layer.sub(norm).has("bias"), (i, norm)
    assert enc.sub("after_norm").has("weight")


# --------------------------------------------------------------------------
# device tier
# --------------------------------------------------------------------------
@needs_golden
@needs_weights
@needs_l1_small
def test_device_flow_encoder_matches_golden(device):
    """All 6 blocks on device against the captured encoder output.

    The reference feeds a full-length single sequence, so its mask is all-ones
    and masking is a no-op; `mask=None` here is that, not a shortcut.
    """
    import ttnn
    from models.demos.cosyvoice.tt.flow.encoder import TtConformerEncoder
    from models.demos.cosyvoice.tt.weights import WeightBag

    g = load_golden("flow.encoder")
    xs = as_torch(g["call0.in_xs"])  # [1, T, 512]
    want = as_torch(g["call0.out_xs"])
    mask = as_torch(g["call0.out_masks"])
    assert bool(mask.all()), "captured mask is not all-ones; masking is no longer a no-op"

    bag = WeightBag.load(FLOW_WEIGHTS)
    meta = bag.meta
    enc = TtConformerEncoder(device, bag.sub("encoder"), meta)

    pos = espnet_rel_positional_encoding(xs.shape[1], meta["d_model"])
    x = ttnn.from_torch(xs, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    pe = ttnn.from_torch(pos, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    got = ttnn.to_torch(enc(x, pe)).float()

    p = pcc(got, want)
    print(f"\n  flow encoder {meta['n_layers']} blocks, T={xs.shape[1]}")
    print(f"  PCC {p:.10f}  max|d| {(got - want).abs().max():.3e}")
    assert got.shape == want.shape, (got.shape, want.shape)
    assert p >= 0.99, p
