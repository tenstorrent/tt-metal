# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""TTNN encoder layer vs the torch reference."""

import pytest
import torch

import ttnn
from models.experimental.modernbert.common import build_inputs, load_config, load_torch_model
from models.experimental.modernbert.reference.modernbert import ModernBertModel, ModernBertRotaryEmbedding, build_masks
from models.experimental.modernbert.tests.pcc_utils import pcc
from models.experimental.modernbert.tt.model_config import ACTIVATIONS_DTYPE
from models.experimental.modernbert.tt.modernbert_layer import TtnnModernBertEncoderLayer
from models.experimental.modernbert.tt.modernbert_masks import build_masks as build_tt_masks
from models.experimental.modernbert.tt.modernbert_rope import TtnnModernBertRotary
from models.experimental.modernbert.tt.weights import prepare_weights

LAYER_PCC = 0.999


@pytest.fixture(scope="module")
def torch_ref():
    config = load_config()
    hf = load_torch_model()
    ref = ModernBertModel(config)
    ref.load_state_dict(hf.state_dict(), strict=True)
    ref.eval()
    return config, ref


def _layer_reference(config, ref, layer_idx, seq_len):
    ids, attention_mask = build_inputs(seq_len=seq_len)
    with torch.no_grad():
        hidden = ref.embeddings(ids)
        layer = ref.layers[layer_idx]
        hd = config.hidden_size // config.num_attention_heads
        theta = config.rope_parameters[layer.attention_type]["rope_theta"]
        pos = ModernBertRotaryEmbedding(hd, theta)(torch.arange(seq_len).unsqueeze(0), torch.float32)
        masks = build_masks(config, attention_mask, seq_len, hidden.dtype, config.local_attention // 2)
        expected = layer(hidden, pos, masks[layer.attention_type])
    return hidden, expected


@pytest.mark.parametrize("layer_idx", [0, 1])
@pytest.mark.parametrize("seq_len", [256])
def test_ttnn_layer_matches_reference(device, torch_ref, layer_idx, seq_len):
    config, ref = torch_ref
    hidden, expected = _layer_reference(config, ref, layer_idx, seq_len)

    params = prepare_weights(ref, device)
    rotary = TtnnModernBertRotary(config, device, seq_len)
    masks = build_tt_masks(config, device, seq_len)
    module = TtnnModernBertEncoderLayer(params["layers"][layer_idx], config, layer_idx)

    tt_h = ttnn.from_torch(hidden, dtype=ACTIVATIONS_DTYPE, layout=ttnn.TILE_LAYOUT, device=device)
    got = ttnn.to_torch(module(tt_h, rotary, attn_mask=masks[module.layer_type])).reshape(expected.shape)

    p = pcc(expected, got.float())
    kind = config.layer_types[layer_idx]
    has_norm = params["layers"][layer_idx]["attn_norm"] is not None
    print(f"\n[layer {layer_idx} ({kind}, attn_norm={has_norm}) seq={seq_len}] PCC={p:.8f}")
    assert p >= LAYER_PCC, f"layer {layer_idx} PCC {p:.8f} < {LAYER_PCC}"


def test_layer0_has_no_attn_norm(torch_ref):
    """The layer-0 Identity quirk, asserted structurally rather than numerically."""
    _, ref = torch_ref
    import torch.nn as nn

    assert isinstance(ref.layers[0].attn_norm, nn.Identity)
    assert isinstance(ref.layers[1].attn_norm, nn.LayerNorm)


def test_negative_control_norm_applied_at_layer0(device, torch_ref):
    """Applying layer 1's norm weights at layer 0 must change the output."""
    config, ref = torch_ref
    seq_len = 256
    hidden, expected = _layer_reference(config, ref, 0, seq_len)

    params = prepare_weights(ref, device)
    rotary = TtnnModernBertRotary(config, device, seq_len)
    masks = build_tt_masks(config, device, seq_len)
    module = TtnnModernBertEncoderLayer(params["layers"][0], config, 0)
    # break it: borrow layer 1's norm instead of the Identity
    module.attn_norm = params["layers"][1]["attn_norm"]

    tt_h = ttnn.from_torch(hidden, dtype=ACTIVATIONS_DTYPE, layout=ttnn.TILE_LAYOUT, device=device)
    got = ttnn.to_torch(module(tt_h, rotary, attn_mask=masks[module.layer_type])).reshape(expected.shape)

    p = pcc(expected, got.float())
    print(f"\n[NC norm-at-layer0] PCC={p:.8f} (must be < {LAYER_PCC})")
    assert p < LAYER_PCC, "applying a norm at layer 0 was not detected - test is blind"
