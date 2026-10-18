# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""TTNN attention vs the torch reference, global and local variants."""

import pytest
import torch

import ttnn
from models.experimental.modernbert.common import (
    FULL_ATTENTION,
    SLIDING_ATTENTION,
    build_inputs,
    load_config,
    load_torch_model,
)
from models.experimental.modernbert.reference.modernbert import ModernBertModel, ModernBertRotaryEmbedding
from models.experimental.modernbert.reference.modernbert import build_masks as torch_build_masks
from models.experimental.modernbert.tests.pcc_utils import pcc
from models.experimental.modernbert.tt.model_config import ACTIVATIONS_DTYPE
from models.experimental.modernbert.tt.modernbert_attention import TtnnModernBertAttention
from models.experimental.modernbert.tt.modernbert_masks import build_masks
from models.experimental.modernbert.tt.modernbert_rope import TtnnModernBertRotary
from models.experimental.modernbert.tt.weights import prepare_weights

ATTENTION_PCC = 0.999

# layer 0 is full_attention, layer 1 is sliding_attention (read from config)
LAYER_FOR = {FULL_ATTENTION: 0, SLIDING_ATTENTION: 1}


@pytest.fixture(scope="module")
def torch_ref():
    config = load_config()
    hf = load_torch_model()
    ref = ModernBertModel(config)
    ref.load_state_dict(hf.state_dict(), strict=True)
    ref.eval()
    return config, ref


def _inputs(ref, seq_len, layer_idx):
    """Real activations entering the given layer's attention."""
    ids, mask = build_inputs(seq_len=seq_len)
    with torch.no_grad():
        hidden = ref.embeddings(ids)
        return ref.layers[layer_idx].attn_norm(hidden), mask


def _torch_rope(config, layer_type, seq_len):
    hd = config.hidden_size // config.num_attention_heads
    theta = config.rope_parameters[layer_type]["rope_theta"]
    return ModernBertRotaryEmbedding(hd, theta)(torch.arange(seq_len).unsqueeze(0), torch.float32)


def _reference(config, ref, layer_idx, layer_type, x, attention_mask, seq_len):
    masks = torch_build_masks(config, attention_mask, seq_len, x.dtype, config.local_attention // 2)
    with torch.no_grad():
        return ref.layers[layer_idx].attn(x, _torch_rope(config, layer_type, seq_len), masks[layer_type])


def _run(device, config, ref, layer_idx, layer_type, x, seq_len, attn_mask):
    params = prepare_weights(ref, device)
    rotary = TtnnModernBertRotary(config, device, seq_len)
    module = TtnnModernBertAttention(params["layers"][layer_idx]["attn"], config, layer_type)
    tt_x = ttnn.from_torch(x, dtype=ACTIVATIONS_DTYPE, layout=ttnn.TILE_LAYOUT, device=device)
    return ttnn.to_torch(module(tt_x, rotary, attn_mask=attn_mask))


@pytest.mark.parametrize("layer_type", [FULL_ATTENTION, SLIDING_ATTENTION])
@pytest.mark.parametrize("seq_len", [256, 512])
def test_ttnn_attention_matches_reference(device, torch_ref, layer_type, seq_len):
    config, ref = torch_ref
    layer_idx = LAYER_FOR[layer_type]
    x, attention_mask = _inputs(ref, seq_len, layer_idx)
    expected = _reference(config, ref, layer_idx, layer_type, x, attention_mask, seq_len)

    masks = build_masks(config, device, seq_len)
    got = _run(device, config, ref, layer_idx, layer_type, x, seq_len, masks[layer_type]).reshape(expected.shape)

    p = pcc(expected, got.float())
    print(f"\n[attn {layer_type} seq={seq_len}] PCC={p:.8f}")
    assert p >= ATTENTION_PCC, f"attention PCC {p:.8f} < {ATTENTION_PCC}"


def test_negative_control_local_without_band(device, torch_ref):
    """Running a sliding layer with no band mask must not match the banded
    reference. This is the control a seq_len=128 test cannot provide."""
    config, ref = torch_ref
    seq_len = 256
    idx = LAYER_FOR[SLIDING_ATTENTION]
    x, attention_mask = _inputs(ref, seq_len, idx)
    expected = _reference(config, ref, idx, SLIDING_ATTENTION, x, attention_mask, seq_len)

    # break it: hand the sliding layer the FULL-attention mask, so it attends
    # everywhere instead of within the +/-64 band. The full mask is used rather
    # than None because the window is carried entirely by the mask - passing None
    # would test that SDPA ignores a missing mask, not that the band is applied.
    masks = build_masks(config, device, seq_len)
    got = _run(device, config, ref, idx, SLIDING_ATTENTION, x, seq_len, masks[FULL_ATTENTION]).reshape(expected.shape)

    p = pcc(expected, got.float())
    print(f"\n[NC local-without-band] PCC={p:.8f} (must be < {ATTENTION_PCC})")
    assert p < ATTENTION_PCC, "dropping the band mask had no effect - test is blind"


def test_negative_control_wrong_band_width(device, torch_ref):
    """A +/-32 band instead of +/-64 must be detected."""
    config, ref = torch_ref
    seq_len = 256
    idx = LAYER_FOR[SLIDING_ATTENTION]
    x, attention_mask = _inputs(ref, seq_len, idx)
    expected = _reference(config, ref, idx, SLIDING_ATTENTION, x, attention_mask, seq_len)

    class _Narrow:
        local_attention = 65  # -> half = 32
        hidden_size = config.hidden_size
        num_attention_heads = config.num_attention_heads

    bad = build_masks(_Narrow, device, seq_len)
    got = _run(device, config, ref, idx, SLIDING_ATTENTION, x, seq_len, bad[SLIDING_ATTENTION]).reshape(expected.shape)

    p = pcc(expected, got.float())
    print(f"\n[NC band=+/-32] PCC={p:.8f} (must be < {ATTENTION_PCC})")
    assert p < ATTENTION_PCC, "a +/-32 band was not distinguished from +/-64 - test is blind"


def test_negative_control_wrong_theta(device, torch_ref):
    """A global layer driven with the sliding rotary theta must be detected."""
    config, ref = torch_ref
    seq_len = 256
    idx = LAYER_FOR[FULL_ATTENTION]
    x, attention_mask = _inputs(ref, seq_len, idx)
    expected = _reference(config, ref, idx, FULL_ATTENTION, x, attention_mask, seq_len)

    masks = build_masks(config, device, seq_len)
    params = prepare_weights(ref, device)
    rotary = TtnnModernBertRotary(config, device, seq_len)
    module = TtnnModernBertAttention(params["layers"][idx]["attn"], config, FULL_ATTENTION)
    module.layer_type = SLIDING_ATTENTION  # break it: wrong rope cache

    tt_x = ttnn.from_torch(x, dtype=ACTIVATIONS_DTYPE, layout=ttnn.TILE_LAYOUT, device=device)
    got = ttnn.to_torch(module(tt_x, rotary, attn_mask=masks[FULL_ATTENTION])).reshape(expected.shape)

    p = pcc(expected, got.float())
    print(f"\n[NC wrong-rope-theta] PCC={p:.8f} (must be < {ATTENTION_PCC})")
    assert p < ATTENTION_PCC, "wrong rope theta was not detected - test is blind"


def test_band_is_invisible_at_very_short_sequence(device, torch_ref):
    """Pins the sequence length below which the band cannot be tested."""
    config, ref = torch_ref
    seq_len = 64
    idx = LAYER_FOR[SLIDING_ATTENTION]
    x, attention_mask = _inputs(ref, seq_len, idx)
    expected = _reference(config, ref, idx, SLIDING_ATTENTION, x, attention_mask, seq_len)

    masks = build_masks(config, device, seq_len)
    got = _run(device, config, ref, idx, SLIDING_ATTENTION, x, seq_len, masks[FULL_ATTENTION]).reshape(expected.shape)

    p = pcc(expected, got.float())
    print(f"\n[seq=64 band-removed] PCC={p:.8f} - band has no effect at this length")
    assert p >= ATTENTION_PCC, "expected the band to be invisible at seq_len 64"
