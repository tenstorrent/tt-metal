# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""TTNN rotary embeddings vs the torch reference."""

import pytest
import torch

import ttnn
from models.experimental.modernbert.common import FULL_ATTENTION, SLIDING_ATTENTION, load_config
from models.experimental.modernbert.reference.modernbert import ModernBertRotaryEmbedding, apply_rotary_pos_emb
from models.experimental.modernbert.tests.pcc_utils import pcc
from models.experimental.modernbert.tt.model_config import ACTIVATIONS_DTYPE
from models.experimental.modernbert.tt.modernbert_rope import TtnnModernBertRotary

ROPE_PCC = 0.999


@pytest.fixture(scope="module")
def config():
    return load_config()


def _qk(config, seq_len, batch_size=1, seed=0):
    torch.manual_seed(seed)
    nh = config.num_attention_heads
    hd = config.hidden_size // nh
    return torch.randn(batch_size, nh, seq_len, hd), torch.randn(batch_size, nh, seq_len, hd)


def _torch_rope(config, layer_type, q, k, seq_len):
    hd = config.hidden_size // config.num_attention_heads
    theta = config.rope_parameters[layer_type]["rope_theta"]
    cos, sin = ModernBertRotaryEmbedding(hd, theta)(torch.arange(seq_len).unsqueeze(0), torch.float32)
    return apply_rotary_pos_emb(q, k, cos, sin)


def test_thetas_are_distinct(config):
    """Guards the whole dual-theta premise. If these ever collapse to one value
    the local/global distinction is gone and NC5 elsewhere would silently pass."""
    full = config.rope_parameters[FULL_ATTENTION]["rope_theta"]
    sliding = config.rope_parameters[SLIDING_ATTENTION]["rope_theta"]
    print(f"\n[thetas] full={full} sliding={sliding}")
    assert full == 160000.0
    assert sliding == 10000.0


@pytest.mark.parametrize("layer_type", [FULL_ATTENTION, SLIDING_ATTENTION])
@pytest.mark.parametrize("seq_len", [256, 512])
def test_ttnn_rope_matches_reference(device, config, layer_type, seq_len):
    q, k = _qk(config, seq_len)
    q_ref, k_ref = _torch_rope(config, layer_type, q, k, seq_len)

    rotary = TtnnModernBertRotary(config, device, seq_len)
    tt_q = ttnn.from_torch(q, dtype=ACTIVATIONS_DTYPE, layout=ttnn.TILE_LAYOUT, device=device)
    tt_k = ttnn.from_torch(k, dtype=ACTIVATIONS_DTYPE, layout=ttnn.TILE_LAYOUT, device=device)

    got_q = ttnn.to_torch(rotary(tt_q, layer_type)).reshape(q_ref.shape)
    got_k = ttnn.to_torch(rotary(tt_k, layer_type)).reshape(k_ref.shape)

    pq, pk = pcc(q_ref, got_q.float()), pcc(k_ref, got_k.float())
    print(f"\n[rope {layer_type} seq={seq_len}] q PCC={pq:.8f} k PCC={pk:.8f}")
    assert pq >= ROPE_PCC and pk >= ROPE_PCC


def test_negative_control_wrong_theta(device, config):
    """Applying the sliding theta where the full theta belongs must change the
    result. Without this, a single-theta implementation would pass unnoticed."""
    seq_len = 256
    q, k = _qk(config, seq_len)
    q_ref, _ = _torch_rope(config, FULL_ATTENTION, q, k, seq_len)

    rotary = TtnnModernBertRotary(config, device, seq_len)
    tt_q = ttnn.from_torch(q, dtype=ACTIVATIONS_DTYPE, layout=ttnn.TILE_LAYOUT, device=device)
    # deliberately use the sliding cache against the full-attention expectation
    got = ttnn.to_torch(rotary(tt_q, SLIDING_ATTENTION)).reshape(q_ref.shape)

    p = pcc(q_ref, got.float())
    print(f"\n[NC rope-wrong-theta] PCC={p:.8f} (must be < {ROPE_PCC})")
    assert p < ROPE_PCC, "swapping rope theta had no effect - test is blind"


def test_negative_control_no_rope(device, config):
    """Skipping rotary entirely must change the result."""
    seq_len = 256
    q, k = _qk(config, seq_len)
    q_ref, _ = _torch_rope(config, FULL_ATTENTION, q, k, seq_len)

    p = pcc(q_ref, q)
    print(f"\n[NC rope-not-applied] PCC={p:.8f} (must be < {ROPE_PCC})")
    assert p < ROPE_PCC, "rope application is a no-op - test is blind"


def test_llama_rope_would_be_wrong(device, config):
    """Documents why rotary_embedding_hf is the correct op."""
    seq_len = 256
    hd = config.hidden_size // config.num_attention_heads
    q, k = _qk(config, seq_len)
    theta = config.rope_parameters[FULL_ATTENTION]["rope_theta"]
    cos, sin = ModernBertRotaryEmbedding(hd, theta)(torch.arange(seq_len).unsqueeze(0), torch.float32)

    hf_style, _ = apply_rotary_pos_emb(q, k, cos, sin)

    # interleaved (Meta) convention: rotate adjacent pairs instead of halves
    c = cos.unsqueeze(1)
    s = sin.unsqueeze(1)
    x1, x2 = q[..., 0::2], q[..., 1::2]
    interleaved = torch.stack((-x2, x1), dim=-1).flatten(-2)
    meta_style = q.float() * c + interleaved.float() * s

    p = pcc(hf_style, meta_style)
    print(f"\n[rope convention] hf(rotate_half) vs meta(interleaved) PCC={p:.8f}")
    assert p < ROPE_PCC, "the two rope conventions are indistinguishable here - unexpected"
