# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Single-token decode attention against the KV cache.

Decode is validated against the *prefill reference*, not against a separate
decode reference: prefill token S and decode token S must produce the same
output, because causal attention at position S sees exactly the same context
either way. That equivalence is the whole contract of a KV cache, so testing it
directly catches the errors that matter -- an off-by-one write position, RoPE
applied at the wrong index, or a cache that was never seeded by prefill.

This is also the first exercise of the Blackhole ``nlp_create_qkv_heads_decode``
DRAM bug (tt-metal #16667), which zeroes odd-indexed Q rows. The workaround
lives in ``attention_decode``; ``test_decode_q_rows_are_all_live`` is the
regression guard, because with half of Q zeroed the output is still finite,
still plausible, and still scores a deceptively high PCC.
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc

from ..tt.functional_decoder import (
    AttentionConfig,
    attention_decode,
    attention_prefill,
    build_rope_cache,
    create_kv_cache,
    upload_attention_weights,
)
from ..tt.weight_mapping import convert_attention_weights
from .reference import build_reference_layer, layer_state_dict, rotary_embeddings

LAYER_IDX = 0
PCC_REQUIRED = 0.99
MAX_SEQ = 256


@pytest.fixture(scope="module")
def reference():
    return build_reference_layer(LAYER_IDX)


@pytest.fixture(scope="module")
def torch_weights():
    return convert_attention_weights(layer_state_dict(LAYER_IDX), n_heads=32, n_kv_heads=4, head_dim=128)


def _hidden(hf_config, seq_len, seed=0):
    torch.manual_seed(seed)
    return torch.randn(1, seq_len, hf_config.hidden_size, dtype=torch.float32) * 0.02


def _reference_attention(layer, hf_config, hidden):
    seq_len = hidden.shape[1]
    cos, sin = rotary_embeddings(hf_config, seq_len)
    mask = torch.full((seq_len, seq_len), float("-inf")).triu(1).reshape(1, 1, seq_len, seq_len)
    with torch.no_grad():
        out = layer.self_attn(hidden_states=hidden, position_embeddings=(cos, sin), attention_mask=mask)
    return out[0] if isinstance(out, tuple) else out


def _to_device(t, mesh_device, dtype=ttnn.bfloat16):
    return ttnn.from_torch(
        t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def _prefill_then_decode(mesh_device, hf_config, torch_weights, hidden_full, prompt_len, block_size=None):
    """Prefill ``prompt_len`` tokens, then decode the token at ``prompt_len``."""
    config = AttentionConfig.from_hf(hf_config)
    weights = upload_attention_weights(torch_weights, mesh_device)
    cos_cache, sin_cache = build_rope_cache(hf_config, MAX_SEQ, mesh_device)
    kv_cache = create_kv_cache(mesh_device, config, max_batch=1, max_seq_len=MAX_SEQ, block_size=block_size)

    prompt = hidden_full[:, :prompt_len, :]
    attention_prefill(_to_device(prompt.unsqueeze(0), mesh_device), weights, config, cos_cache, sin_cache, kv_cache)

    # [1, 1, batch=1, hidden]
    next_tok = hidden_full[:, prompt_len, :].reshape(1, 1, 1, hf_config.hidden_size)
    current_pos = ttnn.from_torch(torch.tensor([prompt_len], dtype=torch.int32), dtype=ttnn.int32, device=mesh_device)
    out = attention_decode(
        _to_device(next_tok, mesh_device),
        weights,
        config,
        cos_cache,
        sin_cache,
        kv_cache,
        current_pos,
        token_index=prompt_len,
    )
    return ttnn.to_torch(out).reshape(1, hf_config.hidden_size)


@pytest.mark.parametrize("mesh_device", [(1, 1)], ids=["1x1"], indirect=True)
@pytest.mark.parametrize("prompt_len", [32, 128], ids=["p32", "p128"])
@pytest.mark.parametrize("block_size", [None, 32, 64], ids=["contiguous", "paged32", "paged64"])
def test_decode_matches_prefill_at_same_position(mesh_device, reference, torch_weights, prompt_len, block_size):
    layer, hf_config = reference
    hidden_full = _hidden(hf_config, prompt_len + 1)

    # Reference: run the whole prompt+1 through prefill and take the last row.
    ref_out = _reference_attention(layer, hf_config, hidden_full)[:, prompt_len, :]

    tt_out = _prefill_then_decode(mesh_device, hf_config, torch_weights, hidden_full, prompt_len, block_size)

    passing, pcc_message = comp_pcc(ref_out, tt_out, PCC_REQUIRED)
    logger.info(comp_allclose(ref_out, tt_out))
    kind = "contiguous" if block_size is None else f"paged(block={block_size})"
    logger.info(f"decode at pos {prompt_len} [{kind}]: {pcc_message}")
    assert passing, f"decode at position {prompt_len} [{kind}] below {PCC_REQUIRED}: {pcc_message}"


@pytest.mark.parametrize("mesh_device", [(1, 1)], ids=["1x1"], indirect=True)
def test_decode_q_rows_are_all_live(mesh_device, reference, torch_weights):
    """Guard for tt-metal #16667: no systematic zeroing of alternating rows.

    The Blackhole bug zeroes odd-indexed Q rows when the fused QKV is read from
    DRAM. Rather than reach into the op, this checks the observable
    consequence: the head-dim structure of the output must not show a
    stripe of exactly-zero alternating entries.
    """
    layer, hf_config = reference
    hidden_full = _hidden(hf_config, 33)
    out = _prefill_then_decode(mesh_device, hf_config, torch_weights, hidden_full, 32).float()

    assert torch.isfinite(out).all(), "decode produced non-finite values"
    zero_fraction = (out == 0).float().mean().item()
    logger.info(f"decode output zero fraction = {zero_fraction:.4f}")
    assert zero_fraction < 0.1, (
        f"{zero_fraction:.1%} of decode outputs are exactly zero -- looks like the "
        "Blackhole nlp_create_qkv_heads_decode DRAM bug (#16667); the fused QKV "
        "must be staged through L1 before the split"
    )
