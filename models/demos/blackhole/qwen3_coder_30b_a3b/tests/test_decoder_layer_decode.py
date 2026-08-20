# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The full decoder layer in decode mode: prefill a prompt, then step tokens.

Validated against the prefill reference at the same absolute position, which is
the KV cache's defining contract -- attending to a cached prompt must equal
attending to it inline.

``test_multi_step_decode`` matters more than the single-step case. One step can
pass while the cache is subtly broken (a write that lands on the position being
read this turn still looks right); errors in the write position only diverge
once a later token has to read what an earlier step wrote. Three consecutive
steps against a three-token-longer reference catches that.
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc

from ..tt.functional_decoder import (
    DecoderLayerConfig,
    build_expert_sparsity,
    build_rope_cache,
    create_kv_cache,
    decoder_layer_decode,
    decoder_layer_prefill,
    upload_layer_weights,
)
from ..tt.weight_mapping import convert_layer_weights
from .reference import build_reference_layer, layer_state_dict, rotary_embeddings

LAYER_IDX = 0
PCC_REQUIRED = 0.99
MAX_SEQ = 256


@pytest.fixture(scope="module")
def reference():
    return build_reference_layer(LAYER_IDX)


@pytest.fixture(scope="module")
def torch_weights(reference):
    _, hf_config = reference
    return convert_layer_weights(layer_state_dict(LAYER_IDX), hf_config)


def _hidden(hf_config, seq_len, seed=0):
    torch.manual_seed(seed)
    return torch.randn(1, seq_len, hf_config.hidden_size, dtype=torch.float32) * 0.02


def _reference_layer(layer, hf_config, hidden):
    seq_len = hidden.shape[1]
    cos, sin = rotary_embeddings(hf_config, seq_len)
    mask = torch.full((seq_len, seq_len), float("-inf")).triu(1).reshape(1, 1, seq_len, seq_len)
    with torch.no_grad():
        out = layer(hidden, position_embeddings=(cos, sin), attention_mask=mask)
    return out[0] if isinstance(out, tuple) else out


def _to_device(t, mesh_device):
    return ttnn.from_torch(
        t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def _setup(mesh_device, hf_config, torch_weights, block_size=None):
    config = DecoderLayerConfig.from_hf(hf_config)
    weights = upload_layer_weights(torch_weights, mesh_device, config)
    cos_cache, sin_cache = build_rope_cache(hf_config, MAX_SEQ, mesh_device)
    sparsity = build_expert_sparsity(mesh_device, config.moe.num_experts)
    kv_cache = create_kv_cache(mesh_device, config.attention, max_batch=1, max_seq_len=MAX_SEQ, block_size=block_size)
    return config, weights, cos_cache, sin_cache, sparsity, kv_cache


def _decode_step(mesh_device, hf_config, ctx, token_hidden, position):
    config, weights, cos_cache, sin_cache, _, kv_cache = ctx
    current_pos = ttnn.from_torch(torch.tensor([position], dtype=torch.int32), dtype=ttnn.int32, device=mesh_device)
    tt_in = _to_device(token_hidden.reshape(1, 1, 1, hf_config.hidden_size), mesh_device)
    out = decoder_layer_decode(
        tt_in, weights, config, cos_cache, sin_cache, kv_cache, current_pos, token_index=position
    )
    return ttnn.to_torch(out).reshape(1, hf_config.hidden_size)


@pytest.mark.parametrize("mesh_device", [(1, 1)], ids=["1x1"], indirect=True)
@pytest.mark.parametrize("prompt_len", [32, 128], ids=["p32", "p128"])
@pytest.mark.parametrize("block_size", [None, 32], ids=["contiguous", "paged32"])
def test_decode_layer_matches_prefill(mesh_device, reference, torch_weights, prompt_len, block_size):
    layer, hf_config = reference
    hidden_full = _hidden(hf_config, prompt_len + 1)
    ref_out = _reference_layer(layer, hf_config, hidden_full)[:, prompt_len, :]

    ctx = _setup(mesh_device, hf_config, torch_weights, block_size)
    config, weights, cos_cache, sin_cache, sparsity, kv_cache = ctx

    # Prefill the prompt through the full layer so the cache is populated.
    decoder_layer_prefill(
        _to_device(hidden_full[:, :prompt_len, :].unsqueeze(0), mesh_device),
        weights,
        config,
        cos_cache,
        sin_cache,
        sparsity,
        kv_cache=kv_cache,
    )

    tt_out = _decode_step(mesh_device, hf_config, ctx, hidden_full[:, prompt_len, :], prompt_len)

    passing, pcc_message = comp_pcc(ref_out, tt_out, PCC_REQUIRED)
    logger.info(comp_allclose(ref_out, tt_out))
    kind = "contiguous" if block_size is None else f"paged(block={block_size})"
    logger.info(f"decode layer at pos {prompt_len} [{kind}]: {pcc_message}")
    assert passing, f"decode layer at pos {prompt_len} [{kind}] below {PCC_REQUIRED}: {pcc_message}"


@pytest.mark.parametrize("mesh_device", [(1, 1)], ids=["1x1"], indirect=True)
def test_multi_step_decode(mesh_device, reference, torch_weights):
    """Three sequential decode steps, each checked against the prefill reference.

    A cache written one position off still passes a single step; it only shows
    up when a later token reads what an earlier step stored.
    """
    prompt_len, steps = 32, 3
    layer, hf_config = reference
    hidden_full = _hidden(hf_config, prompt_len + steps)
    ref_out = _reference_layer(layer, hf_config, hidden_full)

    # Paged: multi-step is where a block-table mapping error would surface.
    ctx = _setup(mesh_device, hf_config, torch_weights, block_size=32)
    config, weights, cos_cache, sin_cache, sparsity, kv_cache = ctx

    decoder_layer_prefill(
        _to_device(hidden_full[:, :prompt_len, :].unsqueeze(0), mesh_device),
        weights,
        config,
        cos_cache,
        sin_cache,
        sparsity,
        kv_cache=kv_cache,
    )

    for step in range(steps):
        pos = prompt_len + step
        tt_out = _decode_step(mesh_device, hf_config, ctx, hidden_full[:, pos, :], pos)
        passing, pcc_message = comp_pcc(ref_out[:, pos, :], tt_out, PCC_REQUIRED)
        logger.info(f"decode step {step} (pos {pos}): {pcc_message}")
        assert passing, f"decode step {step} at position {pos} below {PCC_REQUIRED}: {pcc_message}"
