# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Shard-advisor capture target for the Llama 3.1 8B optimized decoder.

Run from the tt-metal checkout after sourcing
`.agents/skills/shard-advise/scripts/bootstrap.sh`:

    ttnn-advise capture \
      models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_decoder/shard_advise/advise_llama31_8b.py:decode \
      --out models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_decoder/shard_advise

The advisor is a dense-path L1 layout seed, not the owner of SDPA decode,
cache updates, or DRAM-sharded weights. This target therefore traces the dense
attention+MLP block around the SDPA result: input norm, packed QKV projection,
attention output projection, post-attention norm, gate/up/down MLP, and
residual adds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import AutoConfig, LlamaConfig

import ttnn

REPO_ROOT = Path(__file__).resolve().parents[6]
sys.path.append(str(REPO_ROOT))

from models.autoports.meta_llama_llama_3_1_8b_instruct.tests.test_functional_decoder import (  # noqa: E402
    EMITTED_BATCH_SIZE,
    EMITTED_DECODE_CACHE_LEN,
    HF_MODEL_ID,
    TARGET_CONFIG,
    _synthetic_state_dict,
    _tt_tensor,
)
from models.autoports.meta_llama_llama_3_1_8b_instruct.tt.optimized_decoder import (  # noqa: E402
    OptimizedDecoder,
    OptimizedDecoderPolicy,
)

_DECODER = None


def _hf_config():
    try:
        return AutoConfig.from_pretrained(HF_MODEL_ID, local_files_only=True)
    except Exception:
        return LlamaConfig.from_dict(TARGET_CONFIG)


def _build(device):
    hf_config = _hf_config()
    state_dict = _synthetic_state_dict(hf_config, layer_idx=0)
    policy = OptimizedDecoderPolicy()
    decoder = OptimizedDecoder.from_state_dict(
        state_dict,
        hf_config=hf_config,
        layer_idx=0,
        mesh_device=device,
        batch=EMITTED_BATCH_SIZE,
        max_seq_len=EMITTED_DECODE_CACHE_LEN,
        policy=policy,
    )
    torch.manual_seed(20260731)
    hidden = torch.randn(EMITTED_BATCH_SIZE, 1, hf_config.hidden_size, dtype=torch.bfloat16)
    tt_hidden = _tt_tensor(hidden.reshape(1, EMITTED_BATCH_SIZE, 1, hf_config.hidden_size), device)
    return decoder, tt_hidden


def decode(hidden):
    cfg = _DECODER.cfg
    normed = ttnn.rms_norm(
        hidden,
        epsilon=cfg.rms_norm_eps,
        weight=_DECODER.input_layernorm_weight,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    qkv = _DECODER._matmul(normed, _DECODER.qkv_proj_weight, transpose_b=False)
    attn_seed = ttnn.slice(
        qkv,
        [0, 0, 0, 0],
        [1, EMITTED_BATCH_SIZE, 1, cfg.hidden_size],
        [1, 1, 1, 1],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    attn_out = _DECODER._matmul(attn_seed, _DECODER.o_proj_weight, transpose_b=True)
    attn_residual = ttnn.add(
        attn_out, hidden, dtype=_DECODER.policy.activation_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    post_norm = ttnn.rms_norm(
        attn_residual,
        epsilon=cfg.rms_norm_eps,
        weight=_DECODER.post_attention_layernorm_weight,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    gate = _DECODER._matmul(post_norm, _DECODER.gate_proj_weight, transpose_b=True)
    up = _DECODER._matmul(post_norm, _DECODER.up_proj_weight, transpose_b=True)
    gated = ttnn.multiply(
        ttnn.silu(gate, memory_config=ttnn.DRAM_MEMORY_CONFIG),
        up,
        dtype=_DECODER.policy.activation_dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    mlp_out = _DECODER._matmul(gated, _DECODER.down_proj_weight, transpose_b=True)
    return ttnn.add(
        mlp_out,
        attn_residual,
        dtype=_DECODER.policy.activation_dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def make_inputs(device):
    global _DECODER
    _DECODER, hidden = _build(device)
    return (hidden,)
