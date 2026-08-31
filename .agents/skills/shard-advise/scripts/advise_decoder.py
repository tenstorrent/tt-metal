# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""`ttnn-advise capture` target for the meta_llama_llama_3_1_8b_instruct OptimizedDecoder.

The advisor traces by EXECUTING the ttnn function, so it needs the decoder built
with weights and the decode inputs on device. This file exposes the two symbols
`ttnn-advise capture` expects:

 * `decode`            -- the ttnn function to trace (one decode step)
 * `make_inputs(device)` -- returns the positional args to call `decode` with

The wiring below mirrors the experiment's own test harness
(`tests/test_optimized_decoder.py` + `tests/test_functional_decoder.py`): it uses
`OptimizedDecoder.from_state_dict` with synthetic weights (shapes/dtypes drive the
layout advice, not values) and the model's own decode-input prep helpers.

Run:

 source .agents/skills/shard-advise/scripts/bootstrap.sh
 ttnn-advise capture .agents/skills/shard-advise/scripts/advise_decoder.py:decode \
     --out /tmp/shard-advice 2>/dev/null

Read /tmp/shard-advice/report.json.
"""
import os
import sys

import torch
import ttnn

# ============================ EDIT: experiment wiring ========================
# tt-metal repo root so `models.autoports...` / `models.common...` import.
MODEL_DIR = os.environ.get("SHARD_ADVISE_MODEL_DIR", "/localdev/mvasiljevic/tt-metal")
HF_MODEL = os.environ.get("SHARD_ADVISE_HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
LAYER_IDX = int(os.environ.get("SHARD_ADVISE_LAYER", "0"))
BATCH = int(os.environ.get("SHARD_ADVISE_BATCH", "32"))
MAX_SEQ_LEN = int(os.environ.get("SHARD_ADVISE_SEQ", "64"))
PAGE_BLOCK = int(os.environ.get("SHARD_ADVISE_PAGE_BLOCK", "32"))


def _synthetic_state_dict(config, layer_idx):
    # Same synthetic weights the experiment's own tests use for a layout probe.
    g = torch.Generator().manual_seed(20260708)
    prefix = f"model.layers.{layer_idx}."

    def randn(*shape):
        return (torch.randn(*shape, generator=g) * 0.02).to(torch.bfloat16)

    return {
        prefix + "input_layernorm.weight": torch.ones(config.hidden_size, dtype=torch.bfloat16),
        prefix + "post_attention_layernorm.weight": torch.ones(config.hidden_size, dtype=torch.bfloat16),
        prefix + "self_attn.q_proj.weight": randn(config.hidden_size, config.hidden_size),
        prefix + "self_attn.k_proj.weight": randn(config.num_key_value_heads * config.head_dim, config.hidden_size),
        prefix + "self_attn.v_proj.weight": randn(config.num_key_value_heads * config.head_dim, config.hidden_size),
        prefix + "self_attn.o_proj.weight": randn(config.hidden_size, config.hidden_size),
        prefix + "mlp.gate_proj.weight": randn(config.intermediate_size, config.hidden_size),
        prefix + "mlp.up_proj.weight": randn(config.intermediate_size, config.hidden_size),
        prefix + "mlp.down_proj.weight": randn(config.hidden_size, config.intermediate_size),
    }


def _rope(config, hidden_states, seq_len):
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

    rotary = LlamaRotaryEmbedding(config).to(dtype=torch.bfloat16)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(hidden_states.shape[0], -1)
    return rotary(hidden_states, position_ids)


def _build(device):
    sys.path.insert(0, MODEL_DIR)
    from transformers import AutoConfig

    from models.autoports.meta_llama_llama_3_1_8b_instruct.tt.optimized_decoder import OptimizedDecoder

    hf_config = AutoConfig.from_pretrained(HF_MODEL, local_files_only=True)
    state_dict = _synthetic_state_dict(hf_config, LAYER_IDX)

    decoder = OptimizedDecoder.from_state_dict(
        state_dict,
        hf_config=hf_config,
        layer_idx=LAYER_IDX,
        mesh_device=device,
        batch=BATCH,
        max_seq_len=MAX_SEQ_LEN,
        page_block_size=PAGE_BLOCK,
    )

    # One decode step (seq_len == 1), inputs built with the model's own helpers.
    torch.manual_seed(9876)
    hidden = torch.randn(BATCH, 1, hf_config.hidden_size, dtype=torch.bfloat16)
    cos, sin = _rope(hf_config, hidden, 1)

    tt_hidden = OptimizedDecoder.prepare_decode_inputs(hidden, device, decoder.decode_input_memcfg)
    tt_pos = OptimizedDecoder.prepare_decode_positions(torch.zeros((BATCH,), dtype=torch.long), device)
    page_table = OptimizedDecoder.build_contiguous_page_table(BATCH, max_seq_len=MAX_SEQ_LEN, block_size=PAGE_BLOCK)
    tt_page_table = OptimizedDecoder.prepare_page_table(page_table, device)
    tt_cos, tt_sin = OptimizedDecoder.prepare_decode_rope(cos[:, 0:1, :], sin[:, 0:1, :], device)

    kwargs = dict(current_pos=tt_pos, rot_mats=(tt_cos, tt_sin), page_table=tt_page_table)
    return decoder, kwargs, tt_hidden


# ========================== END experiment wiring ===========================

_DECODER = None
_KWARGS = None


def decode(hidden):
    # One decode step through the model's real ttnn path. The tracer records the
    # ttnn ops; the advisor's optimizer re-derives the L1 layout.
    return _DECODER.decode_forward(hidden, **_KWARGS)


def make_inputs(device):
    global _DECODER, _KWARGS
    _DECODER, _KWARGS, hidden = _build(device)
    return (hidden,)
