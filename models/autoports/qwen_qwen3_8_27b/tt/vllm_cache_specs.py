# SPDX-License-Identifier: Apache-2.0
"""Upstream hybrid-cache specifications for the TP4 text adapter.

The byte sizes are planner units, not physical TT allocation measurements.
Attention uses the model's global heads while recurrence shapes are per device;
all groups use an equal padded planner page so vLLM preserves4096-token blocks.
The adapter separately records real BFP8 attention and FP32/BF16 state bytes.
"""
import torch
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec


BLOCK_SIZE = 4096
CONV_SHAPE = (3, 2560)
RECURRENT_SHAPE = (12, 128, 128)
STATE_BYTES_PER_LAYER = 3 * 2560 * 2 + 12 * 128 * 128 * 4


def get_kv_cache_spec(vllm_config):
    config = vllm_config.model_config.hf_config.text_config
    cache = vllm_config.cache_config
    if cache.block_size != BLOCK_SIZE:
        raise ValueError(f'Hybrid Qwen cache requires scheduler block_size={BLOCK_SIZE}')
    if cache.mamba_cache_mode != 'align':
        raise ValueError('Hybrid Qwen prefix caching requires mamba_cache_mode=align')
    if cache.mamba_block_size not in (None, BLOCK_SIZE):
        raise ValueError('Attention and recurrent checkpoint boundaries must agree')
    attention = FullAttentionSpec(block_size=BLOCK_SIZE,
        num_kv_heads=config.num_key_value_heads, head_size=config.head_dim,
        dtype=torch.bfloat16)
    recurrent = MambaSpec(block_size=BLOCK_SIZE,
        shapes=(CONV_SHAPE, RECURRENT_SHAPE),
        dtypes=(torch.bfloat16, torch.float32),
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
        page_size_padded=attention.page_size_bytes,
        mamba_cache_mode='align', num_prefill_checkpoint_blocks=0,
        prefill_checkpoint_alignment=BLOCK_SIZE)
    return {
        f'model.layers.{index}.state':
            attention if kind == 'full_attention' else recurrent
        for index, kind in enumerate(config.layer_types)
    }
