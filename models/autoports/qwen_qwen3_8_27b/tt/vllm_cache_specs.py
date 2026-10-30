# SPDX-License-Identifier: Apache-2.0
"""Upstream hybrid-cache specifications for the TP4 text adapter.

The byte sizes are planner units, not physical TT allocation measurements.
Attention uses the model's global heads while recurrence shapes are per device;
all groups use an equal padded planner page so vLLM preserves the selected scheduler blocks.
The adapter separately records real BFP8 attention and FP32/BF16 state bytes.
"""
import torch
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec


from .cache_block_config import DEFAULT_BLOCK_SIZE, scheduler_block_size

# Compatibility default; runtime callers must use the resolved selection.
BLOCK_SIZE = DEFAULT_BLOCK_SIZE
CONV_SHAPE = (3, 2560)
RECURRENT_SHAPE = (12, 128, 128)
STATE_BYTES_PER_LAYER = 3 * 2560 * 2 + 12 * 128 * 128 * 4


def get_kv_cache_spec(vllm_config):
    block_size = scheduler_block_size()
    config = vllm_config.model_config.hf_config.text_config
    cache = vllm_config.cache_config
    if cache.block_size != block_size:
        raise ValueError(f'Hybrid Qwen cache requires scheduler block_size={block_size}')
    if cache.mamba_cache_mode != 'align':
        raise ValueError('Hybrid Qwen prefix caching requires mamba_cache_mode=align')
    if cache.mamba_block_size not in (None, block_size):
        raise ValueError('Attention and recurrent checkpoint boundaries must agree')
    attention = FullAttentionSpec(block_size=block_size,
        num_kv_heads=config.num_key_value_heads, head_size=config.head_dim,
        dtype=torch.bfloat16)
    recurrent = MambaSpec(block_size=block_size,
        shapes=(CONV_SHAPE, RECURRENT_SHAPE),
        dtypes=(torch.bfloat16, torch.float32),
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
        page_size_padded=attention.page_size_bytes,
        mamba_cache_mode='align', num_prefill_checkpoint_blocks=0,
        prefill_checkpoint_alignment=block_size)
    return {
        f'model.layers.{index}.state':
            attention if kind == 'full_attention' else recurrent
        for index, kind in enumerate(config.layer_types)
    }
