# SPDX-License-Identifier: Apache-2.0
"""Upstream hybrid-cache specifications for the TP4 text adapter.

The byte sizes are planner units, not physical TT allocation measurements.
Attention uses the model's global heads while recurrence shapes are per device;
all groups use an equal padded planner page so vLLM preserves checkpoint blocks.
The adapter separately records real BFP8 attention and FP32/BF16 state bytes.
"""
import os
import torch
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec


BLOCK_SIZE = 4096
CONV_SHAPE = (3, 2560)
RECURRENT_SHAPE = (12, 128, 128)
STATE_BYTES_PER_LAYER = 3 * 2560 * 2 + 12 * 128 * 128 * 4
ATTENTION_LAYERS = 16
RECURRENT_LAYERS = 48
RECURRENT_GROUPS = 3
# TP4, one local KV head, dim256: eight BFP8 tiles of1088 bytes per32 tokens.
ATTENTION_BYTES_PER_TOKEN = ATTENTION_LAYERS * 2 * 8 * 1088 // 32


def shared_recurrent_snapshots():
    value = os.environ.get('QWEN38_SHARED_RECURRENT_SNAPSHOTS', '0')
    if value not in ('0', '1'):
        raise ValueError('QWEN38_SHARED_RECURRENT_SNAPSHOTS must be0 or1')
    if value == '1' and physical_cache_budget() is None:
        raise ValueError('Shared snapshots require an explicit physical cache byte budget')
    return value == '1'


def selected_block_size():
    block = int(os.environ.get('QWEN38_CACHE_BLOCK_SIZE', BLOCK_SIZE))
    if block not in (256, 512, 1024, 1536, 2048, 4096):
        raise ValueError('Qwen checkpoint block size must be256/512/1024/1536/2048/4096')
    return block


def union_state_backing():
    """Back recurrent snapshot rows by the same block ID's attention K/V pages.

    vLLM gives each block ID to exactly one group at a time, so a block's
    attention bytes are free whenever it holds a snapshot. Needs shared
    snapshots and blocks whose per-layer K+V pages hold one state row.
    """
    value = os.environ.get('QWEN38_UNION_STATE_BACKING', '0')
    if value not in ('0', '1'):
        raise ValueError('QWEN38_UNION_STATE_BACKING must be0 or1')
    if value == '1':
        if not shared_recurrent_snapshots():
            raise ValueError('Union state backing requires QWEN38_SHARED_RECURRENT_SNAPSHOTS=1')
        if selected_block_size() * ATTENTION_BYTES_PER_TOKEN // ATTENTION_LAYERS < STATE_BYTES_PER_LAYER:
            raise ValueError('Union state backing requires blocks of at least 1536 tokens')
    return value == '1'


def physical_cache_budget():
    raw = os.environ.get('QWEN38_CACHE_BUDGET_BYTES_PER_DEVICE')
    if raw is None:
        if selected_block_size() <= 512:
            raise ValueError('Fine checkpoints require an explicit physical cache byte budget')
        return None
    budget = int(raw)
    if budget <= 0:
        raise ValueError('Physical cache budget must be positive bytes per device')
    return budget


def physical_bytes_per_block(block):
    """Selected BFP8 attention plus BF16/FP32 snapshots; excludes active state/scratch."""
    if union_state_backing():
        return block * ATTENTION_BYTES_PER_TOKEN
    pools = RECURRENT_LAYERS // RECURRENT_GROUPS if shared_recurrent_snapshots() else RECURRENT_LAYERS
    return block * ATTENTION_BYTES_PER_TOKEN + pools * STATE_BYTES_PER_LAYER


def bounded_token_budget(max_model_len, max_num_seqs):
    """Invert the TT worker's output + hybrid-state reservations, not context length.

    The worker overwrites num_gpu_blocks_override. Its model token hook is the
    supported input to pool sizing. An explicit byte budget bounds physical KV
    and snapshot backing only, and makes no claim about transient serving peak.
    """
    block, budget = selected_block_size(), physical_cache_budget()
    if budget is None:
        return max_model_len
    blocks = budget // physical_bytes_per_block(block)
    # Three align groups each retain current/previous state, plus global null.
    reserved = max_num_seqs * (1 + 2 * RECURRENT_GROUPS) + 1
    admission = (max_model_len + block - 1) // block + 2 * RECURRENT_GROUPS + 1 + max_num_seqs
    if blocks < max(admission, reserved + 1):
        raise ValueError(f'Physical cache budget permits{blocks} blocks but native context/output '
                         f'admission needs{admission} and worker reservations need{reserved + 1}')
    return (blocks - reserved) * block


def validate_budget_geometry(layer_types, num_kv_heads, head_dim):
    if physical_cache_budget() is not None and (
            len(layer_types) != 64 or layer_types.count('full_attention') != ATTENTION_LAYERS
            or layer_types.count('linear_attention') != RECURRENT_LAYERS
            or num_kv_heads != 4 or head_dim != 256):
        raise ValueError('Physical budget requires all64 Qwen27B layers,16 attention/48 recurrent,4 KV heads,dim256')


def get_kv_cache_spec(vllm_config):
    config = vllm_config.model_config.hf_config.text_config
    cache = vllm_config.cache_config
    block = selected_block_size()
    if shared_recurrent_snapshots() and (getattr(vllm_config, 'kv_transfer_config', None)
            or getattr(vllm_config, 'speculative_config', None)):
        raise ValueError('Shared snapshots support only local non-speculative cache ownership')
    if cache.block_size != block:
        raise ValueError(f'Hybrid Qwen CLI block_size must match QWEN38_CACHE_BLOCK_SIZE={block}')
    if cache.mamba_cache_mode != 'align':
        raise ValueError('Hybrid Qwen prefix caching requires mamba_cache_mode=align')
    if cache.mamba_block_size not in (None, block):
        raise ValueError('Attention and recurrent checkpoint boundaries must agree')
    validate_budget_geometry(config.layer_types, config.num_key_value_heads, config.head_dim)
    attention = FullAttentionSpec(block_size=block,
        num_kv_heads=config.num_key_value_heads, head_size=config.head_dim,
        dtype=torch.bfloat16)
    recurrent = MambaSpec(block_size=block,
        shapes=(CONV_SHAPE, RECURRENT_SHAPE),
        dtypes=(torch.bfloat16, torch.float32),
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
        page_size_padded=attention.page_size_bytes,
        mamba_cache_mode='align', num_prefill_checkpoint_blocks=0,
        prefill_checkpoint_alignment=block)
    return {
        f'model.layers.{index}.state':
            attention if kind == 'full_attention' else recurrent
        for index, kind in enumerate(config.layer_types)
    }
