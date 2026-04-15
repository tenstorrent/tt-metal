# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Tests self-attention, GLM4 Flash attention, and paged attention KV cache."""

import pytest
import torch

from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor
from models.experimental.tt_symbiote.core.utils import compare_fn_outputs
from models.experimental.tt_symbiote.modules.attention import (
    SelfAttention,
    SelfAttentionConfig,
    TTNNSelfAttention,
    TTNNGlm4MoeLiteAttention,
    PagedAttentionConfig,
    TTNNPagedAttentionKVCache,
)
from models.experimental.tt_symbiote.utils.device_management import set_device


@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
def test_self_attention(device):
    config = SelfAttentionConfig(
        hidden_size=768,
        num_attention_heads=12,
    )
    model = SelfAttention(config).to(dtype=torch.bfloat16)
    model.eval()
    torch.set_grad_enabled(False)
    inputs = TorchTTNNTensor(torch.randn((1, 5, 768), dtype=torch.bfloat16))
    outputs_torch = model(inputs)

    ttnn_model = TTNNSelfAttention.from_torch(model)
    set_device(ttnn_model, device)
    ttnn_model.preprocess_weights()
    ttnn_model.move_weights_to_device()
    outputs_ttnn = ttnn_model(inputs)
    compare_fn_outputs(outputs_torch, outputs_ttnn, "SelfAttention")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
def test_glm4_flash_attention_with_paged_kv_cache(device):
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained("zai-org/GLM-4.7-Flash", trust_remote_code=True)
    config.num_hidden_layers = 2
    config.num_attention_heads = 4
    config.num_key_value_heads = 4
    config.hidden_size = 512
    config.intermediate_size = 1024
    config.kv_lora_rank = 128
    config.q_lora_rank = 192
    config.qk_nope_head_dim = 64
    config.qk_rope_head_dim = 64
    config.qk_head_dim = 128
    config.v_head_dim = 128
    config.moe_intermediate_size = 384
    config.num_local_experts = 4
    config.num_experts_per_tok = 2

    model = AutoModelForCausalLM.from_config(config).to(dtype=torch.bfloat16).eval()
    torch.set_grad_enabled(False)
    torch_attn = model.model.layers[0].self_attn

    paged_config = PagedAttentionConfig(block_size=32, max_num_blocks=64, batch_size=1)
    paged_cache = TTNNPagedAttentionKVCache(
        num_layers=2,
        num_kv_heads=4,
        head_dim=128,
        config=paged_config,
        device=None,
        dtype=torch.bfloat16,
    ).to_device(device)

    from transformers.cache_utils import DynamicCache

    dynamic_cache = DynamicCache()

    hidden = torch.randn(1, 5, 512, dtype=torch.bfloat16)
    pos = torch.arange(5).unsqueeze(0)
    cos, sin = model.model.rotary_emb(hidden, pos)

    torch_out_prefill = torch_attn(
        hidden,
        attention_mask=None,
        position_embeddings=(cos, sin),
        past_key_values=dynamic_cache,
    )

    ttnn_attn = TTNNGlm4MoeLiteAttention.from_torch(torch_attn, distributed=False)
    set_device(ttnn_attn, device)
    ttnn_attn.preprocess_weights()
    ttnn_attn.move_weights_to_device()

    inputs = TorchTTNNTensor(hidden)
    cache_position = torch.arange(5).unsqueeze(0)
    ttnn_out_prefill = ttnn_attn(
        inputs,
        position_embeddings=(cos, sin),
        past_key_values=paged_cache,
        cache_position=cache_position,
    )

    compare_fn_outputs(torch_out_prefill, ttnn_out_prefill, "Glm4MoeLiteAttention_PagedPrefill")

    assert paged_cache.get_seq_length(0) == 5
    assert dynamic_cache.get_seq_length(0) == 5
