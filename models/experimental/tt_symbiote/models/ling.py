# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Ling-mini-2.0 model setup for TTNN backend.

Centralizes module replacement tables, weight preprocessing, paged KV
cache creation, and forward-patching so that tests and demos import a
single entry point instead of duplicating setup logic.
"""

import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.experimental.tt_symbiote.modules.activation import TTNNSilu
from models.experimental.tt_symbiote.modules.attention import (
    PagedAttentionConfig,
    TTNNPagedAttentionKVCache,
)
from models.experimental.tt_symbiote.modules.decoder_layer import TTNNBailingMoEDecoderLayerPadded
from models.experimental.tt_symbiote.modules.embedding import TTNNBailingPaddedEmbedding, TTNNBailingRotaryEmbedding
from models.experimental.tt_symbiote.modules.linear import TTNNLinearIColShardedWRowSharded, TTNNLinearLmHeadBf8
from models.experimental.tt_symbiote.modules.normalization import TTNNDistributedRMSNorm
from models.experimental.tt_symbiote.utils.device_management import set_device
from models.experimental.tt_symbiote.utils.module_replacement import register_module_replacement_dict

from models.experimental.tt_symbiote.models.bailing_moe_v2 import TTNNBailingMoeV2ForCausalLM, TTNNBailingMoeV2Model

DEFAULT_MODEL_NAME = "inclusionAI/Ling-mini-2.0"


def create_paged_kv_cache(model_config, device, batch_size=1):
    """Create a paged attention KV cache sized for *model_config*."""
    config = PagedAttentionConfig(
        block_size=64,
        max_num_blocks=32,
        batch_size=batch_size,
    )
    return TTNNPagedAttentionKVCache(
        num_layers=model_config.num_hidden_layers,
        num_kv_heads=model_config.num_key_value_heads,
        head_dim=model_config.head_dim,
        config=config,
        device=None,
    ).to_device(device)


def replace_modules(model):
    """Register all TTNN module replacements for Ling and return the merged dict."""
    nn_to_ttnn_1 = {
        model.model.layers[0].__class__: TTNNBailingMoEDecoderLayerPadded,
        model.model.norm.__class__: TTNNDistributedRMSNorm,
        nn.Embedding: TTNNBailingPaddedEmbedding,
        model.model.rotary_emb.__class__: TTNNBailingRotaryEmbedding,
    }
    nn_to_ttnn_2 = {
        nn.Linear: TTNNLinearIColShardedWRowSharded,
        nn.SiLU: TTNNSilu,
    }
    nn_to_ttnn_3 = {
        model.model.__class__: TTNNBailingMoeV2Model,
    }
    m1 = register_module_replacement_dict(model, nn_to_ttnn_1, model_config=None)
    m2 = register_module_replacement_dict(model, nn_to_ttnn_2, model_config=None, exclude_replacement={"lm_head"})
    nn_to_ttnn_lmhead = {nn.Linear: TTNNLinearLmHeadBf8}
    m_lm = register_module_replacement_dict(model, nn_to_ttnn_lmhead, model_config=None)
    m3 = register_module_replacement_dict(model, nn_to_ttnn_3, model_config=None)
    return {**m1, **m2, **m_lm, **m3}


def preprocess_weights(modules, mesh_device):
    """Set devices, preprocess and move weights for all TTNN modules."""
    print(f"Preprocessing {len(modules)} TTNN module weights...")
    for _k, v in tqdm(modules.items()):
        v.preprocess_weights()
        v.move_weights_to_device()


def load_model(mesh_device, model_name=DEFAULT_MODEL_NAME, batch_size=1):
    """Load, convert, and prepare a Ling model for TTNN inference.

    Returns (model, tokenizer, paged_cache).
    """
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

    modules = replace_modules(model)

    type(model).device = property(lambda self: torch.device("cpu"))
    set_device(model, mesh_device)
    preprocess_weights(modules, mesh_device)

    model.eval()
    torch.set_grad_enabled(False)

    wrapper = TTNNBailingMoeV2ForCausalLM.patch_forward(model, mesh_device)
    model._ttnn_causal_lm_wrapper = wrapper
    paged_cache = create_paged_kv_cache(model.config, mesh_device, batch_size=batch_size)
    return model, tokenizer, paged_cache
