# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Test for dots.ocr model on N300 (1x2 Wormhole mesh) with col-sharded residual flow.

Phase 3+4: Full col-sharded pipeline. Residual stream is sharded [B, S, 768] per device.
"""

import os

import pytest
import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import ttnn
from models.experimental.tt_symbiote.core.run_config import (
    DispatchManager,
    TracedRun,
)
from models.experimental.tt_symbiote.modules.dots_ocr_decoder_layer import TTNNDotsOCRDecoderLayer
from models.experimental.tt_symbiote.modules.embedding import TTNNEmbedding
from models.experimental.tt_symbiote.modules.attention import (
    PagedAttentionConfig,
    TTNNPagedAttentionKVCache,
)
from models.experimental.tt_symbiote.modules.linear import TTNNLinearIColShardedWRowSharded
from models.experimental.tt_symbiote.modules.normalization import TTNNDistributedRMSNorm
from models.experimental.tt_symbiote.utils.device_management import (
    DeviceInit,
    set_device,
)
from models.experimental.tt_symbiote.utils.module_replacement import (
    register_module_replacement_dict,
)


MESH_DEVICE_MAP = {
    "N150": (1, 1),
    "N300": (1, 2),
    "N150x4": (1, 4),
    "T3K": (1, 8),
    "TG": (8, 4),
    "P150": (1, 1),
    "P300": (1, 2),
    "P150x4": (1, 4),
    "P150x8": (1, 8),
    "BHGLX": (8, 4),
}

DOTS_OCR_LOCAL_PATH = "/home/salnahari/.cache/huggingface/hub/models--rednote-hilab--dots.ocr/snapshots/c0111ce6bc07803dbc267932ffef0ae3a51dc951"


def create_paged_kv_cache(model_config, device, batch_size=1):
    head_dim = model_config.hidden_size // model_config.num_attention_heads
    config = PagedAttentionConfig(
        block_size=64,
        max_num_blocks=32,
        batch_size=batch_size,
    )
    return TTNNPagedAttentionKVCache(
        num_layers=model_config.num_hidden_layers,
        num_kv_heads=model_config.num_key_value_heads,
        head_dim=head_dim,
        config=config,
        device=None,
    ).to_device(device)


@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 200000000, "num_command_queues": 1, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [MESH_DEVICE_MAP.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))],
    indirect=True,
)
def test_dots_ocr_n300(mesh_device):
    """Test dots.ocr on N300 (1x2 mesh) with col-sharded residual flow."""
    model_name = DOTS_OCR_LOCAL_PATH

    print("Loading dots.ocr from local cache...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    decoder_class = model.model.layers[0].__class__
    norm_class = model.model.layers[0].input_layernorm.__class__
    embed_class = model.model.embed_tokens.__class__

    # Pass 1: Replace decoder layers, final norm, and embedding
    nn_to_ttnn = {
        decoder_class: TTNNDotsOCRDecoderLayer,
        norm_class: TTNNDistributedRMSNorm,
        embed_class: TTNNEmbedding,
    }
    modules = register_module_replacement_dict(model, nn_to_ttnn, model_config=None)

    # Convert ModuleList to plain list — Qwen2Model.forward slices self.layers[:n],
    # which constructs a new ModuleList and fails isinstance(TTNNModule, nn.Module).
    layers_list = list(model.model.layers)
    del model.model._modules["layers"]
    model.model.layers = layers_list

    # Pass 2: Replace lm_head
    nn_to_ttnn2 = {
        nn.Linear: TTNNLinearIColShardedWRowSharded,
    }
    modules2 = register_module_replacement_dict(model, nn_to_ttnn2, model_config=None)

    type(model).device = property(lambda self: torch.device("cpu"))

    messages = [
        {"role": "user", "content": "What is optical character recognition and how does it work?"},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]

    set_device(model, mesh_device, device_init=DeviceInit)

    all_modules = {**modules, **modules2}
    print(f"Preprocessing {len(all_modules)} TTNN modules weights...")
    for k, v in tqdm(all_modules.items()):
        v.preprocess_weights()
        v.move_weights_to_device()

    print("Running inference with mesh device...")
    model.eval()
    torch.set_grad_enabled(False)

    kv_cache = create_paged_kv_cache(model.config, mesh_device)
    outputs = model.generate(**inputs, max_new_tokens=2, use_cache=True, past_key_values=kv_cache)

    kv_cache = create_paged_kv_cache(model.config, mesh_device)
    outputs = model.generate(**inputs, max_new_tokens=4, use_cache=True, past_key_values=kv_cache)

    kv_cache = create_paged_kv_cache(model.config, mesh_device)
    DispatchManager.clear_timings()
    outputs = model.generate(**inputs, max_new_tokens=128, use_cache=True, past_key_values=kv_cache)
    ttnn.synchronize_device(mesh_device)

    decoded = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :])
    print(f"dots.ocr N300 OUTPUT: {decoded}")

    assert len(decoded.strip()) > 0, "Generated output should not be empty"

    DispatchManager.save_stats_to_file("dots_ocr_n300_timing_stats.csv")
    TracedRun.release_all()
