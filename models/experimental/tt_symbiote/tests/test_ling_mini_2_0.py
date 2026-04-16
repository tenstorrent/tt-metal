# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Test for Ling-mini-2.0 with TTNN backend."""

import os

import pytest
import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import ttnn
from models.experimental.tt_symbiote.core.run_config import DispatchManager, TracedRun
from models.experimental.tt_symbiote.modules.activation import TTNNSilu
from models.experimental.tt_symbiote.modules.linear import (
    TTNNLinearIColShardedWRowSharded,
)
from models.experimental.tt_symbiote.utils.device_management import set_device
from models.experimental.tt_symbiote.utils.module_replacement import register_module_replacement_dict
from models.experimental.tt_symbiote.modules.decoder_layer import TTNNBailingMoEDecoderLayerPadded
from models.experimental.tt_symbiote.modules.normalization import TTNNDistributedRMSNorm
from models.experimental.tt_symbiote.modules.embedding import TTNNBailingPaddedEmbedding, TTNNBailingRotaryEmbedding
from models.experimental.tt_symbiote.models.bailing_moe_v2 import TTNNBailingMoeV2Model
from models.experimental.tt_symbiote.models.ling import (
    DecodeParams,
    create_paged_kv_cache,
    decode_with_logit_postprocess,
    replicated_mesh_tt_to_torch,
    token_ids_list_for_hf_decode,
)


@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 200000000, "num_command_queues": 1, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {
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
        }.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))
    ],
    indirect=True,
)
def test_ling_mini_2_0(mesh_device):
    """Test Ling-mini-2.0 model with TTNN acceleration."""

    tokenizer = AutoTokenizer.from_pretrained("inclusionAI/Ling-mini-2.0", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("inclusionAI/Ling-mini-2.0", trust_remote_code=True)
    nn_to_ttnn = {
        model.model.layers[0].__class__: TTNNBailingMoEDecoderLayerPadded,
        model.model.norm.__class__: TTNNDistributedRMSNorm,
        nn.Embedding: TTNNBailingPaddedEmbedding,
        model.model.rotary_emb.__class__: TTNNBailingRotaryEmbedding,
    }
    nn_to_ttnn2 = {
        nn.Linear: TTNNLinearIColShardedWRowSharded,
        nn.SiLU: TTNNSilu,
    }
    nn_to_ttnn_3 = {
        model.model.__class__: TTNNBailingMoeV2Model,
    }
    messages = [
        {
            "role": "user",
            "content": "What is your favorite condiment? There are so many condiments to choose from, each bringing its unique flavor and texture to enhance different dishes. Do you prefer the classic taste of ketchup, the creamy richness of mayonnaise, the spicy kick of mustard, or perhaps something more exotic like sriracha or hoisin sauce? Maybe you enjoy the tangy zest of salsa or the smooth and savory taste of aioli. Share what your favorite condiment is and why you love it. Does it remind you of a specific dish or meal?",
        },
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
    modules1 = register_module_replacement_dict(model, nn_to_ttnn, model_config=None)
    modules2 = register_module_replacement_dict(model, nn_to_ttnn2, model_config=None)
    modules3 = register_module_replacement_dict(model, nn_to_ttnn_3, model_config=None)
    # After replacing all nn.Modules with TTNNModules, HF's model.device
    # (which calls next(self.parameters())) fails since no nn.Module params remain.
    # Patch it to return cpu — HF uses this for placing generated token tensors.
    type(model).device = property(lambda self: torch.device("cpu"))
    set_device(model, mesh_device)
    all_modules = {**modules1, **modules2, **modules3}
    print(f"Preprocessing {len(all_modules)} TTNN modules weights...")
    for k, v in tqdm(all_modules.items()):
        v.preprocess_weights()
        v.move_weights_to_device()

    # Create paged KV cache
    paged_cache = create_paged_kv_cache(model.config, mesh_device, batch_size=1)

    print("Running inference with paged attention...")
    model.eval()
    torch.set_grad_enabled(False)

    decode_params = DecodeParams()  # greedy: temperature=0

    # Warmup
    decode_with_logit_postprocess(
        model, inputs["input_ids"], inputs.get("attention_mask"), paged_cache, 2, decode_params, mesh_device
    )
    paged_cache.reset()
    # Short decode (program cache hot)
    decode_with_logit_postprocess(
        model, inputs["input_ids"], inputs.get("attention_mask"), paged_cache, 4, decode_params, mesh_device
    )
    paged_cache.reset()

    DispatchManager.clear_timings()
    output_ids_tt = decode_with_logit_postprocess(
        model, inputs["input_ids"], inputs.get("attention_mask"), paged_cache, 128, decode_params, mesh_device
    )
    output_ids = replicated_mesh_tt_to_torch(output_ids_tt, mesh_device).long()
    ttnn.deallocate(output_ids_tt)
    prompt_len = inputs["input_ids"].shape[-1]
    gen_list = output_ids.reshape(-1)[prompt_len:].tolist()
    decoded = tokenizer.decode(token_ids_list_for_hf_decode(gen_list, tokenizer))
    print(f"Ling-mini-2.0 PAGED ATTENTION OUTPUT: {decoded}")

    # Verify output is coherent (non-empty generated text)
    assert len(decoded.strip()) > 0, "Generated output should not be empty"

    DispatchManager.save_stats_to_file("ling_mini_2_0_paged_attention_timing_stats.csv")
    TracedRun.release_all()
