# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Test for OLMo-3.1-32B-Think with TTNN backend."""

import os

import pytest
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import ttnn
from models.experimental.tt_symbiote.core.run_config import DispatchManager
from models.experimental.tt_symbiote.utils.device_management import set_device
from models.experimental.tt_symbiote.utils.module_replacement import register_module_replacement_dict


@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 50000000, "num_command_queues": 1, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}],
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
def test_olmo_3_1_32b_think(mesh_device):
    """Test OLMo-3.1-32B-Think model with TTNN acceleration."""

    tokenizer = AutoTokenizer.from_pretrained("allenai/Olmo-3.1-32B-Think", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("allenai/Olmo-3.1-32B-Think", trust_remote_code=True).to(
        torch.bfloat16
    )

    nn_to_ttnn = {
        # model.model.layers[0].self_attn.__class__: TTNNOlmoAttention,  # Add TTNNOlmoAttention module for OLMo attention
        # model.model.layers[0].input_layernorm.__class__: TTNNRMSNorm,  # Add RMSNorm or LayerNorm
        # model.model.layers[0].post_attention_layernorm.__class__: TTNNRMSNorm,
    }
    nn_to_ttnn2 = {
        # nn.Linear: TTNNLinearIColShardedWRowSharded,
        # nn.SiLU: TTNNSilu,  # Or appropriate activation function
    }

    messages = [
        {
            "role": "user",
            "content": "Solve this step-by-step: What is 15% of 240? Show your reasoning.",
        },
    ]

    text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device).to(torch.bfloat16)

    modules1 = register_module_replacement_dict(model, nn_to_ttnn, model_config=None)
    modules2 = register_module_replacement_dict(model, nn_to_ttnn2, model_config=None)
    set_device(model, mesh_device)
    all_modules = {**modules1, **modules2}

    print(f"Preprocessing {len(all_modules)} TTNN modules weights...")
    for k, v in tqdm(all_modules.items()):
        v.preprocess_weights()
        v.move_weights_to_device()

    print("Running inference...")
    model.eval()  # Disables dropout, batch norm updates
    torch.set_grad_enabled(False)  # Disables autograd overhead

    # Warmup
    outputs = model.generate(**inputs, max_new_tokens=2, use_cache=True)

    DispatchManager.clear_timings()
    outputs = model.generate(**inputs, max_new_tokens=128, use_cache=True)

    output_ids = outputs[0][len(inputs["input_ids"][0]) :].tolist()

    content = tokenizer.decode(output_ids, skip_special_tokens=True)

    print(f"OLMo-3.1-32B-Think OUTPUT: {content}")
    DispatchManager.save_stats_to_file("olmo_3_1_32b_think_timing_stats.csv")
