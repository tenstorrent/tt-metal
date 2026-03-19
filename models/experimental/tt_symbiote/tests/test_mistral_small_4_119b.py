# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Test for Mistral-Small-4-119B with TTNN backend.

This test implements support for the mistralai/Mistral-Small-4-119B-2603 model
which uses:
- 128 experts with top-4 routing
- Softmax-based routing (not sigmoid)
- No group routing (n_group=1)
- 4096 hidden_size, 2048 moe_intermediate_size

Note: This model requires transformers>=5.3.0 and uses the mistral3 multimodal
architecture with a mistral4 text backbone (which maps to DeepseekV3 architecture).
"""

import os

import pytest
import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoTokenizer, AutoConfig

# Import Mistral4 MoE class (requires transformers with mistral4 support)
from transformers.models.mistral4.modeling_mistral4 import Mistral4MoE

import ttnn
from models.experimental.tt_symbiote.core.run_config import DispatchManager
from models.experimental.tt_symbiote.modules.activation import TTNNSilu
from models.experimental.tt_symbiote.modules.linear import (
    TTNNLinearIColShardedWRowSharded,
)
from models.experimental.tt_symbiote.utils.device_management import set_device
from models.experimental.tt_symbiote.utils.module_replacement import register_module_replacement_dict
from models.experimental.tt_symbiote.core.run_config import TracedRun
from models.experimental.tt_symbiote.modules.moe import TTNNMistralMoE_BF4


# Model identifier
MODEL_ID = "mistralai/Mistral-Small-4-119B-2603"


def get_moe_class_from_model(model):
    """
    Dynamically get the MoE block class from the loaded model.

    This handles Mistral4MoE for Mistral-Small-4-119B.
    """
    # For Mistral3 models with Mistral4 text backbone
    if hasattr(model, "model"):
        # Check for language_model (Mistral3ForConditionalGeneration structure)
        lang_model = getattr(model.model, "language_model", None)
        if lang_model is not None:
            layers = getattr(lang_model, "layers", None)
            if layers is not None:
                for layer in layers:
                    if hasattr(layer, "mlp") and isinstance(layer.mlp, Mistral4MoE):
                        return Mistral4MoE

        # Check direct layers (for causal LM models)
        layers = getattr(model.model, "layers", None)
        if layers is not None:
            for layer in layers:
                if hasattr(layer, "block_sparse_moe"):
                    return layer.block_sparse_moe.__class__
                if hasattr(layer, "mlp"):
                    if isinstance(layer.mlp, Mistral4MoE):
                        return Mistral4MoE
                    if hasattr(layer.mlp, "experts"):
                        return layer.mlp.__class__

    # Fallback: return Mistral4MoE for this model
    return Mistral4MoE


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
def test_mistral_small_4_119b(mesh_device):
    """Test Mistral-Small-4-119B model with TTNN acceleration.

    This test:
    1. Loads the Mistral-Small-4-119B model
    2. Replaces MoE layers with TTNNMistralMoE (softmax routing)
    3. Replaces linear layers with TTNN accelerated versions
    4. Runs inference and validates output
    """

    # Load tokenizer and model
    # Note: Mistral-Small-4-119B uses Mistral3 multimodal architecture with DeepseekV3 text backbone
    # It must be loaded with AutoModelForImageTextToText, not AutoModelForCausalLM
    # We disable FP8 quantization since it requires triton/GPU - load in BF16 directly
    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    # Load config and disable FP8 quantization by deleting the quantization_config
    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    if hasattr(config, "quantization_config"):
        delattr(config, "quantization_config")

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,  # Use BF16 precision
    )

    # Dynamically get the MoE class from the loaded model
    moe_class = get_moe_class_from_model(model)
    if moe_class is None:
        pytest.skip("Could not find MoE class in model architecture")

    print(f"Found MoE class: {moe_class.__name__}")

    # Module replacement mappings
    nn_to_ttnn_moe = {
        moe_class: TTNNMistralMoE_BF4,  # Use bfloat4_b experts (fits T3K)
    }
    nn_to_ttnn_linear = {
        nn.Linear: TTNNLinearIColShardedWRowSharded,
        nn.SiLU: TTNNSilu,
    }

    # Test prompt
    messages = [
        {
            "role": "user",
            "content": "What is your favorite condiment? There are so many condiments to choose from, each bringing its unique flavor and texture to enhance different dishes. Do you prefer the classic taste of ketchup, the creamy richness of mayonnaise, the spicy kick of mustard, or perhaps something more exotic like sriracha or hoisin sauce? Maybe you enjoy the tangy zest of salsa or the smooth and savory taste of aioli. Share what your favorite condiment is and why you love it. Does it remind you of a specific dish or meal?",
        },
    ]

    # Tokenize input
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    # Register module replacements
    modules_moe = register_module_replacement_dict(model, nn_to_ttnn_moe, model_config=None)
    modules_linear = register_module_replacement_dict(model, nn_to_ttnn_linear, model_config=None)

    # Set device for all modules
    set_device(model, mesh_device)

    # Combine all modules
    all_modules = {**modules_moe, **modules_linear}

    # Preprocess and move weights
    print(f"Preprocessing {len(all_modules)} TTNN modules weights...")
    for k, v in tqdm(all_modules.items()):
        v.preprocess_weights()
        v.move_weights_to_device()

    # Run inference
    print("Running inference...")
    model.eval()
    torch.set_grad_enabled(False)

    # Warmup run
    outputs = model.generate(**inputs, max_new_tokens=2, use_cache=True)

    # Clear timings from warmup
    DispatchManager.clear_timings()

    # Actual inference
    outputs = model.generate(**inputs, max_new_tokens=64, use_cache=True)

    # Decode and print output
    generated_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :])
    print(f"Mistral-Small-4-119B OUTPUT: {generated_text}")

    # Save timing statistics
    DispatchManager.save_stats_to_file("mistral_small_4_119b_timing_stats.csv")

    # Cleanup
    TracedRun.release_all()

    # Basic validation
    assert len(generated_text) > 0, "Model should generate some output"
    print("Test passed!")
