# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Test for Molmo2-8B vision-language model with TTNN backend."""


import pytest
import torch
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image
import requests

from models.experimental.tt_symbiote.core.run_config import DispatchManager
from models.experimental.tt_symbiote.utils.device_management import set_device
from models.experimental.tt_symbiote.utils.module_replacement import register_module_replacement_dict


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 245760}],
    indirect=True,
)
def test_molmo2_8b(device):
    """Test Molmo2-8B vision-language model with TTNN acceleration."""

    processor = AutoProcessor.from_pretrained(
        "allenai/Molmo2-8B", trust_remote_code=True, dtype="auto", device_map="auto"
    )
    model = AutoModelForImageTextToText.from_pretrained(
        "allenai/Molmo2-8B",
        trust_remote_code=True,
        dtype="auto",
        device_map="auto",
    )

    nn_to_ttnn = {
        # Vision encoder modules
        # model.vision_backbone.trunk.blocks[0].attn.__class__: TTNNViTSelfAttention,  # Vision attention
        # model.vision_backbone.trunk.blocks[0].norm1.__class__: TTNNLayerNorm,  # Vision normalization
        # Language model modules
        # model.model.layers[0].self_attn.__class__: TTNNMolmoAttention,  # Language attention
        # model.model.layers[0].input_layernorm.__class__: TTNNRMSNorm,  # Language normalization
    }
    nn_to_ttnn2 = {
        # nn.Linear: TTNNLinearIColShardedWRowSharded,
        # nn.GELU: TTNNGelu,  # Or appropriate activation
    }

    # Use a simpler image input instead of video for more reliable testing
    # Download a test image
    image_url = (
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
    )
    image = Image.open(requests.get(image_url, stream=True).raw)

    # Create multimodal conversation with image
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What do you see in this image?"},
                {"type": "image", "image": image},
            ],
        }
    ]

    # Process inputs with processor using apply_chat_template directly
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    # Move inputs to model device and ensure correct shapes
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    print(f"Input shapes: {[(k, v.shape) for k, v in inputs.items()]}")

    # Only perform module replacement if replacement dicts are non-empty
    all_modules = {}
    modules1 = register_module_replacement_dict(
        model,
        nn_to_ttnn,
        model_config=None,
    )
    modules2 = register_module_replacement_dict(
        model,
        nn_to_ttnn2,
        model_config=None,
    )
    set_device(model, device)
    all_modules = {**modules1, **modules2}
    print(f"Preprocessing {len(all_modules)} TTNN modules weights...")
    for k, v in tqdm(all_modules.items()):
        v.preprocess_weights()
        v.move_weights_to_device()

    print("Running inference...")
    model.eval()  # Disables dropout, batch norm updates
    torch.set_grad_enabled(False)  # Disables autograd overhead

    # Warmup
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=2)

    DispatchManager.clear_timings()

    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=100)

    # Extract only generated tokens (skip input)
    generated_tokens = outputs[0][inputs["input_ids"].size(1) :]
    content = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    print(f"Molmo2-8B OUTPUT: {content}")

    DispatchManager.save_stats_to_file("molmo2_8b_timing_stats.csv")
