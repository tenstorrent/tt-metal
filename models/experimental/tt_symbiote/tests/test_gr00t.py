# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Test for GR00T N1.6 model with TTNN backend."""

import sys
from pathlib import Path

# Add Isaac-GR00T to path
gr00t_path = Path(__file__).parent.parent / "Isaac-GR00T"
sys.path.insert(0, str(gr00t_path))

import pytest
import torch
from torch import nn
from tqdm import tqdm

try:
    from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6
except:
    print(
        "groot import failed. Make sure you have the Isaac-GR00T https://github.com/NVIDIA/Isaac-GR00T.git in models.tt_symbiote.groot"
    )
    exit(1)

from models.tt_symbiote.core.run_config import DispatchManager
from models.tt_symbiote.modules.activation import TTNNGelu, TTNNSilu
from models.tt_symbiote.modules.linear import TTNNLinear
from models.tt_symbiote.modules.normalization import TTNNLayerNorm
from models.tt_symbiote.utils.device_management import set_device
from models.tt_symbiote.utils.module_replacement import register_module_replacement_dict


@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
def test_gr00t_inference(device):
    """Test GR00T N1.6 model with TTNN acceleration for inference."""

    # Define module replacement mapping
    nn_to_ttnn = {
        nn.Linear: TTNNLinear,
        nn.LayerNorm: TTNNLayerNorm,
        nn.GELU: TTNNGelu,
        nn.SiLU: TTNNSilu,
    }

    torch_dtype = torch.bfloat16

    # Load model configuration
    # Note: Update this path to point to your actual checkpoint or config
    model_path = "nvidia/GR00T-N1.6-3B"

    print(f"Loading GR00T N1.6 model from {model_path}...")
    model = Gr00tN1d6.from_pretrained(model_path, torch_dtype=torch_dtype)

    model.to(dtype=torch_dtype)
    model.eval()
    torch.set_grad_enabled(False)

    # Register module replacements
    print("Registering TTNN module replacements...")
    modules = register_module_replacement_dict(model, nn_to_ttnn, model_config=None)

    # Set device for all TTNN modules
    print("Setting device for TTNN modules...")
    set_device(model, device)

    # Preprocess and move weights to device
    print("Preprocessing and moving weights to device...")
    for k, v in tqdm(modules.items(), desc="Processing modules"):
        v.preprocess_weights()
        v.move_weights_to_device()

    # Create dummy inputs for testing
    # Based on EagleBackbone.forward() - expects input_ids, attention_mask, pixel_values
    # and action head expects: state, action, embodiment_id, action_mask

    # Get dimensions from model config
    config = model.config
    batch_size = 1
    seq_len = 256  # Sequence length for input_ids
    num_images = 1  # Number of images
    img_channels = 3
    img_height, img_width = 224, 224
    state_dim = config.max_state_dim  # Use config's max_state_dim (29)
    action_dim = config.max_action_dim  # Use config's max_action_dim (29)
    action_horizon = config.action_horizon  # 16

    # Get image token index from backbone model config
    image_token_index = model.backbone.model.config.image_token_index  # 151669

    # Create input_ids with proper image tokens
    # Format: [tokens... <image> tokens...]
    input_ids = torch.randint(0, 32000, (batch_size, seq_len), dtype=torch.long)
    # Insert image token at position 128 (middle of sequence)
    input_ids[0, 128] = image_token_index

    # Create dummy inputs matching what the model expects
    dummy_inputs = {
        # Eagle backbone inputs
        "input_ids": input_ids,
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
        "pixel_values": torch.randn(
            batch_size,
            num_images,
            img_channels,
            img_height,
            img_width,
            dtype=torch.float32,  # Use float32 for vision inputs
        ),
        # Action head inputs
        "state": torch.randn(batch_size, 1, state_dim, dtype=torch_dtype),
        "embodiment_id": torch.zeros(batch_size, dtype=torch.long),
    }

    print("Running inference...")
    DispatchManager.clear_timings()

    # Use get_action instead of forward to avoid training-time Beta sampling
    # which doesn't support bfloat16
    output = model.get_action(dummy_inputs)
    DispatchManager.save_stats_to_file("gr00t_timing_stats.csv")
    print("Inference completed successfully!")
