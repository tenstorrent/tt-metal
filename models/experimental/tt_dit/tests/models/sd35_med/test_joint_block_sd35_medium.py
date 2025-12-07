# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Single test function with 3 parametrized test cases for JointTransformerBlock
Tests early, middle, and final block types with real weights
"""

import pytest
import torch
import ttnn
from loguru import logger

from tests.ttnn.utils_for_testing import assert_with_pcc

# Import the three block implementations
from models.experimental.tt_dit.models.transformers.sd35_med.joint_block_sd35_medium import (
    JointTransformerBlockEarly,
    JointTransformerBlockMiddle,
    JointTransformerBlockFinal,
)
from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel


@pytest.mark.parametrize(
    "block_type,layer_idx,description",
    [
        ("early", 0, "Blocks 0-12: SD35AdaLayerNormZeroX + AdaLayerNormZero"),
        ("middle", 13, "Blocks 13-22: AdaLayerNormZero + AdaLayerNormZero"),
        ("final", 23, "Block 23: AdaLayerNormZero + AdaLayerNormContinuous"),
    ],
    ids=["early_block", "middle_block", "final_block"],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_joint_transformer_block_real_weights(device, reset_seeds, block_type, layer_idx, description):
    """
    Test JointTransformerBlock with real weights from SD3.5 Medium.
    """
    dim = 1536
    num_heads = 24
    seq_len = 1024
    context_seq_len = 77
    batch_size = 1

    # Create the appropriate block based on test case
    if block_type == "early":
        block = JointTransformerBlockEarly(dim=dim, num_heads=num_heads, mesh_device=device)
    elif block_type == "middle":
        block = JointTransformerBlockMiddle(dim=dim, num_heads=num_heads, mesh_device=device)
    elif block_type == "final":
        block = JointTransformerBlockFinal(dim=dim, num_heads=num_heads, mesh_device=device)
    else:
        raise ValueError(f"Unknown block_type: {block_type}")

    # Load real weights from SD3.5 Medium model
    model_id = "stabilityai/stable-diffusion-3.5-medium"
    torch_model = SD3Transformer2DModel.from_pretrained(model_id, subfolder="transformer", torch_dtype=torch.bfloat16)
    torch_model.eval()

    # Extract weights for the specific layer
    layer_prefix = f"transformer_blocks.{layer_idx}"
    layer_state_dict = {
        key.replace(f"{layer_prefix}.", ""): value
        for key, value in torch_model.state_dict().items()
        if key.startswith(layer_prefix)
    }

    # Load weights into TTNN block
    block.load_torch_state_dict(layer_state_dict)
    reference_block = torch_model.transformer_blocks[layer_idx]

    # Create test inputs
    torch.manual_seed(42)
    hidden_states = torch.randn((batch_size, seq_len, dim), dtype=torch.bfloat16)
    encoder_hidden_states = torch.randn((batch_size, context_seq_len, dim), dtype=torch.bfloat16)
    temb = torch.randn((batch_size, dim), dtype=torch.bfloat16)

    # Run reference PyTorch forward pass
    with torch.no_grad():
        ref_output = reference_block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb=temb,
        )

    # Handle different return formats
    if block_type == "final":
        # Final block returns just hidden_states
        if isinstance(ref_output, tuple):
            ref_x = ref_output[0] if ref_output[0] is not None else ref_output[1]
        else:
            ref_x = ref_output
        ref_context = None
    else:
        # Early/middle blocks return (encoder_hidden_states, hidden_states) - swapped!
        ref_context, ref_x = ref_output

    # Convert to TTNN tensors
    x = ttnn.from_torch(hidden_states.unsqueeze(1), dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    context = ttnn.from_torch(
        encoder_hidden_states.unsqueeze(1), dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT
    )
    conditioning = ttnn.from_torch(temb.unsqueeze(1), dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

    # Run TT forward pass
    output_x, output_context = block(
        x=x, context=context, conditioning=conditioning, seq_len=seq_len, context_seq_len=context_seq_len
    )

    # Convert outputs back to torch
    output_x_torch = ttnn.to_torch(ttnn.from_device(output_x)).squeeze(1)
    output_context_torch = ttnn.to_torch(ttnn.from_device(output_context)).squeeze(1)

    # PCC comparison
    pcc_threshold = 0.99

    if block_type == "final":
        # Final block: only compare x
        passing, pcc = assert_with_pcc(ref_x, output_x_torch, pcc=pcc_threshold)
        logger.info(f"✓ {block_type} block: PCC={pcc:.6f}")
        assert passing, f"{block_type} block PCC test FAILED: pcc={pcc:.6f}"
    else:
        # Early/middle blocks: compare concatenated output
        ref_full = torch.cat([ref_x, ref_context], dim=1)
        output_full = torch.cat([output_x_torch, output_context_torch], dim=1)
        passing, pcc = assert_with_pcc(ref_full, output_full, pcc=pcc_threshold)
        logger.info(f"✓ {block_type} block: PCC={pcc:.6f}")
        assert passing, f"{block_type} block PCC test FAILED: pcc={pcc:.6f}"
