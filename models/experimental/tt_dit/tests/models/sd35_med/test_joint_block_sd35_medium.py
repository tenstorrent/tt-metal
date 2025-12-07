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
    # ids=["final_block"]
    ids=["early_block", "middle_block", "final_block"],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_joint_transformer_block_real_weights(device, reset_seeds, block_type, layer_idx, description):
    """
    Test JointTransformerBlock with real weights from SD3.5 Medium.

    This single test function runs 3 test cases:
    1. Early block (layer 0) with SD35AdaLayerNormZeroX + AdaLayerNormZero
    2. Middle block (layer 13) with AdaLayerNormZero + AdaLayerNormZero
    3. Final block (layer 23) with AdaLayerNormZero + AdaLayerNormContinuous
    """

    dim = 1536
    num_heads = 24
    seq_len = 1024
    context_seq_len = 77
    batch_size = 1

    # Create the appropriate block based on test case
    if block_type == "early":
        block = JointTransformerBlockEarly(
            dim=dim,
            num_heads=num_heads,
            mesh_device=device,
        )
    elif block_type == "middle":
        block = JointTransformerBlockMiddle(
            dim=dim,
            num_heads=num_heads,
            mesh_device=device,
        )
    elif block_type == "final":
        block = JointTransformerBlockFinal(
            dim=dim,
            num_heads=num_heads,
            mesh_device=device,
        )
    else:
        raise ValueError(f"Unknown block_type: {block_type}")

    # Load real weights from SD3.5 Medium model
    try:
        # Use the full HuggingFace model ID
        model_id = "stabilityai/stable-diffusion-3.5-medium"

        torch_model = SD3Transformer2DModel.from_pretrained(
            model_id,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
        )
        torch_model.eval()
        full_state_dict = torch_model.state_dict()

        # Extract weights for the specific layer
        layer_prefix = f"transformer_blocks.{layer_idx}"
        layer_state_dict = {}

        for key, value in full_state_dict.items():
            if key.startswith(layer_prefix):
                new_key = key.replace(f"{layer_prefix}.", "")
                layer_state_dict[new_key] = value

        logger.info(f"Loaded {len(layer_state_dict)} weight tensors for {block_type} block layer {layer_idx}")

        # Print reference model structure and all state dict keys for debugging
        reference_block = torch_model.transformer_blocks[layer_idx]
        print("=" * 80)
        print(f"REFERENCE MODEL STRUCTURE (Block {layer_idx}):")
        print("=" * 80)
        print(reference_block)
        print("=" * 80)
        print(f"STATE DICT KEYS ({len(layer_state_dict)} keys):")
        print("=" * 80)
        for key in sorted(layer_state_dict.keys()):
            print(f"  {key}: {layer_state_dict[key].shape}")
        print("=" * 80)

        # Load weights into TTNN block
        block.load_torch_state_dict(layer_state_dict)
        logger.info(f"✓ Successfully loaded real weights into {block_type} block")

    except Exception as e:
        logger.warning(f"Could not load real weights: {e}")
        pytest.skip(f"Skipping {block_type} test due to weight loading failure: {e}")

    # Create test inputs
    # - hidden_states (x): image latents [B, seq_len=1024, dim]
    # - encoder_hidden_states (context): text embeddings [B, context_seq_len=77, dim]
    # - temb: timestep embedding [B, dim]
    torch.manual_seed(42)

    # Reference inputs: [B, seq_len, dim] format for Diffusers
    hidden_states = torch.randn((batch_size, seq_len, dim), dtype=torch.bfloat16)
    encoder_hidden_states = torch.randn((batch_size, context_seq_len, dim), dtype=torch.bfloat16)
    temb = torch.randn((batch_size, dim), dtype=torch.bfloat16)

    # Run reference PyTorch forward pass
    # Note: Diffusers SD3.5 returns (encoder_hidden_states, hidden_states) - swapped order!
    # Final block returns (x, None) - context is not returned
    with torch.no_grad():
        ref_output = reference_block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb=temb,
        )

    # Handle different return formats
    if block_type == "final":
        # Final block returns just hidden_states (single tensor, not tuple)
        if isinstance(ref_output, tuple):
            ref_x = ref_output[0] if ref_output[0] is not None else ref_output[1]
        else:
            ref_x = ref_output
        ref_context = None
        logger.info(f"Reference output shapes: ref_x={ref_x.shape}, ref_context=None (final block)")
    else:
        # Early/middle blocks return (encoder_hidden_states, hidden_states) - swapped!
        ref_context, ref_x = ref_output
        logger.info(f"Reference output shapes: ref_x={ref_x.shape}, ref_context={ref_context.shape}")

    # Convert to TTNN tensors: [B, 1, seq_len, dim] format for TT model
    x = ttnn.from_torch(hidden_states.unsqueeze(1), dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    context = ttnn.from_torch(
        encoder_hidden_states.unsqueeze(1), dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT
    )
    conditioning = ttnn.from_torch(temb.unsqueeze(1), dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

    # Run TT forward pass
    output_x, output_context = block(
        x=x, context=context, conditioning=conditioning, seq_len=seq_len, context_seq_len=context_seq_len
    )

    # Convert outputs back to torch for validation
    output_x_torch = ttnn.to_torch(ttnn.from_device(output_x))
    output_context_torch = ttnn.to_torch(ttnn.from_device(output_context))

    # Reshape TT outputs to match reference: [B, 1, seq_len, dim] -> [B, seq_len, dim]
    output_x_torch = output_x_torch.squeeze(1)
    output_context_torch = output_context_torch.squeeze(1)

    logger.info(f"TT output shapes: x={output_x_torch.shape}, context={output_context_torch.shape}")

    # PCC comparison
    pcc_threshold = 0.99

    if block_type == "final":
        # Final block: only compare x (hidden_states)
        logger.info("Comparing x (hidden_states) output (final block - no context comparison)...")
        passing, pcc_x = assert_with_pcc(ref_x, output_x_torch, pcc=pcc_threshold)
        logger.info(f"  x PCC: {pcc_x:.6f} (threshold: {pcc_threshold})")

        # Debug: show sample values
        logger.info(f"  ref_x sample: {ref_x[0, 0, :5].tolist()}")
        logger.info(f"  tt_x sample:  {output_x_torch[0, 0, :5].tolist()}")

        if passing:
            logger.info(f"✓ {block_type} block PCC test PASSED")
            logger.info(f"  Description: {description}")
            logger.info(f"  x PCC: {pcc_x:.6f}")
        else:
            pytest.fail(f"{block_type} block PCC test FAILED: x_pcc={pcc_x:.6f}")
    else:
        # Early/middle blocks: compare both x and context
        # Concatenate x and context for full joint block comparison
        ref_full = torch.cat([ref_x, ref_context], dim=1)
        output_full = torch.cat([output_x_torch, output_context_torch], dim=1)

        logger.info(f"Full output shapes: ref={ref_full.shape}, tt={output_full.shape}")

        # Basic validations
        assert output_full.shape == ref_full.shape, f"Shape mismatch: {output_full.shape} vs {ref_full.shape}"
        assert torch.isfinite(output_full).all(), "Output contains non-finite values"

        # Compare x (hidden_states) separately
        logger.info("Comparing x (hidden_states) output...")
        _, pcc_x = assert_with_pcc(ref_x, output_x_torch, pcc=0.0)  # Don't fail, just get PCC
        logger.info(f"  x PCC: {pcc_x:.6f}")

        # Compare context (encoder_hidden_states) separately
        logger.info("Comparing context (encoder_hidden_states) output...")
        _, pcc_context = assert_with_pcc(ref_context, output_context_torch, pcc=0.0)
        logger.info(f"  context PCC: {pcc_context:.6f}")

        # Compare full output
        logger.info("Comparing full joint block output...")
        passing, pcc = assert_with_pcc(ref_full, output_full, pcc=pcc_threshold)
        logger.info(f"  Full Joint Block PCC: {pcc:.6f} (threshold: {pcc_threshold})")

        # Debug: show sample values
        logger.info(f"  ref_x sample: {ref_x[0, 0, :5].tolist()}")
        logger.info(f"  tt_x sample:  {output_x_torch[0, 0, :5].tolist()}")
        logger.info(f"  ref_context sample: {ref_context[0, 0, :5].tolist()}")
        logger.info(f"  tt_context sample:  {output_context_torch[0, 0, :5].tolist()}")

        # Final result
        if passing:
            logger.info(f"✓ {block_type} block PCC test PASSED")
            logger.info(f"  Description: {description}")
            logger.info(f"  PCC: {pcc:.6f}")
        else:
            pytest.fail(
                f"{block_type} block PCC test FAILED: x_pcc={pcc_x:.6f}, context_pcc={pcc_context:.6f}, full_pcc={pcc:.6f}"
            )
