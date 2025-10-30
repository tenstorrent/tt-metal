# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Test TTNN SpeechT5 Encoder implementation against PyTorch reference.
"""

import pytest
import torch
import ttnn
from loguru import logger
from transformers import SpeechT5ForTextToSpeech

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from tt.ttnn_speecht5_encoder import (
    TtSpeechT5Encoder,
    TtSpeechT5Config,
)
from reference.speecht5_model import SpeechT5ModelReference


def comp_pcc(golden, calculated, pcc=0.99):
    """
    Compute Pearson Correlation Coefficient (PCC) between two tensors.

    Returns:
        passing: bool - whether PCC meets threshold
        message: str - detailed message with PCC value
    """
    # Flatten tensors
    if isinstance(golden, torch.Tensor):
        golden_flat = golden.flatten().float()
        calculated_flat = calculated.flatten().float()
    else:
        golden_flat = torch.tensor(golden).flatten().float()
        calculated_flat = torch.tensor(calculated).flatten().float()

    # Compute PCC
    cal_mean = torch.mean(calculated_flat)
    golden_mean = torch.mean(golden_flat)

    cal_var = calculated_flat - cal_mean
    golden_var = golden_flat - golden_mean

    cov = torch.sum(cal_var * golden_var)
    cal_std = torch.sqrt(torch.sum(cal_var**2))
    golden_std = torch.sqrt(torch.sum(golden_var**2))

    pcc_value = cov / (cal_std * golden_std + 1e-10)
    pcc_value = pcc_value.item()

    passing = pcc_value >= pcc
    message = f"PCC: {pcc_value:.6f} (threshold: {pcc:.6f})"

    return passing, message


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_ttnn_encoder_shape(device):
    """Test that TTNN encoder produces correct output shape"""
    logger.info("=== Test TTNN Encoder Shape ===")

    # Load HuggingFace model for config
    hf_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    config = TtSpeechT5Config.from_hf_config(hf_model.config)

    # Create test input
    batch_size = 1
    seq_len = 20
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    # Convert to TTNN
    input_ids_tt = ttnn.from_torch(
        input_ids,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    # Load encoder
    logger.info("Loading TTNN encoder...")
    encoder = TtSpeechT5Encoder.from_pretrained(
        "microsoft/speecht5_tts",
        dtype=ttnn.bfloat16,
        device=device,
    )

    # Forward pass
    logger.info("Running forward pass...")
    output_tt = encoder(input_ids_tt)

    # Convert back to PyTorch
    output_torch = ttnn.to_torch(output_tt)

    # Check shape
    expected_shape = (batch_size, seq_len, config.hidden_size)
    assert output_torch.shape == expected_shape, f"Output shape mismatch: {output_torch.shape} != {expected_shape}"

    logger.info(f"✓ Output shape correct: {output_torch.shape}")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("seq_len", [10, 20, 32])
def test_ttnn_encoder_vs_pytorch(device, batch_size, seq_len):
    """Test TTNN encoder output against PyTorch reference with PCC validation"""
    logger.info(f"\n=== Test TTNN Encoder vs PyTorch (batch={batch_size}, seq_len={seq_len}) ===")

    # Load reference model (HuggingFace wrapper with PCC ≈ 1.0)
    logger.info("Loading reference model...")
    ref_model = SpeechT5ModelReference("microsoft/speecht5_tts")
    config = TtSpeechT5Config.from_hf_config(ref_model.config)

    # Create test input
    torch.manual_seed(42)
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    # PyTorch reference
    logger.info("Running PyTorch reference...")
    with torch.no_grad():
        ref_hidden = ref_model.forward_encoder(input_ids)

    # TTNN implementation
    logger.info("Loading TTNN encoder...")
    encoder_tt = TtSpeechT5Encoder.from_pretrained(
        "microsoft/speecht5_tts",
        dtype=ttnn.bfloat16,
        device=device,
    )

    # Convert input to TTNN
    input_ids_tt = ttnn.from_torch(
        input_ids,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    # Forward pass
    logger.info("Running TTNN forward pass...")
    output_tt = encoder_tt(input_ids_tt)

    # Convert back to PyTorch
    output_torch = ttnn.to_torch(output_tt)

    # Compute metrics
    max_diff = torch.max(torch.abs(ref_hidden - output_torch)).item()
    mean_diff = torch.mean(torch.abs(ref_hidden - output_torch)).item()

    logger.info(f"Max absolute difference: {max_diff:.6f}")
    logger.info(f"Mean absolute difference: {mean_diff:.6f}")

    # Compute PCC
    passing, pcc_message = comp_pcc(ref_hidden, output_torch, pcc=0.94)
    logger.info(pcc_message)

    if passing:
        logger.info("✓ Encoder test PASSED! PCC > 0.94")
    else:
        logger.warning("✗ Encoder test FAILED! PCC < 0.94")

    # Assert PCC threshold
    assert passing, f"PCC test failed: {pcc_message}"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_ttnn_encoder_single_layer(device):
    """Test a single encoder layer in isolation"""
    logger.info("\n=== Test Single TTNN Encoder Layer ===")

    from tt.ttnn_speecht5_encoder import TtSpeechT5EncoderLayer, TtSpeechT5EncoderLayerParameters

    # Load HuggingFace model
    hf_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    hf_model.eval()
    state_dict = hf_model.state_dict()
    config = TtSpeechT5Config.from_hf_config(hf_model.config)

    # Test first layer
    layer_idx = 0
    logger.info(f"Testing encoder layer {layer_idx}...")

    # Load TTNN layer parameters
    layer_params = TtSpeechT5EncoderLayerParameters.from_torch(
        state_dict, layer_idx, dtype=ttnn.bfloat16, device=device
    )

    # Create TTNN layer
    layer_tt = TtSpeechT5EncoderLayer(
        layer_params,
        num_heads=config.num_heads,
        hidden_size=config.hidden_size,
        layer_norm_epsilon=config.layer_norm_epsilon,
    )

    # Create test input
    batch_size = 1
    seq_len = 10
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

    # Convert to TTNN
    hidden_states_tt = ttnn.from_torch(
        hidden_states,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    # Forward pass
    output_tt = layer_tt(hidden_states_tt)
    output_torch = ttnn.to_torch(output_tt)

    # Check shape
    assert output_torch.shape == hidden_states.shape
    logger.info(f"✓ Layer output shape correct: {output_torch.shape}")

    # Check that output is different from input (layer is doing something)
    diff = torch.max(torch.abs(hidden_states - output_torch)).item()
    assert diff > 1e-6, "Layer output is identical to input!"
    logger.info(f"✓ Layer is transforming input (max diff: {diff:.6f})")


if __name__ == "__main__":
    # Run tests manually (requires device setup)
    logger.info("To run tests, use: pytest test_ttnn_encoder.py -v")
    logger.info("Make sure to set environment variables:")
    logger.info("  export ARCH_NAME=wormhole_b0")
    logger.info("  export TT_METAL_HOME=$(pwd)")
    logger.info("  export PYTHONPATH=$(pwd)")
