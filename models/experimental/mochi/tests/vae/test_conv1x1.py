import torch
import torch.nn as nn
import pytest
from loguru import logger
import os
import ttnn
import pickle
from pathlib import Path
from models.experimental.mochi.vae.conv1x1 import TtConv1x1
from genmo.mochi_preview.vae.models import Conv1x1 as RefConv1x1, Decoder

from models.experimental.mochi.common import (
    compute_metrics,
    to_tt_tensor,
    to_torch_tensor,
)
from models.experimental.mochi.vae.common import load_decoder_weights

# Common test configurations
PCC_REQUIRED = 0.99


class Conv3d1x1(nn.Conv3d):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__(in_channels, out_channels, kernel_size=(1, 1, 1), bias=bias)


def extract_conv1x1_weights(decoder, layer_name):
    """Extract weights for a specific Conv1x1 layer from the decoder"""
    # For common layers like output_proj
    if f"{layer_name}.weight" in decoder:
        return {
            "weight": decoder[f"{layer_name}.weight"],
            "bias": decoder[f"{layer_name}.bias"] if f"{layer_name}.bias" in decoder else None,
        }
    # For layers in blocks
    for key in decoder:
        if layer_name in key:
            prefix = key.split(layer_name)[0] + layer_name
            return {
                "weight": decoder[f"{prefix}.weight"],
                "bias": decoder[f"{prefix}.bias"] if f"{prefix}.bias" in decoder else None,
            }

    raise ValueError(f"Could not find {layer_name} in decoder")


def create_random_conv3d_models(mesh_device, in_channels, out_channels, bias=True, use_real_weights=False):
    """Initialize both reference Conv3d and TT models."""
    # Create reference model
    reference_model = Conv3d1x1(in_channels, out_channels, bias=bias)

    # Optionally load real weights
    if use_real_weights:
        try:
            decoder = load_decoder_weights()
            # First layer in decoder is usually Conv3d 1x1x1
            weights = extract_conv1x1_weights(decoder, "blocks.0.0")
            reference_model.load_state_dict(weights)
            logger.info("Loaded real weights for Conv3d 1x1x1")
        except (FileNotFoundError, ValueError) as e:
            logger.warning(f"Could not load real weights, using random: {e}")
            use_real_weights = False

    ref_state_dict = reference_model.state_dict()

    # Create TT model
    tt_model = TtConv1x1(
        mesh_device=mesh_device,
        state_dict=ref_state_dict,
        state_dict_prefix="",
        in_channels=in_channels,
        out_channels=out_channels,
        bias=bias,
    )

    return reference_model, tt_model


def create_random_conv1x1_models(mesh_device, in_channels, out_channels, bias=True, use_real_weights=False):
    """Initialize both reference Conv1x1 and TT models."""
    # Create reference model
    reference_model = RefConv1x1(in_channels, out_channels, bias=bias)

    # Optionally load real weights
    if use_real_weights:
        try:
            decoder = load_decoder_weights()
            # output_proj is typically Conv1x1
            weights = extract_conv1x1_weights(decoder, "output_proj")
            reference_model.load_state_dict(weights)
            logger.info("Loaded real weights for Conv1x1")
        except (FileNotFoundError, ValueError) as e:
            logger.warning(f"Could not load real weights, using random: {e}")
            use_real_weights = False

    ref_state_dict = reference_model.state_dict()

    # Create TT model
    tt_model = TtConv1x1(
        mesh_device=mesh_device,
        state_dict=ref_state_dict,
        state_dict_prefix="",
        in_channels=in_channels,
        out_channels=out_channels,
        bias=bias,
    )

    return reference_model, tt_model


def validate_outputs(tt_output, ref_output, test_name):
    """Validate and compare model outputs."""
    pcc, mse, mae = compute_metrics(ref_output, tt_output)
    logger.info(f"Output - PCC: {pcc}, MSE: {mse}, MAE: {mae}")

    passing = pcc >= PCC_REQUIRED

    if passing:
        logger.info(f"{test_name} Passed!")
    else:
        logger.warning(f"{test_name} Failed!")
        logger.error(f"Output failed with PCC: {pcc}, MSE: {mse}, MAE: {mae}")

    assert passing, f"{test_name} output does not meet PCC requirement {PCC_REQUIRED}"


@pytest.mark.parametrize(
    "N, C_in, C_out, T, H, W",
    [
        (1, 12, 768, 28, 60, 106),
    ],
    ids=["12->768"],
)
@pytest.mark.parametrize("divide_T", [8, 1], ids=["T8", "T1"])  # Emulate T fracturing
@pytest.mark.parametrize("use_real_weights", [False, True], ids=["random_weights", "real_weights"])
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("FAKE_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
def test_tt_conv3d_1x1x1(
    mesh_device, N, C_in, C_out, T, H, W, use_program_cache, reset_seeds, divide_T, use_real_weights
):
    """Test forward pass of TtConv1x1 against Conv3d with 1x1x1 kernel."""
    T = T // divide_T
    reference_model, tt_model = create_random_conv3d_models(mesh_device, C_in, C_out, use_real_weights=use_real_weights)

    # Create input tensor
    torch_input = torch.randn(N, C_in, T, H, W)
    tt_input = torch_input.permute(0, 2, 3, 4, 1)  # [N, T, H, W, C]
    tt_input = ttnn.from_torch(
        tt_input,
        device=mesh_device,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    logger.info("Run TtConv1x1 forward (Conv3d mode)")
    tt_output = tt_model.forward(tt_input)

    # Convert TT output to torch tensor
    tt_output_torch = to_torch_tensor(tt_output, mesh_device)
    tt_output_torch = tt_output_torch.permute(0, 4, 1, 2, 3)  # [N, C, T, H, W]

    # Get reference output
    with torch.no_grad():
        ref_output = reference_model(torch_input)

    validate_outputs(tt_output_torch, ref_output, "TtConv1x1 (Conv3d mode) forward")


@pytest.mark.parametrize(
    "N, C_in, C_out, T, H, W",
    [
        (1, 128, 3, 163, 480, 848),
    ],
    ids=["128->3"],
)
@pytest.mark.parametrize("divide_T", [8, 1], ids=["T8", "T1"])  # Emulate T fracturing
@pytest.mark.parametrize("use_real_weights", [False, True], ids=["random_weights", "real_weights"])
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("FAKE_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
def test_tt_conv1x1_linear(
    mesh_device, N, C_in, C_out, T, H, W, use_program_cache, reset_seeds, divide_T, use_real_weights
):
    """Test forward pass of TtConv1x1 against Conv1x1 (linear implementation)."""
    T = T // divide_T
    reference_model, tt_model = create_random_conv1x1_models(
        mesh_device, C_in, C_out, use_real_weights=use_real_weights
    )

    # Create input tensor
    torch_input = torch.randn(N, C_in, T, H, W)
    tt_input = torch_input.permute(0, 2, 3, 4, 1)  # [N, T, H, W, C]
    tt_input = ttnn.from_torch(
        tt_input,
        device=mesh_device,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    logger.info("Run TtConv1x1 forward (Linear mode)")
    tt_output = tt_model.forward(tt_input)

    # Convert TT output to torch tensor
    tt_output_torch = to_torch_tensor(tt_output, mesh_device)
    tt_output_torch = tt_output_torch.permute(0, 4, 1, 2, 3)  # [N, C, T, H, W]

    # Get reference output
    with torch.no_grad():
        ref_output = reference_model(torch_input)

    validate_outputs(tt_output_torch, ref_output, "TtConv1x1 (Linear mode) forward")


@pytest.mark.parametrize(
    "texp, sexp, B, C_in, C_out, T, H, W",
    [
        (3, 2, 1, 768, 6144, 28, 60, 106),
        (2, 2, 1, 512, 2048, 82, 120, 212),
        (1, 2, 1, 256, 512, 163, 240, 424),
    ],
    ids=["texp3_sexp2", "texp2_sexp2", "texp1_sexp2"],
)
@pytest.mark.parametrize("divide_T", [8, 1], ids=["T8", "T1"])  # Emulate T fracturing
@pytest.mark.parametrize("use_real_weights", [False, True], ids=["random_weights", "real_weights"])
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("FAKE_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
def test_tt_conv1x1_expand(
    mesh_device, texp, sexp, B, C_in, C_out, T, H, W, use_program_cache, reset_seeds, divide_T, use_real_weights
):
    """Test TtConv1x1 with channel expansion similar to test_conv1x1.py."""
    T = T // divide_T

    # Create input tensor
    torch_input = torch.randn(B, C_in, T, H, W)

    # Create Conv1x1 module for reference
    reference_model = RefConv1x1(C_in, C_out)

    # Optionally load real weights
    if use_real_weights:
        try:
            state_dict = load_decoder_weights()
            # Find matching proj layer based on dimensions
            found = False
            for block_idx in range(1, 4):  # Check blocks 1-3 which contain CausalUpsampleBlocks
                weight_key = f"blocks.{block_idx}.proj.weight"
                bias_key = f"blocks.{block_idx}.proj.bias"

                if (
                    weight_key in state_dict
                    and bias_key in state_dict
                    and state_dict[weight_key].shape[0] == C_out
                    and state_dict[weight_key].shape[1] == C_in
                ):
                    weights = {"weight": state_dict[weight_key], "bias": state_dict[bias_key]}
                    reference_model.load_state_dict(weights)
                    logger.info(f"Loaded real weights for expansion Conv1x1 from {weight_key}")
                    found = True
                    break

            if not found:
                logger.warning("Could not find matching expansion layer, using random weights")
                use_real_weights = False
        except (FileNotFoundError, ValueError) as e:
            logger.warning(f"Could not load real weights, using random: {e}")
            use_real_weights = False

    # Run reference forward
    torch_output = reference_model(torch_input)

    # Reshape for expansion (similar to test_conv1x1.py)
    out_chan_dim = C_out // (texp * sexp * sexp)

    def swizzle_weight(w):
        # X (C texp sexp sexp) -> X (texp sexp sexp C)
        w = w.reshape(-1, out_chan_dim, texp, sexp, sexp)
        w = w.permute(0, 2, 3, 4, 1)
        w = w.reshape(-1, texp * sexp * sexp * out_chan_dim)
        return w.squeeze()

    tt_model = TtConv1x1(
        mesh_device=mesh_device,
        state_dict=reference_model.state_dict(),
        state_dict_prefix="",
        in_channels=C_in,
        out_channels=C_out,
        bias=reference_model.bias is not None,
        swizzle_weight=swizzle_weight,
    )

    # Run TT forward
    tt_input = torch_input.permute(0, 2, 3, 4, 1)  # [B, T, H, W, C]
    tt_input = ttnn.from_torch(
        tt_input,
        device=mesh_device,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    logger.info("Run TtConv1x1 forward with channel expansion")
    tt_output = tt_model.forward(tt_input)

    # Convert TT output to torch tensor
    tt_output_torch = to_torch_tensor(tt_output, mesh_device)
    tt_output_torch = tt_output_torch.permute(0, 4, 1, 2, 3)  # [B, C, T, H, W]

    # Undo output channel swizzling to compare to ground truth
    tt_output_torch = tt_output_torch.reshape(B, texp, sexp, sexp, out_chan_dim, T, H, W)
    tt_output_torch = tt_output_torch.permute(0, 4, 1, 2, 3, 5, 6, 7).reshape(B, C_out, T, H, W)

    validate_outputs(tt_output_torch, torch_output, "TtConv1x1 with channel expansion")
