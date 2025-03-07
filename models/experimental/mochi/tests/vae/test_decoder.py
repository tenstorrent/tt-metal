import torch
import pytest
from loguru import logger
import os
import ttnn
from models.experimental.mochi.tt.vae.decoder import Decoder as TtDecoder
from genmo.mochi_preview.vae.models import Decoder as RefDecoder

from models.experimental.mochi.common import (
    compute_metrics,
    to_torch_tensor,
)
from models.experimental.mochi.tt.vae.common import load_decoder_weights

# Common test configurations
PCC_REQUIRED = 0.99

# Basic decoder configuration that aligns with typical decoder settings
decoder_base_args = {
    "out_channels": 3,
    "base_channels": 128,
    "channel_multipliers": [1, 2, 4, 6],
    "temporal_expansions": [1, 2, 3],
    "spatial_expansions": [2, 2, 2],
    "num_res_blocks": [3, 3, 4, 6, 3],
    "latent_dim": 12,
    "has_attention": [False, False, False, False, False],
    "output_norm": False,
    "nonlinearity": "silu",
    "output_nonlinearity": "silu",
    "causal": True,
}

# Test case configurations for different input sizes
test_configs = [
    {
        "name": "small_latent",
        "input_shape": (1, 12, 28, 30, 53),
        # Expected output will be approximately: (1, 3, 81, 240, 424)
    },
    {
        "name": "medium_latent",
        "input_shape": (1, 12, 28, 60, 106),
        # Expected output will be approximately: (1, 3, 163, 480, 848)
    },
]


def create_decoder_models(mesh_device, use_real_weights=False, **model_args):
    """Initialize both reference and TT decoder models with optional real weights."""
    # Create reference model
    reference_model = RefDecoder(**model_args)

    # Try to load real weights if requested
    if use_real_weights:
        state_dict = load_decoder_weights()
        if state_dict:
            try:
                # Load weights into reference model
                reference_model.load_state_dict(state_dict, strict=True)
                logger.info(f"Loaded real weights for reference decoder model")
            except Exception as e:
                logger.warning(f"Failed to load weights for reference decoder: {e}")

    # Create TT model with same weights
    tt_model = TtDecoder(
        mesh_device=mesh_device,
        state_dict=reference_model.state_dict(),
        state_dict_prefix="",
        **model_args,
    )

    return reference_model, tt_model


def validate_outputs(tt_output, ref_output, test_name):
    """Validate and compare model outputs."""
    pcc, mse, mae = compute_metrics(ref_output, tt_output)
    logger.info(f"Output - PCC: {pcc}, MSE: {mse}, MAE: {mae}")

    assert pcc > PCC_REQUIRED, f"{test_name} failed with PCC {pcc} (required > {PCC_REQUIRED})"


@pytest.mark.parametrize(
    "config",
    test_configs,
    ids=[cfg["name"] for cfg in test_configs],
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
def test_decoder(mesh_device, config, divide_T, use_program_cache, reset_seeds, use_real_weights):
    """Test TtDecoder against reference implementation."""
    input_shape = config["input_shape"]
    N, C, T, H, W = input_shape
    T = T // divide_T
    input_shape = (N, C, T, H, W)

    # Initialize model arguments
    model_args = decoder_base_args.copy()

    logger.info(
        f"Testing decoder with latent_dim={model_args['latent_dim']}, "
        f"base_channels={model_args['base_channels']}, "
        f"channel_multipliers={model_args['channel_multipliers']}, "
        f"use_real_weights={use_real_weights}"
    )

    # Create models
    reference_model, tt_model = create_decoder_models(mesh_device, use_real_weights=use_real_weights, **model_args)

    # Create input tensor (latent representation)
    N, C, T, H, W = input_shape
    torch_input = torch.randn(N, C, T, H, W)

    # Convert to TTNN format [N, T, H, W, C]
    tt_input = torch_input.permute(0, 2, 3, 4, 1)
    tt_input = ttnn.from_torch(
        tt_input,
        device=mesh_device,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=1),
    )

    logger.info(f"Input shape: {torch_input.shape}")
    logger.info("Run TtDecoder forward")
    tt_output = tt_model.forward(tt_input)

    # Convert TT output to torch tensor (from NTHWC to NCTHW)
    tt_output_torch = to_torch_tensor(tt_output, mesh_device, dim=1)
    tt_output_torch = tt_output_torch.permute(0, 4, 1, 2, 3)  # [N, C, T, H, W]
    logger.info(f"TT output shape: {tt_output_torch.shape}")

    # Get reference output
    with torch.no_grad():
        ref_output = reference_model(torch_input)
    logger.info(f"Reference output shape: {ref_output.shape}")

    # Verify output shapes match
    assert (
        ref_output.shape == tt_output_torch.shape
    ), f"Output shapes do not match: {ref_output.shape} vs {tt_output_torch.shape}"

    validate_outputs(tt_output_torch, ref_output, "TtDecoder forward")
