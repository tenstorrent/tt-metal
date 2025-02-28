import torch
import pytest
from loguru import logger
import os
import ttnn
from models.experimental.mochi.vae.upsample import TtCausalUpsampleBlock
from genmo.mochi_preview.vae.models import CausalUpsampleBlock as RefCausalUpsampleBlock

from models.experimental.mochi.common import (
    compute_metrics,
    to_tt_tensor,
    to_torch_tensor,
)

# Common test configurations
PCC_REQUIRED = 0.99

# Base configuration that applies to all test cases
upsample_base_args = {
    "affine": True,
    "causal": True,
    "prune_bottleneck": False,
    "padding_mode": "replicate",
    "bias": True,
    "has_attention": False,
}

# Test case configurations from decoder
test_configs = [
    # First upsample block (768->512)
    {
        "name": "block1_768-512",
        "in_channels": 768,
        "out_channels": 512,
        "num_res_blocks": 6,
        "temporal_expansion": 3,
        "spatial_expansion": 2,
        "input_shape": (1, 768, 28, 60, 106),
        # "expected_output_shape": (1, 512, 82, 120, 212),
    },
    # Second upsample block (512->256)
    {
        "name": "block2_512-256",
        "in_channels": 512,
        "out_channels": 256,
        "num_res_blocks": 4,
        "temporal_expansion": 2,
        "spatial_expansion": 2,
        "input_shape": (1, 512, 82, 120, 212),
        # "expected_output_shape": (1, 256, 163, 240, 424),
    },
    # Third upsample block (256->128)
    {
        "name": "block3_256-128",
        "in_channels": 256,
        "out_channels": 128,
        "num_res_blocks": 3,
        "temporal_expansion": 1,
        "spatial_expansion": 2,
        "input_shape": (1, 256, 163, 240, 424),
        # "expected_output_shape": (1, 128, 163, 480, 848),
    },
]


def create_random_models(mesh_device, in_channels, out_channels, **model_args):
    """Initialize both reference and TT models."""
    # Create reference model
    reference_model = RefCausalUpsampleBlock(in_channels=in_channels, out_channels=out_channels, **model_args)
    ref_state_dict = reference_model.state_dict()

    # Create TT model
    tt_model = TtCausalUpsampleBlock(
        mesh_device=mesh_device,
        state_dict=ref_state_dict,
        state_dict_prefix="",
        in_channels=in_channels,
        out_channels=out_channels,
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
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("FAKE_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
def test_upsample(mesh_device, config, divide_T, use_program_cache, reset_seeds):
    """Test TtCausalUpsampleBlock against reference implementation."""
    in_channels = config["in_channels"]
    out_channels = config["out_channels"]
    num_res_blocks = config["num_res_blocks"]
    temporal_expansion = config["temporal_expansion"]
    spatial_expansion = config["spatial_expansion"]
    input_shape = config["input_shape"]
    N, C, T, H, W = input_shape
    T = T // divide_T
    input_shape = (N, C, T, H, W)
    # expected_output_shape = config["expected_output_shape"]

    # Prepare model args
    block_args = upsample_base_args.copy()
    block_args.update(
        {
            "temporal_expansion": temporal_expansion,
            "spatial_expansion": spatial_expansion,
            "num_res_blocks": num_res_blocks,
        }
    )

    logger.info(
        f"Testing upsample with in_channels={in_channels}, out_channels={out_channels}, "
        f"temporal_expansion={temporal_expansion}, "
        f"spatial_expansion={spatial_expansion}, "
        f"num_res_blocks={num_res_blocks}"
    )

    reference_model, tt_model = create_random_models(
        mesh_device, in_channels=in_channels, out_channels=out_channels, **block_args
    )

    # Create input tensor with correct shape from the decoder
    N, C, T, H, W = input_shape
    torch_input = torch.randn(N, C, T, H, W)
    tt_input = torch_input.permute(0, 2, 3, 4, 1)  # [N, T, H, W, C]
    tt_input = ttnn.from_torch(
        tt_input,
        device=mesh_device,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    logger.info(f"Input shape: {torch_input.shape}")
    logger.info("Run TtCausalUpsampleBlock forward")
    tt_output = tt_model.forward(tt_input)

    # Convert TT output to torch tensor
    tt_output_torch = to_torch_tensor(tt_output, mesh_device)
    tt_output_torch = tt_output_torch.permute(0, 4, 1, 2, 3)  # [N, C, T, H, W]
    logger.info(f"TT output shape: {tt_output_torch.shape}")

    # Get reference output
    with torch.no_grad():
        ref_output = reference_model(torch_input)
    logger.info(f"Reference output shape: {ref_output.shape}")

    # Verify output shapes match expected shapes from decoder
    # assert tt_output_torch.shape == expected_output_shape, \
    # f"Expected shape {expected_output_shape}, got {tt_output_torch.shape}"
    # assert ref_output.shape == tt_output_torch.shape, \
    # f"Reference output shape {ref_output.shape} doesn't match TT output shape {tt_output_torch.shape}"

    validate_outputs(tt_output_torch, ref_output, "TtCausalUpsampleBlock forward")
