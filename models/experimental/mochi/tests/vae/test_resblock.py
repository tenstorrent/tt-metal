import torch
import pytest
from loguru import logger
import os
import ttnn
from models.experimental.mochi.vae.resblock import TtResBlock
from genmo.mochi_preview.vae.models import ResBlock as RefResBlock

from models.experimental.mochi.common import (
    compute_metrics,
    to_tt_tensor,
    to_torch_tensor,
)

# Common test configurations
PCC_REQUIRED = 0.99

resblock_args = {
    "affine": True,
    "attn_block": None,
    "causal": True,
    "prune_bottleneck": False,
    "padding_mode": "replicate",
    "bias": True,
}


def create_random_models(mesh_device, **model_args):
    """Initialize both reference and TT models."""
    # Create reference model
    reference_model = RefResBlock(**model_args)
    ref_state_dict = reference_model.state_dict()

    # Create TT model
    tt_model = TtResBlock(mesh_device=mesh_device, state_dict=ref_state_dict, state_dict_prefix="", **model_args)

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
    "N, C, T, H, W",
    [
        (1, 768, 28, 60, 106),
        (1, 512, 82, 120, 212),
        (1, 256, 163, 240, 424),
        (1, 128, 163, 480, 848),
    ],
    ids=["768", "512", "256", "128"],
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
def test_tt_resblock_forward(mesh_device, N, C, T, H, W, use_program_cache, reset_seeds, divide_T):
    """Test complete forward pass of TtResBlock."""
    T = T // divide_T
    block_args = resblock_args.copy()
    block_args["channels"] = C
    reference_model, tt_model = create_random_models(mesh_device, **block_args)

    # Create input tensor
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
    logger.info("Run TtResBlock forward")
    tt_output = tt_model.forward(tt_input)

    # Convert TT output to torch tensor
    tt_output_torch = to_torch_tensor(tt_output, mesh_device)
    tt_output_torch = tt_output_torch.permute(0, 4, 1, 2, 3)  # [N, C, T, H, W]

    # Get reference output
    with torch.no_grad():
        ref_output = reference_model(torch_input)

    validate_outputs(tt_output_torch, ref_output, "TtResBlock forward")
