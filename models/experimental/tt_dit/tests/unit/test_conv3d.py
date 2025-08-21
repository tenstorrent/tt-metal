import torch
import pytest
from loguru import logger
import os
import ttnn
from ...layers.conv3d import ContextParallelConv3d as TtContextParallelConv3d
from genmo.mochi_preview.vae.models import ContextParallelConv3d as RefContextParallelConv3d
from models.utility_functions import skip_for_grayskull
from ...utils.check import assert_quality

# Common test configurations
PCC_REQUIRED = 0.999

conv3d_args = {
    "context_parallel": True,
    "causal": True,
    "padding_mode": "replicate",
    "bias": True,
    "groups": 1,
}


def create_random_models(
    mesh_device,
    **model_args,
):
    """Initialize both reference and TT models."""
    # Create reference model
    reference_model = RefContextParallelConv3d(**model_args)

    # Create TT model
    tt_model = TtContextParallelConv3d(mesh_device=mesh_device, torch_ref=reference_model, **model_args)

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


@torch.no_grad()
@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "input_shape, out_channels, kernel_size, stride",
    [
        [(1, 768, 28, 60, 106), 768, (3, 3, 3), (1, 1, 1)],
        [(1, 512, 82, 120, 212), 512, (3, 3, 3), (1, 1, 1)],
        [(1, 256, 163, 240, 424), 256, (3, 3, 3), (1, 1, 1)],
        [(1, 128, 163, 480, 848), 128, (3, 3, 3), (1, 1, 1)],
    ],
    ids=["768", "512", "256", "128"],
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 1), "TG": (8, 4)}.get(
            os.environ.get("FAKE_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
def test_context_parallel_conv3d_forward(mesh_device, input_shape, out_channels, kernel_size, stride, reset_seeds):
    """Test complete forward pass of TtContextParallelConv3d."""
    input_channels = input_shape[1]

    model_args = conv3d_args.copy()
    model_args.update(
        {
            "in_channels": input_channels,
            "out_channels": out_channels,
            "kernel_size": kernel_size,
            "stride": stride,
        }
    )
    # Create the models
    reference_model, tt_model = create_random_models(mesh_device, **model_args)

    # Create input tensor (NCTHW format for PyTorch)
    torch_input = torch.randn(input_shape)

    # Convert to NTHWC format for TT
    tt_input_NTHWC = torch_input.permute(0, 2, 3, 4, 1)
    tt_input_NTHWC = ttnn.from_torch(
        tt_input_NTHWC,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=[None, 1]),
    )

    logger.info("Run TtContextParallelConv3d forward")
    tt_output = tt_model(tt_input_NTHWC)

    # Convert TT output to torch tensor (from NTHWC to NCHW format)
    tt_output_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_output)[0])
    tt_output_torch = tt_output_torch.permute(0, 4, 1, 2, 3)  # NTHWC -> NCHW

    # Get reference output
    logger.info("Run reference model forward")
    with torch.no_grad():
        ref_output = reference_model(torch_input)

    # Compare shapes
    logger.info(f"TT output shape: {tt_output_torch.shape}, Ref output shape: {ref_output.shape}")
    assert tt_output_torch.shape == ref_output.shape, "Output shapes don't match"

    assert_quality(ref_output, tt_output_torch, pcc=PCC_REQUIRED)
