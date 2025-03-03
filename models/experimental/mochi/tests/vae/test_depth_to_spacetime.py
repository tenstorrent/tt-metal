import torch
import ttnn
import pytest
from loguru import logger
from models.experimental.mochi.common import compute_metrics
from genmo.mochi_preview.vae.models import DepthToSpaceTime

import math
import os


@pytest.mark.parametrize(
    "B,C,T,H,W,texp,sexp",
    [
        # From first block
        (1, 6144, 28, 60, 106, 3, 2),
        # From second block
        (1, 2048, 82, 120, 212, 2, 2),
        # From third block
        (1, 512, 163, 240, 424, 1, 2),
    ],
)
def test_depth_to_spacetime_torch(B, C, T, H, W, texp, sexp):
    # Set manual seed for reproducibility
    torch.manual_seed(42)

    # Create input tensor
    input_shape = (B, C, T, H, W)
    x = torch.randn(*input_shape, dtype=torch.float32)

    # Create DepthToSpaceTime module
    d2st = DepthToSpaceTime(texp, sexp)

    # Get output
    output = d2st(x)

    # Manual computation for verification
    out_channels = C // (texp * sexp * sexp)
    manual_output = x.reshape(B, out_channels, texp, sexp, sexp, T, H, W)
    manual_output = manual_output.permute(0, 1, 5, 2, 6, 3, 7, 4)
    manual_output = manual_output.reshape(B, out_channels, T * texp, H * sexp, W * sexp)

    # For first block with texp > 1, drop first texp-1 frames
    if texp > 1:
        manual_output = manual_output[:, :, texp - 1 :]

    # Compare outputs
    pcc, mse, mae = compute_metrics(output, manual_output)
    logger.info(f"PCC = {pcc}, MSE = {mse}, MAE = {mae}")

    # Verify shapes
    expected_T = T * texp if texp == 1 else T * texp - (texp - 1)
    expected_shape = (B, out_channels, expected_T, H * sexp, W * sexp)
    assert output.shape == expected_shape
    assert torch.allclose(output, manual_output, rtol=1e-5, atol=1e-5)
    assert pcc > 0.99


@pytest.mark.parametrize(
    "B,C,T,H,W,texp,sexp",
    [
        # Small test case
        (1, 192, 16, 5, 5, 3, 2),
        # From first block
        (1, 6144, 28, 60, 106, 3, 2),
        # From second block
        (1, 2048, 82, 120, 212, 2, 2),
        # From third block
        (1, 512, 163, 240, 424, 1, 2),
    ],
)
@pytest.mark.parametrize("parallel_factor", [1, 8], ids=["T1", "T8"])
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("FAKE_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
def test_depth_to_spacetime_tt(mesh_device, B, C, T, H, W, texp, sexp, parallel_factor):
    # Set manual seed for reproducibility
    torch.manual_seed(42)

    T = math.ceil(T / parallel_factor)
    # Create input tensor
    input_shape = (B, C, T, H, W)
    x = torch.randn(*input_shape, dtype=torch.float32)
    # x = torch.ones(*input_shape, dtype=torch.float32)

    # Create DepthToSpaceTime module for ground truth
    d2st = DepthToSpaceTime(texp, sexp)
    torch_output = d2st(x)

    # ttnn input will be of shape: B T (H W) (texp sexp sexp C)
    # it will also be tilized, coming out of the conv1x1 linear layer
    out_channels = C // (texp * sexp * sexp)
    x_perm = x.permute(0, 2, 3, 4, 1).reshape(B, T, H * W, out_channels, texp, sexp, sexp)
    x_perm = x_perm.permute(0, 1, 2, 4, 5, 6, 3).reshape(B, T, H, W, texp * sexp**2 * out_channels)

    tt_input = ttnn.from_torch(
        x_perm,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=1),
    )

    def depth_to_spacetime(x_NTHWC):
        if mesh_device.get_num_devices() == 1:
            B, T, H, W, C = x_NTHWC.shape
            x = ttnn.reshape(x_NTHWC, [B, T, H, W, texp, sexp, sexp, out_channels])
            x = ttnn.permute(x, [0, 1, 4, 2, 5, 3, 6, 7])  # (B T texp H sexp W sexp C_out)

            x = ttnn.reshape(x, [B, T * texp, H * sexp, W * sexp, out_channels])
            if texp > 1:
                # Drop the first texp - 1 frames.
                x = ttnn.slice(x, [0, texp - 1, 0, 0, 0], [B, T * texp, H * sexp, W * sexp, out_channels])
            return x

        else:
            # Workaround for 1) issue #17535 for multi-device reshape,
            # and 2) slicing only the first shard.
            x_tensors = ttnn.get_device_tensors(x_NTHWC)
            for i in range(len(x_tensors)):
                B, T, H, W, C = x_tensors[i].shape
                x_tensors[i] = ttnn.reshape(x_tensors[i], [B, T, H, W, texp, sexp, sexp, out_channels])
                x_tensors[i] = ttnn.permute(x_tensors[i], [0, 1, 4, 2, 5, 3, 6, 7])  # (B T texp H sexp W sexp C_out)

                x_tensors[i] = ttnn.reshape(x_tensors[i], [B, T * texp, H * sexp, W * sexp, out_channels])

                if texp > 1 and i == 0:
                    x_tensors[i] = ttnn.slice(
                        x_tensors[i], [0, texp - 1, 0, 0, 0], [B, T * texp, H * sexp, W * sexp, out_channels]
                    )
                    x = ttnn.aggregate_as_tensor(x_tensors)
                    # TODO: This messes up the shape of the tensor...
            x = ttnn.aggregate_as_tensor(x_tensors)
            return x

    tt_output = depth_to_spacetime(tt_input)

    # Convert back to torch
    tt_output = ttnn.to_torch(tt_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=1))

    tt_output = tt_output.permute(0, 4, 1, 2, 3)

    assert tt_output.shape == torch_output.shape, f"{tt_output.shape=} != {torch_output.shape=}"
    # Compare outputs
    pcc, mse, mae = compute_metrics(torch_output, tt_output)
    logger.info(f"PCC = {pcc}, MSE = {mse}, MAE = {mae}")

    assert pcc > 0.99
