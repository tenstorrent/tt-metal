# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import csv
import os

import pytest
import torch
from loguru import logger

import ttnn

from ....models.vae.vae_wan2_1 import WanCausalConv3d
from ....parallel.config import ParallelFactor, VaeHWParallelConfig
from ....parallel.manager import CCLManager
from ....utils.conv3d import conv_pad_height, conv_pad_in_channels, conv_unpad_height
from ....utils.tensor import bf16_tensor_2dshard


def _print_conv3d_error_analysis(ref, tt, label="", shape_str="", math_fidelity="", csv_path="conv3d_errors.csv"):
    """Print detailed error analysis for conv3d outputs (BCTHW format)."""
    diff = ref.float() - tt.float()
    abs_err = diff.abs()
    pcc_val = torch.corrcoef(torch.stack([ref.float().flatten(), tt.float().flatten()]))[0, 1].item()
    rmse = diff.pow(2).mean().sqrt().item()
    max_err = abs_err.max().item()
    ref_std = ref.float().std().item()
    print(
        f"  {label}: PCC={pcc_val:.6f}  RMSE={rmse:.4e}  maxerr={max_err:.4e}  ref_std={ref_std:.4e}  shape={list(ref.shape)}"
    )

    # Top-20 worst elements
    flat_abs = abs_err.flatten()
    top_vals, top_flat_idx = flat_abs.topk(min(20, flat_abs.numel()))
    if top_vals[0] > 0.01:
        print(f"         worst elements (>{0.01:.2f}):")
        for k in range(len(top_vals)):
            if top_vals[k] < 0.01:
                break
            coords = torch.unravel_index(top_flat_idx[k], abs_err.shape)
            idx = tuple(c.item() for c in coords)
            rv = ref[idx].float().item()
            tv = tt[idx].float().item()
            print(f"           {list(idx)} ref={rv:+.4f} tt={tv:+.4f} err={top_vals[k]:.4f}")

    # Save worst 20 values to CSV
    if csv_path and shape_str:
        file_exists = os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as csvfile:
            fieldnames = ["shape", "math_fidelity", "abs_error", "ref_value", "tt_value", "coordinates"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            for k in range(min(20, len(top_vals))):
                coords = torch.unravel_index(top_flat_idx[k], abs_err.shape)
                idx = tuple(c.item() for c in coords)
                rv = ref[idx].float().item()
                tv = tt[idx].float().item()

                writer.writerow(
                    {
                        "shape": shape_str,
                        "math_fidelity": math_fidelity,
                        "abs_error": top_vals[k].item(),
                        "ref_value": rv,
                        "tt_value": tv,
                        "coordinates": str(list(idx)),
                    }
                )


@pytest.mark.parametrize(
    ("B, C_in, C_out, T, H, W, kernel_size, stride, padding"),
    [
        (1, 16, 384, 1, 90, 160, 3, 1, 1),  # decoder.conv_in
        (1, 384, 384, 1, 90, 160, 3, 1, 1),  # decoder.mid_block.resnets.0.conv1
        (1, 192, 384, 2, 180, 320, 3, 1, 1),  # decoder.up_blocks.1.resnets.0.conv1
        (1, 384, 384, 2, 180, 320, 3, 1, 1),  # decoder.up_blocks.1.resnets.0.conv2
        (1, 192, 192, 4, 360, 640, 3, 1, 1),  # decoder.up_blocks.2.resnets.0.conv1
        (1, 96, 96, 4, 720, 1280, 3, 1, 1),  # decoder.up_blocks.3.resnets.0.conv1
        (1, 96, 3, 4, 720, 1280, 3, 1, 1),  # decoder.conv_out
        (1, 384, 768, 1, 90, 160, (3, 1, 1), 1, (1, 0, 0)),  # decoder.up_blocks.0.upsamplers.0.time_conv
        (1, 384, 768, 2, 180, 320, (3, 1, 1), 1, (1, 0, 0)),  # decoder.up_blocks.0.upsamplers.0.time_conv
    ],
    ids=[
        "conv_0",
        "conv_1",
        "conv_2",
        "conv_3",
        "conv_4",
        "conv_5",
        "conv_6",
        "conv_7",
        "conv_8",
    ],
)
@pytest.mark.parametrize("cache_len", [None, 1, 2], ids=["cache_none", "cache_1", "cache_2"])
@pytest.mark.parametrize("mean, std", [(0, 1)])
@pytest.mark.parametrize(
    "mesh_device, h_axis, w_axis",
    [
        ((1, 1), 0, 1),
    ],
    ids=[
        "1x1_h0_w1",
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize(
    "math_fidelity",
    [ttnn.MathFidelity.HiFi4, ttnn.MathFidelity.HiFi3, ttnn.MathFidelity.HiFi2],
    ids=["HiFi4", "HiFi3", "HiFi2"],
)
def test_wan_conv3d(
    mesh_device,
    B,
    C_in,
    C_out,
    T,
    H,
    W,
    kernel_size,
    stride,
    padding,
    cache_len,
    mean,
    std,
    h_axis,
    w_axis,
    math_fidelity,
):
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d as TorchWanCausalConv3d

    torch_dtype = torch.float32
    torch_model = TorchWanCausalConv3d(
        in_channels=C_in, out_channels=C_out, kernel_size=kernel_size, stride=stride, padding=padding
    )
    torch_model.eval()

    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)
    parallel_config = VaeHWParallelConfig(
        height_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[h_axis], mesh_axis=h_axis),
        width_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[w_axis], mesh_axis=w_axis),
    )
    tt_model = WanCausalConv3d(
        in_channels=C_in,
        out_channels=C_out,
        kernel_size=kernel_size,
        mesh_device=mesh_device,
        stride=stride,
        padding=padding,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
    )
    # Force HiFi4 so the test explicitly exercises the HiFi4 path
    # (fails with HiFi4, works with HiFi3/HiFi2)
    tt_model.compute_kernel_config = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=math_fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    tt_model.load_state_dict(torch_model.state_dict())

    torch_input_tensor = torch.randn(B, C_in, T, H, W, dtype=torch_dtype) * std + mean
    tt_input_tensor = torch_input_tensor.permute(0, 2, 3, 4, 1)
    tt_input_tensor = conv_pad_in_channels(tt_input_tensor)
    tt_input_tensor, logical_h = conv_pad_height(tt_input_tensor, parallel_config.height_parallel.factor)
    if logical_h != tt_input_tensor.shape[2]:
        logger.info(f"padding from {logical_h} to {tt_input_tensor.shape[2]}")
    tt_input_tensor = bf16_tensor_2dshard(
        tt_input_tensor, mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, shard_mapping={h_axis: 2, w_axis: 3}
    )
    logger.info(f"torch_input_tensor.shape: {torch_input_tensor.shape}")
    logger.info(f"tt_input_tensor.shape: {tt_input_tensor.shape}")

    if cache_len is not None:
        torch_cache_tensor = torch.randn(B, C_in, cache_len, H, W, dtype=torch_dtype) * std + mean
        tt_cache_tensor = torch_cache_tensor.permute(0, 2, 3, 4, 1)
        tt_cache_tensor = conv_pad_in_channels(tt_cache_tensor)
        tt_cache_tensor, logical_h = conv_pad_height(tt_cache_tensor, parallel_config.height_parallel.factor)
        tt_cache_tensor = bf16_tensor_2dshard(
            tt_cache_tensor, mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, shard_mapping={h_axis: 2, w_axis: 3}
        )
    else:
        torch_cache_tensor = tt_cache_tensor = None

    with torch.no_grad():
        torch_output = torch_model(torch_input_tensor, cache_x=torch_cache_tensor)
    tt_output = tt_model(tt_input_tensor, cache_x_BTHWC=tt_cache_tensor, logical_h=logical_h)

    concat_dims = [None, None]
    concat_dims[h_axis] = 2
    concat_dims[w_axis] = 3
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=concat_dims),
    )
    if logical_h != tt_output_torch.shape[2]:
        logger.info(f"Checking that output padded portion is zeros")
        padding = tt_output_torch[:, :, logical_h:, :, :]
        assert torch.all(padding == 0.0), f"Padding must be zero, got {padding}"

    tt_output_torch = conv_unpad_height(tt_output_torch, logical_h)
    tt_output_torch = tt_output_torch.permute(0, 4, 1, 2, 3)

    if tt_output_torch.shape != torch_output.shape:
        logger.warning(
            f"tt_output_torch.shape != torch_output.shape, got {tt_output_torch.shape} != {torch_output.shape}"
        )
        tt_output_torch = tt_output_torch[:, :C_out]
        logger.warning(f"Trimmed tt_output_torch to {tt_output_torch.shape}")

    shape_str = f"B{B}_Cin{C_in}_Cout{C_out}_T{T}_H{H}_W{W}"
    math_fidelity_str = str(math_fidelity).split(".")[-1] if hasattr(math_fidelity, "__str__") else str(math_fidelity)

    _print_conv3d_error_analysis(
        torch_output,
        tt_output_torch,
        label=f"conv3d C_in={C_in} C_out={C_out} T={T} H={H} W={W} k={kernel_size}",
        shape_str=shape_str,
        math_fidelity=math_fidelity_str,
        csv_path="conv3d_errors.csv",
    )
