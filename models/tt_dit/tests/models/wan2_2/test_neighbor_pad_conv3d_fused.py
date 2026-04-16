# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Production-shape tests for fused NeighborPad + Conv3d correctness.

Validates the fused NP+Conv3d path against PyTorch reference on all
VAE decoder shapes that trigger the halo path.
"""

import pytest
import torch
from loguru import logger

import ttnn

from ....models.vae.vae_wan2_1 import WanCausalConv3d
from ....parallel.config import ParallelFactor, VaeHWParallelConfig
from ....parallel.manager import CCLManager
from ....utils.check import assert_quality
from ....utils.conv3d import ConvDims, conv_pad_height, conv_pad_in_channels
from ....utils.tensor import typed_tensor_2dshard

TORCH_REF_PCC = 0.99999
MAX_RMSE = 0.010


def _make_model_and_manager(mesh_device, B, C_in, C_out, kernel_size, padding, h_axis, w_axis, num_links, dtype):
    """Create ONE CCLManager + WanCausalConv3d pair (matches production usage)."""
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d as TorchWanCausalConv3d

    h_factor = tuple(mesh_device.shape)[h_axis]
    w_factor = tuple(mesh_device.shape)[w_axis]
    stride = 1

    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear, num_links=num_links)
    parallel_config = VaeHWParallelConfig(
        height_parallel=ParallelFactor(factor=h_factor, mesh_axis=h_axis),
        width_parallel=ParallelFactor(factor=w_factor, mesh_axis=w_axis),
    )

    torch.manual_seed(42)
    torch_model = TorchWanCausalConv3d(
        in_channels=C_in, out_channels=C_out, kernel_size=kernel_size, stride=stride, padding=padding
    )
    torch_model.eval()

    return torch_model, ccl_manager, parallel_config, h_factor, w_factor


def _run_one_seed(
    mesh_device,
    torch_model,
    ccl_manager,
    parallel_config,
    B,
    C_in,
    C_out,
    T,
    H,
    W,
    kernel_size,
    padding,
    h_axis,
    w_axis,
    h_factor,
    w_factor,
    dtype,
    seed,
    force_pipelining=False,
):
    """Run one inference call with a specific seed; return (torch_output, tt_output_torch)."""
    tt_input_dtype = ttnn.bfloat16 if dtype == ttnn.DataType.BFLOAT16 else ttnn.float32
    torch_dtype = torch.float32
    stride = 1

    H_out_per_device = H // h_factor
    W_out_per_device = W // w_factor

    tt_model = WanCausalConv3d(
        in_channels=C_in,
        out_channels=C_out,
        kernel_size=kernel_size,
        mesh_device=mesh_device,
        stride=stride,
        padding=padding,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        dtype=dtype,
        conv_dims=ConvDims(T=T, H=H_out_per_device, W=W_out_per_device),
    )
    tt_model.load_torch_state_dict(torch_model.state_dict())

    assert tt_model.conv_config.use_h_halo_buffer, (
        f"Fused path not enabled for this shape (T_out_block={tt_model.conv_config.T_out_block}). "
        "Adjust T so that T_out >= T_out_block > 1."
    )

    if force_pipelining and tt_model.conv_config.T_out_block > 0:
        tt_model.conv_config.input_progress_t_batch_size = tt_model.conv_config.T_out_block

    logger.info(
        f"seed={seed} use_h_halo_buffer={tt_model.conv_config.use_h_halo_buffer} "
        f"input_progress_t_batch_size={tt_model.conv_config.input_progress_t_batch_size} "
        f"T_out_block={tt_model.conv_config.T_out_block} force_pipelining={force_pipelining}"
    )

    torch.manual_seed(seed + 999)
    torch_input = torch.randn(B, C_in, T, H, W, dtype=torch_dtype)

    tt_input = torch_input.permute(0, 2, 3, 4, 1)
    tt_input = conv_pad_in_channels(tt_input)
    tt_input, logical_h = conv_pad_height(tt_input, parallel_config.height_parallel.factor)
    tt_input = typed_tensor_2dshard(
        tt_input,
        mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        shard_mapping={h_axis: 2, w_axis: 3},
        dtype=tt_input_dtype,
    )

    with torch.no_grad():
        torch_output = torch_model(torch_input)

    tt_output = tt_model(tt_input, logical_h=logical_h)

    concat_dims = [None, None]
    concat_dims[h_axis] = 2
    concat_dims[w_axis] = 3
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=concat_dims),
    )
    if logical_h != tt_output_torch.shape[2]:
        tt_output_torch = tt_output_torch[:, :logical_h, :, :, :]
    tt_output_torch = tt_output_torch.permute(0, 4, 1, 2, 3)
    if tt_output_torch.shape[1] != C_out:
        tt_output_torch = tt_output_torch[:, :C_out]

    ttnn.deallocate(tt_input)
    ttnn.deallocate(tt_output)
    return torch_output, tt_output_torch


def _diagnose_pcc_failure(torch_ref, tt_out, h_factor, w_factor, pad_h=1, pad_w=1, top_k=20):
    """Print top-K error positions with boundary classification on PCC failure."""
    diff = (torch_ref - tt_out).abs()
    H, W = torch_ref.shape[3], torch_ref.shape[4]
    H_dev, W_dev = H // h_factor, W // w_factor

    flat_idx = diff.flatten().topk(min(top_k, diff.numel())).indices
    for rank, idx in enumerate(flat_idx):
        coords = []
        tmp = idx.item()
        for dim_size in reversed(diff.shape):
            coords.append(tmp % dim_size)
            tmp //= dim_size
        bi, c, t, h, w = list(reversed(coords))

        local_h = h % H_dev
        local_w = w % W_dev
        dev_h, dev_w = h // H_dev, w // W_dev

        regions = []
        if dev_h > 0 and local_h < pad_h:
            regions.append("H-top")
        if dev_h < h_factor - 1 and local_h >= H_dev - pad_h:
            regions.append("H-bot")
        if dev_w > 0 and local_w < pad_w:
            regions.append("W-left")
        if dev_w < w_factor - 1 and local_w >= W_dev - pad_w:
            regions.append("W-right")
        region = "+".join(regions) if regions else "interior"

        logger.error(
            f"  #{rank:2d} B={bi} C={c} T={t} H={h} W={w} "
            f"(dev[{dev_h},{dev_w}] lh={local_h} lw={local_w}) "
            f"diff={diff[bi, c, t, h, w]:.6g}  "
            f"ref={torch_ref[bi, c, t, h, w]:.6g}  tt={tt_out[bi, c, t, h, w]:.6g}  "
            f"[{region}]"
        )


@pytest.mark.parametrize(
    "B, C_in, C_out, T, H, W, kernel_size, padding, mesh_device, h_axis, w_axis, num_links",
    [
        (1, 96, 96, 3, 480, 832, 3, 1, (2, 4), 0, 1, 2),
        (1, 96, 96, 83, 736, 1280, 3, 1, (4, 8), 0, 1, 2),
        (1, 192, 192, 83, 368, 640, 3, 1, (4, 8), 0, 1, 2),
        (1, 96, 96, 13, 736, 1280, 3, 1, (4, 8), 0, 1, 2),
    ],
    ids=["up3_res_bh_lb_2x4", "up3_res_bh_glx_4x8", "up2_res_bh_glx_4x8", "up3_res_fast_4x8"],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("dtype", [ttnn.DataType.BFLOAT16], ids=["bf16"])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.timeout(900)
def test_fused_production_shapes(
    mesh_device, B, C_in, C_out, T, H, W, kernel_size, padding, h_axis, w_axis, num_links, dtype
):
    """Fused NP+Conv3d vs PyTorch reference on production VAE decoder shapes."""
    torch_model, ccl_manager, parallel_config, h_factor, w_factor = _make_model_and_manager(
        mesh_device, B, C_in, C_out, kernel_size, padding, h_axis, w_axis, num_links, dtype
    )

    torch_ref, tt_out = _run_one_seed(
        mesh_device,
        torch_model,
        ccl_manager,
        parallel_config,
        B,
        C_in,
        C_out,
        T,
        H,
        W,
        kernel_size,
        padding,
        h_axis,
        w_axis,
        h_factor,
        w_factor,
        dtype,
        seed=0,
        force_pipelining=True,
    )

    try:
        assert_quality(torch_ref, tt_out, pcc=TORCH_REF_PCC, relative_rmse=MAX_RMSE)
    except Exception:
        logger.error(f"PCC failure — top-20 error positions (H_dev={H // h_factor}, W_dev={W // w_factor}):")
        _diagnose_pcc_failure(torch_ref, tt_out, h_factor, w_factor, pad_h=padding, pad_w=padding)
        raise

    logger.info("PASSED")
