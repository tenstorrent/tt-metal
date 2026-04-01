# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Conv3D T_out_block > 1 correctness tests.

The fused tilize+matmul kernel reduced cb_vol2col_tiled from M_t*K_t to K_t tiles,
enabling T_out_block > 1. Larger T_out_block gives a fatter matmul (M_t > 1),
fewer spatial blocks, and reduced per-block + host dispatch overhead.

Measured ~20-55% speedup on WAN 2.2 VAE decoder bottleneck layers (BH Galaxy 6U).
"""

from loguru import logger
import torch
import pytest
import ttnn
from tests.ttnn.utils_for_testing import check_with_pcc
from tests.ttnn.unit_tests.operations.conv.test_conv3d import (
    prepare_input_tensor,
    reshape_output,
    ALIGNMENT,
)


def run_conv3d_t_block_test(
    device,
    input_shape,
    out_channels,
    kernel_size,
    padding,
    T_out_block,
    C_in_block,
    C_out_block,
    H_out_block,
    W_out_block,
):
    """Run conv3d with explicit blocking and verify against PyTorch."""
    torch.manual_seed(42)

    N, C, D, H, W = input_shape
    kD, kH, kW = kernel_size
    pD, pH, pW = padding
    D_out = D + 2 * pD - kD + 1
    H_out = H + 2 * pH - kH + 1
    W_out = W + 2 * pW - kW + 1

    # PyTorch reference
    input_tensor = torch.randn(N, C, D, H, W, dtype=torch.float32)
    conv3d_module = torch.nn.Conv3d(C, out_channels, kernel_size, padding=padding, bias=True)
    gt_output = conv3d_module(input_tensor)

    # TT input
    tt_input = prepare_input_tensor(input_tensor, C, device)

    # TT config
    grid_size = device.compute_with_storage_grid_size()
    config = ttnn.Conv3dConfig(
        weights_dtype=ttnn.bfloat16,
        output_layout=ttnn.ROW_MAJOR_LAYOUT,
        T_out_block=T_out_block,
        W_out_block=W_out_block,
        H_out_block=H_out_block,
        C_out_block=C_out_block,
        C_in_block=C_in_block,
        compute_with_storage_grid_size=grid_size,
    )

    kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )

    # TT weights
    w = conv3d_module.weight.data
    padded_C = ((C + ALIGNMENT - 1) // ALIGNMENT) * ALIGNMENT
    if padded_C > C:
        w = torch.nn.functional.pad(w, (0, 0, 0, 0, 0, 0, 0, padded_C - C))
    tt_weight = ttnn.from_torch(w, dtype=ttnn.bfloat16, pad_value=0)
    tt_weight = ttnn.experimental.prepare_conv3d_weights(
        weight_tensor=tt_weight, C_in_block=C_in_block, alignment=ALIGNMENT, device=device
    )
    tt_bias = ttnn.from_torch(
        conv3d_module.bias.data.reshape(1, -1),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        pad_value=0,
    )

    # Run
    tt_output = ttnn.experimental.conv3d(
        input_tensor=tt_input,
        weight_tensor=tt_weight,
        device=device,
        bias_tensor=tt_bias,
        dtype=ttnn.bfloat16,
        output_channels=out_channels,
        kernel_size=kernel_size,
        stride=(1, 1, 1),
        groups=1,
        padding=padding,
        padding_mode="zeros",
        config=config,
        compute_kernel_config=kernel_config,
    )

    tt_output = reshape_output(tt_output, N, D_out, H_out, W_out, out_channels, device)
    assert tt_output.shape == gt_output.shape

    pcc_passed, pcc_message = check_with_pcc(gt_output, tt_output, pcc=0.999)
    logger.info(f"Conv3d T_block={T_out_block}: {pcc_message}")
    assert pcc_passed, pcc_message


# WAN 2.2 VAE decoder shapes (BH Galaxy 6U 4x32 720p proxy, 1x1 mesh)
# Best T_out_block values from joint spatial+temporal sweep.
@pytest.mark.parametrize(
    "input_shape, out_channels, kernel_size, padding, T_out_block, C_in_block, C_out_block, H_out_block, W_out_block",
    [
        # up3_res: 96->96 k333, T=6 gives -19% vs T=1 (compute-bound, 6x repeated in decoder)
        ((1, 96, 83, 186, 42), 96, (3, 3, 3), (1, 1, 1), 6, 96, 96, 8, 4),
        # up2_res: 192->192 k333, T=3 gives -13% vs T=1 (compute-bound, 6x repeated)
        ((1, 192, 83, 94, 22), 192, (3, 3, 3), (1, 1, 1), 3, 96, 96, 8, 4),
        # up1_res: 384->384 k333, T=1 optimal (multi C_in block, L1 pressure at higher T)
        ((1, 384, 43, 48, 12), 384, (3, 3, 3), (1, 1, 1), 1, 96, 128, 16, 2),
        # conv_out: 96->3 k333, T=9 gives -55% vs T=1 (biggest single-layer win)
        ((1, 96, 83, 186, 42), 3, (3, 3, 3), (1, 1, 1), 9, 96, 32, 8, 4),
        # up1_res0: 192->384 k333, T=3 gives -4%
        ((1, 192, 43, 48, 12), 384, (3, 3, 3), (1, 1, 1), 3, 96, 128, 8, 4),
        # lat_res: 384->384 k333, T=1 optimal (small shape, overhead dominates at higher T)
        ((1, 384, 23, 25, 7), 384, (3, 3, 3), (1, 1, 1), 1, 96, 96, 32, 1),
        # up1_tconv: 384->768 k311, T=3 gives -6% (temporal-only kernel)
        ((1, 384, 42, 46, 8), 768, (3, 1, 1), (1, 0, 0), 3, 192, 384, 16, 2),
        # up0_tconv: 384->768 k311, T=3 gives -8%
        ((1, 384, 22, 23, 4), 768, (3, 1, 1), (1, 0, 0), 3, 128, 256, 8, 4),
        # conv_in: 32->384 k333, T=1 optimal (already fast)
        ((1, 32, 23, 25, 7), 384, (3, 3, 3), (1, 1, 1), 1, 32, 128, 16, 2),
        # up0_spatial: 384->192 k133 (kT=1, no T blocking possible)
        ((1, 384, 41, 48, 12), 192, (1, 3, 3), (0, 1, 1), 1, 96, 96, 16, 4),
        # up1_spatial: 384->192 k133
        ((1, 384, 81, 94, 22), 192, (1, 3, 3), (0, 1, 1), 1, 192, 96, 32, 4),
        # up2_spatial: 192->96 k133
        ((1, 192, 81, 186, 42), 96, (1, 3, 3), (0, 1, 1), 1, 192, 96, 4, 8),
    ],
    ids=[
        "wan_up3_res_T6",
        "wan_up2_res_T3",
        "wan_up1_res_T1",
        "wan_conv_out_T9",
        "wan_up1_res0_T3",
        "wan_lat_res_T1",
        "wan_up1_tconv_T3",
        "wan_up0_tconv_T3",
        "wan_conv_in_T1",
        "wan_up0_spatial_T1",
        "wan_up1_spatial_T1",
        "wan_up2_spatial_T1",
    ],
)
def test_conv3d_t_block_wan_vae(
    device,
    input_shape,
    out_channels,
    kernel_size,
    padding,
    T_out_block,
    C_in_block,
    C_out_block,
    H_out_block,
    W_out_block,
):
    run_conv3d_t_block_test(
        device,
        input_shape,
        out_channels,
        kernel_size,
        padding,
        T_out_block,
        C_in_block,
        C_out_block,
        H_out_block,
        W_out_block,
    )
