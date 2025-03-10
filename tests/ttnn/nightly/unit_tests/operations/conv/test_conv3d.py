# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import torch
import pytest
import ttnn
import torch.nn as nn
from tests.ttnn.utils_for_testing import check_with_pcc

from tests.ttnn.unit_tests.operations.conv.test_conv3d import (
    setup_conv3d_test,
    create_conv3d_config,
    prepare_weights,
    reshape_output,
    run_conv3d_test,
)


@pytest.mark.parametrize("B", [1])
@pytest.mark.parametrize("C_in", [12, 32, 64])
@pytest.mark.parametrize("C_out", [32, 64])
@pytest.mark.parametrize("T", [3, 5])
@pytest.mark.parametrize("H", [4, 6])
@pytest.mark.parametrize("W", [5, 7])
@pytest.mark.parametrize("kernel_size", [(3, 3, 3), (1, 1, 1)], ids=["kernel_333", "kernel_111"])
@pytest.mark.parametrize("stride", [(1, 1, 1)], ids=["stride_111"])
@pytest.mark.parametrize("padding", [(0, 0, 0), (0, 1, 1)], ids=["padding_000", "padding_011"])
@pytest.mark.parametrize("padding_mode", ["zeros", "replicate"])
def test_conv3d_sweep_shapes(device, B, C_in, C_out, T, H, W, kernel_size, stride, padding, padding_mode):
    if padding == (0, 0, 0) and padding_mode == "replicate":
        pytest.skip("Skipping padding (0, 0, 0) and padding_mode replicate because it's duplicate")
    input_shape = (B, C_in, T, H, W)
    out_channels = C_out
    kernel_size = kernel_size
    stride = stride
    padding = padding
    padding_mode = padding_mode
    grid_size = device.compute_with_storage_grid_size()
    run_conv3d_test(device, input_shape, out_channels, kernel_size, stride, padding, padding_mode, grid_size=grid_size)


@pytest.mark.parametrize(
    "input_shape, out_channels, kernel_size, stride, padding, padding_mode",
    [
        [(1, 128, 16, 16, 16), 128, (3, 3, 3), (1, 1, 1), (0, 1, 1), "replicate"],
    ],
)
def test_conv3d_sweep_blocks(
    device, input_shape, out_channels, kernel_size, stride, padding, padding_mode, use_program_cache
):
    """
    For a specific shape, sweep through different block sizes.
    Constrain the sweep such that the num_patches in a block doesn't exceed 64
    """
    import math

    grid_size = device.compute_with_storage_grid_size()
    tt_input, conv3d_module, gt_output, kernel_config, output_dims = setup_conv3d_test(
        input_shape, out_channels, kernel_size, stride, padding, padding_mode, device
    )
    N, D_out, H_out, W_out = output_dims
    C = input_shape[1]
    C_in_blocks = filter(lambda x: C % x == 0, range(32, C + 1, 32))
    C_out_blocks = filter(lambda x: out_channels % x == 0, range(32, out_channels + 1, 32))
    T_out_blocks = [2**i for i in range(int(math.log2(D_out)))]
    H_out_blocks = [2**i for i in range(int(math.log2(H_out)))]
    W_out_blocks = [2**i for i in range(int(math.log2(W_out)))]

    MAX_NUM_PATCHES_IN_BLOCK = 64
    prev_C_in_block = None

    import itertools

    for C_in_block, C_out_block, T_out_block, H_out_block, W_out_block in itertools.product(
        C_in_blocks, C_out_blocks, T_out_blocks, H_out_blocks, W_out_blocks
    ):
        num_patches_in_block = T_out_block * H_out_block * W_out_block
        if num_patches_in_block > MAX_NUM_PATCHES_IN_BLOCK:
            continue
        if (C_in_block == 128 or C_out_block == 128) and num_patches_in_block > 32:
            continue

        logger.info(f"Testing {C_in_block}, {C_out_block}, {T_out_block}, {H_out_block}, {W_out_block}")
        # Prepare weights with specified C_in_block
        if prev_C_in_block != C_in_block:
            # Only prepare if changing C_in_block
            tt_weight, tt_bias = prepare_weights(conv3d_module, C, out_channels, device, C_in_block=C_in_block)
            prev_C_in_block = C_in_block

        config = create_conv3d_config(
            out_channels,
            kernel_size,
            stride,
            padding,
            padding_mode,
            T_out_block=T_out_block,
            H_out_block=H_out_block,
            W_out_block=W_out_block,
            C_out_block=C_out_block,
            C_in_block=C_in_block,
            compute_with_storage_grid_size=grid_size,
        )

        tt_output = ttnn.experimental.conv3d(
            input_tensor=tt_input,
            weight_tensor=tt_weight,
            bias_tensor=tt_bias,
            config=config,
            compute_kernel_config=kernel_config,
        )
        # Reshape output and verify results
        tt_output = reshape_output(tt_output, N, D_out, H_out, W_out, out_channels, device)

        assert tt_output.shape == gt_output.shape
        pcc_passed, pcc_message = check_with_pcc(gt_output, tt_output, pcc=0.99)
        assert pcc_passed, (
            f"{pcc_message} on "
            f"C_out_block={C_out_block}, T_out_block={T_out_block}, "
            f"W_out_block={W_out_block}, H_out_block={H_out_block}, "
            f"C_in_block={C_in_block}"
        )


@pytest.mark.parametrize(
    "input_shape, out_channels, kernel_size, stride, padding, padding_mode, blocking",
    [
        [
            (1, 768, 4, 60, 106),
            768,
            (3, 3, 3),
            (1, 1, 1),
            (0, 1, 1),
            "replicate",
            (128, 96, 1, 2, 16),
        ],  # Best blocking found so far
        [
            (1, 512, 11, 120, 212),
            512,
            (3, 3, 3),
            (1, 1, 1),
            (0, 1, 1),
            "replicate",
            (128, 128, 1, 8, 4),
        ],  # Best blocking found so far
        [
            (1, 256, 21, 240, 424),
            256,
            (3, 3, 3),
            (1, 1, 1),
            (0, 1, 1),
            "replicate",
            (128, 128, 4, 4, 2),
        ],  # Best blocking found so far
        [
            (1, 128, 21, 480, 848),
            128,
            (3, 3, 3),
            (1, 1, 1),
            (0, 1, 1),
            "replicate",
            (128, 128, 1, 2, 16),
        ],  # Best blocking found so far
    ],
    ids=["variant1", "variant2", "variant3", "variant4"],
)
def test_conv3d_mochi_shapes(
    device,
    input_shape,
    out_channels,
    kernel_size,
    stride,
    padding,
    padding_mode,
    blocking,
    use_program_cache,
    is_ci_env,
):
    if out_channels == 128 or out_channels == 256:
        pytest.skip("Skipping test for 128 out channels on CI due to host OOM")

    C_in_block, C_out_block, T_out_block, H_out_block, W_out_block = blocking
    tt_input, conv3d_module, gt_output, kernel_config, output_dims = setup_conv3d_test(
        input_shape, out_channels, kernel_size, stride, padding, padding_mode, device
    )
    N, D_out, H_out, W_out = output_dims
    C = input_shape[1]

    # Prepare weights with specified C_in_block
    tt_weight, tt_bias = prepare_weights(conv3d_module, C, out_channels, device, C_in_block=C_in_block)

    config = create_conv3d_config(
        out_channels,
        kernel_size,
        stride,
        padding,
        padding_mode,
        T_out_block=T_out_block,
        H_out_block=H_out_block,
        W_out_block=W_out_block,
        C_out_block=C_out_block,
        C_in_block=C_in_block,
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
    )

    tt_output = ttnn.experimental.conv3d(
        input_tensor=tt_input,
        weight_tensor=tt_weight,
        bias_tensor=tt_bias,
        config=config,
        compute_kernel_config=kernel_config,
    )
    # Reshape output and verify results
    tt_output = reshape_output(tt_output, N, D_out, H_out, W_out, out_channels, device)

    assert tt_output.shape == gt_output.shape
    pcc_passed, pcc_message = check_with_pcc(gt_output, tt_output, pcc=0.999)
    logger.info(f"{pcc_message}")
    assert pcc_passed, pcc_message
