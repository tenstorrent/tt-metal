# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from ...tt.tt_upsample import BilinearUpsampleMatmulTTNN
from ...tt.tt_upsample import BilinearUpsampleTorch
from ...tt.common import PDL_L1_SMALL_SIZE


@pytest.mark.parametrize(
    "b, h, w, c, scale, input_channels_first, output_channels_first",
    [
        (1, 128, 64, 32, 2, False, False),  # single batch, NHWC -> NHWC, scale 2
        (1, 128, 64, 32, 2, True, True),  # single batch, NCHW -> NCHW, scale 2
        (1, 128, 64, 32, 2, False, True),  # single batch, NHWC -> NCHW, scale 2
        (1, 128, 64, 32, 2, True, False),  # single batch, NCHW -> NHWC, scale 2
        (1, 128, 64, 32, 4, False, False),  # single batch, NHWC -> NHWC, scale 4
        (1, 128, 64, 32, 4, False, True),  # single batch, NHWC -> NHWC, scale 4
        (1, 128, 64, 32, 4, True, False),  # single batch, NHWC -> NHWC, scale 4
        (1, 128, 64, 32, 4, True, True),  # single batch, NCHW -> NCHW, scale 4
        (3, 64, 32, 16, 2, False, False),  # multi-batch, NHWC -> NHWC, scale 2
        (3, 64, 32, 16, 4, True, True),  # multi-batch, NCHW -> NCHW, scale 4
        (8, 32, 32, 8, 3, False, False),  # larger batch, NHWC -> NHWC, scale 3
    ],
)
def test_bilinear_upsample_matmul_vs_torch(b, h, w, c, scale, input_channels_first, output_channels_first):
    torch.manual_seed(0)

    if input_channels_first:
        # Generate channels-first input: (B, C, H, W)
        img_torch = torch.rand(b, c, h, w, dtype=torch.float32)
        img_nchw = img_torch  # Already (B, C, H, W)
    else:
        # Generate channels-last input: (B, H, W, C)
        img_torch = torch.rand(b, h, w, c, dtype=torch.float32)
        img_nchw = img_torch.permute(0, 3, 1, 2)  # (B, C, H, W)

    # PyTorch matmul upsampling using the class
    upsampler = BilinearUpsampleTorch(
        h, w, scale=scale, input_channels_first=input_channels_first, output_channels_first=output_channels_first
    )
    out_torch = upsampler.forward(img_torch)

    # Torch bilinear upsampling (always works with channels-first)
    out_t = torch.nn.functional.interpolate(img_nchw, scale_factor=scale, mode="bilinear", align_corners=True)

    if output_channels_first:
        # Keep channels-first format
        out_t_formatted = out_t  # (B, C, H_out, W_out)
        expected_shape = (b, c, h * scale, w * scale)
    else:
        # Convert back to channels-last format
        out_t_formatted = out_t.permute(0, 2, 3, 1)  # (B, H_out, W_out, C)
        expected_shape = (b, h * scale, w * scale, c)

    # Validate numerical closeness
    torch.testing.assert_close(out_torch, out_t_formatted, rtol=1e-3, atol=1e-3)

    # Verify output shapes
    assert out_torch.shape == expected_shape, f"Expected shape {expected_shape}, got {out_torch.shape}"


@pytest.mark.parametrize(
    "b, h, w, c, scale, input_channels_first, output_channels_first, memory_config",
    # fmt: off
    [
        (1, 128, 64, 32, 2, False, False, ttnn.DRAM_MEMORY_CONFIG),# single batch, NHWC -> NHWC, scale 2
        (1, 128, 64, 32, 2, True, True, ttnn.DRAM_MEMORY_CONFIG),# single batch, NCHW -> NCHW, scale 2
        (1, 128, 64, 32, 2, False, True, ttnn.DRAM_MEMORY_CONFIG),# single batch, NHWC -> NCHW, scale 2
        (1, 128, 64, 32, 2, True, False, ttnn.DRAM_MEMORY_CONFIG),# single batch, NCHW -> NHWC, scale 2
        (1, 128, 64, 32, 4, False, False, ttnn.DRAM_MEMORY_CONFIG),# single batch, NHWC -> NHWC, scale 4
        (1, 128, 64, 32, 4, False, True, ttnn.DRAM_MEMORY_CONFIG),# single batch, NHWC -> NHWC, scale 4
        (1, 128, 64, 32, 4, True, False, ttnn.DRAM_MEMORY_CONFIG),# single batch, NHWC -> NHWC, scale 4
        (1, 128, 64, 32, 4, True, True, ttnn.DRAM_MEMORY_CONFIG),# single batch, NCHW -> NCHW, scale 4
        (3, 64, 32, 16, 2, False, False, ttnn.DRAM_MEMORY_CONFIG),# multi-batch, NHWC -> NHWC, scale 2
        (3, 64, 32, 16, 4, True, True, ttnn.DRAM_MEMORY_CONFIG),# multi-batch, NCHW -> NCHW, scale 4
        (8, 32, 32, 8, 3, False, False, ttnn.DRAM_MEMORY_CONFIG),# larger batch, NHWC -> NHWC, scale 3
        (1, 128, 256, 32, 4, False, False, ttnn.DRAM_MEMORY_CONFIG),
        (1, 128, 256, 32, 4, True, True, ttnn.DRAM_MEMORY_CONFIG),
        (1, 128, 256, 19, 4, False, False, ttnn.DRAM_MEMORY_CONFIG),
        (1, 128, 256, 32, 4, False, True, ttnn.DRAM_MEMORY_CONFIG),
        (1, 128, 256, 32, 4, False, True, ttnn.L1_MEMORY_CONFIG), # Matmul 512 x 128 x 1024 forced to HiFi4 and perf breaks
    ],
    # fmt: on
)
@pytest.mark.parametrize("output_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
def test_bilinear_upsample_ttnn_matmul_vs_ttnn_upsample(
    device, b, h, w, c, scale, input_channels_first, output_channels_first, memory_config, output_dtype
):
    torch.manual_seed(0)

    img_torch_nchw = torch.rand(b, c, h, w, dtype=torch.bfloat16)
    img_torch_nhwc = img_torch_nchw.permute(0, 2, 3, 1)

    # Create TTNN tensor (always expects NHWC format)
    ttnn_input_tensor = ttnn.from_torch(
        img_torch_nchw if input_channels_first else img_torch_nhwc,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=memory_config,
    )

    # Method 1: Custom matrix multiplication implementation
    upsampler = BilinearUpsampleMatmulTTNN(
        device,
        b,
        c,
        h,
        w,
        scale=scale,
        input_channels_first=input_channels_first,
        output_channels_first=output_channels_first,
        output_dtype=output_dtype,
        mm1_program_config=None,
        mm2_program_config=None,
    )
    output_matmul = upsampler(ttnn_input_tensor)
    output_matmul_torch = ttnn.to_torch(output_matmul)

    # Method 2: PyTorch reference with align_corners=True
    torch_result_nchw = torch.nn.functional.interpolate(
        img_torch_nchw, scale_factor=scale, mode="bilinear", align_corners=True
    )

    if output_channels_first:
        torch_result = torch_result_nchw  # Keep NCHW
        expected_shape = (b, c, h * scale, w * scale)
    else:
        torch_result = torch_result_nchw.permute(0, 2, 3, 1)  # Convert to NHWC
        expected_shape = (b, h * scale, w * scale, c)

    # Compare matmul implementation with PyTorch reference
    pcc_passed, pcc_message = assert_with_pcc(torch_result, output_matmul_torch, pcc=0.99)
    assert pcc_passed, f"Matmul implementation differs from PyTorch: {pcc_message}"

    # Verify output shapes
    assert output_matmul_torch.shape == expected_shape


@pytest.mark.parametrize(
    "b, h, w, c, scale, input_channels_first, output_channels_first, memory_config, output_dtype",
    # fmt: off
    [
        (1, 128, 256, 32, 4, False, True, ttnn.L1_MEMORY_CONFIG, ttnn.bfloat8_b),
    ],
    # fmt: on
)
@pytest.mark.parametrize("output_memory_config", [ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG])
@pytest.mark.parametrize("device_params", [{"l1_small_size": PDL_L1_SMALL_SIZE}], indirect=True)
def test_bilinear_upsample_l1_ttnn_matmul_vs_ttnn_upsample(
    device,
    b,
    h,
    w,
    c,
    scale,
    input_channels_first,
    output_channels_first,
    memory_config,
    output_dtype,
    output_memory_config,
):
    torch.manual_seed(0)

    img_torch_nchw = torch.rand(b, c, h, w, dtype=torch.bfloat16)
    img_torch_nhwc = img_torch_nchw.permute(0, 2, 3, 1)

    # Create TTNN tensor (always expects NHWC format)
    ttnn_input_tensor = ttnn.from_torch(
        img_torch_nchw if input_channels_first else img_torch_nhwc,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=memory_config,
    )

    # First config
    config1 = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(5, 4),
        in0_block_w=2,
        out_subblock_h=4,
        out_subblock_w=2,
        out_block_h=4,
        out_block_w=2,
        per_core_M=4,
        per_core_N=2,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
        gather_in0=False,
        num_global_cb_receivers=0,
        untilize_out=False,
    )

    # Second config
    config2 = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(5, 4),
        in0_block_w=2,
        out_subblock_h=4,
        out_subblock_w=2,
        out_block_h=16,
        out_block_w=2,
        per_core_M=16,
        per_core_N=2,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
        gather_in0=False,
        num_global_cb_receivers=0,
        untilize_out=False,
    )

    # Method 1: Custom matrix multiplication implementation
    upsampler = BilinearUpsampleMatmulTTNN(
        device,
        b,
        c,
        h,
        w,
        scale=scale,
        input_channels_first=input_channels_first,
        output_channels_first=output_channels_first,
        output_dtype=output_dtype,
        mm1_program_config=config1,
        mm2_program_config=config2,
        output_memory_config=output_memory_config,
    )
    output_matmul = upsampler(ttnn_input_tensor)
    output_matmul_torch = ttnn.to_torch(output_matmul)

    # Method 2: PyTorch reference with align_corners=True
    torch_result_nchw = torch.nn.functional.interpolate(
        img_torch_nchw, scale_factor=scale, mode="bilinear", align_corners=True
    )

    if output_channels_first:
        torch_result = torch_result_nchw  # Keep NCHW
        expected_shape = (b, c, h * scale, w * scale)
    else:
        torch_result = torch_result_nchw.permute(0, 2, 3, 1)  # Convert to NHWC
        expected_shape = (b, h * scale, w * scale, c)

    # Compare matmul implementation with PyTorch reference
    pcc_passed, pcc_message = assert_with_pcc(torch_result, output_matmul_torch, pcc=0.99)
    assert pcc_passed, f"Matmul implementation differs from PyTorch: {pcc_message}"

    # Verify output shapes
    assert output_matmul_torch.shape == expected_shape
