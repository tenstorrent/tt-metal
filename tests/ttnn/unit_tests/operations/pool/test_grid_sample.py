# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F
from loguru import logger

import ttnn
from ttnn.operations.pool import golden_grid_sample
from tests.ttnn.utils_for_testing import assert_with_pcc

pytestmark = pytest.mark.use_module_device


@pytest.mark.parametrize(
    "input_shape, grid_shape",
    [
        ((1, 32, 8, 8), (1, 6, 6, 2)),
        ((2, 64, 16, 16), (2, 12, 12, 2)),
        ((1, 96, 24, 24), (1, 20, 20, 2)),
        ((4, 128, 32, 32), (4, 28, 28, 2)),
        ((8, 160, 48, 48), (8, 40, 40, 2)),
        ((2, 192, 64, 64), (2, 56, 56, 2)),
        ((1, 96, 8, 32), (1, 6, 28, 2)),
        ((2, 160, 32, 8), (2, 28, 6, 2)),
    ],
)
@pytest.mark.parametrize(
    "mode",
    [
        "bilinear",
    ],
)
@pytest.mark.parametrize(
    "align_corners",
    [
        False,
        True,
    ],
)
@pytest.mark.parametrize("grid_dtype", [ttnn.bfloat16, ttnn.float32])
def test_grid_sample_random_grid(device, input_shape, mode, align_corners, grid_shape, grid_dtype):
    """Test grid_sample with completely random grid coordinates"""

    torch.manual_seed(0)

    batch_size, channels, height, width = input_shape
    _, grid_h, grid_w, _ = grid_shape

    torch_input_nhwc = torch.randn((batch_size, height, width, channels), dtype=torch.bfloat16)

    # Create a random grid with coordinates in range [-1, 1]
    torch_grid_f32 = torch.rand(grid_shape, dtype=torch.float32) * 2.0 - 1.0

    golden_grid_dtype = torch.float32 if grid_dtype == ttnn.float32 else torch.bfloat16
    torch_output_nhwc = golden_grid_sample(
        input_tensor=torch_input_nhwc,
        grid=torch_grid_f32.to(golden_grid_dtype),
        mode=mode,
        padding_mode="zeros",
        align_corners=align_corners,
    )

    ttnn_input = ttnn.from_torch(torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    ttnn_grid = ttnn.from_torch(torch_grid_f32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=grid_dtype)

    ttnn_output = ttnn.grid_sample(ttnn_input, ttnn_grid, mode=mode, align_corners=align_corners)
    ttnn_output_torch = ttnn.to_torch(ttnn_output)
    # Test allclose with grid type specific tolerances
    if grid_dtype == ttnn.float32:
        atol, rtol = 0.02, 1e-2
    else:  # bfloat16
        atol, rtol = 1.0, 1e-1

    allclose_passed = torch.allclose(torch_output_nhwc, ttnn_output_torch, atol=atol, rtol=rtol)

    assert allclose_passed, f"Test failed allclose comparison (atol={atol}, rtol={rtol})"


@pytest.mark.parametrize(
    "input_shape, grid_shape",
    [
        ((1, 32, 16, 16), (1, 12, 12, 2)),
        ((2, 64, 20, 20), (2, 16, 16, 2)),
        ((1, 96, 24, 24), (1, 18, 18, 2)),
        ((4, 128, 28, 28), (4, 22, 22, 2)),
        ((2, 160, 32, 32), (2, 24, 24, 2)),
        ((1, 192, 36, 36), (1, 28, 28, 2)),
        ((3, 224, 40, 40), (3, 32, 32, 2)),
    ],
)
@pytest.mark.parametrize(
    "mode",
    [
        "bilinear",
    ],
)
@pytest.mark.parametrize(
    "align_corners",
    [
        False,
        True,
    ],
)
@pytest.mark.parametrize("grid_dtype", [ttnn.bfloat16, ttnn.float32])
def test_grid_sample_near_uniform_grid(device, input_shape, mode, align_corners, grid_shape, grid_dtype):
    torch.manual_seed(0)

    batch_size, channels, height, width = input_shape
    _, grid_h, grid_w, _ = grid_shape

    torch_input_nhwc = torch.randn((batch_size, height, width, channels), dtype=torch.bfloat16)

    # Generates a uniform grid using torch affine grid
    theta = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
    theta_batched = theta.unsqueeze(0).expand(batch_size, -1, -1)
    shape = (batch_size, 1, grid_h, grid_w)
    torch_grid = F.affine_grid(theta_batched, shape, align_corners=align_corners)

    # Add small noise to the grid
    torch_grid += torch.randn(grid_shape) * 0.05

    golden_grid_dtype = torch.float32 if grid_dtype == ttnn.float32 else torch.bfloat16
    torch_output_nhwc = golden_grid_sample(
        input_tensor=torch_input_nhwc,
        grid=torch_grid.to(golden_grid_dtype),
        mode=mode,
        padding_mode="zeros",
        align_corners=align_corners,
    )

    ttnn_input = ttnn.from_torch(torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    ttnn_grid = ttnn.from_torch(torch_grid, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=grid_dtype)

    ttnn_output = ttnn.grid_sample(ttnn_input, ttnn_grid, mode=mode, align_corners=align_corners)
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    pcc_passed, pcc_message = assert_with_pcc(torch_output_nhwc, ttnn_output_torch, pcc=0.99)

    # Test allclose with grid type specific tolerances
    if grid_dtype == ttnn.float32:
        atol, rtol = 0.02, 1e-2
    else:  # bfloat16
        atol, rtol = 1.0, 1e-1

    allclose_passed = torch.allclose(torch_output_nhwc, ttnn_output_torch, atol=atol, rtol=rtol)

    assert pcc_passed, f"Test failed with PCC below threshold"
    assert allclose_passed, f"Test failed allclose comparison (atol={atol}, rtol={rtol})"


@pytest.mark.parametrize(
    "input_shape",
    [
        (1, 16, 16, 32),
        (1, 32, 32, 64),
        (2, 16, 16, 32),
    ],
)
@pytest.mark.parametrize(
    "mode",
    [
        "bilinear",
    ],
)
@pytest.mark.parametrize(
    "align_corners",
    [
        False,
        True,
    ],
)
def test_grid_sample_identity_transform(device, input_shape, mode, align_corners):
    """Test grid_sample with identity transformation (should return original image)"""

    torch.manual_seed(0)

    # input_shape is NHWC format: (batch_size, height, width, channels)
    batch_size, height, width, channels = input_shape

    torch_input_nhwc = torch.randn((batch_size, height, width, channels), dtype=torch.bfloat16)

    # Generates a grid that corresponds to the identity transformation
    theta = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
    theta_batched = theta.unsqueeze(0).expand(batch_size, -1, -1)
    shape = (batch_size, 1, height, width)
    torch_grid = F.affine_grid(theta_batched, shape, align_corners=align_corners)

    torch_output_nhwc = golden_grid_sample(
        input_tensor=torch_input_nhwc,
        grid=torch_grid.to(torch.bfloat16),
        mode=mode,
        padding_mode="zeros",
        align_corners=align_corners,
    )

    ttnn_input = ttnn.from_torch(torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_grid = ttnn.from_torch(torch_grid, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    ttnn_output = ttnn.grid_sample(ttnn_input, ttnn_grid, mode=mode, align_corners=align_corners)
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    pcc_passed, pcc_message = assert_with_pcc(torch_output_nhwc, ttnn_output_torch, pcc=0.99)
    logger.info(pcc_message)

    pcc_identity, identity_message = assert_with_pcc(torch_input_nhwc, ttnn_output_torch, pcc=0.99)
    logger.info(f"Identity check: {identity_message}")


@pytest.mark.parametrize(
    "input_shape",
    [
        (1, 16, 16, 32),
        (1, 32, 32, 64),
        (2, 16, 16, 32),
    ],
)
@pytest.mark.parametrize(
    "scale_factor",
    [0.5, 2.0, 1.5],  # Downsampling, upsampling, and mixed scaling
)
@pytest.mark.parametrize(
    "mode",
    [
        "bilinear",
    ],
)
@pytest.mark.parametrize(
    "align_corners",
    [
        False,
        True,
    ],
)
def test_grid_sample_scaling_patterns(device, input_shape, mode, align_corners, scale_factor):
    """Test grid_sample with different scaling patterns"""

    torch.manual_seed(0)

    # input_shape is NHWC format: (batch_size, height, width, channels)
    batch_size, height, width, channels = input_shape

    # Calculate output size based on scale factor
    output_h = int(height * scale_factor)
    output_w = int(width * scale_factor)

    torch_input_nhwc = torch.randn((batch_size, height, width, channels), dtype=torch.bfloat16)

    # Generates a grid that corresponds to downsampling / integer factor upscaling and fractional factor upsampling
    theta = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
    theta_batched = theta.unsqueeze(0).expand(batch_size, -1, -1)
    shape = (batch_size, 1, output_h, output_w)
    torch_grid = F.affine_grid(theta_batched, shape, align_corners=align_corners)

    torch_output_nhwc = golden_grid_sample(
        input_tensor=torch_input_nhwc,
        grid=torch_grid.to(torch.bfloat16),
        mode=mode,
        padding_mode="zeros",
        align_corners=align_corners,
    )

    torch_grid_bf16 = torch_grid.to(torch.bfloat16)

    ttnn_input = ttnn.from_torch(torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_grid = ttnn.from_torch(torch_grid_bf16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    ttnn_output = ttnn.grid_sample(ttnn_input, ttnn_grid, mode=mode, align_corners=align_corners)
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    pcc_passed, pcc_message = assert_with_pcc(torch_output_nhwc, ttnn_output_torch, pcc=0.99)
    logger.info(pcc_message)


# Channel widths beyond the 256-element single-reduction limit exercise the reader's chunked
# (wide-reduction) path: 384 = 12 tiles (partial last chunk + tilize_reconfig), 512 = 16 tiles
# (two full chunks), 1024 = 32 tiles (four full chunks). With fp32_dest_acc_en=True the host
# additionally clamps the chunk size to 4 tiles so each chunk fits in fp32 half-sync DEST without
# forcing dst_full_sync_en.
@pytest.mark.parametrize(
    "input_shape, grid_shape",
    [
        ((1, 384, 16, 16), (1, 12, 12, 2)),
        ((1, 512, 16, 16), (1, 12, 12, 2)),
        ((2, 1024, 8, 8), (2, 6, 6, 2)),
    ],
)
@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize("grid_dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("fp32_dest_acc_en", [False, True])
def test_grid_sample_wide_reduction(device, input_shape, grid_shape, align_corners, grid_dtype, fp32_dest_acc_en):
    """Channel widths > 256 must route through the chunked reader path, including under fp32 acc."""
    torch.manual_seed(0)

    batch_size, channels, height, width = input_shape
    torch_input_nhwc = torch.randn((batch_size, height, width, channels), dtype=torch.bfloat16)
    torch_grid_f32 = torch.rand(grid_shape, dtype=torch.float32) * 2.0 - 1.0

    golden_grid_dtype = torch.float32 if grid_dtype == ttnn.float32 else torch.bfloat16
    torch_output_nhwc = golden_grid_sample(
        input_tensor=torch_input_nhwc,
        grid=torch_grid_f32.to(golden_grid_dtype),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=align_corners,
    )

    ttnn_input = ttnn.from_torch(torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_grid = ttnn.from_torch(torch_grid_f32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=grid_dtype)

    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=fp32_dest_acc_en,
        packer_l1_acc=False,
        math_approx_mode=False,
    )

    ttnn_output = ttnn.grid_sample(
        ttnn_input,
        ttnn_grid,
        mode="bilinear",
        align_corners=align_corners,
        compute_kernel_config=compute_kernel_config,
    )
    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    atol, rtol = (0.02, 1e-2) if grid_dtype == ttnn.float32 else (1.0, 1e-1)
    assert torch.allclose(
        torch_output_nhwc, ttnn_output_torch, atol=atol, rtol=rtol
    ), f"Wide-reduction test failed (atol={atol}, rtol={rtol})"


@pytest.mark.parametrize("input_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("grid_kind", ["precomputed", "float32", "bfloat16"])
@pytest.mark.parametrize("sharded_grid", [False, True])
@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize("channels", [32, 384])
def test_grid_sample_bilinear_weight_format(device, input_dtype, grid_kind, sharded_grid, align_corners, channels):
    torch.manual_seed(42)
    input_tensor = torch.randn(1, 8, 16, channels).to(input_dtype)
    # Dyadic coordinates keep weights exactly representable while covering image boundaries.
    grid = torch.randint(-12, 13, (1, 2, 32, 2)).float() / 8
    expected = golden_grid_sample(input_tensor, grid, align_corners=align_corners)
    use_precomputed_grid = grid_kind == "precomputed"
    grid_dtype = ttnn.float32 if grid_kind == "float32" else ttnn.bfloat16
    grid_host = ttnn.from_torch(grid, dtype=ttnn.float32 if use_precomputed_grid else grid_dtype)
    if use_precomputed_grid:
        grid_host = ttnn.prepare_grid_sample_grid(
            grid_host,
            list(input_tensor.shape),
            mode="bilinear",
            align_corners=align_corners,
            output_dtype=ttnn.bfloat16,
        )
    # Pack four points per row so both grid representations have aligned shard widths.
    elements_per_point = 6 if use_precomputed_grid else 2
    grid_host = ttnn.reshape(grid_host, (1, 2, 8, 4 * elements_per_point))
    grid_device = ttnn.to_device(grid_host, device)
    if sharded_grid:
        memory_config = ttnn.create_sharded_memory_config(
            (8, 4 * elements_per_point),
            ttnn.CoreGrid(y=1, x=2),
            ttnn.ShardStrategy.HEIGHT,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        grid_device = ttnn.to_memory_config(grid_device, memory_config)
    input_device = ttnn.from_torch(input_tensor, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)
    output = ttnn.grid_sample(
        input_device,
        grid_device,
        mode="bilinear",
        align_corners=align_corners,
        use_precomputed_grid=use_precomputed_grid,
    )
    torch.testing.assert_close(ttnn.to_torch(output), expected, atol=0.02, rtol=0.02)


@pytest.mark.parametrize("input_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("batch_output_channels", [False, True])
@pytest.mark.parametrize(
    "grid_batching_factor, placement, batch_size",
    [(k, "interleaved", n) for n in (1, 2, 3) for k in (1, 2, 3, 4)]
    + [(4, "sharded", 1)]
    + [(k, "custom_output", 2) for k in (1, 2, 3, 4)],
)
def test_grid_sample_nearest_batched_work_partition(
    device, input_dtype, batch_output_channels, grid_batching_factor, placement, batch_size
):
    k = grid_batching_factor
    if batch_output_channels and k == 1:
        pytest.skip("Channel batching requires K > 1")
    input_tensor = torch.arange(batch_size * 128).reshape(batch_size, 8, 16, 1)
    input_tensor = input_tensor.expand(-1, -1, -1, 32).to(input_dtype).contiguous()
    indices = torch.arange(192) % 128
    grid = torch.stack(((indices % 16).float() / 15 * 2 - 1, (indices // 16).float() / 7 * 2 - 1), -1)
    grid = grid.reshape(1, 2, 96, 2).repeat(batch_size, 1, 1, 1)
    grid[:, 0, 5] = 3  # Include an out-of-bounds sample in each batch.
    expected = input_tensor.reshape(batch_size, 128, 32)[:, indices].clone()
    expected[:, 5] = 0
    expected = (
        expected.reshape(batch_size, 2, 96 // k, 32 * k)
        if batch_output_channels
        else expected.reshape(batch_size, 2, 96, 32)
    )
    grid_host = ttnn.prepare_grid_sample_grid(
        ttnn.from_torch(grid, dtype=ttnn.float32),
        list(input_tensor.shape),
        mode="nearest",
        align_corners=True,
        output_dtype=ttnn.bfloat16,
    )
    grid_host = ttnn.reshape(grid_host, (batch_size, 2, 96 // k, 2 * k))
    grid_device = ttnn.to_device(grid_host, device)
    if placement == "sharded":
        grid_memory_config = ttnn.create_sharded_memory_config(
            (batch_size * 192 // k // 2, 2 * k),
            ttnn.CoreGrid(y=1, x=2),
            ttnn.ShardStrategy.HEIGHT,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        grid_device = ttnn.to_memory_config(grid_device, grid_memory_config)
    output_kwargs = {}
    if placement == "custom_output":
        cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(2, 0))})
        shard_shape = [
            batch_size * 192 // 2 // (k if batch_output_channels else 1),
            32 * (k if batch_output_channels else 1),
        ]
        output_kwargs["memory_config"] = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(cores, shard_shape, ttnn.ShardOrientation.COL_MAJOR),
        )
    input_device = ttnn.from_torch(input_tensor, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)
    output = ttnn.grid_sample(
        input_device,
        grid_device,
        mode="nearest",
        align_corners=True,
        use_precomputed_grid=True,
        batch_output_channels=batch_output_channels,
        **output_kwargs,
    )
    torch.testing.assert_close(ttnn.to_torch(output), expected, atol=0, rtol=0)
    # A previous overrun corrupted the grid allocation and affected subsequent operations.
    assert torch.equal(ttnn.to_torch(grid_device).view(torch.int16), ttnn.to_torch(grid_host).view(torch.int16))
