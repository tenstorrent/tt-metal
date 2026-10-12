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


# =====================================================================================================================
# NEAREST mode: the paths added for the BEV blocks (2026-10): exact integer coordinate math, the index computation on
# the compute RISCs for a height-sharded bf16 grid, and a rank-5 (N, H_out, W_out, K, 2) grid gathered inside the op.
# Nearest sampling is an exact gather, so besides the PCC the outputs must be bit-identical to the torch reference.
# =====================================================================================================================
import struct


def _hs_mem(num_cores, shard_shape):
    """HEIGHT_SHARDED L1 memory config on the first num_cores cores of the 8x8 grid (row-major core order)."""
    ranges = []
    full_rows, rest = divmod(num_cores, 8)
    if full_rows:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, full_rows - 1)))
    if rest:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, full_rows), ttnn.CoreCoord(rest - 1, full_rows)))
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(ttnn.CoreRangeSet(set(ranges)), shard_shape, ttnn.ShardOrientation.ROW_MAJOR),
    )


def _nearest_output_mem(n, h_out, w_out, c, k, batch_output_channels, num_cores=64):
    """The height-sharded output the nearest kernel writes: one row per grid stick (or per sample when the width is
    extended), the sticks of each core contiguous."""
    sticks = n * h_out * w_out
    cores = min(num_cores, sticks)
    sticks_per_core = (sticks + cores - 1) // cores
    # width-extended: K rows per stick, so the shard height must be a multiple of K
    return _hs_mem(cores, (sticks_per_core * (1 if batch_output_channels else k), c * (k if batch_output_channels else 1))), cores


def _nearest_golden(torch_input_nhwc, grid4d_f32, align_corners, batch_output_channels, k):
    """torch.grid_sample(mode='nearest') on the K coordinate sets packed in the grid's last dim (bf16 values)."""
    return golden_grid_sample(
        input_tensor=torch_input_nhwc,
        grid=grid4d_f32.to(torch.bfloat16),
        mode="nearest",
        padding_mode="zeros",
        align_corners=align_corners,
        batch_output_channels=batch_output_channels,
        grid_batching_factor=k,
    )


def _check_exact(expected, actual, what):
    """Nearest is an exact gather: PCC and bit-equality."""
    assert expected.shape == actual.shape, f"{what}: shape {tuple(actual.shape)} != {tuple(expected.shape)}"
    assert_with_pcc(expected.float(), actual.float(), 0.999)
    diff = (expected.float() - actual.float()).abs()
    assert torch.equal(expected.to(torch.bfloat16), actual.to(torch.bfloat16)), (
        f"{what}: {int((diff > 0).sum())} of {diff.numel()} values differ, max |diff| = {diff.max().item()}"
    )


_NEAREST_SHAPES = [  # (input NCHW, h_out, w_out)
    ((1, 64, 80, 144), 128, 64),  # BEV block D: non-tile-aligned image, 8192 sticks = 128 per core
    ((1, 64, 96, 96), 128, 64),  # BEV block B
    ((2, 32, 20, 36), 30, 25),  # batch 2, 1500 sticks: 24 per core, the 3:2:3 groups of 8 and the 12 DRAM banks do not divide it
    ((1, 160, 7, 13), 9, 11),  # tiny image, 99 sticks on 64 cores: 2 per core, partial last shard
    ((1, 32, 3, 5), 4, 4),  # smaller than a tile in every dimension
]


@pytest.mark.parametrize("input_shape, h_out, w_out", _NEAREST_SHAPES)
@pytest.mark.parametrize("k", [2, 4, 8, 16])
@pytest.mark.parametrize("align_corners", [True, False])
@pytest.mark.parametrize("image_in_l1", [False, True])
def test_grid_sample_nearest_rank5_grid(device, input_shape, h_out, w_out, k, align_corners, image_in_l1):
    """Rank-5 (N, H_out, W_out, K, 2) bf16 grid in DRAM consumed directly (no reshape, no sharding op): the op gathers
    each core's 4-byte grid pages itself. Checked bit-exact against torch and against the rank-4 sharded-grid path."""
    torch.manual_seed(0)
    n, c, h, w = input_shape
    torch_input_nhwc = torch.randn((n, h, w, c), dtype=torch.bfloat16)
    torch_input_nhwc[..., 0] = 1.0  # channel 0 never 0: a sampled pixel is distinguishable from the zero stick
    grid5d = (torch.rand((n, h_out, w_out, k, 2), dtype=torch.float32) * 2.4 - 1.2).to(torch.bfloat16).float()
    expected = _nearest_golden(torch_input_nhwc, grid5d.reshape(n, h_out, w_out, 2 * k), align_corners, True, k)

    if image_in_l1:
        rows = n * h * w
        cores = min(64, rows)
        ttnn_input = ttnn.reshape(
            ttnn.from_torch(torch_input_nhwc.reshape(1, 1, rows, c), layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=_hs_mem(cores, ((rows + cores - 1) // cores, c))),
            [n, h, w, c],
        )
    else:
        ttnn_input = ttnn.from_torch(torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    out_mem, _ = _nearest_output_mem(n, h_out, w_out, c, k, True)

    ttnn_grid5d = ttnn.from_torch(grid5d, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    out5 = ttnn.to_torch(
        ttnn.grid_sample(ttnn_input, ttnn_grid5d, mode="nearest", align_corners=align_corners, batch_output_channels=True, memory_config=out_mem)
    )
    _check_exact(expected, out5, "rank-5 grid")

    # the same grid as a rank-4 height-sharded tensor (the compiler's layout): must give the identical result
    if 2 * k * 2 >= 16:  # an L1 shard page must be at least 16 bytes
        rows = n * h_out * w_out
        cores = min(64, rows)
        ttnn_grid4d = ttnn.from_torch(grid5d.reshape(n, h_out, w_out, 2 * k), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=_hs_mem(cores, ((rows + cores - 1) // cores, 2 * k)))
        out4 = ttnn.to_torch(
            ttnn.grid_sample(ttnn_input, ttnn_grid4d, mode="nearest", align_corners=align_corners, batch_output_channels=True, memory_config=out_mem)
        )
        _check_exact(expected, out4, "rank-4 sharded grid")
        assert torch.equal(out4, out5), "rank-5 and rank-4 paths disagree"


@pytest.mark.parametrize("input_shape, h_out, w_out", _NEAREST_SHAPES[:3])
@pytest.mark.parametrize("k, batch_output_channels", [(1, False), (2, False), (8, False), (4, True), (8, True)])
@pytest.mark.parametrize("grid_placement", ["dram", "l1_sharded"])
@pytest.mark.parametrize("precomputed", [False, True])
@pytest.mark.parametrize("align_corners", [True, False])
def test_grid_sample_nearest_rank4_grid(device, input_shape, h_out, w_out, k, batch_output_channels, grid_placement, precomputed, align_corners):
    """Rank-4 grids on every nearest path: DRAM interleaved raw grid (coordinate math on the data-movement RISCs),
    height-sharded raw grid (coordinate math on the compute RISCs), and the host-precomputed int16 grid; K = 1 and
    batched K with the width-extended and the channel-batched outputs."""
    torch.manual_seed(1)
    n, c, h, w = input_shape
    if grid_placement == "l1_sharded" and 2 * k * 2 < 16:
        pytest.skip("an L1 shard page must be at least 16 bytes")
    if grid_placement == "dram" and n > 1:
        pytest.skip("pre-existing limitation: the op's automatic output sharding for an interleaved grid splits per batch and exceeds the core count for N > 1 (the rank-5 and sharded-grid paths cover N > 1)")
    torch_input_nhwc = torch.randn((n, h, w, c), dtype=torch.bfloat16)
    torch_input_nhwc[..., 0] = 1.0
    grid4d = (torch.rand((n, h_out, w_out, 2 * k), dtype=torch.float32) * 2.4 - 1.2).to(torch.bfloat16).float()
    expected = _nearest_golden(torch_input_nhwc, grid4d, align_corners, batch_output_channels, k)

    ttnn_input = ttnn.from_torch(torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    if precomputed:
        host_grid = ttnn.from_torch(grid4d, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT)
        host_grid = ttnn.prepare_grid_sample_grid(host_grid, [n, h, w, c], mode="nearest", padding_mode="zeros", align_corners=align_corners)
    else:
        host_grid = ttnn.from_torch(grid4d, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    rows = n * h_out * w_out
    cores = min(64, rows)
    grid_mem = _hs_mem(cores, ((rows + cores - 1) // cores, 2 * k)) if grid_placement == "l1_sharded" else ttnn.DRAM_MEMORY_CONFIG
    ttnn_grid = ttnn.to_device(host_grid, device, memory_config=grid_mem)
    # a sharded grid defines the work split, so the output sharding is given; for an interleaved grid the op splits
    # the work itself and writes an interleaved output
    out_mem = _nearest_output_mem(n, h_out, w_out, c, k, batch_output_channels)[0] if grid_placement == "l1_sharded" else None
    out = ttnn.to_torch(
        ttnn.grid_sample(ttnn_input, ttnn_grid, mode="nearest", align_corners=align_corners, use_precomputed_grid=precomputed, batch_output_channels=batch_output_channels, memory_config=out_mem)
    )
    _check_exact(expected, out, f"rank-4 {grid_placement} precomputed={precomputed}")


@pytest.mark.parametrize("grid_rank", [4, 5])
@pytest.mark.parametrize("align_corners", [True, False])
@pytest.mark.parametrize("h, w", [(96, 96), (80, 144), (1, 1), (2, 3)])
def test_grid_sample_nearest_edge_coordinates(device, grid_rank, align_corners, h, w):
    """Every one of the 65,536 bf16 bit patterns as the x coordinate (y fixed) and as the y coordinate (x fixed):
    exact ties (0, +-1), values just inside and outside [-1, 1], tiny and denormal values, huge values, and the
    size-1 image. Finite coordinates must match torch bit for bit; inf / nan must sample the zero stick."""
    torch.manual_seed(2)
    n, c, h_out, w_out, k = 1, 32, 128, 64, 8  # 65,536 samples
    torch_input_nhwc = torch.randn((n, h, w, c), dtype=torch.bfloat16)
    torch_input_nhwc[..., 0] = 1.0
    all_bits = torch.tensor([struct.unpack(">f", struct.pack(">H", b) + b"\x00\x00")[0] for b in range(65536)], dtype=torch.float32)
    finite = torch.isfinite(all_bits)
    ttnn_input = ttnn.from_torch(torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    out_mem, _ = _nearest_output_mem(n, h_out, w_out, c, k, True)
    for axis in (0, 1):
        grid = torch.full((65536, 2), 0.3, dtype=torch.float32)
        grid[:, axis] = all_bits
        grid5d = grid.reshape(n, h_out, w_out, k, 2)
        expected = _nearest_golden(torch_input_nhwc, grid5d.reshape(n, h_out, w_out, 2 * k), align_corners, True, k)
        if grid_rank == 5:
            ttnn_grid = ttnn.from_torch(grid5d, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        else:
            ttnn_grid = ttnn.from_torch(grid5d.reshape(n, h_out, w_out, 2 * k), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=_hs_mem(64, (128, 2 * k)))
        out = ttnn.to_torch(ttnn.grid_sample(ttnn_input, ttnn_grid, mode="nearest", align_corners=align_corners, batch_output_channels=True, memory_config=out_mem))
        exp_s = expected.reshape(65536, c).float()
        out_s = out.reshape(65536, c).float()
        bad = ((exp_s[finite] - out_s[finite]).abs().amax(-1) > 0).sum().item()
        assert bad == 0, f"axis {'xy'[axis]}: {bad} finite bf16 coordinates sample a different pixel than torch"
        assert torch.equal(out_s[~finite], torch.zeros_like(out_s[~finite])), f"axis {'xy'[axis]}: inf / nan coordinates must sample zeros"
        assert_with_pcc(exp_s[finite], out_s[finite], 0.999)


def test_grid_sample_nearest_rank5_grid_rejects_unsupported(device):
    """A rank-5 grid is only for nearest, bf16, interleaved, not precomputed, batch_output_channels=True and a
    height-sharded output: everything else must be rejected with a clear error, not silently mis-sampled."""
    n, c, h, w, h_out, w_out, k = 1, 32, 16, 16, 8, 8, 4
    ttnn_input = ttnn.from_torch(torch.randn((n, h, w, c), dtype=torch.bfloat16), layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    grid5d = (torch.rand((n, h_out, w_out, k, 2)) * 2 - 1).to(torch.bfloat16)
    ttnn_grid = ttnn.from_torch(grid5d, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    out_mem, _ = _nearest_output_mem(n, h_out, w_out, c, k, True)
    # the supported combination works
    ttnn.grid_sample(ttnn_input, ttnn_grid, mode="nearest", align_corners=True, batch_output_channels=True, memory_config=out_mem)
    for kwargs in (
        dict(mode="bilinear", batch_output_channels=True, memory_config=out_mem),
        dict(mode="nearest", batch_output_channels=False, memory_config=out_mem),
        dict(mode="nearest", batch_output_channels=True, use_precomputed_grid=True, memory_config=out_mem),
        dict(mode="nearest", batch_output_channels=True, memory_config=ttnn.DRAM_MEMORY_CONFIG),
    ):
        with pytest.raises(RuntimeError):
            ttnn.grid_sample(ttnn_input, ttnn_grid, align_corners=True, **kwargs)


# =====================================================================================================================
# Full matrix: both modes, align_corners, use_precomputed_grid, grid rank 4 / 5, batching (batch_output_channels, K),
# grid dtype, several input / grid shapes, and the memory configs of input / grid / output plus compute kernel configs.
# Every case is checked against the torch golden (golden_grid_sample) with a PCC bound; nearest cases must also be
# bit-exact (exact gather). Combinations the op rejects in its validation are reported as skips with the op's message.
# =====================================================================================================================
_M_SHAPES = {  # name: (input NCHW, h_out, w_out)
    "s16": ((1, 32, 16, 16), 8, 8),
    "s2x20x36": ((2, 64, 20, 36), 30, 25),
    "s80x144": ((1, 96, 80, 144), 64, 32),
    "s7x13": ((1, 128, 7, 13), 9, 11),
}
_M_BATCHING = [(False, 1), (False, 4), (True, 4), (True, 8)]  # (batch_output_channels, K)


def _m_semantic_cases():
    cases = []
    for shape in _M_SHAPES:
        for mode in ("bilinear", "nearest"):
            for align in (True, False):
                for precomputed in (False, True):
                    for grid_rank in (4, 5):
                        for boc, k in _M_BATCHING:
                            for gdt in ("bf16", "f32"):
                                if grid_rank == 5 and (mode != "nearest" or precomputed or not boc or gdt != "bf16"):
                                    continue  # a rank-5 grid is only the nearest / bf16 / channel-batched form
                                if precomputed and gdt == "f32":
                                    continue  # a precomputed grid is bf16 by definition
                                if precomputed and mode == "bilinear" and k > 1:
                                    continue  # prepare_grid_sample_grid batches only nearest grids
                                cases.append(dict(shape=shape, mode=mode, align=align, precomputed=precomputed, grid_rank=grid_rank, boc=boc, k=k, gdt=gdt,
                                                  in_mem="dram", grid_mem="dram", out_mem="default", compute="default"))
    return cases


def _m_memory_cases():
    cases = []
    for shape in ("s16", "s80x144"):
        for mode in ("bilinear", "nearest"):
            boc, k = (True, 4) if mode == "nearest" else (False, 1)
            for in_mem in ("dram", "l1", "hs"):
                for grid_mem in ("dram", "l1", "hs", "dram5", "l15"):
                    if grid_mem.endswith("5") and mode != "nearest":
                        continue
                    for out_mem in ("default", "dram", "l1", "hs"):
                        for compute in ("default", "lofi", "hifi4_fp32", "hifi2_l1acc"):
                            cases.append(dict(shape=shape, mode=mode, align=True, precomputed=False, grid_rank=5 if grid_mem.endswith("5") else 4, boc=boc, k=k, gdt="bf16",
                                              in_mem=in_mem, grid_mem=grid_mem.rstrip("5"), out_mem=out_mem, compute=compute))
    return cases


_M_CASES = _m_semantic_cases() + _m_memory_cases()
_M_IDS = [f"{c['shape']}-{c['mode']}-ac{int(c['align'])}-pre{int(c['precomputed'])}-r{c['grid_rank']}-boc{int(c['boc'])}-k{c['k']}-{c['gdt']}-in_{c['in_mem']}-grid_{c['grid_mem']}-out_{c['out_mem']}-{c['compute']}" for c in _M_CASES]


def _m_compute(device, name):
    if name == "default":
        return None
    fid, fp32, l1acc = {"lofi": (ttnn.MathFidelity.LoFi, False, False), "hifi4_fp32": (ttnn.MathFidelity.HiFi4, True, False), "hifi2_l1acc": (ttnn.MathFidelity.HiFi2, False, True)}[name]
    return ttnn.init_device_compute_kernel_config(device.arch(), math_fidelity=fid, fp32_dest_acc_en=fp32, packer_l1_acc=l1acc, math_approx_mode=False)


@pytest.mark.parametrize("case", _M_CASES, ids=_M_IDS)
def test_grid_sample_matrix(device, case):
    torch.manual_seed(7)
    (n, c, h, w), h_out, w_out = _M_SHAPES[case["shape"]]
    mode, align, precomputed, grid_rank, boc, k, gdt = (case[x] for x in ("mode", "align", "precomputed", "grid_rank", "boc", "k", "gdt"))
    in_mem, grid_mem, out_mem, compute = (case[x] for x in ("in_mem", "grid_mem", "out_mem", "compute"))

    torch_input_nhwc = torch.randn((n, h, w, c), dtype=torch.bfloat16)
    grid4d = torch.rand((n, h_out, w_out, 2 * k), dtype=torch.float32) * 2.4 - 1.2
    if gdt == "bf16":
        grid4d = grid4d.to(torch.bfloat16).float()
    expected = golden_grid_sample(
        input_tensor=torch_input_nhwc, grid=grid4d.to(torch.bfloat16 if gdt == "bf16" else torch.float32), mode=mode, padding_mode="zeros",
        align_corners=align, batch_output_channels=boc, grid_batching_factor=k,
    )

    # ---- input
    if in_mem == "hs":
        rows = n * h * w
        cores = min(64, rows)
        ttnn_input = ttnn.reshape(
            ttnn.from_torch(torch_input_nhwc.reshape(1, 1, rows, c), layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=_hs_mem(cores, ((rows + cores - 1) // cores, c))),
            [n, h, w, c],
        )
    else:
        ttnn_input = ttnn.from_torch(torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG if in_mem == "dram" else ttnn.L1_MEMORY_CONFIG)
    # ---- grid
    if precomputed:
        host_grid = ttnn.prepare_grid_sample_grid(ttnn.from_torch(grid4d, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT), [n, h, w, c], mode=mode, padding_mode="zeros", align_corners=align)
        grid_width = 6 if mode == "bilinear" else 2 * k
    elif grid_rank == 5:
        host_grid = ttnn.from_torch(grid4d.reshape(n, h_out, w_out, k, 2), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        grid_width = 2
    else:
        host_grid = ttnn.from_torch(grid4d, dtype=ttnn.bfloat16 if gdt == "bf16" else ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT)
        grid_width = 2 * k
    if grid_mem == "hs":
        elem = 4 if gdt == "f32" else 2
        if grid_width * elem < 16:
            pytest.skip("an L1 shard page must be at least 16 bytes")
        rows = n * h_out * w_out
        cores = min(64, rows)
        grid_memcfg = _hs_mem(cores, ((rows + cores - 1) // cores, grid_width))
    else:
        grid_memcfg = ttnn.DRAM_MEMORY_CONFIG if grid_mem == "dram" else ttnn.L1_MEMORY_CONFIG
    ttnn_grid = ttnn.to_device(host_grid, device, memory_config=grid_memcfg)
    # ---- output
    if out_mem == "hs":
        out_memcfg = _nearest_output_mem(n, h_out, w_out, c, k, boc)[0]
    elif out_mem == "default":
        if mode == "nearest" and (grid_rank == 5 or grid_mem == "hs" or n > 1):
            out_memcfg = _nearest_output_mem(n, h_out, w_out, c, k, boc)[0]  # nearest writes a height-sharded output; the op only derives one itself for an interleaved grid with N = 1
        else:
            out_memcfg = None
    else:
        out_memcfg = ttnn.DRAM_MEMORY_CONFIG if out_mem == "dram" else ttnn.L1_MEMORY_CONFIG

    try:
        out = ttnn.grid_sample(
            ttnn_input, ttnn_grid, mode=mode, padding_mode="zeros", align_corners=align, use_precomputed_grid=precomputed,
            batch_output_channels=boc, memory_config=out_memcfg, compute_kernel_config=_m_compute(device, compute),
        )
    except RuntimeError as e:
        msg = str(e).splitlines()
        reason = next((l for l in msg if "TT_FATAL" in l or "TT_THROW" in l), msg[0])
        if "TT_FATAL" in reason and "assert.hpp" not in reason.split("TT_FATAL")[0]:
            pytest.skip(f"op rejects this combination: {reason[:200]}")
        raise
    actual = ttnn.to_torch(out)
    assert expected.shape == actual.shape, f"shape {tuple(actual.shape)} != {tuple(expected.shape)}"
    if mode == "nearest":
        assert_with_pcc(expected.float(), actual.float(), 0.999)
        assert torch.equal(expected.to(torch.bfloat16), actual.to(torch.bfloat16)), "nearest must be bit-exact"
    else:
        pcc = 0.98 if compute == "lofi" else (0.995 if gdt == "f32" else 0.99)
        assert_with_pcc(expected.float(), actual.float(), pcc)
