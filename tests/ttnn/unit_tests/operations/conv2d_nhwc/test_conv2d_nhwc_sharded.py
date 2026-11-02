# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Refinement 1: BLOCK_SHARDED / WIDTH_SHARDED placement (regimes R4 / R5).

The input arrives sharded in L1 (same geometry rule as the golden harness: whole
32-channel slices, sticks over grid rows for BLOCK, all sticks per core for WIDTH);
the output must come back in the same TensorMemoryLayout. Covers ragged stick
splits, multi-row WIDTH groups (irregular multicast sets), uneven C_out over the
slices (zero-padded column tiles), a WIDTH output whose M is not a tile multiple
(staged tail), strided, 1x1 and bias / no bias.
"""

import pytest
import torch
import ttnn

from ttnn.operations.conv2d_nhwc import conv2d_nhwc, prepare_conv2d_bias, prepare_conv2d_weights

ML = ttnn.TensorMemoryLayout


def pcc(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _ceil(a, b):
    return -(-a // b)


def _row_major_cores(ncores, gx):
    full, rem = divmod(ncores, gx)
    ranges = []
    if full:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, full - 1)))
    if rem:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, full), ttnn.CoreCoord(rem - 1, full)))
    return ttnn.CoreRangeSet(ranges)


def _sharded_config(memory_layout, shape, device, shard_shape=None, core_grid=None, orientation=None, num_cores=None):
    if memory_layout == ML.INTERLEAVED:
        return ttnn.DRAM_MEMORY_CONFIG
    N, H, W, C = shape
    grid = device.compute_with_storage_grid_size()
    rows, c_tiles = N * H * W, C // 32
    if memory_layout == ML.HEIGHT_SHARDED:
        # Golden-harness rule: ragged stick split over the whole grid, row-major (or pinned).
        if shard_shape is not None:
            sshape, ncores = list(shard_shape), num_cores
        else:
            per_h = _ceil(rows, grid.x * grid.y)
            ncores = _ceil(rows, per_h)
            sshape = [per_h, C]
        spec = ttnn.ShardSpec(_row_major_cores(ncores, grid.x), sshape, ttnn.ShardOrientation.ROW_MAJOR)
        return ttnn.MemoryConfig(memory_layout, ttnn.BufferType.L1, spec)
    if memory_layout == ML.WIDTH_SHARDED:
        per_c = _ceil(c_tiles, grid.x * grid.y)
        ncores = _ceil(c_tiles, per_c)
        full, rem = divmod(ncores, grid.x)
        ranges = []
        if full:
            ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, full - 1)))
        if rem:
            ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, full), ttnn.CoreCoord(rem - 1, full)))
        cores = ttnn.CoreRangeSet(ranges)
        sshape = [rows, per_c * 32]
    else:
        if shard_shape is not None:
            sshape = list(shard_shape)
            nx, ny = core_grid
        else:
            per_h = _ceil(rows, grid.y)
            ny = _ceil(rows, per_h)
            per_c = _ceil(c_tiles, grid.x)
            nx = _ceil(c_tiles, per_c)
            sshape = [per_h, per_c * 32]
        if orientation == ttnn.ShardOrientation.COL_MAJOR:
            nx, ny = ny, nx  # transposed grid: shard rows along x, channel slices along y
        cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(nx - 1, ny - 1))])
    spec = ttnn.ShardSpec(cores, sshape, orientation or ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(memory_layout, ttnn.BufferType.L1, spec)


_TORCH = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16, ttnn.bfloat8_b: torch.bfloat16}
# (pcc, rel-RMS) gates per activation dtype (eval helpers.TOLERANCES).
_TOL = {ttnn.float32: (0.999, 0.02), ttnn.bfloat16: (0.995, 0.05)}


def run_sharded(
    device,
    input_shape,
    C_out,
    k,
    padding,
    stride,
    memory_layout,
    with_bias,
    dtype=ttnn.float32,
    weight_dtype=ttnn.float32,
    activation=None,
    **cfg,
):
    torch.manual_seed(0)
    N, H, W, C_in = input_shape
    x = torch.randn(input_shape, dtype=torch.float32).to(_TORCH[dtype]).float()
    w = torch.randn(C_out, C_in, k, k, dtype=torch.float32).to(_TORCH[weight_dtype]).float()
    b = torch.randn(C_out, dtype=torch.float32).to(_TORCH[weight_dtype]).float() if with_bias else None
    ref = torch.nn.functional.conv2d(x.permute(0, 3, 1, 2), w, b, stride=stride, padding=padding)
    ref = ref.permute(0, 2, 3, 1).contiguous()
    if activation == "relu":
        ref = torch.relu(ref)

    mc = _sharded_config(memory_layout, input_shape, device, **cfg)
    tt_x = ttnn.from_torch(x, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mc)
    kw = dict(
        kernel_size=k,
        stride=stride,
        padding=padding,
        input_dtype=dtype,
        weight_dtype=weight_dtype,
        input_memory_config=tt_x.memory_config(),
        device=device,
    )
    tt_w = prepare_conv2d_weights(w, **kw)
    tt_b = prepare_conv2d_bias(b, **kw) if with_bias else None
    out = conv2d_nhwc(tt_x, tt_w, k, bias=tt_b, padding=padding, stride=stride, activation=activation)
    assert out.memory_config().memory_layout == memory_layout
    assert out.dtype == dtype
    got = ttnn.to_torch(out).float()
    assert list(got.shape) == list(ref.shape)
    if activation == "relu":
        assert got.min().item() >= 0.0
    p = pcc(got, ref)
    rel_rms = ((got - ref).pow(2).mean().sqrt() / ref.std()).item()
    min_pcc, max_rms = _TOL[dtype]
    print(f"PRECISION pcc={p:.6f} rel_rms={rel_rms:.5f}")
    assert p >= min_pcc and rel_rms <= max_rms, f"pcc={p} rel_rms={rel_rms}"


CASES = [
    # (input_shape, C_out, k, padding, stride, memory_layout)
    pytest.param((1, 8, 8, 64), 64, 3, 1, 1, ML.BLOCK_SHARDED, id="block_small_2slices"),
    pytest.param((2, 14, 14, 128), 128, 3, 1, 1, ML.BLOCK_SHARDED, id="block_n2_ragged_rows"),
    pytest.param((1, 32, 32, 256), 320, 3, 1, 1, ML.BLOCK_SHARDED, id="block_cout_uneven_over_slices"),
    pytest.param((1, 28, 28, 256), 256, 3, 1, 2, ML.BLOCK_SHARDED, id="block_strided"),
    pytest.param((1, 7, 7, 512), 512, 3, 1, 1, ML.WIDTH_SHARDED, id="width_two_grid_rows_tail"),
    pytest.param((1, 14, 14, 256), 512, 3, 1, 2, ML.WIDTH_SHARDED, id="width_strided"),
    pytest.param((1, 14, 14, 1024), 2048, 1, 0, 2, ML.WIDTH_SHARDED, id="width_1x1_s2"),
]


@pytest.mark.parametrize("with_bias", [True, False], ids=["bias", "no_bias"])
@pytest.mark.parametrize("input_shape, C_out, k, padding, stride, memory_layout", CASES)
def test_conv2d_nhwc_sharded(device, input_shape, C_out, k, padding, stride, memory_layout, with_bias):
    run_sharded(device, input_shape, C_out, k, padding, stride, memory_layout, with_bias)


def test_conv2d_nhwc_block_pinned_grid(device):
    """Pinned 10x10 BLOCK geometry spanning several input rows per output row's window."""
    grid = device.compute_with_storage_grid_size()
    if grid.x < 10 or grid.y < 10:
        pytest.skip("needs a 10x10 grid")
    run_sharded(
        device,
        (1, 64, 64, 640),
        640,
        3,
        1,
        1,
        ML.BLOCK_SHARDED,
        True,
        shard_shape=(410, 64),
        core_grid=(10, 10),
    )


# Refinement 2: mixed precision (dtype x weight_dtype) + fused ReLU on every built path.
MIXED_CASES = [
    pytest.param((1, 16, 16, 64), 64, 3, 1, 1, ML.INTERLEAVED, id="interleaved"),
    pytest.param((1, 28, 28, 256), 256, 3, 1, 1, ML.BLOCK_SHARDED, id="block"),
    pytest.param((1, 7, 7, 512), 512, 3, 1, 1, ML.WIDTH_SHARDED, id="width_tail"),
]


@pytest.mark.parametrize("activation", [None, "relu"], ids=["none", "relu"])
@pytest.mark.parametrize(
    "dtype, weight_dtype",
    [
        (ttnn.bfloat16, ttnn.bfloat16),
        (ttnn.bfloat16, ttnn.bfloat8_b),
        (ttnn.bfloat16, ttnn.float32),
        (ttnn.float32, ttnn.bfloat8_b),
    ],
    ids=["bf16xbf16", "bf16xbf8b", "bf16xfp32", "fp32xbf8b"],
)
@pytest.mark.parametrize("input_shape, C_out, k, padding, stride, memory_layout", MIXED_CASES)
def test_conv2d_nhwc_mixed_precision(
    device, input_shape, C_out, k, padding, stride, memory_layout, dtype, weight_dtype, activation
):
    run_sharded(
        device, input_shape, C_out, k, padding, stride, memory_layout, True, dtype, weight_dtype, activation=activation
    )


def test_conv2d_nhwc_relu_no_bias_block(device):
    """ReLU without bias: the packer ReLU wraps the untilize pack instead of the bias pack."""
    run_sharded(
        device,
        (1, 28, 28, 256),
        256,
        3,
        1,
        1,
        ML.BLOCK_SHARDED,
        False,
        ttnn.bfloat16,
        ttnn.bfloat8_b,
        activation="relu",
    )


@pytest.mark.parametrize(
    "input_shape, C_out, stride",
    [((1, 32, 32, 128), 128, 1), ((2, 14, 14, 128), 256, 2)],
    ids=["s1", "s2_ragged"],
)
def test_conv2d_nhwc_block_col_major(device, input_shape, C_out, stride):
    """BLOCK with COL_MAJOR shard orientation: the transposed grid (K exchange down grid columns,
    weight multicast along grid rows); output keeps the input's orientation."""
    run_sharded(
        device,
        input_shape,
        C_out,
        3,
        1,
        stride,
        ML.BLOCK_SHARDED,
        True,
        ttnn.bfloat16,
        ttnn.bfloat16,
        orientation=ttnn.ShardOrientation.COL_MAJOR,
    )


# Refinement 5: HEIGHT_SHARDED (regime R3) — resident shard + NoC halo runs, weight multicast
# (R1) from core 0 to every compute core, HEIGHT-sharded output on the first g_m input cores.
HEIGHT_CASES = [
    pytest.param((1, 56, 56, 64), 64, 3, 1, 1, id="resnet_stage1"),
    pytest.param((1, 56, 56, 64), 128, 3, 1, 2, id="strided"),
    pytest.param((2, 28, 28, 64), 64, 3, 1, 1, id="n2_straddle"),
    pytest.param((1, 17, 17, 32), 64, 3, 1, 1, id="ragged_odd_w"),
    pytest.param((1, 8, 8, 32), 32, 3, 1, 1, id="shard_shorter_than_row"),
    pytest.param((1, 56, 56, 256), 64, 1, 0, 1, id="pointwise"),
]


@pytest.mark.parametrize("with_bias", [True, False], ids=["bias", "no_bias"])
@pytest.mark.parametrize("input_shape, C_out, k, padding, stride", HEIGHT_CASES)
def test_conv2d_nhwc_height(device, input_shape, C_out, k, padding, stride, with_bias):
    run_sharded(device, input_shape, C_out, k, padding, stride, ML.HEIGHT_SHARDED, with_bias)


@pytest.mark.parametrize("activation", [None, "relu"], ids=["none", "relu"])
def test_conv2d_nhwc_height_bf16_bf8b(device, activation):
    run_sharded(
        device, (2, 28, 28, 64), 64, 3, 1, 1, ML.HEIGHT_SHARDED, True, ttnn.bfloat16, ttnn.bfloat8_b, activation
    )


@pytest.mark.parametrize(
    "input_shape, C_out, k, padding, stride, shard_shape, num_cores",
    [
        # Strided: output shards placed on the input cores covering their windows (gappy core set,
        # weight multicast over the input grid with receive-only sinks), split-free single-K pipeline.
        pytest.param((16, 56, 56, 128), 128, 3, 1, 2, (457, 128), 110, id="resnet_layer2_ds"),
        pytest.param((16, 56, 56, 64), 256, 1, 0, 2, (457, 64), 110, id="resnet_shortcut_1x1_s2"),
        # Stride 1, tap-major gather, ragged last core (fewer m-blocks than the rest).
        pytest.param((16, 56, 56, 64), 64, 3, 1, 1, (480, 64), 105, id="resnet_layer1"),
    ],
)
def test_conv2d_nhwc_height_resnet_pinned(device, input_shape, C_out, k, padding, stride, shard_shape, num_cores):
    """Refinement 6: the ResNet50 HEIGHT perf configs (bf16 x bf8b, bias) at their pinned geometry."""
    if device.compute_with_storage_grid_size().x * device.compute_with_storage_grid_size().y < num_cores:
        pytest.skip(f"needs {num_cores} cores")
    run_sharded(
        device,
        input_shape,
        C_out,
        k,
        padding,
        stride,
        ML.HEIGHT_SHARDED,
        True,
        ttnn.bfloat16,
        ttnn.bfloat8_b,
        shard_shape=shard_shape,
        num_cores=num_cores,
    )


def test_conv2d_nhwc_height_weights_streamed(device):
    """Weights too large to keep resident (fp32, 512->512 3x3): streamed K-blocks, unaligned output."""
    run_sharded(device, (1, 14, 14, 512), 512, 3, 1, 2, ML.HEIGHT_SHARDED, True)


def test_conv2d_nhwc_height_resnet_layer2_pinned(device):
    """ResNet50 layer2 3x3 at its production HEIGHT geometry (128-stick shards on 98 cores)."""
    if device.compute_with_storage_grid_size().x * device.compute_with_storage_grid_size().y < 98:
        pytest.skip("needs 98 cores")
    run_sharded(
        device,
        (16, 28, 28, 128),
        128,
        3,
        1,
        1,
        ML.HEIGHT_SHARDED,
        True,
        ttnn.bfloat16,
        ttnn.bfloat8_b,
        shard_shape=(128, 128),
        num_cores=98,
    )
