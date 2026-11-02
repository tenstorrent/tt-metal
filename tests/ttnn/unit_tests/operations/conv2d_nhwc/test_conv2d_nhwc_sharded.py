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


def _sharded_config(memory_layout, shape, device, shard_shape=None, core_grid=None, orientation=None):
    if memory_layout == ML.INTERLEAVED:
        return ttnn.DRAM_MEMORY_CONFIG
    N, H, W, C = shape
    grid = device.compute_with_storage_grid_size()
    rows, c_tiles = N * H * W, C // 32
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
