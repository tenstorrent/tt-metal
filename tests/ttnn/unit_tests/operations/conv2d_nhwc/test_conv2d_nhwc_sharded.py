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


def _sharded_config(memory_layout, shape, device, shard_shape=None, core_grid=None):
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
        cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(nx - 1, ny - 1))])
    spec = ttnn.ShardSpec(cores, sshape, ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(memory_layout, ttnn.BufferType.L1, spec)


def run_sharded(device, input_shape, C_out, k, padding, stride, memory_layout, with_bias, **cfg):
    torch.manual_seed(0)
    N, H, W, C_in = input_shape
    x = torch.randn(input_shape, dtype=torch.float32)
    w = torch.randn(C_out, C_in, k, k, dtype=torch.float32)
    b = torch.randn(C_out, dtype=torch.float32) if with_bias else None
    ref = torch.nn.functional.conv2d(x.permute(0, 3, 1, 2), w, b, stride=stride, padding=padding)
    ref = ref.permute(0, 2, 3, 1).contiguous()

    mc = _sharded_config(memory_layout, input_shape, device, **cfg)
    tt_x = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mc)
    kw = dict(
        kernel_size=k,
        stride=stride,
        padding=padding,
        input_dtype=ttnn.float32,
        weight_dtype=ttnn.float32,
        input_memory_config=tt_x.memory_config(),
        device=device,
    )
    tt_w = prepare_conv2d_weights(w, **kw)
    tt_b = prepare_conv2d_bias(b, **kw) if with_bias else None
    out = conv2d_nhwc(tt_x, tt_w, k, bias=tt_b, padding=padding, stride=stride)
    assert out.memory_config().memory_layout == memory_layout
    got = ttnn.to_torch(out)
    assert list(got.shape) == list(ref.shape)
    p = pcc(got, ref)
    rel_rms = ((got.float() - ref).pow(2).mean().sqrt() / ref.std()).item()
    assert p >= 0.999 and rel_rms <= 0.02, f"pcc={p} rel_rms={rel_rms}"


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
