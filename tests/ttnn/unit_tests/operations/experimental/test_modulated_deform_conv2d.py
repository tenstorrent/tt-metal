# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""ttnn.experimental.modulated_deform_conv2d against a float64 host reference: bilinear columns
times the weight, with torchvision's deform_conv2d pinning the reference's (kk, c) column order."""

import math

import pytest
import torch
import torch.nn.functional as F
import torchvision
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc

# Relative error along the sign of the reference, which PCC cannot see: a uniform scale error. The
# op's HiFi2 products and its bfloat16 sum over kernel positions shrink outputs by about 0.55%; a
# coefficient truncated instead of rounded to bfloat16 adds about 0.2% more.
MAX_RELATIVE_BIAS = 1e-2


def out_size(size, kernel, stride, padding, dilation):
    return (size + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1


def reference_columns(x, offset, mask, kernel_size, stride, padding, dilation):
    """``x`` (B, H, W, C), ``offset`` (B, Ho, Wo, 2*K*K) as (y, x) pixels, ``mask`` (B, Ho, Wo, K*K)
    -> (B*Ho*Wo, K*K*C), bilinear with zero padding outside the input."""
    batch, in_h, in_w, _ = x.shape
    _, out_h, out_w, _ = offset.shape
    k_h, k_w = kernel_size
    h_o = torch.arange(out_h, dtype=x.dtype).view(1, -1, 1)
    w_o = torch.arange(out_w, dtype=x.dtype).view(1, 1, -1)
    columns = []
    for kk in range(k_h * k_w):
        y = h_o * stride[0] - padding[0] + (kk // k_w) * dilation[0] + offset[..., 2 * kk]
        x_ = w_o * stride[1] - padding[1] + (kk % k_w) * dilation[1] + offset[..., 2 * kk + 1]
        # grid_sample with align_corners=False puts pixel p at (2p + 1) / size - 1.
        grid = torch.stack([(2 * x_ + 1) / in_w - 1, (2 * y + 1) / in_h - 1], dim=-1)
        sampled = F.grid_sample(x.permute(0, 3, 1, 2), grid, mode="bilinear", align_corners=False)
        columns.append(sampled.permute(0, 2, 3, 1) * mask[..., kk : kk + 1])
    return torch.cat(columns, dim=-1).reshape(batch * out_h * out_w, -1)


def random_inputs(batch, in_h, in_w, channels, c_out, kernel_size, stride, padding, dilation, offset_scale):
    """Inputs rounded to bfloat16 first, so the reference sees the device's values: the input, the
    offsets, the mask logits, the (K*K*C, C_out) weight and the bias."""
    torch.manual_seed(0)
    k_h, k_w = kernel_size
    out_h = out_size(in_h, k_h, stride[0], padding[0], dilation[0])
    out_w = out_size(in_w, k_w, stride[1], padding[1], dilation[1])
    kk = k_h * k_w
    x = torch.randn(batch, in_h, in_w, channels).bfloat16().double()
    offset = (offset_scale * torch.randn(batch, out_h, out_w, 2 * kk)).bfloat16().double()
    logits = (2.0 * torch.randn(batch, out_h, out_w, kk)).bfloat16().double()
    weight = (torch.randn(kk * channels, c_out) / (kk * channels) ** 0.5).bfloat16().double()
    bias = torch.randn(1, c_out).bfloat16().double()
    return x, offset, logits, weight, bias


def grid_shard_rows(device, rows):
    """Rows per core for ``rows`` height-sharded evenly over the whole grid in whole 32-row blocks."""
    grid = device.compute_with_storage_grid_size()
    return math.ceil(rows / (grid.x * grid.y) / 32) * 32


def height_sharded(device, tensor, shard_rows):
    """``tensor`` (B, H, W, C) height-sharded in L1 over the whole grid, ``shard_rows`` rows per core."""
    batch, height, width, channels = tensor.shape
    grid = device.compute_with_storage_grid_size()
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    spec = ttnn.ShardSpec(cores, [shard_rows, channels], ttnn.ShardOrientation.ROW_MAJOR)
    sharded = ttnn.to_memory_config(
        ttnn.reshape(tensor, (1, 1, batch * height * width, channels)),
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, spec),
    )
    return ttnn.reshape(sharded, (batch, height, width, channels))


def device_inputs(device, x, offset, logits, weight, bias, shard_rows=None):
    """The op's input, offset_mask, weight and bias on ``device``; ``bias`` may be None. With
    ``shard_rows``, the input goes height-sharded with that many rows per core."""
    row_major = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tiled = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_x = row_major(x) if shard_rows is None else height_sharded(device, row_major(x), shard_rows)
    return tt_x, row_major(torch.cat([offset, logits], dim=-1)), tiled(weight), None if bias is None else tiled(bias)


def call_conv(inputs, kernel_size, stride, padding, dilation, relu):
    return ttnn.experimental.modulated_deform_conv2d(
        *inputs, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, relu=relu
    )


def output_rows(out, offset, weight):
    """The op's output as a (B*Ho*Wo, C_out) float64 tensor."""
    rows = offset.shape[0] * offset.shape[1] * offset.shape[2]
    assert tuple(out.shape) == (1, 1, rows, weight.shape[1])
    return ttnn.to_torch(out).double().reshape(rows, -1)


def run_conv(device, x, offset, logits, weight, bias, kernel_size, stride, padding, dilation, relu, shard_rows=None):
    """The op's output as a (B*Ho*Wo, C_out) float64 tensor; see ``device_inputs``."""
    inputs = device_inputs(device, x, offset, logits, weight, bias, shard_rows)
    out = call_conv(inputs, kernel_size, stride, padding, dilation, relu)
    return output_rows(out, offset, weight)


def assert_close(expected, actual):
    passed, pcc = comp_pcc(expected, actual, 0.999)
    assert torch.isfinite(actual).all()
    assert passed, f"PCC {pcc}"
    bias = ((actual - expected) * expected.sign()).mean() / expected.abs().mean()
    logger.info(f"PCC {pcc}, relative bias {bias:.2e}")
    assert abs(bias) < MAX_RELATIVE_BIAS, f"relative bias {bias:.2e}"


@pytest.mark.parametrize(
    "batch, in_h, in_w, channels, c_out, has_bias, relu, offset_scale, kernel_size, stride, padding, dilation",
    [
        (1, 8, 8, 32, 32, False, False, 2.0, (3, 3), (1, 1), (1, 1), (1, 1)),  # corners pushed outside
        (2, 5, 7, 64, 96, True, False, 1.0, (3, 3), (1, 1), (1, 1), (1, 1)),  # a partial last block
        (1, 24, 24, 32, 64, True, True, 20.0, (3, 3), (1, 1), (1, 1), (1, 1)),  # many corners far outside
        (1, 20, 40, 256, 256, True, True, 4.0, (3, 3), (1, 1), (1, 1), (1, 1)),
        # C_out over several 4-tile dest halves: 320 is two full ones and a partial last, 512 four full.
        (1, 16, 20, 64, 320, True, True, 2.0, (3, 3), (1, 1), (1, 1), (1, 1)),
        (1, 16, 20, 64, 512, True, False, 2.0, (3, 3), (1, 1), (1, 1), (1, 1)),
        (1, 19, 21, 64, 64, True, False, 1.0, (3, 3), (2, 2), (1, 1), (1, 1)),
        (1, 16, 16, 32, 64, False, True, 1.0, (3, 3), (1, 1), (2, 2), (2, 2)),
        (1, 6, 10, 96, 64, True, False, 1.0, (1, 3), (1, 1), (0, 1), (1, 1)),
        # Inputs over one 8-tile chunk of channels: several passes per kernel position.
        (1, 12, 20, 512, 64, True, True, 4.0, (3, 3), (1, 1), (1, 1), (1, 1)),
        (1, 10, 14, 384, 96, True, False, 2.0, (3, 3), (1, 1), (1, 1), (1, 1)),
        (1, 10, 14, 320, 64, False, True, 2.0, (3, 3), (1, 1), (1, 1), (1, 1)),
        (6, 58, 100, 256, 256, True, True, 4.0, (3, 3), (1, 1), (1, 1), (1, 1)),  # BEVFormer-base layer3
        (6, 29, 50, 512, 512, True, True, 4.0, (3, 3), (1, 1), (1, 1), (1, 1)),  # BEVFormer-base layer4
    ],
)
def test_modulated_deform_conv2d(
    device, batch, in_h, in_w, channels, c_out, has_bias, relu, offset_scale, kernel_size, stride, padding, dilation
):
    x, offset, logits, weight, bias = random_inputs(
        batch, in_h, in_w, channels, c_out, kernel_size, stride, padding, dilation, offset_scale
    )
    bias = bias if has_bias else None
    actual = run_conv(device, x, offset, logits, weight, bias, kernel_size, stride, padding, dilation, relu)
    expected = reference_columns(x, offset, logits.sigmoid(), kernel_size, stride, padding, dilation) @ weight
    if bias is not None:
        expected = expected + bias
    if relu:
        expected = expected.clamp(min=0)
    assert_close(expected, actual)


@pytest.mark.parametrize(
    "batch, in_h, in_w, channels, c_out, shard_rows",
    [
        # 10 shards: most cores have no rows and only join the weight multicast.
        (1, 16, 20, 64, 64, 32),
        (6, 58, 100, 256, 256, None),  # BEVFormer-base layer3, as the model shards it
        (6, 29, 50, 512, 512, None),  # two channel passes per kernel position
    ],
)
def test_modulated_deform_conv2d_sharded_input(device, batch, in_h, in_w, channels, c_out, shard_rows):
    """A height-sharded input gives the interleaved input's output exactly: only where the corners
    are read from changes. ``shard_rows`` None shards the rows as the model does, evenly over the
    whole grid in whole 32-row blocks."""
    if shard_rows is None:
        shard_rows = grid_shard_rows(device, batch * in_h * in_w)
    conv = ((3, 3), (1, 1), (1, 1), (1, 1))
    x, offset, logits, weight, bias = random_inputs(batch, in_h, in_w, channels, c_out, *conv, 4.0)
    interleaved = run_conv(device, x, offset, logits, weight, bias, *conv, True)
    sharded = run_conv(device, x, offset, logits, weight, bias, *conv, True, shard_rows=shard_rows)
    assert torch.equal(sharded, interleaved)
    expected = (reference_columns(x, offset, logits.sigmoid(), *conv) @ weight + bias).clamp(min=0)
    assert_close(expected, sharded)


def test_modulated_deform_conv2d_weight_multicast(device):
    """The blocks fill two whole columns of the grid, a rectangle short of the grid, so one core
    multicasts each weight slice to those cores only."""
    grid = device.compute_with_storage_grid_size()
    conv = ((3, 3), (1, 1), (1, 1), (1, 1))
    # 16 rows of 4 * grid.y pixels: 2 * grid.y blocks of 32 output rows, which the interleaved
    # split lays out column by column.
    x, offset, logits, weight, bias = random_inputs(1, 16, 4 * grid.y, 64, 64, *conv, 1.0)
    actual = run_conv(device, x, offset, logits, weight, bias, *conv, True)
    expected = (reference_columns(x, offset, logits.sigmoid(), *conv) @ weight + bias).clamp(min=0)
    assert_close(expected, actual)


def test_modulated_deform_conv2d_matches_torchvision(device):
    """The op is torchvision's modulated deformable conv, with the weight laid out (kk, c) x C_out."""
    kernel_size, stride, padding, dilation = (3, 3), (1, 1), (1, 1), (1, 1)
    x, offset, logits, _, bias = random_inputs(2, 16, 20, 64, 64, kernel_size, stride, padding, dilation, 3.0)
    torch_weight = (torch.randn(64, 64, 3, 3) / 24).bfloat16().double()
    expected = torchvision.ops.deform_conv2d(
        x.permute(0, 3, 1, 2),
        offset.permute(0, 3, 1, 2),
        torch_weight,
        bias[0],
        stride=stride,
        padding=padding,
        dilation=dilation,
        mask=logits.sigmoid().permute(0, 3, 1, 2),
    )
    weight = torch_weight.permute(2, 3, 1, 0).reshape(-1, torch_weight.shape[0])
    actual = run_conv(device, x, offset, logits, weight, bias, kernel_size, stride, padding, dilation, False)
    assert_close(expected.permute(0, 2, 3, 1).reshape(actual.shape), actual)


@pytest.mark.parametrize("sharded", [False, True])
def test_modulated_deform_conv2d_cache_hit_fresh_buffers(device, sharded):
    """A second call of the same specification on other tensors, the first call's still live, reuses
    the cached program and reads and writes the new tensors' buffers."""
    conv = ((3, 3), (1, 1), (1, 1), (1, 1))
    cases = [random_inputs(1, 16, 20, 64, 64, *conv, 2.0) for _ in range(2)]
    shard_rows = grid_shard_rows(device, 16 * 20) if sharded else None
    inputs = [device_inputs(device, *case, shard_rows) for case in cases]
    outs, entries = [], []
    for case_inputs in inputs:
        outs.append(call_conv(case_inputs, *conv, True))
        entries.append(device.num_program_cache_entries())
    assert entries[1] == entries[0], f"the second call added {entries[1] - entries[0]} program cache entries"
    for (x, offset, logits, weight, bias), out in zip(cases, outs):
        expected = (reference_columns(x, offset, logits.sigmoid(), *conv) @ weight + bias).clamp(min=0)
        assert_close(expected, output_rows(out, offset, weight))


def test_modulated_deform_conv2d_less_l1_misses_cache(device):
    """Less free L1 gives the op a smaller budget, which keys the program cache: the program sized for
    more L1 is not reused, and the new one gives the same output."""
    conv = ((3, 3), (1, 1), (1, 1), (1, 1))
    x, offset, logits, weight, bias = random_inputs(1, 16, 20, 64, 64, *conv, 2.0)
    inputs = device_inputs(device, x, offset, logits, weight, bias)
    first = output_rows(call_conv(inputs, *conv, True), offset, weight)
    entries = device.num_program_cache_entries()
    # 128 KB per L1 bank, several times the budget's granule, written from the host so that no
    # program of its own enters the cache.
    grid = device.compute_with_storage_grid_size()
    channels = 1024
    filler = ttnn.from_torch(
        torch.zeros(1, 1, 128 * 1024 * grid.x * grid.y // (channels * 2), channels),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    out = call_conv(inputs, *conv, True)
    assert device.num_program_cache_entries() > entries, "the smaller budget reused the cached program"
    assert torch.equal(output_rows(out, offset, weight), first)
    ttnn.deallocate(filler)


def test_modulated_deform_conv2d_l1_too_small(device, expect_error):
    """Each core keeps all its blocks' outputs and two weight slices; past L1 the op says so."""
    x, offset, logits, weight, bias = random_inputs(6, 58, 100, 256, 1024, (3, 3), (1, 1), (1, 1), (1, 1), 1.0)
    with expect_error(RuntimeError, "needs .* KB of L1 per core"):
        run_conv(device, x, offset, logits, weight, bias, (3, 3), (1, 1), (1, 1), (1, 1), False)


def test_modulated_deform_conv2d_kernel_too_large(device, expect_error):
    x, offset, logits, weight, bias = random_inputs(1, 16, 16, 32, 32, (7, 7), (1, 1), (3, 3), (1, 1), 1.0)
    with expect_error(RuntimeError, "more than 32 positions"):
        run_conv(device, x, offset, logits, weight, None, (7, 7), (1, 1), (3, 3), (1, 1), False)
