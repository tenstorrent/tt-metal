# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""conv2d with fixed 2D DRAM slices (Conv2dSliceConfig(slice_height=..., slice_width=...)): correct against torch at
several input sizes in one process, and, after the first size, without new program-cache entries for the same layer."""

import math

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

COMPUTE_CONFIG = dict(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=False)
TILE = 32


def _run(device, x: torch.Tensor, weight, bias, *, out_channels, kernel, stride, dilation, padding, slice_hw, shard):
    n, c, h, w = x.shape
    tt_x = ttnn.from_torch(
        x.permute(0, 2, 3, 1).contiguous(),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out, (out_h, out_w), (weight, bias) = ttnn.conv2d(
        input_tensor=tt_x,
        weight_tensor=weight,
        bias_tensor=bias,
        in_channels=c,
        out_channels=out_channels,
        device=device,
        kernel_size=kernel,
        stride=stride,
        dilation=dilation,
        padding=padding,
        batch_size=n,
        input_height=h,
        input_width=w,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat16,
            shard_layout=shard,
            act_block_h_override=32,
            config_tensors_in_dram=True,
        ),
        compute_config=ttnn.init_device_compute_kernel_config(device.arch(), **COMPUTE_CONFIG),
        slice_config=ttnn.Conv2dSliceConfig(slice_height=slice_hw[0], slice_width=slice_hw[1]),
        return_output_dim=True,
        return_weights_and_bias=True,
    )
    result = ttnn.to_torch(out).reshape(n, out_h, out_w, -1)[..., :out_channels].permute(0, 3, 1, 2)
    ttnn.deallocate(out)
    ttnn.deallocate(tt_x)
    return result, weight, bias


BS = ttnn.TensorMemoryLayout.BLOCK_SHARDED
HS = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
# padding is (top, bottom, left, right). Padding 0 stands for an input that comes padded (as after a halo exchange);
# with padding, the first slices along each axis have programs of their own.
BASE = dict(n=1, kernel=(3, 3), stride=(1, 1), dilation=(1, 1), bias=True)
CASES = {
    "64-bs-pad0": dict(BASE, c=(64, 64), shard=BS, slice_hw=(32, 64), padding=(0, 0, 0, 0)),
    "64-bs-pad1": dict(BASE, c=(64, 64), shard=BS, slice_hw=(32, 64), padding=(1, 1, 1, 1)),
    "128-bs-pad0": dict(BASE, c=(128, 128), shard=BS, slice_hw=(48, 96), padding=(0, 0, 0, 0)),
    "128-bs-pad1": dict(BASE, c=(128, 128), shard=BS, slice_hw=(48, 96), padding=(1, 1, 1, 1)),
    "64to3-hs-pad0": dict(BASE, c=(64, 3), shard=HS, slice_hw=(64, 64), padding=(0, 0, 0, 0)),
    "64to3-hs-pad1": dict(BASE, c=(64, 3), shard=HS, slice_hw=(64, 64), padding=(1, 1, 1, 1)),
    "stride2": dict(BASE, c=(64, 64), shard=BS, slice_hw=(32, 64), stride=(2, 2), padding=(1, 1, 1, 1)),
    "dilation2": dict(BASE, c=(64, 64), shard=BS, slice_hw=(32, 64), dilation=(2, 2), padding=(2, 2, 2, 2)),
    "batch2": dict(BASE, c=(64, 64), shard=BS, slice_hw=(32, 64), n=2, padding=(1, 1, 1, 1)),
    # Padding wider than the stride: the last slice, shifted back to end at the edge, would start in the padding.
    "5x5-pad2": dict(BASE, c=(64, 64), shard=BS, slice_hw=(32, 64), kernel=(5, 5), padding=(2, 2, 2, 2)),
    "asym-pad": dict(BASE, c=(64, 64), shard=BS, slice_hw=(32, 64), padding=(2, 0, 0, 1)),
    "no-bias": dict(BASE, c=(64, 64), shard=BS, slice_hw=(32, 64), padding=(1, 1, 1, 1), bias=False),
}


def _flatten_key(n: int, out_h: int, out_w: int, out_channels: int, num_cores: int) -> int:
    """The core-set part of the key of conv2d's final flatten to [1, 1, n * h * w, c]: it runs on every core once
    there are at least as many output tiles as cores, otherwise on one core per tile."""
    pages = math.ceil(n * out_h * out_w / TILE) * math.ceil(out_channels / TILE)
    return 0 if pages >= num_cores else pages


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
def test_conv2d_fixed_slices(device, case):
    torch.manual_seed(0)
    in_channels, out_channels = case["c"]
    kernel, stride, dilation, padding = case["kernel"], case["stride"], case["dilation"], case["padding"]
    slice_h, slice_w = case["slice_hw"]
    n = case["n"]
    # Output sizes. The first has at least two slices per axis and a width that is not whole tiles, so it builds
    # every program the layer has. Then one slice exactly, one more than a slice (the last slice shifted back
    # almost to the start), rows and columns not whole slices or tiles, and smaller than one slice in both axes.
    out_sizes = [
        (103, 200),
        (slice_h, slice_w),
        (slice_h + 1, slice_w + 1),
        (70, 138),
        (40, 74),
        (17, 20),
        (103, 200),
    ]
    weight = (torch.randn(out_channels, in_channels, *kernel) / (math.prod(kernel) * in_channels) ** 0.5).bfloat16()
    bias = torch.randn(1, 1, 1, out_channels).bfloat16() if case["bias"] else None
    tt_weight = ttnn.from_torch(weight, dtype=ttnn.bfloat16)
    tt_bias = ttnn.from_torch(bias, dtype=ttnn.bfloat16) if bias is not None else None

    grid = device.compute_with_storage_grid_size()
    num_cores = grid.x * grid.y
    entries = []
    for oh, ow in out_sizes:
        h = (oh - 1) * stride[0] + dilation[0] * (kernel[0] - 1) + 1 - padding[0] - padding[1]
        w = (ow - 1) * stride[1] + dilation[1] * (kernel[1] - 1) + 1 - padding[2] - padding[3]
        x = torch.randn(n, in_channels, h, w).bfloat16()
        before = device.num_program_cache_entries()
        result, tt_weight, tt_bias = _run(
            device,
            x,
            tt_weight,
            tt_bias,
            out_channels=out_channels,
            kernel=kernel,
            stride=stride,
            dilation=dilation,
            padding=padding,
            slice_hw=case["slice_hw"],
            shard=case["shard"],
        )
        entries.append(device.num_program_cache_entries() - before)
        padded = torch.nn.functional.pad(x.float(), (padding[2], padding[3], padding[0], padding[1]))
        expected = torch.nn.functional.conv2d(
            padded,
            weight.float(),
            bias.reshape(-1).float() if bias is not None else None,
            stride=stride,
            dilation=dilation,
        )
        assert result.shape == expected.shape == (n, out_channels, oh, ow)
        result = result.float()
        assert_with_pcc(expected, result, 0.999)
        # Elementwise, so that one wrong border row or column can't hide in the PCC. The conv's own error is spread
        # evenly, up to about 1.3% of the output's largest magnitude (5x5 with 64 channels); a pixel computed from
        # the wrong window or padding is off by about the output's typical magnitude, several times more.
        scale = expected.abs().max().item()
        error = (result - expected).abs()
        assert torch.all(error <= 0.02 * scale), (
            f"size {oh}x{ow}: max error {error.max().item():.4f} at {torch.nonzero(error == error.max())[0].tolist()}, "
            f"output scale {scale:.3f}"
        )
    print(f"new program-cache entries per size: {entries}")
    # The first size builds the layer's programs and prepares the weights. After it, the slices' programs are reused,
    # and only the final flatten can add an entry: its core set follows the output tile count when there are fewer
    # tiles than cores, so each such new tile count adds one.
    seen = {_flatten_key(n, *out_sizes[0], out_channels, num_cores)}
    expected_entries = []
    for oh, ow in out_sizes[1:]:
        key = _flatten_key(n, oh, ow, out_channels, num_cores)
        expected_entries.append(0 if key in seen else 1)
        seen.add(key)
    assert entries[1:] == expected_entries, (entries, expected_entries)
