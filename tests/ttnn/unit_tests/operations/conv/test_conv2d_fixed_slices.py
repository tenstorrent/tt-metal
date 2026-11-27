# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""conv2d with fixed 2D DRAM slices (Conv2dSliceConfig(slice_height=..., slice_width=...)): correct against torch at
several input sizes in one process, and, after the first size, without new program-cache entries for the same layer."""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

COMPUTE_CONFIG = dict(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=False)


def _run(device, x: torch.Tensor, weight, bias, *, out_channels, padding, slice_hw, shard_layout):
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
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=padding,
        batch_size=n,
        input_height=h,
        input_width=w,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat16,
            shard_layout=shard_layout,
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


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "in_channels, out_channels, shard_layout, slice_hw",
    [
        (64, 64, ttnn.TensorMemoryLayout.BLOCK_SHARDED, (32, 64)),
        (128, 128, ttnn.TensorMemoryLayout.BLOCK_SHARDED, (48, 96)),
        (64, 3, ttnn.TensorMemoryLayout.HEIGHT_SHARDED, (64, 64)),
    ],
    ids=["64-bs", "128-bs", "64to3-hs"],
)
# 0: the input comes padded (as after a halo exchange); 1: the conv pads, so the first slices along each axis have
# programs of their own.
@pytest.mark.parametrize("padding", [0, 1], ids=["pad0", "pad1"])
def test_conv2d_fixed_slices(device, in_channels, out_channels, shard_layout, slice_hw, padding):
    torch.manual_seed(0)
    # Output sizes (pad 0: the input is 2 more per axis). The first has at least two slices per axis and a width that
    # is not whole tiles, so it builds every program the layer has (with padding, the slices at the top and left edge
    # have their own). Then one slice exactly, rows and columns not whole slices or tiles, and smaller than one slice
    # in both axes.
    out_sizes = [(103, 200), (slice_hw[0], slice_hw[1]), (70, 138), (40, 74), (17, 20), (103, 200)]
    weight = torch.randn(out_channels, in_channels, 3, 3) / (3 * in_channels**0.5)
    bias = torch.randn(1, 1, 1, out_channels)
    tt_weight = ttnn.from_torch(weight, dtype=ttnn.bfloat16)
    tt_bias = ttnn.from_torch(bias, dtype=ttnn.bfloat16)

    entries = []
    for oh, ow in out_sizes:
        h, w = oh + 2 - 2 * padding, ow + 2 - 2 * padding
        x = torch.randn(1, in_channels, h, w)
        before = device.num_program_cache_entries()
        result, tt_weight, tt_bias = _run(
            device,
            x,
            tt_weight,
            tt_bias,
            padding=(padding, padding),
            slice_hw=slice_hw,
            out_channels=out_channels,
            shard_layout=shard_layout,
        )
        entries.append(device.num_program_cache_entries() - before)
        expected = torch.nn.functional.conv2d(x, weight, bias.reshape(-1), padding=padding)
        assert result.shape == expected.shape
        assert_with_pcc(expected, result.float(), 0.999)
    print(f"new program-cache entries per size: {entries}")
    # The first size builds the layer's programs and prepares the weights. The output smaller than one slice reuses
    # the slices' programs, but its flatten has fewer output tiles than there are cores, which sets the flatten's core
    # set and so adds one entry.
    assert entries[1:] == [0, 0, 0, 1, 0], entries
