# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Standalone test for the Quasar "C=8" fold path — the change that lets the direct data-movement fold
(`ttnn.experimental.quasar.fold`, use_transpose_as_fold=False) feed the first conv WITHOUT the slow
per-group padding-strip tail (reshape/slice/reshape), which times out the resnet50 stem on the 2-core
emulator.

BACKGROUND
----------
Quasar row-major shards need a 16B-aligned page width, so bf16 fold channels must be a multiple of 8.
The resnet stem image is C=3; the fold pads it to an aligned width and gathers a stride_h*stride_w
window, producing `groups * C_aligned` output channels. With the historical align=4, that is
4*4 = 16, but the aligned width on Quasar is 8, so the direct fold produces 4*8 = 32 and then STRIPS
back to 16 (a reshape/slice/reshape over tens of thousands of narrow rows, minutes/op on the sim).

The fix mirrors what the WH/BH transpose fold already does — keep the aligned width and absorb the
padding channels as ZERO channels in the folded conv weights (pad_and_fold_conv_filters_for_unity_
stride at align_c=8 -> groups*8 = 32 input channels). Then the direct fold's natural 32-wide output
feeds conv1 unchanged (fold.cpp skips the strip when c_keep == c_aligned).

WHAT IT VALIDATES
-----------------
- test_c8_conv_weight_fold_equivalence (host-only): folding the first-conv weights AND the activation
  at align_c=8 reproduces the original 7x7/stride-2 conv bit-for-bit (same as align_c=4). This is the
  model-side correctness claim: the 5 zero pad channels/group contribute nothing.
- test_c8_direct_fold_no_strip (device): the direct fold with an output_shape whose C == groups*8
  returns the aligned [N,115,115,32] output directly (no strip), and it matches the CPU fold. Exercises
  the fold.cpp c_keep == c_aligned fast path.

RUN
---
  # host-only equivalence (no device):
  pytest models/demos/vision/classification/resnet50/quasar/tests/ops/test_fold_c8.py -k equivalence
  # device no-strip path:
  TT_METAL_SIMULATOR=~/sim/libttsim.so \
  TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_FORCE_JIT_COMPILE=1 \
  pytest models/demos/vision/classification/resnet50/quasar/tests/ops/test_fold_c8.py -k no_strip
"""

import pytest
import torch

import ttnn
from models.common.utility_functions import _nearest_y, pad_and_fold_conv_filters_for_unity_stride
from tests.ttnn.utils_for_testing import assert_with_pcc


def _cpu_fold_activation(activation_nchw, pad_h, pad_w, stride_h, stride_w, align_c):
    """CPU reference fold (NCHW in, NCHW folded out), parameterized by the channel alignment.

    Identical layout to models.common pad_and_fold_conv_filters_for_unity_stride: channels are padded
    to `align_c`, then the stride_h*stride_w window is packed group-interleaved as
    [grp0: align_c][grp1: align_c]..., real channels first then zero padding within each group.
    """
    assert stride_h == stride_w
    assert activation_nchw.shape[2] == activation_nchw.shape[3]
    C = _nearest_y(activation_nchw.shape[1], align_c)
    padded = torch.nn.functional.pad(activation_nchw, (pad_w, pad_w, pad_h, pad_h, 0, C - activation_nchw.shape[1]))
    assert padded.shape[2] % stride_h == 0
    folded = torch.zeros(
        [padded.shape[0], C * stride_h * stride_w, padded.shape[2] // stride_h, padded.shape[3] // stride_w]
    )
    for h in range(0, padded.shape[2], stride_h):
        for w in range(0, padded.shape[3], stride_w):
            fh, fw = h // stride_h, w // stride_w
            for i in range(stride_h * stride_w):
                start_c = i * C
                folded[:, start_c : start_c + C, fh, fw] = padded[:, :, h + i // stride_w, w + i % stride_w]
    return folded


def _fit_cores(total_rows, device):
    """Largest core count <= device cores that divides total_rows (so the height shards are exact)."""
    grid = device.compute_with_storage_grid_size()
    cap = min(total_rows, grid.x * grid.y)
    num_cores = cap
    while num_cores > 1 and total_rows % num_cores != 0:
        num_cores -= 1
    return num_cores, grid


# --------------------------------------------------------------------------------------------------
# Host-only: the align_c=8 weight+activation fold reproduces the original conv (model-side math).
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("align_c", [4, 8], ids=["align4", "align8"])
def test_c8_conv_weight_fold_equivalence(align_c):
    torch.manual_seed(0)
    # resnet50 stem conv: Conv2d(3, 64, kernel_size=7, stride=2, padding=3).
    c_in, c_out, k, stride, pad = 3, 64, 7, 2, 3
    x = torch.rand((1, c_in, 224, 224), dtype=torch.float32)
    weight = torch.rand((c_out, c_in, k, k), dtype=torch.float32)
    bias = torch.rand((c_out,), dtype=torch.float32)

    # Reference: the real strided conv.
    ref = torch.nn.functional.conv2d(x, weight, bias, stride=stride, padding=pad)

    # Folded: pad+fold weights and activation to `align_c`, then a unity-stride k'xk' conv.
    folded_w = pad_and_fold_conv_filters_for_unity_stride(weight, stride, stride, align_c=align_c)
    folded_x = _cpu_fold_activation(x, pad, pad, stride, stride, align_c=align_c)

    groups = stride * stride
    C_aligned = _nearest_y(c_in, align_c)
    assert folded_w.shape[1] == groups * C_aligned, folded_w.shape
    assert folded_x.shape[1] == groups * C_aligned, folded_x.shape

    folded_out = torch.nn.functional.conv2d(folded_x, folded_w, bias, stride=1, padding=0)

    assert folded_out.shape == ref.shape, (folded_out.shape, ref.shape)
    # Pure rearrangement of the same MACs -> bit-exact up to float rounding.
    torch.testing.assert_close(folded_out, ref, rtol=1e-4, atol=1e-4)


# --------------------------------------------------------------------------------------------------
# Device: the direct fold no-strip fast path (c_keep == c_aligned) returns the aligned output.
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("batch_size", [1], ids=["b1"])
@pytest.mark.timeout(900)
def test_c8_direct_fold_no_strip(mesh_device, batch_size):
    device = mesh_device
    torch.manual_seed(0)

    # resnet50 stem fold params, but aligned to 8 (Quasar C=8 path).
    c, h, w = 3, 224, 224
    kernel_size = 3
    stride_h = stride_w = 2
    pad_h = pad_w = kernel_size  # fold_pad_h/w = kernel_size = 3
    align_c = 8
    C = _nearest_y(c, align_c)  # 8
    pad_c = C - c  # 5
    groups = stride_h * stride_w  # 4

    torch_input = torch.rand((batch_size, c, h, w), dtype=torch.bfloat16)

    # Golden: CPU fold at align_c=8 -> [N, groups*8, 115, 115], permuted to NHWC to match device output.
    golden = _cpu_fold_activation(torch_input, pad_h, pad_w, stride_h, stride_w, align_c=align_c)
    golden = torch.permute(golden, (0, 2, 3, 1))  # [N, 115, 115, groups*8 = 32]
    out_c = groups * C  # 32

    # Upload channels-last [N,H,W,C] host-padded to the aligned width (Quasar RM shards need width % 8 == 0),
    # interleaved L1, so the direct fold skips the on-device NCHW->NHWC transpose (no Quasar kernel).
    torch_input_nhwc = torch_input.permute(0, 2, 3, 1).contiguous()  # [N,H,W,c]
    torch_input_nhwc = torch.nn.functional.pad(torch_input_nhwc, (0, C - c))  # C: c -> C_aligned(8)
    tt_input = ttnn.from_torch(
        torch_input_nhwc,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    # A device-sized shard grid (the fold reshards the halo/fold intermediates onto it).
    total_rows = batch_size * (h + pad_h * 2) // stride_h * ((w + pad_w * 2) // stride_w)
    num_cores, grid = _fit_cores(total_rows, device)
    shard_grid = ttnn.num_cores_to_corerangeset(num_cores, grid, row_wise=True)

    # output_shape C == groups*C_aligned == 32 -> c_keep == c_aligned -> fold skips the padding strip.
    tt_out = ttnn.experimental.quasar.fold(
        tt_input,
        stride_h,
        stride_w,
        use_transpose_as_fold=False,
        padding=[pad_h, pad_h, pad_w, pad_w, 0, pad_c],
        grid_size=shard_grid,
        input_is_nhwc=True,
        output_shape=ttnn.Shape([batch_size, 115, 115, out_c]),
    )

    got = ttnn.to_torch(tt_out).reshape(golden.shape).float()
    assert_with_pcc(golden.float(), got, 0.9999)
