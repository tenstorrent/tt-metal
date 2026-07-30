# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Standalone per-op PCC test for the Quasar resnet50 conv2d op.

This exercises ``ttnn.experimental.quasar.conv2d`` in isolation (no test_infra / no full-model
dependency) so the LLK team can iterate on / fix the conv kernels with a simple golden check.
The cases mirror the convs the resnet50/quasar model actually issues
(models/demos/vision/classification/resnet50/quasar/tt/ttnn_functional_resnet50.py):

  * STEM conv1        : 7x7, stride 2, pad 3, 3 -> 64, input 224x224 (HEIGHT_SHARDED)
  * bottleneck conv1  : 1x1, stride 1, pad 0, 64 -> 64            (mm_conv / matmul path)
  * bottleneck conv3  : 1x1, stride 1, pad 0, 64 -> 256           (mm_conv, channel expansion)
  * bottleneck conv2  : 3x3, stride 1, pad 1, 64 -> 64            (halo + conv_bmm_tilize path)
  * bottleneck conv2  : 3x3, stride 2, pad 1, 64 -> 64            (strided/downsample variant)
  * layer3 conv2      : 3x3, stride 1, pad 1, 256 -> 256, 14x14   (BLOCK_SHARDED, see caveat)

Dtype / fidelity match the model's batch-1 config (bfloat16 activations + weights, MathFidelity.LoFi;
see test_resnet50_functional.py). Golden is torch.nn.functional.conv2d.

LAYOUT CONVERSION (documented):
  - torch reference input is NCHW; ttnn conv2d takes an NHWC-flattened activation, so we permute
    NCHW -> NHWC and hand the op a row-major [N, H, W, C] host tensor (the op flattens to
    [1, 1, N*H*W, C] and shards it internally per conv_config.shard_layout).
  - the ttnn output is a (possibly tile-padded) NHWC-flattened tensor; we reshape it back to
    [N, out_h, out_w, C_padded], slice off the channel padding, and permute NHWC -> NCHW to line up
    with the torch golden before assert_with_pcc.

STEM NOTE: the real model folds the 224x224 image (fold + 4x4 stride-1 conv) instead of issuing a
plain 7x7/stride-2 conv, purely as an L1/perf transform. Here we issue the mathematically-equivalent
7x7 conv directly so the torch golden is trivial; this still drives the same stem conv kernels
(halo + conv_bmm_tilize) the LLK team cares about.

Run (craq-sim / emulator, slow dispatch + forced JIT):
  TT_METAL_SIMULATOR=~/sim/libttsim.so \
  TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_FORCE_JIT_COMPILE=1 \
  pytest models/demos/vision/classification/resnet50/quasar/tests/ops/test_conv2d.py

CAVEATS / known-problematic variants on Quasar:
  - The STEM conv (kernel 4x4/7x7, packer_l1_acc + fused bias) has hung in conv_bmm_tilize_metal2 on
    Quasar (3-thread MATH<->PACK<->UNPACK cycle over the matmul-partials DFB); see the sibling repro
    models/.../quasar/tests/test_conv_hang.py. If the stem case hangs rather than fails, that is the
    same deadlock.
  - The layer3/4 BLOCK_SHARDED convs (512->1024, 1024->2048) overflow the uint16_t weights-DFB ring
    unless the input is resharded to the optimal block layout; the model sets reshard_if_not_optimal
    for those on Quasar. The block-sharded case below uses reshard_if_not_optimal=True and a small
    (256->256, 14x14) shape so it fits a modest grid; the large layer4 shapes are intentionally not
    included here (they need the full 32-core part to fit L1).
  - On a tiny emulator grid (1-2 cores) the per-core height shard for the 224x224 stem / 56x56
    bottleneck cases can OOM L1; the full 32-core Quasar grid is the intended target.
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

# bf16 + MathFidelity.LoFi -> looser PCC (matches the model's batch-1 LoFi config).
PCC = 0.97


def _run_conv2d(
    mesh_device,
    *,
    batch_size,
    in_channels,
    out_channels,
    input_height,
    input_width,
    kernel_size,
    stride,
    padding,
    shard_layout,
    reshard_if_not_optimal=False,
    pcc=PCC,
):
    torch.manual_seed(0)
    device = mesh_device

    kh, kw = kernel_size

    # --- torch golden (NCHW) ---
    torch_input_nchw = torch.randn((batch_size, in_channels, input_height, input_width), dtype=torch.bfloat16).float()
    torch_weight = torch.randn((out_channels, in_channels, kh, kw), dtype=torch.bfloat16).float()
    torch_bias = torch.randn((1, 1, 1, out_channels), dtype=torch.bfloat16).float()

    torch_golden = torch.nn.functional.conv2d(
        torch_input_nchw,
        torch_weight,
        bias=torch_bias.reshape(-1),
        stride=stride,
        padding=padding,
    )

    # --- ttnn inputs: NCHW -> NHWC row-major host activation; op flattens + shards internally ---
    torch_input_nhwc = torch.permute(torch_input_nchw, (0, 2, 3, 1))
    tt_input = ttnn.from_torch(torch_input_nhwc, dtype=ttnn.bfloat16)
    tt_weight = ttnn.from_torch(torch_weight, dtype=ttnn.bfloat16)
    tt_bias = ttnn.from_torch(torch_bias, dtype=ttnn.bfloat16)

    conv_config = ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat16,
        shard_layout=shard_layout,
        reshard_if_not_optimal=reshard_if_not_optimal,
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        packer_l1_acc=True,
    )

    out, [out_h, out_w], [tt_weight, tt_bias] = ttnn.experimental.quasar.conv2d(
        input_tensor=tt_input,
        weight_tensor=tt_weight,
        bias_tensor=tt_bias,
        in_channels=in_channels,
        out_channels=out_channels,
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=(1, 1),
        groups=1,
        device=device,
        conv_config=conv_config,
        compute_config=compute_config,
        return_output_dim=True,
        return_weights_and_bias=True,
        dtype=ttnn.bfloat16,
    )

    # --- de-shard / un-tile back to torch, then NHWC-flattened -> NCHW ---
    tt_out = ttnn.to_torch(ttnn.from_device(out))
    tt_out = tt_out.reshape(batch_size, out_h, out_w, tt_out.shape[-1])
    tt_out = tt_out[:, :, :, :out_channels]  # drop channel (tile) padding
    tt_out = torch.permute(tt_out, (0, 3, 1, 2))  # NHWC -> NCHW

    assert_with_pcc(torch_golden, tt_out.float(), pcc=pcc)


# The stem conv routes through conv_bmm_tilize which currently deadlocks on Quasar (see test_conv_hang.py);
# cap the run so that hang surfaces as a timeout, not a suite block.
@pytest.mark.timeout(600)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, in_channels, out_channels, input_height, input_width, kernel_size, stride, padding, shard_layout, reshard_if_not_optimal",
    [
        # STEM conv1 (logical 7x7/stride-2 equivalent of the model's fold+4x4 stem).
        pytest.param(
            1,
            3,
            64,
            224,
            224,
            (7, 7),
            (2, 2),
            (3, 3),
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            False,
            id="stem_7x7_s2_3to64_224",
        ),
        # bottleneck conv1: 1x1, 64->64 (matmul / mm_conv path).
        pytest.param(
            1,
            64,
            64,
            56,
            56,
            (1, 1),
            (1, 1),
            (0, 0),
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            False,
            id="bottleneck_1x1_64to64_56",
        ),
        # bottleneck conv3: 1x1, 64->256 (channel-expansion matmul).
        pytest.param(
            1,
            64,
            256,
            56,
            56,
            (1, 1),
            (1, 1),
            (0, 0),
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            False,
            id="bottleneck_1x1_64to256_56",
        ),
        # bottleneck conv2: 3x3, 64->64, stride 1 (halo + conv_bmm_tilize).
        pytest.param(
            1,
            64,
            64,
            56,
            56,
            (3, 3),
            (1, 1),
            (1, 1),
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            False,
            id="bottleneck_3x3_64to64_s1_56",
        ),
        # bottleneck conv2: 3x3, 64->64, stride 2 (strided / downsample variant, 56 -> 28).
        pytest.param(
            1,
            64,
            64,
            56,
            56,
            (3, 3),
            (2, 2),
            (1, 1),
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            False,
            id="bottleneck_3x3_64to64_s2_56",
        ),
        # layer3-style BLOCK_SHARDED 3x3 conv (small stand-in; reshard_if_not_optimal like the model).
        pytest.param(
            1,
            256,
            256,
            14,
            14,
            (3, 3),
            (1, 1),
            (1, 1),
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            True,
            id="layer3_3x3_256to256_block_14",
        ),
    ],
)
def test_quasar_conv2d(
    mesh_device,
    batch_size,
    in_channels,
    out_channels,
    input_height,
    input_width,
    kernel_size,
    stride,
    padding,
    shard_layout,
    reshard_if_not_optimal,
):
    # Emulator guard: conv2d auto-sizes its shard grid, so on a small grid (the 2-core emulator) a
    # large-spatial conv gets a huge per-core activation shard (activation + halo + weights + interm CBs)
    # and OOMs L1 -- e.g. the 224x224 stem = 50176 sticks / 2 cores = 25088/core. Skip when the grid can't
    # spread it thin enough; the full 32-core Quasar part (50176/32 = 1568/core) still runs every case.
    grid = mesh_device.compute_with_storage_grid_size()
    max_cores = grid.x * grid.y
    sticks = batch_size * input_height * input_width
    MAX_STICKS_PER_CORE = 2048
    if sticks / max_cores > MAX_STICKS_PER_CORE:
        pytest.skip(
            f"conv activation = {sticks} sticks over {max_cores} cores = {sticks // max_cores}/core "
            f"(> {MAX_STICKS_PER_CORE}); OOMs L1 on this small grid -- run on the full Quasar part."
        )
    _run_conv2d(
        mesh_device,
        batch_size=batch_size,
        in_channels=in_channels,
        out_channels=out_channels,
        input_height=input_height,
        input_width=input_width,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        shard_layout=shard_layout,
        reshard_if_not_optimal=reshard_if_not_optimal,
    )
