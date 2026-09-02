# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Standalone test for the Quasar BLOCK_SHARDED conv2d path.

WHERE IT COMES FROM
-------------------
The resnet layer1 bottleneck conv2 (3x3, 64->64, stride 1, pad 1) runs BLOCK_SHARDED. On the emulator's
small (single-row / 2x1) core grid it crashed at PROGRAM CREATION:

    RuntimeError: No core coordinate found at location: (1, 1, TENSIX, LOGICAL)
    ... Conv2dShardedProgramFactory::create_program_artifacts

Cause: the block-sharded weights-mcast corner `top_left_core_plus_one = grid_start + {1,1}` was resolved
unconditionally, but on a degenerate grid (only 1 core in x or y) that names a logical core that does not
exist (e.g. (1,1) on a 2x1 grid). FIX (conv2d_op_sharded_program_factory.cpp): clamp the +1 per dimension
so it only steps into a dim that spans >1 core (mirrors the matmul 2D-mcast single-row/col fix). On a full
2D grid the corner is unchanged.

WHAT THIS VALIDATES
-------------------
This forces the BLOCK_SHARDED path (shard_layout=BLOCK_SHARDED + reshard_if_not_optimal) with the exact
layer1.conv2 shape, so it exercises the clamped corner on the emulator's grid. Before the fix it throws the
"No core at (1,1)" error at program creation; after the fix it builds and the conv output must match a torch
golden (PCC ~1.0).

RUN (emulator, forced JIT):
  TT_METAL_FORCE_JIT_COMPILE=1 \
    pytest -q models/demos/vision/classification/resnet50/quasar/tests/ops/test_conv2d_block_sharded.py
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("with_bias_relu", [False, True], ids=["pure", "bias_relu"])
def test_quasar_conv2d_block_sharded(mesh_device, with_bias_relu):
    """layer1.conv2 shape (3x3, 64->64, stride1, pad1) on the BLOCK_SHARDED path — exercises the clamped
    degenerate-grid weights-mcast corner. RELU + bias mirrors the model's folded-BN conv2."""
    device = mesh_device
    torch.manual_seed(0)

    batch_size = 1
    in_channels = 64
    out_channels = 64
    kernel_size = (3, 3)
    stride = (1, 1)
    padding = (1, 1)
    # resnet layer1 spatial (224 -> stem/2 -> 112 -> maxpool/2 -> 56). stride1+pad1+k3 keeps H,W.
    input_height = 56
    input_width = 56

    torch_input_nchw = torch.randn((batch_size, in_channels, input_height, input_width), dtype=torch.bfloat16).float()
    torch_weight = torch.randn((out_channels, in_channels, *kernel_size), dtype=torch.bfloat16).float()
    torch_bias = torch.randn((out_channels,), dtype=torch.bfloat16).float() if with_bias_relu else None
    torch_golden = torch.nn.functional.conv2d(
        torch_input_nchw, torch_weight, bias=torch_bias, stride=stride, padding=padding
    )
    if with_bias_relu:
        torch_golden = torch.relu(torch_golden)

    # Pre-shard the activation HEIGHT_SHARDED into L1; the conv reshards to BLOCK_SHARDED
    # (reshard_if_not_optimal), which is the path that hit the degenerate-grid mcast corner.
    nhw = batch_size * input_height * input_width
    flat = torch.permute(torch_input_nchw, (0, 2, 3, 1)).reshape(1, 1, nhw, in_channels).contiguous()
    grid = device.compute_with_storage_grid_size()
    max_cores = grid.x * grid.y
    num_cores = max(c for c in range(1, max_cores + 1) if nhw % c == 0)
    shard_h = nhw // num_cores
    core_grid = ttnn.num_cores_to_corerangeset(num_cores, grid, True)
    in_mem = ttnn.create_sharded_memory_config(
        shape=(1, 1, shard_h, in_channels),
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    tt_input = ttnn.from_torch(flat, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT).to(device, in_mem)
    tt_weight = ttnn.from_torch(torch_weight, dtype=ttnn.bfloat16)
    tt_bias = (
        ttnn.from_torch(torch_bias.reshape(1, 1, 1, out_channels), dtype=ttnn.bfloat16) if with_bias_relu else None
    )

    conv_config = ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat16,
        shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,  # <-- the path that crashed on the degenerate grid
        reshard_if_not_optimal=True,
        deallocate_activation=True,
        activation=(ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU) if with_bias_relu else None),
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.LoFi, packer_l1_acc=True
    )

    # Block-sharded conv uses the plain sharded factory, not the DRAM-slicing split path.
    out, [oh, ow], _wb = ttnn.experimental.quasar.conv2d(
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

    tt_out = ttnn.to_torch(ttnn.from_device(out))
    tt_out = tt_out.reshape(batch_size, oh, ow, tt_out.shape[-1])[:, :, :, :out_channels]
    tt_out = torch.permute(tt_out, (0, 3, 1, 2))  # NHWC -> NCHW
    assert_with_pcc(torch_golden, tt_out, 0.99)
