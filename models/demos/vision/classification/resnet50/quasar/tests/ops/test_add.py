# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Standalone per-op test for the resnet50/quasar residual ADD op.

This isolates the in-place binary add the model makes at the end of every bottleneck
(resnet50Bottleneck.__call__):

    out = ttnn.experimental.quasar.add_(
        out, ds_out, activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)]
    )

i.e. the residual add of the conv3 output (`out`) and the (downsample/identity) skip branch
(`ds_out`), with a fused RELU. `add_` is the in-place variant (input_tensor_a is overwritten and
returned). This lets the LLK team test / fix the residual add in isolation with a PCC check.

Operand layout (verbatim from the bottleneck path): both operands are the conv-output layout
[1, 1, N*H*W, C], TILE, HEIGHT_SHARDED, and the model forces them to the same memory config before
the add (`if ds_out.memory_config() != out.memory_config(): ds_out = ...to_memory_config(...)`), so
we build both with one identical height-sharded config here.

Shapes swept are the batch-1 bottleneck residual outputs of each stage:
  layer1: [1,1,3136,256]  (56x56, 256ch)   layer3: [1,1,196,1024]  (14x14,1024ch)
  layer2: [1,1, 784,512]  (28x28, 512ch)   layer4: [1,1, 49,2048]  ( 7x7,2048ch)

RELU is a real part of the op here: inputs are torch.randn (can be negative) so the fused RELU
actually clamps, exercising the activation path rather than being a no-op.

Run (craq-sim, slow dispatch + forced JIT):
  TT_METAL_SIMULATOR=~/sim/libttsim.so \
  TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_FORCE_JIT_COMPILE=1 \
  pytest models/demos/vision/classification/resnet50/quasar/tests/ops/test_add.py
"""

import math

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "height, channels",
    [
        pytest.param(3136, 256, id="layer1_56x56_256c"),
        pytest.param(784, 512, id="layer2_28x28_512c"),
        pytest.param(196, 1024, id="layer3_14x14_1024c"),
        pytest.param(49, 2048, id="layer4_7x7_2048c"),
    ],
)
def test_resnet50_residual_add(mesh_device, height, channels):
    device = mesh_device
    torch.manual_seed(0)

    # --- build both operands in the conv-output layout: [1,1,N*H*W,C], TILE, HEIGHT_SHARDED ---
    # Height-shard over the device grid, padding the height to a (num_cores * TILE) boundary and
    # sharding it evenly (mirrors the model's conv-output sharding). Grid-adaptive: full 32-core grid
    # on Quasar, fewer on the 2x3 emulator / craq-sim.
    compute_grid = device.compute_with_storage_grid_size()
    num_cores = compute_grid.x * compute_grid.y
    core_grid = ttnn.num_cores_to_corerangeset(num_cores, compute_grid, row_wise=True)

    tile = 32
    padded_height = math.ceil(height / (num_cores * tile)) * (num_cores * tile)
    shard_height = padded_height // num_cores

    mem_config = ttnn.create_sharded_memory_config(
        shape=(1, 1, shard_height, channels),
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    a_torch = torch.randn((1, 1, height, channels), dtype=torch.bfloat16)
    b_torch = torch.randn((1, 1, height, channels), dtype=torch.bfloat16)

    # golden: RELU(a + b)  -- fused activation the op applies
    golden = torch.relu(a_torch.float() + b_torch.float())

    a = ttnn.from_torch(a_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT).to(device, mem_config)
    b = ttnn.from_torch(b_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT).to(device, mem_config)

    # --- the exact residual add (in-place, fused RELU) ---
    out = ttnn.experimental.quasar.add_(
        a,
        b,
        activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)],
    )

    got = ttnn.to_torch(out).float()
    assert tuple(got.shape) == (1, 1, height, channels), got.shape
    # elementwise add + relu in bf16: nearly exact -> 0.99.
    assert_with_pcc(golden, got, pcc=0.99)
