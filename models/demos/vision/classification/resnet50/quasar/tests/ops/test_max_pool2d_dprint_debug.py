# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Standalone DPRINT debug repro for the Quasar max_pool2d VALUE-INFLATION bug.

Context (from the resnet50/quasar per-stage PCC trace): conv1/stem output is numerically
perfect, but the quasar max_pool2d inflates its output (mean ~2x golden, and produces values
ABOVE the input max) -> the -inf reduce init/pad is not taking effect, so a phantom value
(stale L1 / the reduce scalar ~1.0) leaks into the max. This corrupts every downstream stage
(PCC-benign for random weights -> resnet passes; PCC-fatal for pretrained -> resnet fails 0.54).

This test isolates it with a CONSTANT input so the expected output is trivially predictable:
  input = all 0.5  ->  a correct max-pool outputs 0.5 EVERYWHERE (max of {0.5,...} and -inf pad).
  Any output value != 0.5 (especially ~1.0) is the phantom leak.

Single-core sharding keeps the kernel DPRINT to a single print. To see the kernel DPRINTs:
  export TT_METAL_DPRINT_CORES=all      # (or the single worker core)
  unset TT_METAL_LLK_ASSERTS; unset TT_METAL_LIGHTWEIGHT_KERNEL_ASSERTS
  pytest models/demos/vision/classification/resnet50/quasar/tests/ops/test_max_pool2d_dprint_debug.py -s

DPRINTs added (bisect fill vs reduce):
  reader_pool_2d.cpp   (ENABLE_DEBUG_PRINT=1): dumps bf16_init_value + clear_value_cb[0] + in_cb[0]
                        after the init clear -> is the -inf fill actually in L1?
  compute_pool_2d.cpp  (DEBUG_PRINT=1): dumps the scalar tile + the reduce-INPUT in_cb tile the
                        reduce reads -> does the padding show -inf (0xFF80) or a phantom value?
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.timeout(300)
@pytest.mark.parametrize(
    "input_h,input_w,single_core",
    [
        (16, 16, True),  # tiny single-core CONTROL: PASSES with the get_entry_size() fix.
        (56, 56, False),  # multi-core mid-size: multi-buffered per-core in_cb (like the model).
        (112, 112, False),  # exact resnet50 stem spatial, multi-core -> reproduces the full-model config.
    ],
    ids=["tiny_1core", "mid_multicore", "stem_multicore"],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_max_pool2d_const_input(mesh_device, input_h, input_w, single_core):
    device = mesh_device
    torch.manual_seed(0)

    batch_size = 1
    channels = 64
    kernel_size = [3, 3]
    stride = [2, 2]
    padding = [1, 1]
    dilation = [1, 1]

    output_h = (input_h - kernel_size[0] + 2 * padding[0]) // stride[0] + 1
    output_w = (input_w - kernel_size[1] + 2 * padding[1]) // stride[1] + 1

    # RANDOM input (NOT a constant): a constant golden makes PCC degenerate (always "passes") and a
    # constant input hides any stale-L1 leak whose value is <= the constant. With a random input in
    # [0,1), a correct max-pool output is also in [0,1); the value-inflation leak injects out-of-range
    # values (we've seen 128), so `got.max() > input.max()` is a hard, unambiguous leak detector, and PCC
    # against the non-constant golden is now meaningful.
    x_nchw = torch.rand((batch_size, channels, input_h, input_w), dtype=torch.bfloat16)
    input_max = x_nchw.float().max().item()
    golden_nchw = torch.nn.functional.max_pool2d(
        x_nchw.float(), kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation
    )
    golden_flat = golden_nchw.permute(0, 2, 3, 1).reshape(1, 1, batch_size * output_h * output_w, channels).contiguous()

    x_nhwc_flat = x_nchw.permute(0, 2, 3, 1).reshape(1, 1, batch_size * input_h * input_w, channels).contiguous()
    tensor_height = batch_size * input_h * input_w
    tensor_width = channels

    # single_core: 1 core (readable DPRINT, small in_cb). multi-core: grid-adaptive tile-aligned shard,
    # mirroring how the model shards the pool input -> a multi-buffered per-core in_cb (the config where
    # the get_entry_size() fix alone was insufficient in the full model). Set TT_METAL_DPRINT_CORES to one
    # core to keep the kernel DPRINT readable in the multi-core cases.
    grid = device.compute_with_storage_grid_size()
    max_cores = grid.x * grid.y
    height_tiles = tensor_height // 32
    num_cores = 1 if single_core else max(c for c in range(1, max_cores + 1) if height_tiles % c == 0)
    shard_height = (height_tiles // num_cores) * 32
    core_grid = ttnn.num_cores_to_corerangeset(num_cores, grid, True)
    mem_config = ttnn.create_sharded_memory_config(
        shape=(1, 1, shard_height, tensor_width),
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    x = ttnn.from_torch(x_nhwc_flat, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    x = x.to(device, mem_config)

    out = ttnn.experimental.quasar.max_pool2d(
        input_tensor=x,
        batch_size=batch_size,
        input_h=input_h,
        input_w=input_w,
        channels=channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    ttnn.synchronize_device(device)

    got = ttnn.to_torch(out).float().reshape(1, 1, batch_size * output_h * output_w, channels)

    # Hard leak detector: a correct max-pool can NEVER output a value above its input max.
    got_max = got.max().item()
    leaked = got_max > input_max + 1e-2
    print(
        f"\n[max_pool2d random-input {input_h}x{input_w}] input_max={input_max:.4f}\n"
        f"  golden  min/mean/max = {golden_flat.min():.4f}/{golden_flat.mean():.4f}/{golden_flat.max():.4f}\n"
        f"  got     min/mean/max = {got.min():.4f}/{got.mean():.4f}/{got_max:.4f}\n"
        f"  LEAK (got.max > input.max)? {leaked}"
    )
    assert not leaked, f"max_pool2d leaked stale L1: got.max={got_max} > input.max={input_max}"
    assert_with_pcc(golden_flat, got, pcc=0.99)
