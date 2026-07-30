# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Standalone per-op test for the Quasar (Metal-2) `ttnn.experimental.quasar.slice_write`.

WHERE IT COMES FROM
-------------------
The write-back half of the Quasar conv2d DRAM-slicing path (`conv2d_DRAM` -> `op_slicing::run_sliced_op`):
after `conv2d_L1` produces an L1-sharded slice result, `slice_write` writes it back into the interleaved
DRAM output at the slice's spatial offset. The shared `slice_write` builds a legacy DataMovementKernel
(rejected on Quasar), so it was ported to a Metal-2 QuasarDataMovementKernel factory (RM sharded-input).
This test exercises that RM path in isolation (the inverse of test_padded_slice.py):

    input  : [1, 59, 115, 32]  bf16, ROW_MAJOR, HEIGHT_SHARDED L1   (a conv-slice-shaped result)
    output : [1, 115, 115, 32] bf16, ROW_MAJOR, INTERLEAVED DRAM    (pre-allocated; written in place)
    write  : output[:, 0:59, :, :] = input   (start=[0,0,0,0], end=[1,59,115,32])

WHAT IT VALIDATES
-----------------
slice_write is a pure data-movement rearrangement, so the written region of `ttnn.to_torch(out)` must
equal the input and the untouched region must keep its initial value. A wrong Metal-2 binding (writer DFB
drain / dst TensorAccessor / vararg dim-walk) corrupts or misplaces sticks.

RUN (craq-sim / emulator, slow dispatch + forced JIT):
  TT_METAL_SIMULATOR=~/sim/libttsim.so \
  TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_FORCE_JIT_COMPILE=1 \
  pytest models/demos/vision/classification/resnet50/quasar/tests/ops/test_slice_write.py
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


def _fit_cores(total_rows, device):
    """Largest core count <= device cores that divides total_rows (exact, unpadded height shards)."""
    grid = device.compute_with_storage_grid_size()
    cap = min(total_rows, grid.x * grid.y)
    num_cores = cap
    while num_cores > 1 and total_rows % num_cores != 0:
        num_cores -= 1
    return num_cores, grid


# (n, in_h, w, c, out_h, id): write an [n,in_h,w,c] slice into an [n,out_h,w,c] output at height 0.
CASES = [
    (1, 59, 115, 32, 115, "stem_h59_into_h115_c32"),  # the resnet-stem conv2d DRAM slice write-back
    (1, 32, 32, 32, 64, "small_h32_into_h64_c32"),  # small aligned control
]


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("n, in_h, w, c, out_h, tid", CASES, ids=[cse[-1] for cse in CASES])
def test_quasar_slice_write(mesh_device, n, in_h, w, c, out_h, tid):
    device = mesh_device
    torch.manual_seed(0)

    input_torch = torch.rand((n, in_h, w, c), dtype=torch.bfloat16)
    # Pre-allocated output; slice_write fills [:, 0:in_h] and leaves the rest at its initial value.
    output_torch = torch.zeros((n, out_h, w, c), dtype=torch.bfloat16)
    golden = output_torch.clone()
    golden[:, :in_h, :, :] = input_torch

    # Input: ROW_MAJOR, HEIGHT_SHARDED L1 over the flattened N*in_h*W rows, width = c.
    total_rows = n * in_h * w
    num_cores, grid = _fit_cores(total_rows, device)
    shard_grid = ttnn.num_cores_to_corerangeset(num_cores, grid, row_wise=True)
    shard_spec = ttnn.ShardSpec(shard_grid, (total_rows // num_cores, c), ttnn.ShardOrientation.ROW_MAJOR)
    in_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    tt_input = ttnn.from_torch(
        input_torch, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=in_mem_config
    )

    # Output: ROW_MAJOR, INTERLEAVED DRAM (the target the writer strides into).
    tt_output = ttnn.from_torch(
        output_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    out = ttnn.experimental.quasar.slice_write(
        tt_input,
        tt_output,
        [0, 0, 0, 0],
        [n, in_h, w, c],
        [1, 1, 1, 1],
    )

    got = ttnn.to_torch(out).reshape(golden.shape).float()
    assert tuple(got.shape) == tuple(golden.shape), (got.shape, golden.shape)
    assert_with_pcc(golden.float(), got, 0.9999)
