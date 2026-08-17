# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Standalone per-op test for the Quasar (Metal-2) `ttnn.experimental.quasar.padded_slice`.

WHERE IT COMES FROM
-------------------
The resnet50 stem conv on the 2-core emulator does not fit L1 in a single slice, so the Quasar
`conv2d` DRAM-slicing path (`conv2d_DRAM` -> `op_slicing::run_sliced_op`) slices the DRAM activation
into L1 height-slices with `padded_slice`, runs `conv2d_L1` on each, and writes back with `slice_write`.
The shared `padded_slice` builds a legacy DataMovementKernel (rejected on Quasar, kernel.hpp:382), so it
was ported to a Metal-2 QuasarDataMovementKernel factory (RM path). This test exercises that RM path in
isolation, at the exact shape/sharding the stem hits:

    input  : [1, 115, 115, 32]  bf16, ROW_MAJOR, INTERLEAVED DRAM   (the reshaped fold output)
    slice  : start=[0,0,0,0]  end=[1, 59, 115, 32]                  (DRAM height-slice 0 of 2)
    output : [1, 59, 115, 32]  HEIGHT_SHARDED L1                    (fed to conv2d_L1)

WHAT IT VALIDATES
-----------------
padded_slice is a pure data-movement rearrangement, so `ttnn.to_torch(out)` must equal the torch slice.
A wrong Metal-2 binding (reader DFB / TensorAccessor / vararg dim-walk) corrupts or drops sticks.

RUN (craq-sim, slow dispatch + forced JIT):
  TT_METAL_SIMULATOR=~/sim/libttsim.so \
  TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_FORCE_JIT_COMPILE=1 \
  pytest models/demos/vision/classification/resnet50/quasar/tests/ops/test_padded_slice.py
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


# (n, h, w, c, out_h, id) -- the stem DRAM height-slice plus a couple of small aligned controls.
CASES = [
    (1, 115, 115, 32, 59, "stem_h115_to_h59_c32"),  # the exact resnet-stem conv2d DRAM height-slice
    (1, 64, 32, 32, 32, "small_h64_to_h32_c32"),  # small aligned control
    (1, 32, 16, 16, 16, "small_h32_to_h16_c16"),  # narrower control (c=16 still 32B-aligned)
]


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("n, h, w, c, out_h, tid", CASES, ids=[c[-1] for c in CASES])
def test_quasar_padded_slice(mesh_device, n, h, w, c, out_h, tid):
    device = mesh_device
    torch.manual_seed(0)

    torch_input = torch.rand((n, h, w, c), dtype=torch.bfloat16)
    golden = torch_input[:, :out_h, :, :]  # height-slice 0

    # Input: ROW_MAJOR, INTERLEAVED DRAM (mirrors the fold output that conv2d_DRAM slices).
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Output: HEIGHT_SHARDED L1 over the flattened N*out_h*W rows, width = c (the padded_slice factory
    # requires a sharded L1 output). Core count tied to the device so it runs on the 2-core emulator too.
    total_out_rows = n * out_h * w
    num_cores, grid = _fit_cores(total_out_rows, device)
    shard_grid = ttnn.num_cores_to_corerangeset(num_cores, grid, row_wise=True)
    shard_spec = ttnn.ShardSpec(shard_grid, (total_out_rows // num_cores, c), ttnn.ShardOrientation.ROW_MAJOR)
    out_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)

    out = ttnn.experimental.quasar.padded_slice(
        tt_input,
        [0, 0, 0, 0],
        [n, out_h, w, c],
        [1, 1, 1, 1],
        memory_config=out_mem_config,
    )

    got = ttnn.to_torch(out).reshape(golden.shape).float()
    assert tuple(got.shape) == tuple(golden.shape), (got.shape, golden.shape)
    assert_with_pcc(golden.float(), got, 0.9999)
