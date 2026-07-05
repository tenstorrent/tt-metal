# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Standalone per-op test for the Quasar resnet50 stem `ttnn.experimental.quasar.fold`.

WHERE IT COMES FROM
-------------------
resnet50 (models/.../quasar/tt/ttnn_functional_resnet50.py, run()) folds the input image into
channels (space-to-depth) as the very first op, so the 4x4 stem conv can run at unity stride:

    fold_output_tensor = ttnn.experimental.quasar.fold(
        input_tensor,
        self.fold_stride_h, self.fold_stride_w,   # = stride = 2
        use_transpose_as_fold=True,
        padding=[self.fold_pad_h, self.fold_pad_h, self.fold_pad_w, self.fold_pad_w, 0, self.fold_pad_c],
        grid_size=self.fold_compute_grid_size,
        override_memory_config=self.override_fold_mem_config,
    )

For resnet50 the constructor is called with kernel_size=3, stride=2 and a (N, 3, 224, 224) image
(see resnet50_test_infra.py), which gives:
    fold_stride_h = fold_stride_w = 2
    fold_pad_h = fold_pad_w = kernel_size = 3
    fold_pad_c = nearest_y(3, 4) - 3 = 1              (channels padded 3 -> 4)
    padding = [3, 3, 3, 3, 0, 1]
    fold_output_shape = (N, 230//2, 230//2, 4*2*2) = (N, 115, 115, 16)

The input to fold is the ROW_MAJOR, HEIGHT_SHARDED NCHW image (setup_l1_sharded_input shards it over
the flattened N*C*H rows). This test reproduces that exact configuration; only the batch and the core
count are tied to the device so it runs on the small Quasar sim grid as well as full silicon.

WHAT IT VALIDATES
-----------------
`use_transpose_as_fold=True` implements the fold as a sequence of transposes/reshapes on device. It
is data-preserving (a pure rearrangement), so `ttnn.to_torch` of the output must equal the reference
CPU fold. A wrong Quasar LLK binding (transpose/reshape data movement, sharded reader/writer)
corrupts the data or hangs. The golden below is the well-established CPU reference used by the
existing WH/BH fold tests (tests/ttnn/unit_tests/operations/conv/data_movement/test_fold_op.py).

RUN
---
  TT_METAL_SIMULATOR=~/sim/libttsim.so \
  TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_FORCE_JIT_COMPILE=1 \
  pytest models/demos/vision/classification/resnet50/quasar/tests/ops/test_fold.py
"""

import pytest
import torch

import ttnn
from models.common.utility_functions import _nearest_y
from tests.ttnn.utils_for_testing import assert_with_pcc


def pad_and_fold_conv_activation_for_unity_stride(activation_pyt_nchw_tensor, pad_h, pad_w, stride_h, stride_w):
    """CPU reference fold (NCHW in, NCHW folded out), copied from the WH/BH fold op test."""
    assert stride_h == stride_w
    assert activation_pyt_nchw_tensor.shape[2] == activation_pyt_nchw_tensor.shape[3]
    # Pad channels to a multiple of 4 (keeps L1 read addresses 16-bit aligned), plus the conv padding.
    C = _nearest_y(activation_pyt_nchw_tensor.shape[1], 4)
    activation_pyt_padded = torch.nn.functional.pad(
        activation_pyt_nchw_tensor, (pad_w, pad_w, pad_h, pad_h, 0, C - activation_pyt_nchw_tensor.shape[1])
    )
    assert activation_pyt_padded.shape[2] % stride_h == 0
    activation_pyt_padded_folded = torch.zeros(
        [
            activation_pyt_padded.shape[0],
            C * stride_h * stride_w,
            (int)(activation_pyt_padded.shape[2] / stride_h),
            (int)(activation_pyt_padded.shape[3] / stride_w),
        ]
    )
    for h in range(0, activation_pyt_padded.shape[2], stride_h):
        for w in range(0, activation_pyt_padded.shape[3], stride_w):
            folded_h = (int)(h / stride_h)
            folded_w = (int)(w / stride_w)
            for i in range(stride_h * stride_w):
                start_c = i * C
                activation_pyt_padded_folded[:, start_c : start_c + C, folded_h, folded_w] = activation_pyt_padded[
                    :, :, h + (int)(i / stride_w), w + (int)(i % stride_w)
                ]
    return activation_pyt_padded_folded


def _fit_cores(total_rows, device):
    """Largest core count <= device cores that divides total_rows (so the height shards are exact)."""
    grid = device.compute_with_storage_grid_size()
    cap = min(total_rows, grid.x * grid.y)
    num_cores = cap
    while num_cores > 1 and total_rows % num_cores != 0:
        num_cores -= 1
    return num_cores, grid


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("batch_size", [1, 2], ids=["b1", "b2"])
def test_quasar_fold(mesh_device, batch_size):
    device = mesh_device
    torch.manual_seed(0)

    # resnet50 stem fold params.
    c, h, w = 3, 224, 224
    kernel_size = 3
    stride_h = stride_w = 2
    pad_h = pad_w = kernel_size  # fold_pad_h/w = kernel_size = 3
    C = _nearest_y(c, 4)  # 4
    pad_c = C - c  # 1

    torch_input = torch.rand((batch_size, c, h, w), dtype=torch.bfloat16)

    # Golden: CPU fold (NCHW folded) then permute to NHWC to match the device output layout.
    golden = pad_and_fold_conv_activation_for_unity_stride(torch_input, pad_h, pad_w, stride_h, stride_w)
    golden = torch.permute(golden, (0, 2, 3, 1))

    # HEIGHT-shard the ROW_MAJOR NCHW image over the flattened N*C*H rows, tied to the device grid
    # (mirrors setup_l1_sharded_input). Use an exact divisor so shards are unpadded.
    total_rows = batch_size * c * h
    num_cores, grid = _fit_cores(total_rows, device)
    shard_grid = ttnn.num_cores_to_corerangeset(num_cores, grid, row_wise=True)
    shard_spec = ttnn.ShardSpec(shard_grid, (total_rows // num_cores, w), ttnn.ShardOrientation.ROW_MAJOR)
    input_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)

    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=input_mem_config,
    )

    tt_out = ttnn.experimental.quasar.fold(
        tt_input,
        stride_h,
        stride_w,
        use_transpose_as_fold=True,
        padding=[pad_h, pad_h, pad_w, pad_w, 0, pad_c],
        grid_size=shard_grid,
    )

    got = ttnn.to_torch(tt_out).to(torch.bfloat16)
    # fold is a pure data-preserving rearrangement.
    assert_with_pcc(golden.to(torch.bfloat16), got, pcc=0.999)
