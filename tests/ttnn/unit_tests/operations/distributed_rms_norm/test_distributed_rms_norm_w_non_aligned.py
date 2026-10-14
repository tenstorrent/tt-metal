# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 2 — W_local non-tile-aligned, with GARBAGE in the tile padding of x.

The golden suite builds x through ttnn.from_torch, which zero-pads the last W tile, so it cannot tell whether the
op masks that padding. Here the padding columns of x hold large values (a view over a padded tile tensor); the op
must keep them out of sum(x^2) on every placement (interleaved, HEIGHT_SHARDED, WIDTH_SHARDED).
"""

import pytest
import torch
import ttnn

from eval.golden_tests.distributed_rms_norm.helpers import (
    _build_memory_config,
    check_output,
    create_ttnn_input_tensor,
    pytorch_distributed_rms_norm,
    random_stacked,
)
from ttnn.operations.distributed_rms_norm import distributed_rms_norm

from .test_distributed_rms_norm import _FABRIC, _MESH

I = ttnn.TensorMemoryLayout.INTERLEAVED
H = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
W = ttnn.TensorMemoryLayout.WIDTH_SHARDED
F32, BF16 = ttnn.float32, ttnn.bfloat16
_TORCH = {F32: torch.float32, BF16: torch.bfloat16}
GARBAGE = 100.0

CASES = [
    ((32, 17), I, BF16, BF16),
    ((1, 1, 64, 1000), I, BF16, BF16),  # several W chunks, partial tile in the last one
    ((4, 128, 47), I, F32, F32),  # r = 15 < 16: partial face-0/2 rows + whole faces 1/3
    ((1, 1, 32, 50), H, BF16, BF16),
    ((2, 1, 128, 100), H, F32, None),
    ((1, 1, 64, 1000), W, BF16, BF16),
    ((1, 1, 32, 50), W, F32, BF16),
]


def _garbage_padded_input(x, mesh_device, dtype, memory_layout):
    """x stacked (R, C, *shape) -> mesh tensor of logical shape `shape` whose tile padding holds GARBAGE."""
    shape = list(x.shape[2:])
    w = shape[-1]
    wp = (w + 31) // 32 * 32
    padded = torch.full((*x.shape[:-1], wp), GARBAGE, dtype=x.dtype)
    padded[..., :w] = x
    t = create_ttnn_input_tensor(padded, mesh_device, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    pshape = list(t.padded_shape)
    t = ttnn.reshape(t, ttnn.Shape(shape), ttnn.Shape(pshape))  # view: same tiles, logical W < padded W
    if memory_layout != I:
        t = ttnn.to_memory_config(
            t, _build_memory_config(memory_layout, ttnn.TILE_LAYOUT, shape, mesh_device, dtype=dtype)
        )
    # precondition: the padding really holds garbage (otherwise this test proves nothing)
    host = ttnn.get_device_tensors(t)[0].cpu().to_torch_with_padded_shape()
    assert host.float()[..., w:wp].abs().max() >= GARBAGE / 2, "tile padding was not preserved"
    return t


@_MESH
@_FABRIC
@pytest.mark.parametrize(
    "shape,memory_layout,dtype,gamma_dtype",
    CASES,
    ids=[f"{'x'.join(map(str, s))}-{str(m).split('.')[-1]}-{d}-{g}" for s, m, d, g in CASES],
)
def test_w_non_aligned_garbage_padding(mesh_device, shape, memory_layout, dtype, gamma_dtype):
    cluster_axis = 1
    if tuple(mesh_device.shape)[cluster_axis] < 2:
        pytest.skip("cluster_axis has size 1")
    x = random_stacked(mesh_device, shape, _TORCH[dtype], seed=3)
    gamma = ttnn_gamma = None
    if gamma_dtype is not None:
        gamma = random_stacked(mesh_device, (1, 1, 1, shape[-1]), _TORCH[gamma_dtype], seed=4)
        ttnn_gamma = create_ttnn_input_tensor(gamma, mesh_device, dtype=gamma_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
    expected = pytorch_distributed_rms_norm(x, gamma, cluster_axis)
    tx = _garbage_padded_input(x, mesh_device, dtype, memory_layout)
    kwargs = {"memory_config": tx.memory_config()} if memory_layout != I else {}
    out = distributed_rms_norm(tx, cluster_axis=cluster_axis, gamma=ttnn_gamma, **kwargs)
    check_output(
        out, expected, shape=tuple(shape), dtype=dtype, fp32_dest_acc_en=True, expected_layout=ttnn.TILE_LAYOUT
    )
