# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 4 — numerical configurability: bfloat8_b input/output and fp32_dest_acc_en=False.

Runs through the golden helper (golden tolerances: bf8b 0.995 / 0.04-0.05, bf16+False 0.999 / 0.015). One case per
code path the format change touches: interleaved prescale (cb_xg narrows to Float16_b when DEST is 16-bit), fused
gamma (no cb_xg), w_non_aligned (the reader's w_tail mask on block-float tiles), HEIGHT / WIDTH sharded (zero-copy
shard CBs + R5 pre-combine).
"""

import pytest
import ttnn

from eval.golden_tests.distributed_rms_norm.helpers import run_distributed_rms_norm
from ttnn.operations.ccl import Topology

from .test_distributed_rms_norm import _FABRIC, _MESH

I = ttnn.TensorMemoryLayout.INTERLEAVED
H = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
W = ttnn.TensorMemoryLayout.WIDTH_SHARDED
F32, BF16, BF8 = ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b

CASES = [
    # (shape, memory_layout, dtype, fp32_dest_acc_en, gamma_dtype)
    ((1, 1, 640, 1792), I, BF8, True, BF16),
    ((1, 1, 640, 1792), I, BF8, False, BF16),
    ((1, 1, 640, 1792), I, BF16, False, BF16),
    ((1, 1, 32, 1792), I, BF16, False, F32),
    ((2, 4, 128, 512), I, BF8, False, None),
    ((1, 1, 64, 1000), I, BF8, True, BF16),  # w_non_aligned: w_tail = 8 (<= 16)
    ((1, 1, 64, 1000), I, BF16, False, BF16),
    ((1, 1, 100, 1272), I, BF8, False, F32),  # w_tail = 24 (> 16), h_non_aligned rows
    ((1, 1, 640, 1792), H, BF8, True, BF16),
    ((1, 1, 640, 1792), H, BF16, False, BF16),
    ((1, 1, 640, 1792), W, BF8, False, BF16),
    ((1, 1, 32, 1792), W, BF16, False, None),
]


def _id(c):
    shape, ml, dt, acc, g = c
    return f"{'x'.join(map(str, shape))}-{str(ml).split('.')[-1]}-{dt}-{'fp32acc' if acc else 'bf16acc'}-{g}"


@_MESH
@_FABRIC
@pytest.mark.parametrize("shape,memory_layout,dtype,fp32_acc,gamma_dtype", CASES, ids=[_id(c) for c in CASES])
def test_numerics(mesh_device, shape, memory_layout, dtype, fp32_acc, gamma_dtype):
    if tuple(mesh_device.shape)[1] < 2:
        pytest.skip(f"cluster_axis=1 has size 1 on mesh {tuple(mesh_device.shape)}")
    run_distributed_rms_norm(
        (shape,),
        mesh_device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        fp32_dest_acc_en=fp32_acc,
        gamma_mode="gamma" if gamma_dtype is not None else "no_gamma",
        gamma_dtype=gamma_dtype if gamma_dtype is not None else "none",
        memory_layout=memory_layout,
        cluster_axis=1,
        topology=Topology.Linear,
        num_links=None,
    )
