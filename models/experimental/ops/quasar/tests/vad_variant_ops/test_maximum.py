# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.maximum`` (bf16) and ``ttnn.atan2`` (fp32) — binary SFPU ops.

Shape source (kernel cache):
  * ``maximum`` bf16 ``eltwise_binary_sfpu_no_bcast/913100``. Paired in the cache with a
    ``fill(1e-5)`` unary (``fill_tile_bitcast 0x3727c5ac``) — the inverse_sigmoid eps clamp
    ``max(x, eps)`` on reference points; shape assumed [1, 10000, 4, 2] (query x level x xy)  [L]
  * ``atan2`` fp32 ``eltwise_binary_sfpu_no_bcast/151084`` (L1) — yaw from (sin, cos)    [L shape]
"""

import pytest
import torch

import ttnn
from models.experimental.ops.quasar.tests.vad_variant_ops import op_utils as U


@U.with_default_mesh()
@pytest.mark.parametrize("a_shape", [pytest.param((1, U.NUM_QUERY, 4, 2), id="B17-maximum-913100")])
def test_maximum(ttnn_mesh_device, reset_seeds, a_shape):
    U.run_binary(
        ttnn_mesh_device, ttnn.maximum, torch.maximum, a_shape, a_shape, "bf16", "bf16", "l1", "l1", "l1", pcc=0.9999
    )


@U.with_default_mesh()
@pytest.mark.parametrize("a_shape", [pytest.param((1, 300, 1), id="B22-atan2-fp32-151084")])
def test_atan2(ttnn_mesh_device, reset_seeds, a_shape):
    U.run_binary(
        ttnn_mesh_device, ttnn.atan2, torch.atan2, a_shape, a_shape, "fp32", "fp32", "l1", "l1", "l1", pcc=0.999
    )
