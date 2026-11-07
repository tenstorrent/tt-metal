# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 1 — native sharded placement. Pins each native shard path and asserts it was taken (the ports were
placed off the member shard cores), not the TensorAccessor fallback:

  HEIGHT regime A   (1,1,640,384)  20 shards, k = 1, own slice read in place from the input shard (cb_x_in)
  HEIGHT regime N   (4,128,47)     16 shards, compose reads the own slice from the input shard
  WIDTH symmetric   (1,1,32,64)    8 one-tile members, k <= K_SYM_MAX
  WIDTH owner       (1,1,32,384)   48 one-tile members, owner combine
  WIDTH ragged      (1,1,32,4096)  103 five-tile members, the last one ragged (pad pages up to the shard pitch)

And one fallback pin: WIDTH (1,1,64,512) — 64 one-tile members do not keep up with the transport (keep-up gate).
"""

from __future__ import annotations

import pytest
import ttnn

import ttnn.operations.all_gather_rms_norm.all_gather_rms_norm_program_descriptor as PD

from tests.ttnn.unit_tests.operations.all_gather_rms_norm.test_all_gather_rms_norm import (
    DEVICE_PARAMS,
    LINE,
    RM,
    TILE,
    run_case,
)

H = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
W = ttnn.TensorMemoryLayout.WIDTH_SHARDED

CASES = [
    pytest.param(H, (1, 1, 640, 384), TILE, ttnn.bfloat16, True, id="height_A_640x384"),
    pytest.param(H, (4, 128, 47), None, None, True, id="height_N_4x128x47"),
    pytest.param(W, (1, 1, 32, 64), RM, ttnn.bfloat16, True, id="width_sym_32x64"),
    pytest.param(W, (1, 1, 32, 384), RM, ttnn.bfloat16, True, id="width_owner_32x384"),
    pytest.param(W, (1, 1, 32, 4096), TILE, ttnn.float32, True, id="width_ragged_32x4096"),
    pytest.param(W, (1, 1, 64, 512), TILE, ttnn.bfloat16, False, id="width_fallback_64x512"),
]


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True, ids=["torus_xy"])
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True, ids=["mesh1x4"])
@pytest.mark.parametrize("memory_layout,shape,gamma_layout,gamma_dtype,expect_native", CASES)
def test_native_shard(mesh_device, memory_layout, shape, gamma_layout, gamma_dtype, expect_native, monkeypatch):
    taken = []
    orig = PD.plan

    def spy(mesh, axis, ring, L, avoid=frozenset()):
        taken.append(bool(avoid))
        return orig(mesh, axis, ring, L, avoid)

    monkeypatch.setattr(PD, "plan", spy)
    run_case(
        mesh_device,
        cluster_axis=1,
        shape=shape,
        dtype=ttnn.bfloat16,
        topology=LINE,
        num_links=1,
        gamma_layout=gamma_layout,
        gamma_dtype=gamma_dtype,
        memory_layout=memory_layout,
    )
    assert taken and taken[-1] == expect_native, f"native placement taken={taken}, expected {expect_native}"
