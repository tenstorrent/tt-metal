# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""SP-over-T for the op-backed NA3D executor: Q sharded over T across the mesh, K/V replicated, each
chip fed its global frame origin; outputs all-gathered along T. Held against the host reference."""

from __future__ import annotations

import pytest
import torch

import ttnn

from ...layers.na3d import na3d_torch, neighborhood_attention_3d_op_sp
from ...parallel.manager import CCLManager
from ...utils.check import assert_quality
from ...utils.tensor import from_torch
from ...utils.tensor import to_torch as to_torch_replicated


@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True, ids=["ring"]
)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True, ids=["4x8"])
@pytest.mark.parametrize("sp_axis", [0, 1], ids=["sp_rows", "sp_cols"])
@pytest.mark.parametrize("dims, kernel", [((8, 8, 8), (3, 3, 3)), ((8, 4, 8), (3, 3, 3))])
def test_na3d_op_sp_matches_host(*, mesh_device, sp_axis, dims, kernel):
    T, H, W = dims
    heads, head_dim = 4, 64
    sp = list(mesh_device.shape)[sp_axis]
    if T % sp != 0:
        pytest.skip(f"T={T} not divisible by sp={sp}")
    if (T // sp) * H * W % 32 != 0:
        pytest.skip(f"shard origin not tile-aligned for dims={dims}, sp={sp}")

    torch.manual_seed(0)
    q, k, v = (torch.randn(1, T, H, W, heads, head_dim, dtype=torch.float32) for _ in range(3))
    expected = na3d_torch(q, k, v, kernel, scale=1.0).reshape(1, T, H, W, heads * head_dim)

    # Q sharded over T (dim 1) along sp_axis; K/V replicated on every chip.
    shard_axes = [None] * 6
    shard_axes[1] = sp_axis
    q_tt = from_torch(q, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_axes=shard_axes)
    k_tt = from_torch(k, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    v_tt = from_torch(v, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    ccl_manager = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)
    actual = neighborhood_attention_3d_op_sp(
        q_tt, k_tt, v_tt, dims=dims, kernel_size=kernel, sp_axis=sp_axis, ccl_manager=ccl_manager, scale=1.0
    )

    assert tuple(actual.shape) == tuple(expected.shape), f"{tuple(actual.shape)} != {tuple(expected.shape)}"
    # After the all-gather the volume is identical on every chip, so extract one replica.
    assert_quality(expected, to_torch_replicated(actual), pcc=0.999)
