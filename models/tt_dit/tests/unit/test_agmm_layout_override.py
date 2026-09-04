# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Tests for the per-shape AGMM non-transposed layout override (agmm_layout_overrides).

The host test pins the config wiring: the override must hand back the exact swept layout, and the
grid table entry for the same key must hold the blocking that was swept AT that layout (the table
key does not encode orientation, so the two flip together). The device test runs ColParallelLinear
end-to-end through the override path — the Aang a2v-1080p qkv / ff1 shapes at TP4 on a 4x8 ring —
and checks numerics against torch, since the perf sweep that produced the entries measures time
only.
"""

import pytest
import torch

import ttnn

from ...layers.linear import ColParallelLinear
from ...parallel.config import DiTParallelConfig
from ...parallel.manager import CCLManager
from ...utils.check import assert_quality
from ...utils.matmul import get_agmm_config, get_agmm_layout_override
from ...utils.tensor import bf16_tensor
from ...utils.test import mesh_device_config_to_string, ring_params_8k


@pytest.mark.parametrize(
    ("M, K, N_per_device, expected_blocks, expected_subblock"),
    [
        (1664, 5120, 3840, (3, 8, 16), (1, 4)),  # Aang a2v-1080p to_qkv
        (1664, 5120, 3456, (3, 8, 10), (3, 1)),  # Aang a2v-1080p ff1
    ],
    ids=["qkv", "ff1"],
)
def test_layout_override_wiring(M, K, N_per_device, expected_blocks, expected_subblock):
    """The override and the grid table entry must resolve to the swept nt configuration."""
    override = get_agmm_layout_override(M, K, N_per_device)
    assert override is not None
    force_transpose, core_grid, num_workers_per_link = override
    assert force_transpose is False
    assert (core_grid.x, core_grid.y) == (12, 9)
    assert num_workers_per_link == 6

    grid, config, _workers = get_agmm_config(
        M,
        K,
        N_per_device,
        full_grid=ttnn.CoreCoord(12, 10),
        cluster_size=4,
        num_links=2,
        core_grid=core_grid,
        force_transpose=force_transpose,
    )
    assert (grid.x, grid.y) == (12, 9)
    assert (config.M_block_size, config.K_block_size, config.N_block_size) == expected_blocks
    assert (config.subblock_h, config.subblock_w) == expected_subblock

    # Shapes without an override keep the default layout untouched.
    assert get_agmm_layout_override(2656, 5120, 3840) is None


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [((4, 8), {**ring_params_8k, "require_exact_physical_num_devices": True})],
    ids=mesh_device_config_to_string,
    indirect=True,
)
@pytest.mark.parametrize(
    ("M, K, N, activation_fn, chunks"),
    [
        (1664, 5120, 15360, None, 3),  # Aang a2v-1080p to_qkv: N/tp4 = 3840 hits the override
        (1664, 5120, 13824, "gelu_tanh", None),  # Aang a2v-1080p ff1: N/tp4 = 3456 hits the override
    ],
    ids=["qkv", "ff1"],
)
def test_col_parallel_linear_nt_override(
    mesh_device: ttnn.MeshDevice,
    M: int,
    K: int,
    N: int,
    activation_fn,
    chunks,
) -> None:
    tp_axis = 0
    tp = tuple(mesh_device.shape)[tp_axis]

    torch_model = torch.nn.Linear(K, N, bias=True).to(dtype=torch.bfloat16)
    torch_model.eval()

    ccl_manager = CCLManager(mesh_device, num_links=2, topology=ttnn.Topology.Ring)
    tt_model = ColParallelLinear(
        K,
        N,
        bias=True,
        activation_fn=activation_fn,
        mesh_device=mesh_device,
        mesh_axis=tp_axis,
        ccl_manager=ccl_manager,
        chunks=chunks,
    )
    tt_model.load_torch_state_dict(torch_model.state_dict())

    parallel_config = DiTParallelConfig.from_tuples(
        cfg=(1, 0), sp=(tuple(mesh_device.shape)[1 - tp_axis], 1 - tp_axis), tp=(tp, tp_axis)
    )

    torch_input = torch.randn((1, 1, M, K), dtype=torch.bfloat16)
    # K-fractured input: forces the gather (AGMM) path, where the override applies.
    tt_input = bf16_tensor(torch_input, device=mesh_device, mesh_axis=tp_axis, shard_dim=-1)

    with torch.no_grad():
        torch_output = torch_model(torch_input)
        if activation_fn == "gelu_tanh":
            torch_output = torch.nn.functional.gelu(torch_output, approximate="tanh")

    tt_output = tt_model(tt_input, parallel_config=parallel_config)
    if chunks:
        # Chunked output splits the per-device N slice into contiguous pieces; concat restores it.
        tt_output = ttnn.concat(list(tt_output), dim=-1)

    shard_dims = [None, None]
    shard_dims[tp_axis] = -1
    shard_dims[1 - tp_axis] = 0
    tt_output = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=shard_dims, mesh_shape=tuple(mesh_device.shape)),
    )
    for i in range(tt_output.shape[0]):
        assert_quality(torch_output.squeeze(), tt_output[i].squeeze(), pcc=0.999_500)
