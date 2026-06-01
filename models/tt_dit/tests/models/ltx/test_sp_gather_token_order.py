# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""SP AllGather token-order guard for LTX audio velocity readback."""

import pytest
import torch

import ttnn
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils.tensor import bf16_tensor
from models.tt_dit.utils.test import line_params


@pytest.mark.parametrize(
    "mesh_device, mesh_shape, sp_axis, tp_axis, num_links, dynamic_load, device_params, topology, is_fsdp",
    [
        [(2, 4), (2, 4), 1, 0, 2, True, line_params, ttnn.Topology.Linear, False],
    ],
    ids=["bh_2x4sp1tp0"],
    indirect=["mesh_device", "device_params"],
)
def test_sp_all_gather_preserves_token_order(
    mesh_device,
    mesh_shape,
    sp_axis,
    tp_axis,
    num_links,
    dynamic_load,
    topology,
    is_fsdp,
):
    """After one SP gather, host readback must be ``0..N-1``; a second gather inflates length."""
    sp_factor = mesh_shape[sp_axis]
    if sp_factor < 2:
        pytest.skip("needs SP > 1")

    N, D = 64, 128
    x = torch.zeros(1, 1, N, D, dtype=torch.bfloat16)
    for i in range(N):
        x[0, 0, i, :] = float(i)

    tt = bf16_tensor(x, device=mesh_device, mesh_axis=sp_axis, shard_dim=2)
    ccl = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)

    gathered = ccl.all_gather_persistent_buffer(tt, dim=2, mesh_axis=sp_axis)
    no_gather_dims: list[int | None] = [None, None]
    sp_gather_dims: list[int | None] = [None, None]
    sp_gather_dims[sp_axis] = 2
    host_once = ccl.device_to_host(gathered, no_gather_dims).float()
    host_twice = ccl.device_to_host(gathered, sp_gather_dims).float()

    idx_once = host_once[0, 0, :, 0].float()
    idx_twice = host_twice[0, 0, :, 0].float()

    assert idx_once.shape[0] == N
    assert torch.allclose(idx_once, torch.arange(N, dtype=torch.float32))
    assert idx_twice.shape[0] > N
