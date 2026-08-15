# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""A DiffVAE NA block computing the same thing sharded across the mesh as replicated on it.

The NA3D tests cover the attention primitive. This covers the block around it — the fused QKV,
the RMS norms, RoPE, the output projection, the two residual adds and the SwiGLU — under the
one change sharding makes to it: the activation carries a halo into attention and comes back
narrower.

The comparison is the same module instance run both ways, so the weights are identical by
construction and the only variable is the sharding.

RoPE is the part most likely to be wrong and isn't. Its tables are built over the local buffer
rather than the global grid, which is sound because attention scores depend on the *difference*
of two positions: shifting a shard's whole buffer by a constant leaves every q-k pair alone.
A test that passed with global RoPE tables and local ones alike would not be testing this, so
the shards here sit at different offsets in the volume.
"""

import pytest
import torch

import ttnn

from ....layers.na3d import (
    AxisShard,
    build_device_plan,
    build_mesh_device_plan,
    plan_na3d,
    plan_na3d_mesh,
    uniform_halo,
)
from ....models.vae.diffvae_ltx import NABlock, rope_tables
from ....parallel.manager import CCLManager
from ....utils.check import assert_quality

_FABRIC = {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "require_exact_physical_num_devices": True}


def _random_state(dim: int, hidden: int, head_dim: int, generator) -> dict[str, torch.Tensor]:
    def randn(*shape):
        return torch.randn(*shape, generator=generator) * 0.05

    return {
        "norm1.weight": randn(dim),
        "norm2.weight": randn(dim),
        "attn.qkv.weight": randn(3 * dim, dim),
        "attn.qkv.bias": randn(3 * dim),
        "attn.proj.weight": randn(dim, dim),
        "attn.proj.bias": randn(dim),
        "attn.q_norm.weight": randn(head_dim),
        "attn.k_norm.weight": randn(head_dim),
        "mlp.w_gate.weight": randn(hidden, dim),
        "mlp.w_up.weight": randn(hidden, dim),
        "mlp.w_down.weight": randn(dim, hidden),
    }


@pytest.mark.parametrize(
    "mesh_device, device_params, dims, kernel",
    [
        ((4, 8), _FABRIC, (4, 32, 64), (3, 7, 7)),
        ((4, 8), _FABRIC, (4, 32, 64), (11, 11, 11)),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("dim, head_dim", [(128, 64)])
def test_sharded_block_matches_replicated(*, mesh_device, dims, kernel, dim, head_dim):
    mesh = tuple(mesh_device.shape)
    hidden = (int(dim * 4.0) + 15) // 16 * 16

    generator = torch.Generator().manual_seed(0)
    block = NABlock(dim, kernel, head_dim=head_dim, mesh_device=mesh_device)
    block.load_torch_state_dict(_random_state(dim, hidden, head_dim, generator))
    activation = torch.randn(1, *dims, dim, generator=generator)

    # Replicated: every device computes the whole volume, so device 0's answer is the reference.
    whole = build_device_plan(plan_na3d(dims, kernel), mesh_device=mesh_device)
    cos, sin = rope_tables(dims, block.attn.rope_dim_split, mesh_device=mesh_device)
    replicated = block(
        ttnn.from_torch(activation.reshape(-1, dim), device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
        dims=dims,
        cos=cos,
        sin=sin,
        device_plan=whole,
    )
    reference = ttnn.to_torch(
        replicated, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=[0, 1], mesh_shape=mesh)
    )
    reference = reference[: dims[0] * dims[1] * dims[2], :dim].reshape(1, *dims, dim)

    # Sharded: H over cluster axis 0, W over axis 1, T local.
    halo = (0, uniform_halo(dims[1], mesh[0], kernel[1]), uniform_halo(dims[2], mesh[1], kernel[2]))
    grid = [
        (AxisShard(dims[0], 0, dims[0]), h, w)
        for h in _axis_shards(dims[1], mesh[0])
        for w in _axis_shards(dims[2], mesh[1])
    ]
    plans = plan_na3d_mesh(grid, kernel, halo=halo)
    buffer_dims = plans[0].dims
    query_dims = plans[0].output_dims

    ccl = CCLManager(mesh_device=mesh_device, num_links=1, topology=ttnn.Topology.Linear)

    def exchange(tokens: ttnn.Tensor) -> ttnn.Tensor:
        """(queries, dim) on this shard -> (buffer, dim) with the neighbours' rows attached."""
        volume = ttnn.reshape(ttnn.to_layout(tokens, ttnn.ROW_MAJOR_LAYOUT), (1, *query_dims, dim))
        padded = ccl.neighbor_pad(
            volume,
            dims=[2, 3],
            pad_left=[halo[1], halo[2]],
            pad_right=[halo[1], halo[2]],
            padding_mode="replicate",
            axes=[0, 1],
            neighbor_sems=[ccl.get_np_ping_pong_semaphore(0), ccl.get_np_ping_pong_semaphore(1)],
            num_links=[1, 1],
        )
        return ttnn.to_layout(ttnn.reshape(padded, (-1, dim)), ttnn.TILE_LAYOUT)

    sharded_in = ttnn.from_torch(
        activation,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh, dims=[2, 3]),
    )
    sharded_in = ttnn.to_layout(ttnn.reshape(sharded_in, (-1, dim)), ttnn.TILE_LAYOUT)

    sharded = block(
        sharded_in,
        dims=buffer_dims,
        cos=rope_tables(buffer_dims, block.attn.rope_dim_split, mesh_device=mesh_device)[0],
        sin=rope_tables(buffer_dims, block.attn.rope_dim_split, mesh_device=mesh_device)[1],
        device_plan=build_mesh_device_plan(plans, mesh_device=mesh_device),
        exchange=exchange,
    )
    gathered = ttnn.to_torch(
        ttnn.reshape(sharded, (1, *query_dims, dim)),
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=[2, 3], mesh_shape=mesh),
    )

    assert tuple(gathered.shape) == tuple(reference.shape), f"{tuple(gathered.shape)} != {tuple(reference.shape)}"
    assert_quality(reference, gathered, pcc=0.999)


def _axis_shards(length: int, parts: int) -> list[AxisShard]:
    edges = [round(i * length / parts) for i in range(parts + 1)]
    return [AxisShard(length=length, start=a, stop=b) for a, b in zip(edges, edges[1:])]
