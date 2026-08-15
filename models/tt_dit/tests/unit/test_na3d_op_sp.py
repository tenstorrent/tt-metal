# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""SP-over-T for the op-backed NA3D executor: Q sharded over T across the mesh, K/V replicated, each
chip fed its global frame origin; outputs all-gathered along T. Held against the host reference."""

from __future__ import annotations

import pytest
import torch

import ttnn

from ...layers.na3d import na3d_torch, neighborhood_attention_3d_op_sp
from ...models.vae.diffvae_ltx import NABlock, default_rope_dim_split, rope_tables
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

    # Replicated in/out (the block's contract): the SP path mesh_partitions Q over T internally.
    q_tt, k_tt, v_tt = (
        from_torch(x, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT) for x in (q, k, v)
    )

    ccl_manager = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)
    actual = neighborhood_attention_3d_op_sp(
        q_tt, k_tt, v_tt, kernel_size=kernel, sp_axis=sp_axis, ccl_manager=ccl_manager, scale=1.0
    )

    assert tuple(actual.shape) == tuple(expected.shape), f"{tuple(actual.shape)} != {tuple(expected.shape)}"
    # After the all-gather the volume is identical on every chip, so extract one replica.
    assert_quality(expected, to_torch_replicated(actual), pcc=0.999)


@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True, ids=["ring"]
)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True, ids=["4x8"])
def test_na_block_op_sp_matches_op(*, mesh_device):
    """A whole DiffVAE NA block with na3d_backend="op_sp" gives the same result as "op" (replicated),
    so a stage can turn on SP-over-T for attention transparently. Verifies the block threading
    (ccl_manager, sp_axis) end-to-end through norm + qkv + q/k-norm + RoPE + NA3D + proj + SwiGLU."""
    torch.manual_seed(0)
    dim, head_dim, kernel = 128, 64, (3, 3, 3)
    dims = (8, 8, 8)
    sp_axis = 0
    tokens = dims[0] * dims[1] * dims[2]
    hidden = (int(dim * 4.0) + 15) // 16 * 16

    weights = {
        "norm1.weight": (dim,),
        "norm2.weight": (dim,),
        "attn.qkv.weight": (3 * dim, dim),
        "attn.qkv.bias": (3 * dim,),
        "attn.proj.weight": (dim, dim),
        "attn.proj.bias": (dim,),
        "attn.q_norm.weight": (head_dim,),
        "attn.k_norm.weight": (head_dim,),
        "mlp.w_gate.weight": (hidden, dim),
        "mlp.w_up.weight": (hidden, dim),
        "mlp.w_down.weight": (dim, hidden),
    }
    state = {name: torch.randn(shape) * 0.1 for name, shape in weights.items()}
    hidden_states = torch.randn(tokens, dim)
    cos, sin = rope_tables(dims, default_rope_dim_split(head_dim), mesh_device=mesh_device)
    ccl_manager = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)

    def run(backend: str) -> torch.Tensor:
        block = NABlock(
            dim,
            kernel,
            head_dim=head_dim,
            mesh_device=mesh_device,
            na3d_backend=backend,
            ccl_manager=ccl_manager if backend == "op_sp" else None,
            sp_axis=sp_axis if backend == "op_sp" else None,
        )
        block.load_state_dict({k: v.clone() for k, v in state.items()})
        tt_hidden = ttnn.from_torch(hidden_states, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        out = block(tt_hidden, dims=dims, cos=cos, sin=sin, device_plan=None)
        return to_torch_replicated(out)

    assert_quality(run("op"), run("op_sp"), pcc=0.999)


@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True, ids=["ring"]
)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True, ids=["4x8"])
@pytest.mark.parametrize("sp_axis", [0, 1], ids=["sp_rows", "sp_cols"])
@pytest.mark.parametrize("dims, kernel", [((8, 8, 8), (3, 3, 3)), ((8, 4, 8), (3, 3, 3))])
def test_na3d_op_sp_sharded_matches_host(*, mesh_device, sp_axis, dims, kernel):
    """Sharded-I/O SP attention (full-stage building block): q/k/v sharded over T, output sharded,
    K/V gathered internally. Reassembled, it matches the host full result."""
    from ...layers.na3d import neighborhood_attention_3d_op_sp_sharded

    T, H, W = dims
    heads, head_dim = 4, 64
    sp = list(mesh_device.shape)[sp_axis]
    if T % sp != 0 or (T // sp) * H * W % 32 != 0:
        pytest.skip(f"dims={dims} not shardable over sp={sp} with tile-aligned origin")

    torch.manual_seed(0)
    q, k, v = (torch.randn(1, T, H, W, heads, head_dim, dtype=torch.float32) for _ in range(3))
    expected = na3d_torch(q, k, v, kernel, scale=1.0).reshape(1, T, H, W, heads * head_dim)

    # Shard input over T (dim 1) along sp_axis; output comes back sharded the same way.
    shard_axes = [None] * 6
    shard_axes[1] = sp_axis
    q_tt, k_tt, v_tt = (
        from_torch(x, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_axes=shard_axes)
        for x in (q, k, v)
    )
    ccl_manager = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)
    out = neighborhood_attention_3d_op_sp_sharded(
        q_tt, k_tt, v_tt, dims=dims, kernel_size=kernel, sp_axis=sp_axis, ccl_manager=ccl_manager, scale=1.0
    )

    # Reassemble the T-sharded output (dim 1 along sp_axis) into the full volume.
    got = to_torch_replicated(out, mesh_axes=[None, sp_axis, None, None, None])
    assert_quality(expected, got, pcc=0.999)
