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


@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True, ids=["ring"]
)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True, ids=["4x8"])
def test_na_block_op_sp_sharded_matches_op(*, mesh_device):
    """SP-3a integration: a full NA block run on a T-SHARDED sequence (na3d_backend="op_sp_sharded",
    x + RoPE tables sharded over T) matches the replicated block, so a stage can run its blocks
    sharded -- pointwise ops stay 1/sp, attention gathers K/V. Reassembled, it matches "op"."""
    torch.manual_seed(0)
    dim, head_dim, kernel = 128, 64, (3, 3, 3)
    dims = (8, 8, 8)
    sp_axis = 0
    T, H, W = dims
    sp = list(mesh_device.shape)[sp_axis]
    t_local = T // sp
    tokens = T * H * W
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

    # Replicated reference.
    block_op = NABlock(dim, kernel, head_dim=head_dim, mesh_device=mesh_device, na3d_backend="op")
    block_op.load_state_dict({k: v.clone() for k, v in state.items()})
    x_full = ttnn.from_torch(hidden_states, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ref = to_torch_replicated(block_op(x_full, dims=dims, cos=cos, sin=sin, device_plan=None))

    # Sharded: x and the RoPE tables split over T; the block runs on this chip's slice.
    block_sp = NABlock(
        dim,
        kernel,
        head_dim=head_dim,
        mesh_device=mesh_device,
        na3d_backend="op_sp_sharded",
        ccl_manager=ccl_manager,
        sp_axis=sp_axis,
    )
    block_sp.load_state_dict({k: v.clone() for k, v in state.items()})
    x_shard = from_torch(
        hidden_states, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_axes=[sp_axis, None]
    )
    cos_shard = ttnn.mesh_partition(cos, dim=1, cluster_axis=sp_axis)
    sin_shard = ttnn.mesh_partition(sin, dim=1, cluster_axis=sp_axis)
    out_shard = block_sp(x_shard, dims=(t_local, H, W), cos=cos_shard, sin=sin_shard, device_plan=None)
    got = to_torch_replicated(out_shard, mesh_axes=[sp_axis, None])

    assert_quality(ref, got, pcc=0.999)


@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True, ids=["ring"]
)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True, ids=["4x8"])
@pytest.mark.parametrize("sp_axis", [0, 1], ids=["sp_rows", "sp_cols"])
@pytest.mark.parametrize("dims, kernel", [((4, 4, 32), (3, 3, 3)), ((2, 8, 32), (3, 3, 5))])
def test_na3d_op_sp_w_matches_host(*, mesh_device, sp_axis, dims, kernel):
    """Spatial-SP over W: replicated in/out, but the attention is split over W across the mesh by
    sharding Q over a W-outer flatten (K/V replicated) with a per-device W origin. Matches host."""
    from ...layers.na3d import neighborhood_attention_3d_op_sp_w

    T, H, W = dims
    heads, head_dim = 4, 64
    sp = list(mesh_device.shape)[sp_axis]
    if W % sp != 0:
        pytest.skip(f"W={W} not divisible by sp={sp}")
    if (W // sp) * T * H % 32 != 0:
        pytest.skip(f"shard origin not tile-aligned for dims={dims}, sp={sp}")

    torch.manual_seed(0)
    q, k, v = (torch.randn(1, T, H, W, heads, head_dim, dtype=torch.float32) for _ in range(3))
    expected = na3d_torch(q, k, v, kernel, scale=1.0).reshape(1, T, H, W, heads * head_dim)

    q_tt, k_tt, v_tt = (
        from_torch(x, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT) for x in (q, k, v)
    )
    ccl_manager = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)
    actual = neighborhood_attention_3d_op_sp_w(
        q_tt, k_tt, v_tt, kernel_size=kernel, sp_axis=sp_axis, ccl_manager=ccl_manager, scale=1.0
    )

    assert tuple(actual.shape) == tuple(expected.shape), f"{tuple(actual.shape)} != {tuple(expected.shape)}"
    assert_quality(expected, to_torch_replicated(actual), pcc=0.999)


@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True, ids=["ring"]
)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True, ids=["4x8"])
@pytest.mark.parametrize("sp_axis", [0, 1], ids=["sp_rows", "sp_cols"])
@pytest.mark.parametrize(
    "dims, kernel",
    # The last case has a non-tile-aligned (W/sp)*T*H -- the deterministic stages are all like this,
    # so it must be exact too (the executor tile-pads the sequence transparently).
    [((4, 4, 32), (3, 3, 3)), ((2, 8, 32), (3, 3, 5)), ((3, 2, 16), (3, 3, 3))],
)
def test_na3d_op_sp_w_sharded_matches_host(*, mesh_device, sp_axis, dims, kernel):
    """Sharded-I/O SP-over-W (full-stage building block): q/k/v sharded over W, output sharded, K/V
    gathered internally over a W-outer flatten. Reassembled, it matches the host full result --
    including a non-tile-aligned shard, which the deterministic stages need."""
    from ...layers.na3d import neighborhood_attention_3d_op_sp_w_sharded

    T, H, W = dims
    heads, head_dim = 4, 64
    sp = list(mesh_device.shape)[sp_axis]
    if W % sp != 0:
        pytest.skip(f"W={W} not divisible by sp={sp}")

    torch.manual_seed(0)
    q, k, v = (torch.randn(1, T, H, W, heads, head_dim, dtype=torch.float32) for _ in range(3))
    expected = na3d_torch(q, k, v, kernel, scale=1.0).reshape(1, T, H, W, heads * head_dim)

    # Shard input over W (dim 3) along sp_axis; output comes back sharded the same way.
    shard_axes = [None] * 6
    shard_axes[3] = sp_axis
    q_tt, k_tt, v_tt = (
        from_torch(x, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_axes=shard_axes)
        for x in (q, k, v)
    )
    ccl_manager = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)
    out = neighborhood_attention_3d_op_sp_w_sharded(
        q_tt, k_tt, v_tt, dims=dims, kernel_size=kernel, sp_axis=sp_axis, ccl_manager=ccl_manager, scale=1.0
    )

    # Reassemble the W-sharded output (dim 3 along sp_axis) into the full volume.
    got = to_torch_replicated(out, mesh_axes=[None, None, None, sp_axis, None])
    assert_quality(expected, got, pcc=0.999)


@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True, ids=["ring"]
)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True, ids=["4x8"])
@pytest.mark.parametrize("sp_axis", [1], ids=["sp_cols"])
@pytest.mark.parametrize(
    "dims, kernel, stride",
    # Strides must divide their axis and stay <= the kernel. W is the sharded axis, so a stride on it
    # also exercises the per-device W origin: each shard's group boundaries have to line up globally.
    [
        ((4, 4, 32), (3, 3, 3), (2, 2, 2)),
        ((4, 4, 32), (3, 3, 3), (1, 1, 2)),
        # stride_w == the 4-wide W shard: the group boundary lands exactly on the shard boundary, so a
        # per-device origin that forgot to be global would snap to the wrong leader.
        ((2, 8, 32), (3, 3, 5), (2, 2, 4)),
    ],
)
def test_na3d_op_sp_w_sharded_gna_stride_matches_host(*, mesh_device, sp_axis, dims, kernel, stride):
    """GNA on the production sharded path: the W-sharded executor at a stride vs the host reference
    planned at the same stride. The stride is given in physical (t,h,w) and the executor permutes it
    to op order alongside the kernel -- a mismatch there would show up as a wrong window, so this is
    the test that pins the axis-order plumbing."""
    from ...layers.na3d import neighborhood_attention_3d_op_sp_w_sharded

    T, H, W = dims
    heads, head_dim = 4, 64
    sp = list(mesh_device.shape)[sp_axis]
    if W % sp != 0:
        pytest.skip(f"W={W} not divisible by sp={sp}")
    if (W // sp) % stride[2] != 0:
        pytest.skip(f"W shard {W // sp} not divisible by stride_w={stride[2]}")

    torch.manual_seed(0)
    q, k, v = (torch.randn(1, T, H, W, heads, head_dim, dtype=torch.float32) for _ in range(3))
    expected = na3d_torch(q, k, v, kernel, scale=1.0, stride=stride).reshape(1, T, H, W, heads * head_dim)

    shard_axes = [None] * 6
    shard_axes[3] = sp_axis
    q_tt, k_tt, v_tt = (
        from_torch(x, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_axes=shard_axes)
        for x in (q, k, v)
    )
    ccl_manager = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)
    out = neighborhood_attention_3d_op_sp_w_sharded(
        q_tt,
        k_tt,
        v_tt,
        dims=dims,
        kernel_size=kernel,
        sp_axis=sp_axis,
        ccl_manager=ccl_manager,
        scale=1.0,
        gna_stride=stride,
    )

    got = to_torch_replicated(out, mesh_axes=[None, None, None, sp_axis, None])
    assert_quality(expected, got, pcc=0.999)


@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True, ids=["ring"]
)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True, ids=["4x8"])
@pytest.mark.parametrize("dims, kernel", [((4, 4, 32), (3, 3, 3)), ((3, 2, 16), (3, 3, 3))])
def test_na3d_op_sp_w_sharded_tp_matches_host(*, mesh_device, dims, kernel):
    """W-SP composed with TENSOR PARALLELISM OVER HEADS on the orthogonal mesh axis. q/k/v are
    W-sharded over sp_axis (cols, 8) and replicated over tp_axis (rows, 4); the executor partitions
    the 4 heads over tp_axis (1 head/chip), runs the flash on that head, and gathers the heads back
    before returning. Reassembled over W, it matches the host full result -- proving the two axes
    compose. The head-gather makes the output replicated over tp_axis, so W-reassembly is unchanged."""
    from ...layers.na3d import neighborhood_attention_3d_op_sp_w_sharded

    sp_axis, tp_axis = 1, 0  # W over the 8-axis, heads over the 4-axis
    T, H, W = dims
    heads, head_dim = 4, 64
    sp = list(mesh_device.shape)[sp_axis]
    tp = list(mesh_device.shape)[tp_axis]
    assert heads % tp == 0 and W % sp == 0

    torch.manual_seed(0)
    q, k, v = (torch.randn(1, T, H, W, heads, head_dim, dtype=torch.float32) for _ in range(3))
    expected = na3d_torch(q, k, v, kernel, scale=1.0).reshape(1, T, H, W, heads * head_dim)

    # Shard W (dim 3) over sp_axis; heads stay full and replicated over tp_axis (the executor slices
    # them). Output comes back W-sharded over sp_axis, replicated over tp_axis.
    shard_axes = [None] * 6
    shard_axes[3] = sp_axis
    q_tt, k_tt, v_tt = (
        from_torch(x, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_axes=shard_axes)
        for x in (q, k, v)
    )
    ccl_manager = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)
    out = neighborhood_attention_3d_op_sp_w_sharded(
        q_tt,
        k_tt,
        v_tt,
        dims=dims,
        kernel_size=kernel,
        sp_axis=sp_axis,
        ccl_manager=ccl_manager,
        scale=1.0,
        tp_axis=tp_axis,
    )

    got = to_torch_replicated(out, mesh_axes=[None, None, None, sp_axis, None])
    assert_quality(expected, got, pcc=0.999)


@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True, ids=["ring"]
)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True, ids=["4x8"])
@pytest.mark.parametrize("sp_axis", [0, 1], ids=["sp_rows", "sp_cols"])
def test_na_block_op_sp_w_sharded_matches_op(*, mesh_device, sp_axis):
    """SP over W for a deterministic-stage NA block: run on a W-SHARDED sequence (x + RoPE tables
    split over W) and match the replicated block. Uses a non-tile-aligned shard, as the det stages
    do. This is the det-stage-SP analog of test_na_block_op_sp_sharded_matches_op (T)."""
    torch.manual_seed(0)
    dim, head_dim, kernel = 128, 64, (3, 3, 3)
    dims = (3, 2, 16)  # (W/sp)*T*H is not tile-aligned -- exactly the det-stage regime
    T, H, W = dims
    sp = list(mesh_device.shape)[sp_axis]
    if W % sp != 0:
        pytest.skip(f"W={W} not divisible by sp={sp}")
    w_local = W // sp
    tokens = T * H * W
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

    # Replicated reference.
    block_op = NABlock(dim, kernel, head_dim=head_dim, mesh_device=mesh_device, na3d_backend="op")
    block_op.load_state_dict({k: v.clone() for k, v in state.items()})
    x_full = ttnn.from_torch(hidden_states, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ref = to_torch_replicated(block_op(x_full, dims=dims, cos=cos, sin=sin, device_plan=None))

    # W-sharded: the flat (t, h, w) rows are reordered to (device, t, h, w_local) contiguous before
    # sharding on the row dim; the RoPE tables split over W with a plain mesh_partition on the W dim.
    block_sp = NABlock(
        dim,
        kernel,
        head_dim=head_dim,
        mesh_device=mesh_device,
        na3d_backend="op_sp_w_sharded",
        ccl_manager=ccl_manager,
        sp_axis=sp_axis,
    )
    block_sp.load_state_dict({k: v.clone() for k, v in state.items()})
    reordered = hidden_states.reshape(T, H, sp, w_local, dim).permute(2, 0, 1, 3, 4).reshape(sp * T * H * w_local, dim)
    x_shard = from_torch(
        reordered, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_axes=[sp_axis, None]
    )
    cos_shard = ttnn.mesh_partition(cos, dim=3, cluster_axis=sp_axis)
    sin_shard = ttnn.mesh_partition(sin, dim=3, cluster_axis=sp_axis)
    out_shard = block_sp(x_shard, dims=(T, H, w_local), cos=cos_shard, sin=sin_shard, device_plan=None)

    got = to_torch_replicated(out_shard, mesh_axes=[sp_axis, None])
    got = got.reshape(sp, T, H, w_local, dim).permute(1, 2, 0, 3, 4).reshape(tokens, dim)
    assert_quality(ref, got, pcc=0.999)


@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True, ids=["ring"]
)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True, ids=["4x8"])
@pytest.mark.parametrize("sp_axis", [0, 1], ids=["sp_rows", "sp_cols"])
def test_diffusion_nablock_op_sp_w_sharded_matches_op(*, mesh_device, sp_axis):
    """SP-over-W integration for stage 5: a full DiffusionNABlock (context injection + AdaLN attn +
    AdaLN SwiGLU) run on a W-SHARDED sequence -- x, context and the RoPE frame table split over W --
    matches the replicated block. Proves the stage's block, RoPE W-sharding and sharded-I/O executor
    compose, so stage 5 can keep its activation W-sharded (pointwise ops 1/sp, K/V gathered in attn).
    """
    from ...models.vae.diffvae_ltx_stage5 import (
        NUM_ADALN_CHUNKS,
        DiffusionNABlock,
        DiffVAEStage5Config,
        Grid,
        _bands,
        _build_rope_tables,
        _RopeParts,
        _RopeTables,
        default_rope_dim_split,
    )

    dim, head_dim = 128, 64
    num_heads = dim // head_dim
    kernel = (3, 3, 3)
    T, H, W = 4, 4, 32
    sp = list(mesh_device.shape)[sp_axis]
    if W % sp != 0 or (W // sp) * T * H % 32 != 0:
        pytest.skip(f"W={W} not shardable over sp={sp} with tile-aligned origin")
    wl = W // sp
    hidden = 4 * dim
    config = DiffVAEStage5Config(
        dim=dim, head_dim=head_dim, kernel_size=kernel, context_channels=dim, mlp_hidden=hidden, num_blocks=1
    )
    grid = Grid(1, T, H, W)
    tokens = T * H * W

    torch.manual_seed(0)
    shapes = {
        "context_proj.weight": (dim, dim),
        "context_proj.bias": (dim,),
        "scale_shift_table": (NUM_ADALN_CHUNKS, dim),
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
    # Small weights: the static gates upstream folds into the residual-write projections are absent
    # here, so unscaled randoms would compound across the residual adds into a meaningless range.
    state = {name: torch.randn(shape) * 0.1 for name, shape in shapes.items()}
    hidden_states = torch.randn(1, 1, tokens, dim)
    context = torch.randn(1, 1, tokens, dim)
    modulation = torch.randn(1, 1, 1, NUM_ADALN_CHUNKS * dim) * 0.1

    ccl_manager = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)
    bands = _bands(T, frames=None, kernel=kernel[0])
    tables = _build_rope_tables(
        grid,
        dim_split=default_rope_dim_split(head_dim),
        base=config.rope_base,
        num_heads=num_heads,
        mesh_device=mesh_device,
        dtype=ttnn.bfloat16,
    )
    mod_tt = ttnn.from_torch(modulation, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    # Replicated reference.
    block_op = DiffusionNABlock(config, mesh_device=mesh_device, dtype=ttnn.bfloat16, na3d_backend="op")
    block_op.load_state_dict({k: v.clone() for k, v in state.items()})
    x_full = ttnn.from_torch(hidden_states, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    ctx_full = ttnn.from_torch(context, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    band_tables = tuple(tables.frames(b.pad_lo, b.pad_hi) for b in bands)
    ref = to_torch_replicated(block_op([x_full], ctx_full, mod_tt, grid, bands, band_tables)[0])

    # W-sharded: x, context and the RoPE frame table split over W (the inner axis), so the flat
    # T-outer rows are reordered to (device, t, h, w_local) contiguous before sharding on the site dim.
    def shard_rows(flat: torch.Tensor, channels: int) -> ttnn.Tensor:
        reordered = flat.reshape(T, H, sp, wl, channels).permute(2, 0, 1, 3, 4).reshape(1, 1, sp * T * H * wl, channels)
        return from_torch(
            reordered,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_axes=[None, None, sp_axis, None],
        )

    def shard_frame(part: ttnn.Tensor) -> ttnn.Tensor:
        host = to_torch_replicated(part).reshape(H, W, num_heads, head_dim)
        host = (
            host.reshape(H, sp, wl, num_heads, head_dim)
            .permute(1, 0, 2, 3, 4)
            .reshape(1, 1, sp * H * wl * num_heads, head_dim)
        )
        return from_torch(
            host,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_axes=[None, None, sp_axis, None],
        )

    sharded_tables = _RopeTables(
        frame=_RopeParts(shard_frame(tables.frame.cos), shard_frame(tables.frame.sin)),
        time=tables.time,
        rows_per_frame=H * wl * num_heads,
    )
    block_sp = DiffusionNABlock(
        config,
        mesh_device=mesh_device,
        dtype=ttnn.bfloat16,
        ccl_manager=ccl_manager,
        na3d_backend="op_sp_w_sharded",
        sp_axis=sp_axis,
    )
    block_sp.load_state_dict({k: v.clone() for k, v in state.items()})
    x_shard = shard_rows(hidden_states, dim)
    ctx_shard = shard_rows(context, dim)
    band_tables_sp = tuple(sharded_tables.frames(b.pad_lo, b.pad_hi) for b in bands)
    out_shard = block_sp([x_shard], ctx_shard, mod_tt, grid, bands, band_tables_sp)[0]

    # Reassemble over W: gather the site dim, then undo the (device, t, h, w_local) reordering.
    got = to_torch_replicated(out_shard, mesh_axes=[None, None, sp_axis, None])
    got = got.reshape(sp, T, H, wl, dim).permute(1, 2, 0, 3, 4).reshape(1, 1, tokens, dim)
    assert_quality(ref, got, pcc=0.999)


@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True, ids=["ring"]
)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True, ids=["4x8"])
@pytest.mark.parametrize("flat", [False, True], ids=["volume", "flat"])
@pytest.mark.parametrize(
    "dims, kernel, heads",
    [
        ((4, 8, 64), (3, 3, 3), 4),
        ((4, 8, 64), (3, 3, 3), 8),
        ((6, 16, 128), (3, 5, 5), 4),
        ((6, 16, 128), (3, 5, 5), 8),
        # The deterministic stages' shard: W_local 15, which only a width-1 brick can plan.
        # Stage 2's window and 4 heads per chip; stage 3's window and 2 per chip; a shallow T
        # that forces the (4,8,1) brick; and stage 4's W_local 30 at its (8,2,2) brick.
        ((8, 16, 120), (3, 7, 7), 16),
        ((8, 16, 120), (3, 5, 5), 8),
        ((4, 8, 120), (3, 7, 7), 16),
        ((8, 8, 240), (3, 5, 5), 8),
    ],
    ids=[
        "w_local8_1head",
        "w_local8_2heads",
        "w_local16_1head",
        "w_local16_2heads",
        "w_local15_win7_4heads",
        "w_local15_win5_2heads",
        "w_local15_shallowT_4heads",
        "w_local30_win5_2heads",
    ],
)
def test_bricked_w_sharded_tp_matches_host(*, mesh_device, heads, dims, kernel, flat):
    """The BRICKED W-sharded executor under TP over heads, with more than one head per chip.

    Stage 5 runs this path at one head per chip (4 heads, TP4), where the site-major op output
    and the head-major layout the head all-gather wants are the same bytes. The deterministic
    stages have 16/8/8 heads, so TP4 leaves 4/2/2 per chip and the reassembly needs a real
    site-major -> head-major permute before the gather (plan B2). The two-head cases are what used
    to trip the ``head_count == 1`` assert; the one-head cases are the control that the stage-5
    path is unchanged. Heads are presharded over tp_axis here because the bricked executor does
    not slice them itself (the column-parallel qkv does that in production).

    ``flat`` hands the executor the HEAD-major ``(1, heads_local, tokens, head_dim)`` TILE that
    ``nlp_create_qkv_heads`` emits -- the deterministic stages' ``flat_seq`` handoff -- instead of
    the 6-D volume. At more than one head per chip the executor must transpose it into its
    site-major volume; a reshape "as a view" was right only at one head and interleaved heads with
    sites otherwise, which this case would catch as a PCC miss.

    The W_local 15 cases are the deterministic stages' geometry (B1): the middle shards' origins
    are odd, so only a width-1 brick is brick-aligned, and the K/V halo has to move in bricked order
    (stick 32 * channels) since a 1-wide natural halo cannot fold up to a safe stick width.
    """
    from ...layers.neighborhood_attention import neighborhood_attention_3d_bricked_w_sharded

    sp_axis, tp_axis = 1, 0  # W over the 8-axis, heads over the 4-axis
    T, H, W = dims
    head_dim = 64
    sp = list(mesh_device.shape)[sp_axis]
    tp = list(mesh_device.shape)[tp_axis]
    assert heads % tp == 0 and W % sp == 0

    torch.manual_seed(0)
    q, k, v = (torch.randn(1, T, H, W, heads, head_dim, dtype=torch.float32) for _ in range(3))
    expected = na3d_torch(q, k, v, kernel, scale=1.0).reshape(1, T, H, W, heads * head_dim)

    if flat:
        # Per W-band, head-major: (sp, heads, T*H*W_local, hd), so sharding dim 0 over sp_axis and
        # dim 1 over tp_axis leaves each chip (1, heads/tp, tokens_local, hd) -- create_heads' shape.
        w_local = W // sp

        def head_major(x):
            bands = x.reshape(1, T, H, sp, w_local, heads, head_dim).permute(3, 5, 1, 2, 4, 0, 6)
            return bands.reshape(sp, heads, T * H * w_local, head_dim).contiguous()

        q_tt, k_tt, v_tt = (
            from_torch(
                head_major(x),
                device=mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_axes=[sp_axis, tp_axis, None, None],
            )
            for x in (q, k, v)
        )
        assert tuple(q_tt.shape) == (1, heads // tp, T * H * w_local, head_dim), tuple(q_tt.shape)
    else:
        shard_axes = [None, None, None, sp_axis, tp_axis, None]  # W over sp_axis, heads over tp_axis
        q_tt, k_tt, v_tt = (
            from_torch(x, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_axes=shard_axes)
            for x in (q, k, v)
        )
    ccl_manager = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)
    out = neighborhood_attention_3d_bricked_w_sharded(
        q_tt,
        k_tt,
        v_tt,
        dims=dims,
        kernel_size=kernel,
        sp_axis=sp_axis,
        ccl_manager=ccl_manager,
        scale=1.0,
        tp_axis=tp_axis,
        heads_presharded=True,
    )

    # The TP branch returns (1, 1, sites_local, heads * head_dim) TILE, W-sharded over sp_axis and
    # replicated over tp_axis, with sites in (t, h, w_local) order -- so concatenating the shards on
    # the site axis interleaves W-bands, and the reassembly has to put W back together explicitly.
    assert tuple(out.shape) == (1, 1, T * H * (W // sp), heads * head_dim), tuple(out.shape)
    got = to_torch_replicated(out, mesh_axes=[None, None, sp_axis, None])
    got = got.reshape(sp, T, H, W // sp, heads * head_dim).permute(1, 2, 0, 3, 4).reshape(1, T, H, W, heads * head_dim)
    assert_quality(expected, got, pcc=0.999)
