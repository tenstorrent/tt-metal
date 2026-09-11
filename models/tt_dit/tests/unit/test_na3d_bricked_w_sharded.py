# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The bricked W-sharded neighborhood-attention executor against the host reference, on the 4x8 mesh:
W over the size-8 axis, heads presharded over the size-4 axis, in the volume and flat input forms."""

from __future__ import annotations

import pytest
import torch

import ttnn

from ...layers.na3d import na3d_torch
from ...parallel.manager import CCLManager
from ...utils.check import assert_quality
from ...utils.tensor import from_torch
from ...utils.tensor import to_torch as to_torch_replicated


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
    ``nlp_create_qkv_heads`` emits, the form the deterministic stages' projections produce, instead of
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
