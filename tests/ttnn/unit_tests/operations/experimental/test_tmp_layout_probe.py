# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TEMPORARY layout probe -- not for commit.

Checks the ONE assumption the ring_mla-side un-stripe rests on: that a plain (un-striped) TP gather of a
block-cyclic shard lands the slab RANK-MAJOR with a per-rank stride of the WHOLE shard, i.e.

    logical  c*(tp*R) + t*R + o      ->    physical  t*(n_chunks*R) + c*R + o

Gathers the same source both ways and compares against host-built expectations, so a mismatch says
whether the layout model is wrong (rather than the kernel plumbing).
"""

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from tests.ttnn.unit_tests.operations.experimental.test_high_bw_all_gather import _device_params, _make_tensor


@run_for_blackhole("high_bw_all_gather requires Blackhole fabric")
@pytest.mark.parametrize("device_params", [_device_params(ttnn.FabricConfig.FABRIC_2D)], indirect=True)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize(
    "layout,dtype", [(ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16), (ttnn.TILE_LAYOUT, ttnn.bfloat8_b)], ids=["rm", "tile"]
)
@pytest.mark.parametrize("num_links", [1, 2], ids=["links1", "links2"])
@pytest.mark.parametrize("slots", [1, 3], ids=["slot1", "slot3"])
@pytest.mark.parametrize("active_chunks", [1, 3], ids=["active1", "active3"])
def test_tmp_layout_probe(mesh_device, active_chunks, layout, dtype, num_links, slots):
    sp, tp = tuple(mesh_device.shape)
    n_chunks, rows_dev, width = 3, 64, 64  # 64 rows = 2 seq tiles
    chunk_global = rows_dev * sp * tp
    rows_sp = chunk_global // sp

    torch.manual_seed(0)
    slot = slots - 1
    host = torch.rand((slots, n_chunks, chunk_global, width), dtype=torch.bfloat16)
    shards = [
        host[:, :, d * rows_dev : (d + 1) * rows_dev].reshape(slots, 1, n_chunks * rows_dev, width)
        for d in range(sp * tp)
    ]
    source = _make_tensor(
        mesh_device,
        torch.cat(shards, dim=2),
        dtype,
        layout,
        ttnn.ShardTensorToMesh(mesh_device, dim=2),
    )
    dist_shape = ttnn.MeshShape(sp, tp)
    source.update_tensor_topology(
        ttnn.TensorTopology(
            dist_shape,
            [ttnn.PlacementShard(2), ttnn.PlacementShard(2)],
            [ttnn.MeshCoordinate([c[i] for i in range(c.dims())]) for c in ttnn.MeshCoordinateRange(dist_shape)],
        )
    )
    sp_only = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(sp, tp), dims=[2, None])

    def gather(striped):
        out = _make_tensor(
            mesh_device,
            torch.zeros((1, 1, n_chunks * rows_sp * sp, width), dtype=torch.bfloat16),
            dtype,
            layout,
            sp_only,
        )
        kw = {"input_stripe_size": rows_dev} if striped else {}
        out = ttnn.experimental.high_bw_all_gather(
            source,
            dim=2,
            output_tensor=out,
            num_links=num_links,
            input_batch_index=slot,
            cluster_axis=1,
            gathered_dim_size=active_chunks * tp * rows_dev,
            **kw,
        )
        # SP rank 0's slab.
        return ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float()[0, 0]

    got_striped = gather(striped=True)
    got_plain = gather(striped=False)

    # SP rank 0 owns rows [s*C/sp, +C/sp) of every chunk -> chunk-major natural order.
    natural = torch.cat([host[slot, c, 0:rows_sp] for c in range(n_chunks)], dim=0).float()

    R = rows_dev
    active_rows = active_chunks * tp * R

    # 1. Striped gather == natural order over the active prefix (the branch's current behaviour).
    torch.testing.assert_close(got_striped[:active_rows], natural[:active_rows], rtol=0, atol=0.2)

    # 2. Plain gather: does logical L sit at t*(n_chunks*R) + c*R + o?
    def physical(L):
        c, rem = divmod(L, tp * R)
        t, o = divmod(rem, R)
        return t * (n_chunks * R) + c * R + o

    remapped = torch.stack([got_plain[physical(L)] for L in range(active_rows)], dim=0)
    torch.testing.assert_close(remapped, natural[:active_rows], rtol=0, atol=0.2)
    print(
        f"LAYOUT_PROBE layout={layout} active={active_chunks} links={num_links} slots={slots}: rank-major + full-shard stride CONFIRMED"
    )
