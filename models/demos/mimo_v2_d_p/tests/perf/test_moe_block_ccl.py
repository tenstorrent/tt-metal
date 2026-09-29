# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bulk-collective alternatives to MoE dispatch / combine, on the SP (dispatch) axis, at MiMo-V2 shapes.

Measures what a whole-block "all-gather x over SP -> local row gather -> experts -> local weighted reduce ->
reduce-scatter over SP" design would pay for its data movement, to compare with deepseek_prefill.dispatch /
combine (per (token, expert) row over the fabric):

  ag_{layout}_S{S}_l{L}      all_gather (1,1,S,H) bf16 over cluster_axis 0 (every chip gets the column's 2S tokens)
  rs_S{S}_l{L}               reduce_scatter (1,1,2S,H) bf16 over cluster_axis 0 (partials back to the token owners)
  gather_S{S}                ttnn.embedding: 4S rows picked by index from the gathered (2S, H) x (the flat buffer
                             built locally; 4S = the expected rows per chip on 2x2, 8 experts x 64/256)

    scripts/run_safe_pytest.sh --profile models/demos/mimo_v2_d_p/tests/perf/test_moe_block_ccl.py
"""

import os

import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.mimo_v2_d_p.tests.mesh import MESH_PARAMS

try:
    from tracy import signpost
except ImportError:  # pragma: no cover
    signpost = lambda *a, **k: None

H = 4096
SEQS = [int(s) for s in os.environ.get("MIMO_CCL_SEQ", "640,2048").split(",")]
LINKS = [int(s) for s in os.environ.get("MIMO_CCL_LINKS", "1,2").split(",")]
ITERS = int(os.environ.get("MIMO_CCL_ITERS", "3"))


def _timed(tag, fn):
    fn()  # warm-up (compile)
    ttnn.synchronize_device(_timed.dev)
    for _ in range(ITERS):
        signpost(f"{tag}_start")
        out = fn()
        ttnn.synchronize_device(_timed.dev)
        signpost(f"{tag}_end")
    return out


@MESH_PARAMS
def test_moe_block_ccl(mesh_device, device_params):
    _timed.dev = mesh_device
    sp_topo, _ = per_axis_topology()
    rows, cols = tuple(mesh_device.shape)
    shard = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=(2, None))
    for S in SEQS:
        xt = torch.randn(1, 1, rows * S, H)
        for layout, lname in ((ttnn.ROW_MAJOR_LAYOUT, "rm"), (ttnn.TILE_LAYOUT, "tile")):
            x = ttnn.from_torch(
                xt,
                device=mesh_device,
                layout=layout,
                dtype=ttnn.bfloat16,
                mesh_mapper=shard,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for L in LINKS:
                out = _timed(
                    f"ag_{lname}_S{S}_l{L}",
                    lambda: ttnn.all_gather(x, dim=2, cluster_axis=0, num_links=L, topology=sp_topo),
                )
                got = ttnn.to_torch(ttnn.get_device_tensors(out)[0])
                assert torch.equal(got.float(), xt.bfloat16().float()), f"all_gather {lname} L{L} mismatch"
                ttnn.deallocate(out)
            ttnn.deallocate(x)

        # partials: every chip holds (2S, H); rs sums over the column and leaves each chip its own S rows
        pt = torch.randn(rows, 1, rows * S, H)
        p = ttnn.from_torch(
            pt,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=(0, None)),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for L in LINKS:
            out = _timed(
                f"rs_S{S}_l{L}",
                lambda: ttnn.reduce_scatter(p, dim=2, cluster_axis=0, num_links=L, topology=sp_topo),
            )
            got = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float()
            ref = pt.bfloat16().float().sum(0, keepdim=True)[:, :, :S]
            assert torch.allclose(got, ref, atol=0.1, rtol=0.05), f"reduce_scatter L{L}: {(got - ref).abs().max()}"
            ttnn.deallocate(out)
        # 2-chip reduce-scatter as an exchange: all_gather each chip's partial for the OTHER row's tokens, keep the
        # peer's piece, add it to the own-row partial (what a p2p send + add costs, via the fast all_gather path)
        if rows == 2:
            other = ttnn.from_torch(
                pt[:, :, :S],
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=(0, None)),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            own = ttnn.slice(p, (0, 0, 0, 0), (1, 1, S, H))

            def xrs():
                g = ttnn.all_gather(other, dim=2, cluster_axis=0, topology=sp_topo)
                peer = ttnn.slice(g, (0, 0, S, 0), (1, 1, 2 * S, H))
                ttnn.deallocate(g)
                o = ttnn.add(own, peer)
                ttnn.deallocate(peer)
                return o

            ttnn.deallocate(_timed(f"xrs_S{S}", xrs))
            ttnn.deallocate(other)
            ttnn.deallocate(own)
        ttnn.deallocate(p)

        # local flat-buffer build from the gathered x: 4S rows by index (sorted by expert = arbitrary token order)
        xa = ttnn.from_torch(
            xt,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        it = torch.randint(0, rows * S, (1, 4 * S), dtype=torch.int32)
        idx = ttnn.from_torch(
            it,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint32,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        out = _timed(
            f"gather_S{S}",
            lambda: ttnn.embedding(idx, xa, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG),
        )
        got = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).reshape(-1, H).float()
        assert torch.equal(got, xt.reshape(-1, H)[it[0].long()].bfloat16().float()), "embedding gather mismatch"
        for t in (out, xa, idx):
            ttnn.deallocate(t)
