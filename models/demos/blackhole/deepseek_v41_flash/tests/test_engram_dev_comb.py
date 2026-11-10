# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Microbench of the cross-chip combine variants of the device Engram rows (partial [Ttot*48, 256] bf16 per chip, mostly zero rows): stock RS(axis0)+AR(axis1) vs wider row-major pages, tile layout, topology.
Each variant is checked equal to the baseline (sum of one non-zero partial and zeros: exact)."""
import os
import pytest
import torch
import ttnn
from models.demos.blackhole.deepseek_v41_flash.tests.test_engram_dev_device import chain_us


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [pytest.param({"l1_small_size": 16384, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 200_000_000}, id="ring")],
    indirect=True,
)
@torch.no_grad()
def test_engram_dev_comb(mesh_device):
    md = mesh_device
    mr, mc = tuple(md.shape)
    LIN, RING = ttnn.Topology.Linear, ttnn.Topology.Ring
    for ttot in [int(x) for x in os.environ.get("DSV41_COMB_TTOT", "64,96,48").split(",")]:
        nrows = ttot * 48
        g = torch.Generator().manual_seed(ttot)
        full = torch.randn(nrows, 256, generator=g).to(torch.bfloat16)
        owner = torch.randint(0, 32, (nrows,), generator=g)  # one owning chip per row, zeros elsewhere
        parts = torch.zeros(32, nrows, 256, dtype=torch.bfloat16)
        for c in range(32):
            m = owner == c
            parts[c][m] = full[m]
        part = ttnn.from_torch(parts.reshape(32 * nrows, 256), device=md, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ShardTensorToMesh(md, dim=0))
        part = ttnn.reshape(part, [1, 1, nrows, 256])
        ch = nrows // mr

        def rsar(x, topo, links, dim=2):
            rs = ttnn.reduce_scatter(x, dim=dim, cluster_axis=0, num_links=links, topology=topo)
            return ttnn.all_reduce(rs, cluster_axis=1, num_links=links, topology=topo)

        variants = {}
        variants["base_lin_l2"] = lambda: rsar(part, LIN, 2)
        variants["base_lin_l1"] = lambda: rsar(part, LIN, 1)
        variants["base_ring_l2"] = lambda: rsar(part, RING, 2)
        for f in (4, 8, 16):
            variants[f"wide{256*f}_lin"] = (lambda f=f: ttnn.reshape(rsar(ttnn.reshape(part, [1, 1, nrows // f, 256 * f]), LIN, 2), [1, 1, ch, 256]))
        variants["tile_lin"] = lambda: ttnn.to_layout(rsar(ttnn.to_layout(part, ttnn.TILE_LAYOUT), LIN, 2), ttnn.ROW_MAJOR_LAYOUT)
        def tile_pre():
            return ttnn.to_layout(part, ttnn.TILE_LAYOUT)
        tp = tile_pre()
        variants["tile_only_comm"] = lambda: rsar(tp, LIN, 2)
        ref = None
        for name, fn in variants.items():
            try:
                out = fn()
                ttnn.synchronize_device(md)
                got = [ttnn.to_torch(t).reshape(ch, 256) for t in ttnn.get_device_tensors(out)]
                if ref is None:
                    ref = got
                ok = all((a.view(torch.int16) == b.view(torch.int16)).all() or True for a, b in zip(got, ref))
                eq = all(((a.float() == b.float()) | (a.float().isnan() & b.float().isnan())).all() for a, b in zip(got, ref))
                us = chain_us(md, fn)
                print(f"COMB ttot={ttot} {name}: {us:.1f} us equal={bool(eq)}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"COMB ttot={ttot} {name}: FAILED {type(e).__name__} {str(e)[:150]}", flush=True)
