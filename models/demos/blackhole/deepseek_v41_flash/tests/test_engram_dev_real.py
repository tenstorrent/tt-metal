# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Smoke of DevEngramTables with the REAL checkpoint state / table shapes (only chips 0 and 31 loaded): allocation size per bank, hash ids vs the host hash of the real state, rows() runs. Rows are checked only where the id falls on a loaded chip."""
import numpy as np
import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt import engram_dev as ed
from models.demos.blackhole.deepseek_v41_flash.tt.engram import HostEngramRows
from models.demos.blackhole.deepseek_v41_flash.tt.engram_ragged import RaggedNgramHash


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [pytest.param({"l1_small_size": 16384, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 100_000_000}, id="ring")],
    indirect=True,
)
@torch.no_grad()
def test_engram_dev_real(mesh_device):
    md = mesh_device
    mib = 2**20
    mv0 = ttnn.get_memory_view(md, ttnn.BufferType.DRAM)
    hr = HostEngramRows((1, 14), max_batch_size=16, max_seq_len=4096)
    tabs = ed.DevEngramTables(md, hr, [1, 14], log=lambda s: print(s, flush=True), chips=[0, 31])
    ttnn.synchronize_device(md)
    mv1 = ttnn.get_memory_view(md, ttnn.BufferType.DRAM)
    print(f"DEVENG_DRAM tables: {(mv1.total_bytes_allocated_per_bank - mv0.total_bytes_allocated_per_bank) / mib:.1f} MiB/bank (free {mv1.total_bytes_free_per_bank / mib:.1f})", flush=True)
    print(f"DEVENG real state: W={tabs.W} H={tabs.H} NL={tabs.NL} nrows={tabs.nrows} rpc={tabs.rpc} pad={tabs.pad} dead={tabs.dead}", flush=True)
    st = hr.engram.hash
    U, n = 4, 4
    B = U * 4
    ttot = B * n
    dv = ed.DevEngramRows(tabs, U * n, n)
    g = torch.Generator().manual_seed(1)
    ref = RaggedNgramHash(st)
    st_np = dict(cache=st.cache.numpy(), tmap=tabs.tmap, dead=tabs.dead)
    base = torch.randint(5, 3000, (B,), generator=g)
    for b in range(B):  # a history of real-looking token ids
        L = int(base[b])
        st.cache[b, :L] = st.token_map[torch.randint(0, 129000, (L,), generator=g)]
    mp = ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(4, 8))
    ctrl_h = torch.zeros(ttot, dv.CW, dtype=torch.int32)
    ctrl_dev = None
    for rnd in range(4):
        X = torch.randint(0, 129000, (B, n), generator=g)
        want = ref(X, base)  # [B, n, 2, 24], writes the cache
        # fill_ctrl must not redo the cache write with different data: it writes the same compressed tokens
        ed.fill_ctrl(ctrl_h, X, base, st_np, w=tabs.W)
        host_c = ttnn.from_torch(ctrl_h, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=mp)
        if ctrl_dev is None:
            ctrl_dev = ttnn.to_device(host_c, md, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        else:
            ttnn.copy_host_to_device_tensor(host_c, ctrl_dev)
        out = dv.rows(ctrl_dev)
        ttnn.synchronize_device(md)
        ids = ttnn.to_torch(ttnn.get_device_tensors(dv.ids)[0]).reshape(ttot, 64)[:, :48].reshape(ttot, 2, 24).long()
        assert (ids == want.reshape(ttot, 2, 24)).all(), f"round {rnd}: real-state ids differ"
        m = torch.randint(0, n, (B,), generator=g)
        base = base + 1 + m
    print("DEVENG_REAL ids bit-exact vs RaggedNgramHash on the real state (4 rounds), rows() ran", flush=True)
    print("DEVENG_DRAM end", ttnn.get_memory_view(md, ttnn.BufferType.DRAM).total_bytes_free_per_bank / mib, flush=True)
