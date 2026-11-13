# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""S1: layer-1 MoE PREFILL (N tokens per mesh row, default 512; unified dispatch/combine pipeline reading the decode ring weights in place) on REAL activations
vs the fp32 oracle; routed + shared (shared on the replicated chunk, summed on host). Env HY4_N, HY4_LAYER, HY4_ROUTER."""
import os
import statistics
import time

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dev_common import ACTS, close_mesh, open_mesh, p, pcc
from models.demos.blackhole.hy4_preview.tt import hy4_dense as D
from models.demos.blackhole.hy4_preview.tt.hy4_moe import Hy4MoEBlock
from models.demos.blackhole.hy4_preview.tt.hy4_moe_weights import load_moe_layer
from models.demos.blackhole.hy4_preview.tt.hy4_prefill_moe import COLS, ROWS, Hy4UnifiedMoE, get_shared, moe_cols
from models.demos.gpt_oss.config import mesh_4x8
from models.demos.gpt_oss.tt.ccl import CCLManager

LAYER = int(os.environ.get("HY4_LAYER", "1"))
N = int(os.environ.get("HY4_N", "512"))
HID = 6144
acts = torch.load(ACTS + "acts_l0_l1.pt")
gs = torch.load(ACTS + "l1_gate_shared.pt")
X = torch.cat([h.reshape(-1, HID) for h, y, r in acts["moe"]])
Y = torch.cat([y.reshape(-1, HID) for h, y, r in acts["moe"]])
md = open_mesh(400_000_000)
try:
    W = load_moe_layer(LAYER)
    blk = Hy4MoEBlock(md, W, batch_per_device=1)
    blk.warmup()
    shared_w = [W[k][256] for k in ("shared_w0", "shared_w1", "shared_w2")]
    t0 = time.time()
    um = Hy4UnifiedMoE(md, LAYER, ring=(blk.decode.expert_state.tt_w0_w1, blk.decode.expert_state.tt_w2))
    sh = get_shared(md, N)
    p(f"unified moe ready {time.time()-t0:.1f}s max_buf={sh.max_buf} max_per_expert={sh.max_per_expert}")
    mc, ccl = mesh_4x8(), CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
    shared = D.make_shared(md, *shared_w, decode=False)
    idx = torch.stack(
        [(torch.arange(N) + 80 * r) % X.shape[0] for r in range(ROWS)]
    )  # row r: N real tokens (offset 80 r)
    x = X[idx.flatten()].reshape(ROWS, N, HID).to(torch.bfloat16)
    xd = ttnn.from_torch(
        x.reshape(ROWS, 1, N, HID),
        device=md,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(ROWS, COLS)),
    )
    n_own = N // COLS
    xo = x.reshape(ROWS, N // 256, COLS, 32, HID).permute(0, 2, 1, 3, 4).reshape(ROWS, COLS, n_own, HID)
    xod = ttnn.from_torch(
        xo,
        device=md,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(0, 1), mesh_shape=(ROWS, COLS)),
    )
    dev = lambda t, r=0, c=0: ttnn.to_torch(ttnn.get_device_tensors(t)[r * COLS + c])

    def routed():
        return moe_cols(um, blk.gate, xod, mc, ccl)

    for it in range(2):
        own = routed()
        ttnn.synchronize_device(md)
    full = torch.zeros(ROWS, N, HID)
    for r in range(ROWS):
        for c in range(COLS):
            o = dev(own, r, c).float().reshape(N // 256, 32, HID)
            for g in range(N // 256):
                full[r, 256 * g + 32 * c : 256 * g + 32 * c + 32] = o[g]
    shp = D.allreduce_cols(shared.forward(xd))
    ttnn.synchronize_device(md)
    sh_full = torch.stack([dev(shp, r, 0).float().reshape(N, HID) for r in range(ROWS)])
    total = full + sh_full
    ref = Y[idx.flatten()].reshape(ROWS, N, HID)
    for r in range(ROWS):
        pt = [pcc(total[r, t], ref[r, t]) for t in range(N)]
        p(
            f"prefill row {r}: routed+shared PCC vs oracle {pcc(total[r], ref[r]):.5f} (per-token min {min(pt):.4f} median {statistics.median(pt):.4f}); finite {bool(torch.isfinite(full[r]).all())}"
        )
    # latency
    ts = []
    for _ in range(5):
        t0 = time.perf_counter()
        o = routed()
        ttnn.synchronize_device(md)
        ts.append((time.perf_counter() - t0) * 1e3)
        ttnn.deallocate(o)
    p(f"prefill MoE routed chunk N={N}/row eager: median {statistics.median(ts):.2f} ms (min {min(ts):.2f})")
finally:
    close_mesh(md)
