# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Attention core at Gemma-4 shapes: block-cyclic KV-cache write + cache-backed ring SDPA over SP.

Sliding layers: 16 Q / 8 KV heads, D=256, window 1024. Full layers: 16 Q / 2 KV heads, D=512, causal.
scale = 1.0 (Gemma-4 normalises Q/K), no sinks. Writes ``n_chunks`` chunks and attends the last one
against the accumulated cache, vs a torch golden.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_d_p.tt.mla.utils import rotated_chip_positions
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.gemma4_26b_d_p.reference.blocks import attention_mask
from models.demos.gemma4_26b_d_p.tests.mesh import MESH_PARAMS, sp_tp
from models.demos.gemma4_26b_d_p.tt.attention.kv_cache import allocate_kv_cache
from models.demos.gemma4_26b_d_p.tt.ccl import CCLManager
from models.demos.gemma4_26b_d_p.tt.attention.sdpa import chunk_attention

NQ = 16
VARIANTS = {"sliding": dict(nkv=8, hd=256, window=1024), "full": dict(nkv=2, hd=512, window=None)}


def _golden(q, k, v, window):
    S = k.shape[2]
    rep = q.shape[1] // k.shape[1]
    k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
    pos = torch.arange(S)
    return torch.softmax(q @ k.transpose(-1, -2) + attention_mask(pos, pos, window), -1) @ v


@MESH_PARAMS
@pytest.mark.parametrize("variant", ["sliding", "full"])
@pytest.mark.parametrize("n_chunks,chunk_global", [(2, 4096), (3, 4096), (2, 8192)], ids=["2x4k", "3x4k", "2x8k"])
def test_ring_sdpa_core(mesh_device, device_params, variant, n_chunks, chunk_global):
    cfg = VARIANTS[variant]
    nkv, hd, window = cfg["nkv"], cfg["hd"], cfg["window"]
    rows, cols = tuple(mesh_device.shape)
    sp, tp, sp_axis, tp_axis = sp_tp(mesh_device)
    C = chunk_global // sp
    cache_global = n_chunks * chunk_global
    kv_last = (n_chunks - 1) * chunk_global
    # kv heads per chip: shard if possible, else replicate GQA-assigned head per chip.
    nkv_local = max(1, nkv // tp)
    nq_local = NQ // tp

    torch.manual_seed(0)
    # Gemma q/k are RMS-normalised per head (|x|~1 per element), so logits ~ O(sqrt(D)); keep that scale.
    q = torch.randn(1, NQ, cache_global, hd) * 0.12
    k = torch.randn(1, nkv, cache_global, hd)
    v = torch.randn(1, nkv, cache_global, hd)
    ref = _golden(q.bfloat16().float(), k.bfloat16().float(), v.bfloat16().float(), window)[:, :, kv_last:]

    sp_topo, _ = per_axis_topology(device_params["fabric_config"])
    ccl = CCLManager(mesh_device, num_links=1, topology=sp_topo)
    kv = allocate_kv_cache(mesh_device, num_layers=1, max_seq_len=cache_global, sp_axis=sp_axis, n_kv_local=nkv_local, head_dim=hd)

    def bc_index(kv_actual):
        pos = rotated_chip_positions(kv_actual, sp, C)
        return torch.tensor([pos[c][r] for c in range(sp) for r in range(C)], dtype=torch.long)

    def kv_heads_per_col(t):
        # [1, nkv, S, D] -> [1, tp*nkv_local, S, D] so a plain head-shard over TP gives each col its GQA head(s).
        if nkv >= tp:
            return t
        group = NQ // nkv
        return torch.cat([t[:, (c * nq_local) // group : (c * nq_local) // group + 1] for c in range(tp)], dim=1)

    dims = [None, None]
    dims[sp_axis], dims[tp_axis] = 2, 1
    mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=dims)

    def chunk_tt(src, kv_actual, dtype=ttnn.bfloat8_b):
        return ttnn.from_torch(src[:, :, bc_index(kv_actual)], device=mesh_device, dtype=dtype, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)

    kx, vx = kv_heads_per_col(k), kv_heads_per_col(v)
    for c in range(n_chunks):
        for cache, src in ((kv.k, kx), (kv.v, vx)):
            ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
                cache, chunk_tt(src, c * chunk_global), slot_idx=0, layer_idx=0, num_layers=1,
                kv_actual_global=c * chunk_global, cluster_axis=sp_axis,
            )
    tt_q = chunk_tt(q, kv_last, dtype=ttnn.bfloat16)

    out = chunk_attention(
        tt_q, kv, chunk_tt(kx, kv_last, dtype=ttnn.bfloat16), chunk_tt(vx, kv_last, dtype=ttnn.bfloat16),
        kv_actual=kv_last, logical_n=cache_global, window=window, layer_slot=0, num_slots_layers=1,
        mesh_device=mesh_device, ccl_manager=ccl, sp_axis=sp_axis, scale=1.0,
    )
    dts = ttnn.get_device_tensors(out)
    coord = lambda r, c: r * cols + c
    per_sp = []
    for s in range(sp):
        heads = [ttnn.to_torch(dts[coord(s, t) if sp_axis == 0 else coord(t, s)]).float() for t in range(tp)]
        per_sp.append(torch.cat(heads, dim=1))
    full_bc = torch.cat(per_sp, dim=2)
    idx = bc_index(kv_last) - kv_last
    inv = torch.empty(chunk_global, dtype=torch.long)
    inv[idx] = torch.arange(chunk_global)
    got = full_bc[:, :, inv]
    ok, pcc = comp_pcc(ref, got, 0.99)
    logger.info(f"ring SDPA {variant} mesh={rows}x{cols} chunks={n_chunks}x{chunk_global}: {pcc}")
    assert ok, pcc
