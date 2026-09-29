# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Ring joint SDPA with the K split (program_config.ring_k_split) vs without, MiMo GA shapes on the 2x2: 64 Q / 4 KV
heads (per chip 32 / 2), qk 192, v 128, HiFi2, q128 / k1024, a random bf8 KV cache and a chunk at ``kv_actual``.
Output of the split (partitions merged by sdpa.merge_k_split) vs the unsplit op: PCC, relative error, norm ratio;
both timed under signposts ``ksplit{s}_C{chunk_local}_ctx{ctx}`` (run with --profile).

    MIMO_KSPLIT=1,2,4 MIMO_KSPLIT_CTX=32768,65536 MIMO_KSPLIT_CHUNK_LOCAL=2048,640 \\
        scripts/run_safe_pytest.sh --profile models/demos/mimo_v2_d_p/tests/perf/test_sdpa_ksplit.py -s
"""

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.mimo_v2_d_p.tests.mesh import MESH_PARAMS
from models.demos.mimo_v2_d_p.tt.attention.kv_cache import MiMoKVCache, _cache_mem
from models.demos.mimo_v2_d_p.tt.attention.sdpa import merge_k_split, ring_attention
from models.demos.mimo_v2_d_p.tt.ccl import CCLManager, default_num_links

try:
    from tracy import signpost
except ImportError:  # pragma: no cover
    signpost = lambda *a, **k: None

SPLITS = [int(s) for s in os.environ.get("MIMO_KSPLIT", "1,2,4").split(",")]
CTXS = [int(c) for c in os.environ.get("MIMO_KSPLIT_CTX", "32768").split(",")]
CHUNKS = [int(c) for c in os.environ.get("MIMO_KSPLIT_CHUNK_LOCAL", "2048,640").split(",")]
ITERS = int(os.environ.get("MIMO_KSPLIT_ITERS", "2"))
REF = os.environ.get("MIMO_KSPLIT_REF", "1") == "1"  # fp32 host reference for chip (0, 0)
NQ, NKV, DK, DV = 64, 4, 192, 128


def stats(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return (
        torch.corrcoef(torch.stack([a, b]))[0, 1].item(),
        ((a - b).norm() / b.norm()).item(),
        (a.norm() / b.norm()).item(),
    )


@pytest.mark.timeout(3600)
@MESH_PARAMS
@pytest.mark.parametrize("chunk_local", CHUNKS)
def test_sdpa_ksplit(mesh_device, device_params, chunk_local):
    torch.manual_seed(0)
    sp, tp = tuple(mesh_device.shape)
    chunk = chunk_local * sp
    sp_topo, _ = per_axis_topology(device_params["fabric_config"])
    ccl = CCLManager(mesh_device, num_links=default_num_links(), topology=sp_topo)
    max_seq = max((c + chunk - 1) // chunk * chunk for c in CTXS)
    nkv_l = NKV // tp
    cache = lambda d: ttnn.from_torch(  # same random cache on every chip (split vs unsplit read the same data)
        torch.randn(1, nkv_l, max_seq // sp, d),
        dtype=ttnn.bfloat8_b,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=_cache_mem(mesh_device, d),
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    kv = MiMoKVCache(cache(DK), cache(DV), 1, 1, max_seq, sp, nkv_l, DK, DV)
    q_host = torch.randn(1, NQ, chunk, DK) * float(os.environ.get("MIMO_KSPLIT_QSCALE", "1"))
    q = ttnn.from_torch(
        q_host,
        dtype=ttnn.bfloat16,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(sp, tp), dims=(2, 1)),
    )
    scale = DK**-0.5
    for ctx in CTXS:
        kv_actual = (ctx + chunk - 1) // chunk * chunk - chunk
        outs, walls, keep = {}, {}, []
        for s in SPLITS:
            tag = f"ksplit{s}_C{chunk_local}_ctx{kv_actual + chunk}"
            walls[s] = []
            for it in range(1 + ITERS):
                if os.environ.get("MIMO_KSPLIT_SHIFT") == "1":  # move buffer addresses between calls (cache hits)
                    keep.append(
                        ttnn.from_torch(
                            torch.zeros(1, 1, 32 * (it + 1), 1024),
                            device=mesh_device,
                            layout=ttnn.TILE_LAYOUT,
                            dtype=ttnn.bfloat16,
                        )
                    )
                ttnn.synchronize_device(mesh_device)
                t0 = time.perf_counter()
                if it:
                    signpost(f"{tag}_start")
                o = ring_attention(
                    q,
                    kv,
                    kv_actual=kv_actual,
                    logical_n=kv_actual + chunk,
                    window=None,
                    sink=None,
                    layer_slot=0,
                    mesh_device=mesh_device,
                    ccl_manager=ccl,
                    sp_axis=0,
                    scale=scale,
                    k_split=s,
                )
                ttnn.synchronize_device(mesh_device)
                if it:
                    walls[s].append((time.perf_counter() - t0) * 1e3)
                    signpost(f"{tag}_end")
                if it == ITERS:
                    outs[s] = torch.cat([ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(o)])
                o.deallocate(True)
        # the op alone (no merge) and the merge alone, for s > 1
        for s in SPLITS[1:]:
            raw_t, merge_t = [], []
            for it in range(1 + ITERS):
                ttnn.synchronize_device(mesh_device)
                t0 = time.perf_counter()
                o, st = ring_attention(
                    q,
                    kv,
                    kv_actual=kv_actual,
                    logical_n=kv_actual + chunk,
                    window=None,
                    sink=None,
                    layer_slot=0,
                    mesh_device=mesh_device,
                    ccl_manager=ccl,
                    sp_axis=0,
                    scale=scale,
                    k_split=s,
                    merge=False,
                )
                ttnn.synchronize_device(mesh_device)
                t1 = time.perf_counter()
                m = merge_k_split(o, st, k_split=s, scale=scale)
                ttnn.synchronize_device(mesh_device)
                t2 = time.perf_counter()
                if it:
                    raw_t.append((t1 - t0) * 1e3)
                    merge_t.append((t2 - t1) * 1e3)
                for t in (o, st, m):
                    t.deallocate(True)
            med = lambda xs: sorted(xs)[len(xs) // 2]
            logger.info(
                f"KTIME C{chunk_local} ctx{kv_actual + chunk} split {s}: op {med(raw_t):.2f} ms + merge {med(merge_t):.2f} ms"
            )
        base = sorted(walls[SPLITS[0]])[len(walls[SPLITS[0]]) // 2]
        for s in SPLITS:
            w = sorted(walls[s])[len(walls[s]) // 2]
            logger.info(
                f"KTIME C{chunk_local} ctx{kv_actual + chunk} split {s}: {w:.2f} ms wall (median), {base / w:.3f}x"
            )
        if REF:
            # chip (0, 0): q heads 0 .. NQ/tp - 1 (KV heads 0 .. nkv_l - 1), q rows 0 .. chunk_local - 1 of the chunk at
            # global positions kv_actual + i; key at global position p = local cache row (p // chunk) * chunk_local +
            # p % chunk_local (block-cyclic, every chip holds the same random cache); causal keys p <= query position
            k_all = ttnn.to_torch(ttnn.get_device_tensors(kv.k)[0]).float()[0]  # [nkv_l, seq_local, DK] (bf8 values)
            v_all = ttnn.to_torch(ttnn.get_device_tensors(kv.v)[0]).float()[0]
            n_keys = kv_actual + chunk
            pos = torch.arange(n_keys)
            rows = (pos // chunk) * chunk_local + pos % chunk_local
            qh = q_host[0, : NQ // tp, :chunk_local].bfloat16().float()
            qpos = kv_actual + torch.arange(chunk_local)
            mask = pos[None, :] <= qpos[:, None]
            ref = torch.empty(NQ // tp, chunk_local, DV)
            g = (NQ // tp) // nkv_l
            for h in range(NQ // tp):
                kk, vv = k_all[h // g, rows], v_all[h // g, rows]
                sc = (qh[h].double() @ kk.double().T) * scale
                sc = sc.masked_fill(~mask, float("-inf"))
                ref[h] = (torch.softmax(sc, -1) @ vv.double()).float()
            for s in SPLITS:
                p, rel, nr = stats(outs[s][0], ref)
                logger.info(
                    f"KSPLIT C{chunk_local} ctx{kv_actual + chunk}: split {s} vs fp32 reference (chip 0): PCC {p:.6f} "
                    f"rel {rel:.2e} norm ratio {nr:.5f}"
                )
        for s in SPLITS[1:]:
            p, rel, nr = stats(outs[s], outs[SPLITS[0]])
            logger.info(
                f"KSPLIT C{chunk_local} ctx{kv_actual + chunk}: split {s} vs {SPLITS[0]}: PCC {p:.6f} rel {rel:.2e} "
                f"norm ratio {nr:.5f}"
            )
            assert p > 0.999, (s, p)
