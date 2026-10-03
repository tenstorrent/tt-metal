# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Decode-time indexer of layer 2 (ratio 2) on the 4x8 mesh vs the checkpoint's Indexer on real weights.

State is built on a CPU host by ``tests/indexer_ref_state.py`` (reference Compressor + Indexer code): DSV41_KV_STATE (default
/mnt/tt-data/ssinghal/dsv4-kv-state/layer2_N{N}.pt). Env: DSV41_IDX_N (comma list of entry counts, default 2048,8192), DSV41_IDX_BACKEND (fused|matmul),
DSV41_IDX_KEYS (bf16|bfp8), DSV41_IDX_WDT (wq_b dtype bfp8|bf16, default bfp8), DSV41_IDX_FP4Q (1: simulate the reference fp4 quantisation of q on the device), DSV41_USERS_PER_ROW (default 4). Prints IDX lines: top-512 SET agreement vs the reference module (and vs the reference
without the q fp4 simulation = the ceiling of a device path that does not simulate it), score PCC, latency per step (one trace) and per stage."""

import os
import time

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.reference import ref_kernels
from models.demos.blackhole.deepseek_v41_flash.tt.indexer import DSV41DecodeIndexer
from models.demos.blackhole.deepseek_v41_flash.tt.loader import rope_freqs
from models.demos.blackhole.deepseek_v41_flash.tt.moe_weights import _Shards

STATE_DIR = os.environ.get("DSV41_KV_STATE", "/mnt/tt-data/ssinghal/dsv4-kv-state")
NS = [int(n) for n in os.environ.get("DSV41_IDX_N", "2048,8192").split(",")]
LAYER = 2


def dq(sh, key):
    w, s = sh.get(key + ".weight"), sh.get(key + ".scale")
    return ref_kernels.dequant_fp8_weight(w, s, w.size(0) // s.size(0)).to(torch.bfloat16)


def timed(md, fn, reps=5):
    fn()
    ttnn.synchronize_device(md)
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    for _ in range(reps):
        fn()
    ttnn.end_trace_capture(md, tid, cq_id=0)
    ts = []
    for _ in range(3):
        t = time.perf_counter()
        ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        ts.append((time.perf_counter() - t) / reps * 1e6)
    ttnn.release_trace(md, tid)
    return min(ts)


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 200_000_000,
            },
            id="ring",
        )
    ],
    indirect=True,
)
@pytest.mark.timeout(3600)
@torch.no_grad()
def test_indexer_layer2(mesh_device):
    md = mesh_device
    U = int(os.environ.get("DSV41_USERS_PER_ROW", "4"))
    rows = md.shape[0]
    B = rows * U
    sh = _Shards()
    p = f"layers.{LAYER}.attn.indexer."
    w = {"wq_b": dq(sh, p + "wq_b"), "weights_proj": sh.get(p + "weights_proj.weight").to(torch.bfloat16)}
    key_dtype = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b}[os.environ.get("DSV41_IDX_KEYS", "bf16")]
    backend = os.environ.get("DSV41_IDX_BACKEND", "fused")
    for N in NS:
        st_ref = torch.load(os.path.join(STATE_DIR, f"layer{LAYER}_N{N}.pt"))
        pos, ratio = st_ref["pos"], st_ref["ratio"]
        n_alloc = -(-(N + 32) // 32) * 32
        idx = DSV41DecodeIndexer(
            md,
            w,
            rope_freqs(LAYER, pos + 64),
            users_per_row=U,
            n_alloc=n_alloc,
            ratio=ratio,
            key_dtype=key_dtype,
            weight_dtype={"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b}[os.environ.get("DSV41_IDX_WDT", "bfp8")],
            backend=backend,
            fp4_q=os.environ.get("DSV41_IDX_FP4Q", "0") == "1",
        )
        idx.load_keys(st_ref["index_k"][None].expand(B, -1, -1))
        st = idx.step_inputs(torch.full((B,), pos))
        rep = ttnn.ReplicateTensorToMesh(md)
        up = lambda t: ttnn.from_torch(
            t.reshape(1, 1, 1, -1).expand(1, 1, U, -1).contiguous().to(torch.bfloat16),
            device=md,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=rep,
        )
        x, qr = up(st_ref["x_q"]), up(st_ref["qr"])
        ids, scores = idx.forward(x, qr, st, return_scores=True)
        ttnn.synchronize_device(md)
        if backend == "fused":
            got_ids = ttnn.to_torch(ttnn.get_device_tensors(ids)[0]).reshape(U, -1).long()
            got_sc = torch.stack(
                [ttnn.to_torch(ttnn.get_device_tensors(s)[0]).reshape(32, -1)[0] for s in scores]
            ).float()[:, :N]
        else:
            got_ids = ttnn.to_torch(ttnn.get_device_tensors(ids)[0]).reshape(U, -1).long()
            got_sc = ttnn.to_torch(ttnn.get_device_tensors(scores)[0]).reshape(U, -1).float()[:, :N]
        ref_set, ref_sc, noq = set(st_ref["topk"].tolist()), st_ref["scores"], st_ref["scores_noq"]
        noq_set = set(noq.topk(512).indices.tolist())
        agree = [len(ref_set & set(got_ids[u].tolist())) / 512 for u in range(U)]
        sc_pcc = [float(torch.corrcoef(torch.stack([got_sc[u], ref_sc]))[0, 1]) for u in range(U)]
        sc_pcc_noq = float(torch.corrcoef(torch.stack([got_sc[0], noq]))[0, 1])
        # score-mass recall: share of the reference's top-512 score mass (above the 512th score) kept by the device selection
        thr = ref_sc.topk(512).values[-1]
        sel = got_ids[0].clamp(max=N - 1)
        mass = float((ref_sc[sel] - thr).clamp(min=0).sum() / (ref_sc.topk(512).values - thr).sum())
        print(
            f"IDX N={N} U={U} backend={backend} keys={os.environ.get('DSV41_IDX_KEYS', 'bf16')} fp4q={os.environ.get('DSV41_IDX_FP4Q', '0')} wq_b={os.environ.get('DSV41_IDX_WDT', 'bfp8')}: top-512 set agreement vs reference {min(agree):.4f} "
            f"(ceiling without q-fp4 sim {len(ref_set & noq_set) / 512:.4f}; device vs noq-set {len(noq_set & set(got_ids[0].tolist())) / 512:.4f}), "
            f"score PCC {min(sc_pcc):.5f} (vs noq {sc_pcc_noq:.5f}), score-mass recall {mass:.4f}, sentinel ids {(got_ids[0] > N).sum().item()}",
            flush=True,
        )
        # latency: whole step and stages (each inside a trace)
        t_all = timed(md, lambda: idx.forward(x, qr, st))
        q, wv = idx.project(x, qr, st)
        t_proj = timed(md, lambda: idx.project(x, qr, st))
        t_score = timed(md, lambda: idx.score(q, wv))
        s = idx.score(q, wv)
        t_topk = timed(md, lambda: idx.select(s, st))
        print(
            f"IDX N={N} U={U} backend={backend} keys={os.environ.get('DSV41_IDX_KEYS', 'bf16')} fp4q={os.environ.get('DSV41_IDX_FP4Q', '0')} wq_b={os.environ.get('DSV41_IDX_WDT', 'bfp8')}: step {t_all:.0f} us = projections+rope {t_proj:.0f} + score {t_score:.0f} + topk {t_topk:.0f} "
            f"(stage sum {t_proj + t_score + t_topk:.0f})",
            flush=True,
        )
        assert min(agree) > 0.85
