# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Milestone 2: paged decode attention at long synthetic context (indexer top-512 + sparse_sdpa over the page pool) vs the checkpoint's reference
Attention on real layer weights, plus the measured decode attention time per layer vs context.

State: the reference layer's caches are filled with random data (window ring, compressed latents, index keys: no prefill is run, 64k-token prefills on the CPU
are not the point of the test), the reference runs ONE decode step at position S = L - 1 (so the step completes a group: the reference writes the new latent
and index key), and everything it produced (latents, keys, its top-512 ids) is copied to the device. The device pool is allocated in shuffled page order.

  PAGED_LONG oracle  : the reference's own top-512 ids drive the device attention  -> numerics of ring + page-table gather + sparse_sdpa + projections
  PAGED_LONG device  : the device indexer (key append, fp4-q scoring, topk_large_indices) selects -> end to end (selection differs from the reference on
                       the near-ties at the 512th score: set agreement is printed)
Env: DSV41_LONG_L (comma list of contexts, default 4096,16384,65536), DSV41_LONG_LAYERS (default 0,2,20), DSV41_PAGED_KV (bf16|fp8).
"""

import os
import random
import time

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.reference import ref_kernels
from models.demos.blackhole.deepseek_v41_flash.reference import ref_layer as R
from models.demos.blackhole.deepseek_v41_flash.tests.idx_state import Capture, make_state, replicate_user0
from models.demos.blackhole.deepseek_v41_flash.tt.indexer import DSV41DecodeIndexer, default_backend
from models.demos.blackhole.deepseek_v41_flash.tt.moe_weights import _Shards
from models.demos.blackhole.deepseek_v41_flash.tt.paged_attention import (
    DSV41PagedAttention,
    DSV41PagedCompressedAttention,
    DSV41PagedStepState,
)
from models.demos.blackhole.deepseek_v41_flash.tt.paged_ops import SRC_OFF, PagedKVPool
from models.demos.gpt_oss.config import mesh_4x8
from models.demos.gpt_oss.tt.ccl import CCLManager

DP = {"l1_small_size": 16384, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 200_000_000}
LS = [int(v) for v in os.environ.get("DSV41_LONG_L", "4096,16384,65536").split(",")]
LAYERS = [int(v) for v in os.environ.get("DSV41_LONG_LAYERS", "0,2,20").split(",")]


def dq(sh, key):
    w, s = sh.get(key + ".weight"), sh.get(key + ".scale")
    return ref_kernels.dequant_fp8_weight(w, s, w.size(0) // s.size(0)).to(torch.bfloat16)


def to_host(out, rows, cols, B):
    devs = ttnn.get_device_tensors(out)
    return torch.cat([ttnn.to_torch(devs[r * cols]).reshape(-1, 5120) for r in range(rows)]).float()[:B]


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
@pytest.mark.parametrize("device_params", [pytest.param(DP, id="ring")], indirect=True)
@pytest.mark.parametrize("layer_id", LAYERS)
@pytest.mark.parametrize("L", LS)
@pytest.mark.timeout(7200)
@torch.no_grad()
def test_paged_long_context(mesh_device, layer_id, L):
    torch.manual_seed(1)
    ref_kernels.FAKE_QUANT = True
    md = mesh_device
    rows, cols = tuple(md.shape)
    U = int(os.environ.get("DSV41_USERS_PER_ROW", "4"))
    B = rows * U
    S = L - 1
    mod = R.load_model_module()
    blk = R.build_layer(layer_id, max_batch_size=B, max_seq_len=L + 64)
    at = blk.attn
    ratio = at.compress_ratio
    N = (
        (S + 1) // ratio if ratio else 0
    )  # compressed entries visible at the decode step (the newest one is produced by this step)
    comp = at.compressor
    real_dir = os.environ.get(
        "DSV41_LONG_REAL"
    )  # dump dir of ref_prefill_dump.py: REAL activations (the dump's S must equal L)
    x1 = make_state(
        blk, L, real_dir, layer_id
    )  # one user's reference state (synthetic caches or a real prefill of S-1 tokens) + the decode input
    cap = Capture(mod) if ratio else None
    # the compressor state BEFORE the decode step (the reference rewrites it while decoding)
    kv_state = comp.kv_state.clone() if ratio > 1 else None
    score_state = comp.score_state.clone() if ratio > 1 else None
    t0 = time.time()
    ref = at(x1, S).float().reshape(1, 5120).expand(B, -1).contiguous()
    print(
        f"reference decode step at S={S} ({'REAL' if real_dir else 'synthetic'} state): {time.time() - t0:.1f}s",
        flush=True,
    )
    if ratio:
        cap.release()
    replicate_user0(
        at
    )  # all users identical (the reference state AFTER its decode step, which holds the entry/key/ring of this step)
    x_dec = x1.expand(B, 1, 5120)
    window = at.window_kv_cache.clone().float()
    if ratio:
        ref_ids = (
            (mod.shared_attn.topk_idxs - 128)[:, 0].long().expand(B, -1).contiguous()
        )  # [B, 512] entry ids (position sorted)
        comp_all = at.compress_kv_cache[:, :N].clone().float()
        keys_all = at.indexer.k_cache[:, :N].clone().float()
    # the device state before the step: the compressor state of the PREVIOUS token
    mesh_config = mesh_4x8()
    ccl = CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
    w = R.dequantized_attention_weights(blk)
    if ratio:
        comp_w = {"wkv": comp.wkv.weight.data.float(), "norm": comp.norm.weight.data.float()}
        if ratio > 1:
            comp_w["wgate"] = comp.wgate.weight.data.float()
        sh = _Shards()
        p = f"layers.{layer_id}.attn.indexer."
        w_idx = {"wq_b": dq(sh, p + "wq_b"), "weights_proj": sh.get(p + "weights_proj.weight").to(torch.bfloat16)}
    shard_x = ttnn.ShardTensor2dMesh(md, dims=(2, None), mesh_shape=(rows, cols))
    tt_x = ttnn.from_torch(
        x_dec.reshape(1, 1, B, 5120).to(torch.bfloat16),
        device=md,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=shard_x,
    )
    rs = ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(rows, cols))
    pos = ttnn.from_torch(
        torch.full((B,), S, dtype=torch.int32),
        device=md,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=rs,
    )

    pages = L // 128
    kvp = PagedKVPool(
        md,
        U,
        num_pages=U * (pages + 1),
        n_ring_layers=1,
        max_ctx=L + 128,
        dtype=ttnn.fp8_e4m3 if os.environ.get("DSV41_PAGED_KV", "bf16") == "fp8" else ttnn.bfloat16,
    )
    for a in kvp.allocs:
        random.Random(2).shuffle(a._free)
    for b in range(B):
        kvp.admit(b, L)
    kvp.sync_page_table()
    kvp.stage_begin()
    pdt = ttnn.fp8_e4m3 if os.environ.get("DSV41_PAGED_KV", "bf16") == "fp8" else ttnn.bfloat16
    if ratio:
        n_alloc = -(-(N + 32) // 32) * 32
        idx = DSV41DecodeIndexer(
            md,
            w_idx,
            at.freqs_cis,
            users_per_row=U,
            n_alloc=n_alloc,
            ratio=ratio,
            key_dtype=ttnn.bfloat8_b,
            fp4_q=True,
            backend=default_backend(N),
        )
        print(f"indexer backend {idx.backend} for N={N}", flush=True)
        idx.set_key_weights(at.indexer.wk.weight.data.float(), at.indexer.k_norm.weight.data.float())
        idx.load_keys(keys_all[:, : N - 1])  # the device appends the key of the entry this step completes
        pa = DSV41PagedCompressedAttention(
            md, mesh_config, ccl, w, at.freqs_cis, ratio, comp_w, kvp, 0, layer_id, users_per_row=U, indexer=None
        )
        pa.load_state(window, comp_all, kv_state, score_state, start_pos=S)
    else:
        pa = DSV41PagedAttention(md, mesh_config, ccl, w, at.freqs_cis, kvp, 0, users_per_row=U)
        pa.load_ring(window)
    kvp.stage_commit()
    ss = DSV41PagedStepState(pa, max_pos=L + 64, with_indexer=bool(ratio))

    def ids_dev(ids):  # [B, 512] long -> uint32 [B,1,1,512] sharded over mesh rows
        return ttnn.from_torch(
            ids.reshape(B, 1, 1, -1).to(torch.int32),
            device=md,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=rs,
        )

    # ---- oracle selection
    snap = pa.snapshot_state() if ratio else None
    st = ss.build(pos)
    if ratio:
        st["topk_ids"] = ids_dev(ref_ids)
    got = to_host(pa.forward(tt_x, st), rows, cols, B)
    p_or = R.pcc(got, ref)
    per_user = [round(R.pcc(got[i], ref[i]), 4) for i in range(B)]
    print(
        f"PAGED_LONG oracle layer {layer_id} ctx {L} kv={'fp8' if pdt != ttnn.bfloat16 else 'bf16'}: attention PCC vs reference {p_or:.5f} (min user {min(per_user)})",
        flush=True,
    )

    if os.environ.get("DSV41_LONG_PROBE") == "1" and ratio:
        # sparse_sdpa vs an fp32 CPU golden on IDENTICAL device inputs (q, index rows, pool rows) of mesh row 0 / column 0, oracle ids
        from models.demos.blackhole.deepseek_v41_flash.tt import paged_ops as P

        pa.restore_state(snap)
        st = ss.build(pos)
        st["topk_ids"] = ids_dev(ref_ids)
        q, kv_, k_ = pa._qkv(tt_x, st)
        lat = pa._compress_step(tt_x, st) if pa.source is None else None
        idx_rows = P.paged_kv_step(
            kvp.pool,
            q,
            lat,
            st["pos"],
            kvp.page_table,
            st["topk_ids"],
            ring_base=kvp.ring_base(0),
            layer_key=0,
            ratio=ratio,
            src_off=P.SRC_OFF[layer_id],
            topk_out=640,
            ring_rows=kvp.ring_rows,
        )
        o = pa._paged_attend(q, lat, st, ratio, P.SRC_OFF[layer_id], 640)
        d0 = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()
        qh, idh, oh = d0(q).reshape(U, -1, 512), d0(idx_rows).reshape(U, 640).long(), d0(o).reshape(U, -1, 512)
        pool_h = d0(kvp.pool).reshape(-1, 512)
        sink = w["attn_sink"].float()[:8]
        res = []
        for u in range(U):
            ii = idh[u][idh[u] != 0xFFFFFFFF]
            Kr = pool_h[ii]
            s_ = (qh[u][1:9] @ Kr.T) * 512**-0.5
            p_ = torch.softmax(torch.cat([s_, sink.reshape(8, 1)], -1), -1)[:, :-1]
            res.append(R.pcc((p_ @ Kr).reshape(-1), oh[u][1:9].reshape(-1)))
        if (
            pa.source is None and ratio
        ):  # the latent THIS step wrote (entry N-1) vs the reference's own entry (fp4-simulated, RoPE'd)
            row = pool_h[kvp.phys_rows(0, layer_id, torch.tensor([N - 1]))[0]]
            print(
                f"PAGED_PROBE new latent (entry {N - 1}) device vs reference: PCC {R.pcc(row, comp_all[0, N - 1]):.5f}, "
                f"norm ratio {float(row.norm() / comp_all[0, N - 1].norm()):.4f}; previous entry's norm for scale {float(comp_all[0, N - 2].norm()):.2f}",
                flush=True,
            )
        print(
            f"PAGED_PROBE layer {layer_id} ctx {L}: sparse_sdpa vs fp32 golden on identical inputs (per user) {[round(float(v), 5) for v in res]}; "
            f"valid index rows {int((idh[0] != 0xFFFFFFFF).sum())}",
            flush=True,
        )
    t_total = None
    if ratio and N > 512:
        # ---- device indexer (end to end): same state, the compressor "previous token" state restored
        pa.restore_state(snap)
        pa.indexer = idx
        st = ss.build(pos)
        got2 = to_host(pa.forward(tt_x, st), rows, cols, B)
        p_dev = R.pcc(got2, ref)
        dev_ids = ttnn.to_torch(ttnn.get_device_tensors(st["topk_ids"])[0]).reshape(U, -1).long()
        agree = [len(set(dev_ids[u].tolist()) & set(ref_ids[u].tolist())) / 512 for u in range(U)]
        print(
            f"PAGED_LONG device-indexer layer {layer_id} ctx {L} ({'REAL' if real_dir else 'synthetic'}): attention PCC vs reference {p_dev:.5f}; top-512 set agreement {min(agree):.4f}; "
            f"attention-mass coverage of the compressed entries: device set {cap.coverage(dev_ids[0]):.4f}, reference set {cap.coverage(ref_ids[0]):.4f}; "
            f"total coverage (window + set) device {cap.total_coverage(dev_ids[0]):.4f}, reference {cap.total_coverage(ref_ids[0]):.4f}",
            flush=True,
        )

    # ---- timing (one captured trace of REPS calls, per layer, per device)
    if ratio and N > 512:
        t_total = timed(md, lambda: pa.forward(tt_x, st))
        pa.indexer = None
        st_no = ss.build(pos)
        st_no["topk_ids"] = st["topk_ids"]
    else:
        st_no = st
    t_attn = timed(md, lambda: pa.forward(tt_x, st_no))
    q_dummy, kv_d, k_d = pa._qkv(tt_x, st_no)
    t_core = timed(
        md, lambda: pa._paged_attend(q_dummy, None, st_no, ratio, SRC_OFF.get(layer_id, 0), 640 if ratio else 128)
    )
    print(
        f"PAGED_TIME layer {layer_id} ctx {L} users/row {U}: attention layer (x -> out, selection from a given id tensor) {t_attn:.0f} us, "
        + (f"with indexer {t_total:.0f} us (indexer {t_total - t_attn:.0f} us), " if t_total else "")
        + f"paged core (index build + ring/latent write + q layout + sparse_sdpa) {t_core:.0f} us",
        flush=True,
    )
    assert p_or > 0.99
