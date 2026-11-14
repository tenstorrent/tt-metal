"""verify-rows chain test (1x8 row, ring fabric, SCALED_FP8 pool): S = k+1 query rows of ONE user at positions P0..P0+S-1 through the pool decode chain
   set_step -> append S kv rows (row_write) + S key rows (append_key_rows) -> indexer score (per-row causal) -> select_v2 -> id_filter (per-row causal) -> q gather -> split kernel -> exchange/merge.
Checks: (1) appended kv / key rows land at P0..P0+S-1 and nothing else changes; (2) indexer: ids of row j are all <= P0+j, overlap with the float oracle top-2048 (dequantised pool keys, causal);
(3) attention output vs torch full softmax (ref_sparse) over the device-selected ids and the oracle ids; >= 2 consecutive calls.
Run: program/bin/vr.sh <tag> tests.test_hy4_vrows_chain_device MESH_DEVICE=1x8 HY4_FABRIC=ring HY4_FUSE=sparse [XM=1 -> HY4_FUSE=sparse,xchg] SS=1,2,3,4 NS=65536 KINDS=random,real"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tt"))
import seqshard as SQ
import test_hy4_decode_kernel_device as DK
from dk_dev import pcc, ref_sparse
from dm_block import close_row, open_row

import ttnn
from models.demos.blackhole.hy4_preview.tt import hy4_pool as HP
from models.demos.blackhole.hy4_preview.tt.hy4_pool_decode import PoolCtx, append_key_rows, select_global
from models.demos.gpt_oss.config import mesh_1x8
from models.demos.gpt_oss.tt.ccl import CCLManager

DK.HR = 64
NC, K = 8, 2048


def emit(**kw):
    print("RES " + json.dumps(kw), flush=True)


def rep(md, t, dt, lay):
    return ttnn.from_torch(
        t,
        device=md,
        dtype=dt,
        layout=lay,
        mesh_mapper=ttnn.ReplicateTensorToMesh(md),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def shard0(md, t, dt, lay=ttnn.ROW_MAJOR_LAYOUT):
    return ttnn.from_torch(
        t,
        device=md,
        dtype=dt,
        layout=lay,
        mesh_mapper=ttnn.ShardTensorToMesh(md, dim=0),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def pool_bytes(md, kv_pool):
    x = ttnn.to_torch(kv_pool, mesh_composer=ttnn.ConcatMeshToTensor(md, dim=0)).contiguous().view(torch.uint8)
    return x.reshape(NC, -1, x.shape[-1])  # [8, P*64, W]


def dequant_pool(b, T):
    """[8, rows, W] bytes -> natural [T, 576] float (position t at column/local row of seqshard.locate)"""
    t = np.arange(T)
    _, col, loc = SQ.locate(t)
    rows = b[col, loc][:, :656].contiguous()  # [T, 656]
    lat8 = rows[:, :512].view(torch.float8_e4m3fn).float()
    scl = rows[:, 512:528].contiguous().view(torch.float32)  # [T,4]
    rope = rows[:, 528:656].contiguous().view(torch.bfloat16).float()
    lat = lat8.reshape(T, 4, 128) * scl[:, :, None]
    return torch.cat([lat.reshape(T, 512), rope], dim=1)


def key_pool_host(md, kp, T):
    x = ttnn.to_torch(kp, mesh_composer=ttnn.ConcatMeshToTensor(md, dim=0)).float()  # [8P,1,64,128]
    P = x.shape[0] // NC
    x = x.reshape(NC, P * 64, 128)
    t = np.arange(T)
    _, col, loc = SQ.locate(t)
    return x[col, loc]


def idx8_to_global(md, idx8, S):
    """idx8 mesh [1,1,8S,K] per column -> per row s: sorted global ids (union over columns of the owner block 0)"""
    x = (
        ttnn.to_torch(idx8, mesh_composer=ttnn.ConcatMeshToTensor(md, dim=0)).reshape(NC, NC * S, K).to(torch.int64)
        & 0xFFFFFFFF
    )
    out = []
    for s in range(S):
        ids = []
        for c in range(NC):
            r = x[c, s].numpy()  # block j=0: row s
            r = r[r != HP.SENT]
            ids.append(SQ.global_pos(r // 64, c, r))
        out.append(np.sort(np.concatenate(ids)))
    return out


def run_case(md, mc, ccl, N, S, kind, xm):
    pages = N // 512
    P0 = N - S - 7
    att = (
        os.environ.get("NOATTN", "0") != "1"
    )  # NOATTN=1: indexer + key append only (1M context: no kv pool, no host dequantised pool)
    g = torch.Generator().manual_seed(11 + S)
    ctx = PoolCtx(
        md,
        mc,
        ccl,
        pages,
        rows=1,
        S=S,
        mode="dense" if os.environ.get("DENSE", "0") == "1" else "topk",
        bucket_tokens=N,
        xm=xm,
    )
    dp = ttnn.experimental.deepseek_prefill
    kvh, qh64 = DK.gen_data(kind, N if att else 4096, S, seed=5)  # [N,576], [64,S,576]
    ch = torch.exp(torch.randn(128, generator=g) * (0.8 if kind != "random" else 0.0))
    keys = (torch.randn(N, 128, generator=g) * ch).bfloat16()
    # device pool fill (natural order -> column slices)
    kv = None
    if att:
        kv = ttnn.empty(
            [1, 1, pages * 64, 656],
            dtype=ttnn.fp8_e4m3,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=md,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        latd = rep(md, kvh[:, :512].reshape(1, 1, N, 512), ttnn.bfloat16, ttnn.TILE_LAYOUT)
        roped = rep(md, kvh[:, 512:].reshape(1, 1, N, 64), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
        f8, sc = dp.per_token_cast_to_fp8(latd, round_scale_to_power_of_two=True)
        nat = dp.pack_scaled_fp8_kv_cache(f8, sc, roped)
        HP.slice_copy(nat, kv, ctx.colid, 64, HP.ROWB, pages)
        for t in (latd, roped, f8, sc, nat):
            ttnn.deallocate(t)
    kp = ttnn.empty(
        [pages, 1, 64, 128],
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=md,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    kview = ttnn.reshape(kp, [1, 1, pages * 64, 128])
    kd = rep(md, keys.reshape(1, 1, N, 128), ttnn.bfloat16, ttnn.TILE_LAYOUT)
    kb = ttnn.typecast(kd, ttnn.bfloat8_b)
    HP.slice_copy(kb, kview, ctx.colid, 8, HP.KEYTILE, pages)
    ttnn.synchronize_device(md)
    for t in (kd, kb):
        ttnn.deallocate(t)
    pre_kv = pool_bytes(md, kv) if att else None
    pre_key = key_pool_host(md, kp, N)
    # ---- append S rows at P0..P0+S-1
    ctx.set_step([P0])
    new_lat = (torch.randn(S, 512, generator=g) * 2).bfloat16()
    new_rope = torch.randn(S, 64, generator=g).bfloat16()
    new_key = (torch.randn(S, 128, generator=g) * ch).bfloat16()
    if att:
        l2 = rep(md, new_lat.reshape(1, 1, S, 512), ttnn.bfloat16, ttnn.TILE_LAYOUT)
        r2 = rep(md, new_rope.reshape(1, 1, S, 64), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
        f2, s2 = dp.per_token_cast_to_fp8(l2, round_scale_to_power_of_two=True)
        packed = dp.pack_scaled_fp8_kv_cache(f2, s2, r2)
        HP.row_write(packed, ctx.wr, kv, n=int(packed.shape[2]))
    k2 = rep(md, new_key.reshape(1, 1, S, 128), ttnn.bfloat16, ttnn.TILE_LAYOUT)
    append_key_rows(ctx, kp, k2)
    ttnn.synchronize_device(md)
    post_key = key_pool_host(md, kp, N)
    app = slice(P0, P0 + S)
    mask = torch.ones(N, dtype=torch.bool)
    mask[app] = False
    key_app_pcc = pcc(post_key[app], new_key.float())
    key_rest_same = bool(torch.equal(post_key[mask], pre_key[mask]))
    kv_app_pcc = kv_rope_ok = kv_rest_same = None
    if att:
        post_kv = pool_bytes(md, kv)
        kvdq = dequant_pool(post_kv, N)  # natural dequantised pool after the append
        kv_app_pcc = round(pcc(kvdq[app, :512], new_lat.float()), 5)
        kv_rope_ok = bool(torch.equal(kvdq[app, 512:], new_rope.float()))
        t = np.arange(N)
        _, col, loc = SQ.locate(t)
        m_ = mask.numpy()
        kv_rest_same = bool(torch.equal(post_kv[col[m_], loc[m_]], pre_kv[col[m_], loc[m_]]))
    emit(
        event="append",
        N=N,
        S=S,
        kind=kind,
        kv_app_pcc=kv_app_pcc,
        kv_rope_exact=kv_rope_ok,
        key_app_pcc=round(key_app_pcc, 5),
        kv_rest_same=kv_rest_same,
        key_rest_same=key_rest_same,
    )
    # ---- indexer + attention, 2 consecutive calls
    qi = (torch.randn(S, 1, 32, 128, generator=g) * ch.sqrt()).bfloat16()
    wi = torch.zeros(1, 1, 32, 32)
    wi[0, 0, :S] = torch.rand(S, 32, generator=g) / 16 + 1.0 / 64
    qh_d = rep(md, qi, ttnn.bfloat16, ttnn.TILE_LAYOUT)
    w_d = rep(md, wi.bfloat16(), ttnn.bfloat16, ttnn.TILE_LAYOUT)
    # oracle top-2048 per row (float, dequantised pool keys, causal)
    oracle, oracle_sc = [], []
    for s in range(S):
        sc_ = (
            torch.relu(torch.einsum("hd,td->ht", qi[s, 0, :, :].float(), post_key[: P0 + s + 1]))
            * wi[0, 0, s, :, None].bfloat16().float()
        ).sum(0)
        top = torch.topk(sc_, min(K, sc_.numel())).indices.numpy()
        oracle.append(np.sort(top))
        oracle_sc.append(sc_)
    q64 = qfull = sink = None
    if att:
        idx_or = torch.full((S, K), -1, dtype=torch.int64)
        for s in range(S):
            idx_or[s, : oracle[s].size] = torch.from_numpy(oracle[s])
        q64 = DK.align_q_to_keys(kind, qh64, kvdq.to(torch.bfloat16), idx_or, 14.0) if kind != "random" else qh64
        qp = torch.zeros(NC, 8, S, 576, dtype=torch.bfloat16)
        for c in range(NC):
            qp[c] = q64[8 * c : 8 * c + 8]
        qfull = shard0(md, qp, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        sl = torch.linspace(-2, 2, 64)
        sink = ctx.make_sink(sl)
    for call in range(2):
        if (
            os.environ.get("DENSE", "0") == "1"
        ):  # dense mode (every row <= 2048 keys): host-built ids 0..pos+j per row, same id_filter
            gids = None
            idx8 = ctx.dense_ids8()
        else:
            gids = select_global(ctx, qh_d, w_d, kview)
            idx8 = HP.id_filter(gids, ctx.colinfo, S, perrow=ctx.perrow)
        out = ctx.attend(qfull, kv, idx8, sink) if att else None
        ttnn.synchronize_device(md)
        dev_ids = idx8_to_global(md, idx8, S)
        ov = [len(np.intersect1d(dev_ids[s], oracle[s])) / max(1, min(K, P0 + s + 1)) for s in range(S)]
        # tie-tolerant: device ids whose float oracle score is within 2 bf16 ulp (2^-7 relative) of the k-th oracle score count as correct (bf16 scores, bfp8 keys)
        ovt = []
        for s in range(S):
            sc_ = oracle_sc[s]
            kth = float(sc_[oracle[s]].min()) if oracle[s].size else 0.0
            thr = kth - abs(kth) * 2.0**-7
            dv = dev_ids[s][dev_ids[s] <= P0 + s]
            ovt.append(float((sc_[dv] >= thr).float().mean()) if dv.size else 1.0)
        causal = all(int(d.max()) <= P0 + s for s, d in enumerate(dev_ids) if d.size)
        cnt_ok = all(d.size <= min(K, P0 + s + 1) for s, d in enumerate(dev_ids))
        r = dict(
            event="chain",
            N=N,
            S=S,
            kind=kind,
            call=call,
            G=ctx.G,
            kc=ctx.kc,
            xm=ctx.xm is not None,
            overlap_min=round(min(ov), 4),
            overlap=[round(x, 4) for x in ov],
            overlap_tol_min=round(min(ovt), 4),
            causal_ok=bool(causal),
            count_ok=bool(cnt_ok),
        )
        if att:
            o = (
                ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(md, dim=0))
                .float()
                .reshape(NC, 8, S, 512)
                .reshape(64, S, 512)
            )
            idx_dev = torch.full((S, K), -1, dtype=torch.int64)
            for s in range(S):
                idx_dev[s, : dev_ids[s].size] = torch.from_numpy(dev_ids[s])
            ref_d = ref_sparse(q64, kvdq.to(torch.bfloat16), idx_dev, sl)
            ref_o = ref_sparse(q64, kvdq.to(torch.bfloat16), idx_or, sl)
            rows_d = [round(pcc(o[:, s], ref_d[:, s]), 6) for s in range(S)]
            r.update(
                pcc_dev_ids=round(pcc(o, ref_d), 6),
                min_row_dev_ids=min(rows_d),
                pcc_oracle_ids=round(pcc(o, ref_o), 6),
                finite=bool(torch.isfinite(o).all()),
            )
            ttnn.deallocate(out)
        emit(**r)
        if gids is not None:
            ttnn.deallocate(gids)
            ttnn.deallocate(idx8)
    for t in (kv, kp, qfull, sink, qh_d, w_d):
        if t is not None:
            ttnn.deallocate(t)
    return ctx


def main():
    full, md = open_row()
    try:
        mc = mesh_1x8()
        ccl = CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
        NS = [int(x) for x in os.environ.get("NS", "65536").split(",")]
        SS = [int(x) for x in os.environ.get("SS", "1,2,3,4").split(",")]
        xm = None
        if "xchg" in os.environ.get("HY4_FUSE", ""):  # ONE XMerge per process: one S per process
            from models.demos.blackhole.hy4_preview.tt import hy4_xmerge

            assert len(SS) == 1
            xm = hy4_xmerge.XMerge(md, SS[0], nsplit=int(os.environ.get("HY4_XM_NSPLIT", "2")))
        for N in NS:
            for kind in os.environ.get("KINDS", "random,real").split(","):
                for S in SS:
                    try:
                        run_case(md, mc, ccl, N, S, kind, xm)
                    except Exception as e:
                        import traceback

                        traceback.print_exc()
                        emit(event="error", N=N, S=S, kind=kind, err=str(e).strip().splitlines()[0][:300])
    finally:
        close_row(full, md)


if __name__ == "__main__":
    main()
