# SPDX-License-Identifier: Apache-2.0
"""
Full 64-layer Qwen3.6-27B decode, 8-way tensor-parallel on a 1x8 T3K mesh.
Dummy weights (correct shapes/sharding, ~7GB/chip) — measures DECODE TPOT.
Correctness of each TP component is validated separately (test_tp_{mlp,attention,deltanet}.py).

Compares against the single-chip baseline (eager 172ms / trace 164ms).

  QWEN_TP_LAYERS=64 python3 tp_model.py --steps 16
"""
import argparse, os, time, statistics
import torch
import ttnn
from models.demos.qwen36_27b.tt.model_config import Qwen36ModelConfig

TP = 8
DT_W = ttnn.bfloat8_b   # weights bf8 → ~7GB/chip
DT_A = ttnn.bfloat16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=int(os.environ.get("QWEN_TP_LAYERS", "64")))
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--ctx", type=int, default=64, help="fixed attention context length")
    args = ap.parse_args()

    cfg = Qwen36ModelConfig()
    NL = args.layers
    H, I = cfg.hidden_size, cfg.intermediate_size
    nh, nkv, hd, rdim = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, cfg.rotary_dim
    nk, nv, dk, dv = cfg.linear_num_key_heads, cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim
    Kc = cfg.linear_conv_kernel_dim
    key_dim, val_dim = dk * nk, dv * nv
    groups = nh // nkv
    nhp, nvp, nkp = nh // TP, nv // TP, nk // TP
    # full_attention GQA: nkv (4) < TP (8) on T3K → replicate each KV head across TP//nkv chips.
    nkvp = max(1, nkv // TP)        # KV heads per chip (1 when replicated)
    kv_slots = nkvp * TP            # total KV-head slots across the mesh (includes replication)
    ratio = nv // nk
    scale = hd ** -0.5
    ltypes = cfg.layer_types[:NL]

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    md = ttnn.open_mesh_device(ttnn.MeshShape(1, TP))
    sh = (1, TP)
    SH = lambda d: ttnn.ShardTensor2dMesh(md, dims=(None, d), mesh_shape=sh)
    REP = ttnn.ReplicateTensorToMesh(md)
    rnd = lambda *s: (torch.randn(*s) * 0.02)
    fW = lambda t, d: ttnn.from_torch(t.unsqueeze(0).unsqueeze(0), dtype=DT_W, layout=ttnn.TILE_LAYOUT, device=md, mesh_mapper=SH(d))
    fR = lambda t: ttnn.from_torch(t, dtype=DT_A, layout=ttnn.TILE_LAYOUT, device=md, mesh_mapper=REP)

    print(f"[build] {NL} layers, TP={TP}, building sharded dummy weights ...", flush=True)
    t0 = time.time()
    layers = []
    for li in range(NL):
        lt = ltypes[li]
        L = {"type": lt}
        L["in_norm"] = fR(torch.ones(1, 1, 1, H))
        L["post_norm"] = fR(torch.ones(1, 1, 1, H))
        # MLP (col gate/up, row down)
        # MLP: fuse gate+up into one column-parallel matmul (split after)
        L["gu"] = fW(rnd(2 * I, H).T, 3); L["down"] = fW(rnd(H, I).T, 2)
        if lt == "full_attention":
            # fuse q+k+v into one column-parallel matmul (heads contiguous → per-chip [q|k|v])
            L["Wqkv"] = fW(rnd(nh * hd * 2 + 2 * kv_slots * hd, H).T, 3)  # K,V replicated to kv_slots (=TP) heads
            L["Wo"] = fW(rnd(H, nh * hd).T, 2)
            L["qn"] = fR((rnd(hd) + 1.0).view(1, 1, 1, hd)); L["kn"] = fR((rnd(hd) + 1.0).view(1, 1, 1, hd))
            # fixed KV cache per chip [1, nkvp, ctx, hd]
            L["kc"] = ttnn.from_torch(torch.randn(1, nkvp, args.ctx, hd).to(torch.bfloat16), dtype=DT_A, layout=ttnn.TILE_LAYOUT, device=md, mesh_mapper=REP)
            L["vc"] = ttnn.from_torch(torch.randn(1, nkvp, args.ctx, hd).to(torch.bfloat16), dtype=DT_A, layout=ttnn.TILE_LAYOUT, device=md, mesh_mapper=REP)
        else:
            # fuse q+k+v into one matmul (per-chip [q|k|v]); fuse z+b+a into one
            L["Wqkv"] = fW(rnd(key_dim * 2 + val_dim, H).T, 3)
            L["Wzba"] = fW(rnd(val_dim + 2 * nv, H).T, 3)
            L["Wo"] = fW(rnd(H, val_dim).T, 2)
            # conv weight must use per-chip [q|k|v] head grouping (like qkv): shard each block, concat
            def cblk(c):
                return ttnn.from_torch(c.unsqueeze(0).unsqueeze(0).to(torch.bfloat16), dtype=DT_A,
                                       layout=ttnn.TILE_LAYOUT, device=md, mesh_mapper=SH(2))
            cwq = rnd(key_dim, Kc); cwk = rnd(key_dim, Kc); cwv = rnd(val_dim, Kc)
            L["conv"] = ttnn.concat([cblk(cwq), cblk(cwk), cblk(cwv)], dim=2)
            L["A"] = ttnn.from_torch(rnd(nv).view(1,1,1,nv).to(torch.bfloat16), dtype=DT_A, layout=ttnn.TILE_LAYOUT, device=md, mesh_mapper=SH(3))
            L["dt"] = ttnn.from_torch(rnd(nv).view(1,1,1,nv).to(torch.bfloat16), dtype=DT_A, layout=ttnn.TILE_LAYOUT, device=md, mesh_mapper=SH(3))
            L["nw"] = fR(torch.ones(1, 1, 1, dv))
            L["rs"] = ttnn.from_torch(torch.zeros(1, nv, dk, dv).to(torch.bfloat16), dtype=DT_A, layout=ttnn.TILE_LAYOUT, device=md, mesh_mapper=SH(1))
            L["cs"] = ttnn.from_torch(torch.zeros(1, 1, key_dim * 2 + val_dim, 32).to(torch.bfloat16), dtype=DT_A, layout=ttnn.TILE_LAYOUT, device=md, mesh_mapper=SH(2))
            # host-side conv1d tap weights + sliding-window state, channel on dim3 (no transpose)
            cdf = key_dim * 2 + val_dim
            for nm in ("w0", "w1", "w2", "w3", "sA", "sB", "sC"):
                L[nm] = ttnn.from_torch((rnd(1, 1, 1, cdf)).to(torch.bfloat16), dtype=DT_A, layout=ttnn.TILE_LAYOUT, device=md, mesh_mapper=SH(3))
        layers.append(L)
    # final norm + lm_head (vocab col-parallel)
    final_norm = fR(torch.ones(1, 1, 1, H))
    Vp = cfg.padded_vocab_size
    lm_head = fW(rnd(Vp, H).T, 3)
    cur_pos = ttnn.from_torch(torch.tensor([args.ctx - 1], dtype=torch.int32), device=md)
    cos = fR(torch.randn(1, 1, 1, rdim) * 0.5); sin = fR(torch.randn(1, 1, 1, rdim) * 0.5)
    conv_dim_p = (key_dim * 2 + val_dim)
    print(f"[build] done in {time.time()-t0:.1f}s", flush=True)

    def rope_tt(t, n):
        tr = ttnn.slice(t, [0, 0, 0, 0], [1, 1, n, rdim]); tp = ttnn.slice(t, [0, 0, 0, rdim], [1, 1, n, hd])
        x1 = ttnn.slice(tr, [0, 0, 0, 0], [1, 1, n, rdim // 2]); x2 = ttnn.slice(tr, [0, 0, 0, rdim // 2], [1, 1, n, rdim])
        rh = ttnn.concat([ttnn.neg(x2), x1], dim=-1)
        return ttnn.concat([ttnn.add(ttnn.mul(tr, cos), ttnn.mul(rh, sin)), tp], dim=-1)
    shard_cfg = ttnn.create_sharded_memory_config(shape=(32, hd), core_grid=ttnn.CoreGrid(y=1, x=1),
        strategy=ttnn.ShardStrategy.HEIGHT, orientation=ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True)
    # sdpa_decode tree-reduction config (batch=1, 1 kv-head/chip)
    sdpa_cfg = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=md.compute_with_storage_grid_size(),
        q_chunk_size=32, k_chunk_size=32)

    AR_TOPO = ttnn.Topology.Linear  # T3K is a 1x8 line (FABRIC_1D) → linear all_reduce
    def AR(t):
        return ttnn.all_reduce(t, cluster_axis=1, topology=AR_TOPO)

    qsz, ksz = nhp * hd * 2, nkvp * hd  # attn fused-qkv per-chip slice sizes

    def attn(h, L):
        qkv = ttnn.linear(h, L["Wqkv"])  # fused q|k|v, one matmul
        qp = ttnn.slice(qkv, [0, 0, 0, 0], [1, 1, 1, qsz])
        kp = ttnn.slice(qkv, [0, 0, 0, qsz], [1, 1, 1, qsz + ksz])
        vp = ttnn.slice(qkv, [0, 0, 0, qsz + ksz], [1, 1, 1, qsz + 2 * ksz])
        q2 = ttnn.reshape(qp, [1, 1, nhp, hd * 2])
        qry = ttnn.slice(q2, [0, 0, 0, 0], [1, 1, nhp, hd]); gt = ttnn.slice(q2, [0, 0, 0, hd], [1, 1, nhp, hd * 2])
        ky = ttnn.reshape(kp, [1, 1, nkvp, hd]); vl = ttnn.reshape(vp, [1, 1, nkvp, hd])
        qry = ttnn.rms_norm(qry, epsilon=1e-6, weight=L["qn"]); ky = ttnn.rms_norm(ky, epsilon=1e-6, weight=L["kn"])
        qry = rope_tt(qry, nhp); ky = rope_tt(ky, nkvp)
        ks = ttnn.to_memory_config(ky, shard_cfg); vs = ttnn.to_memory_config(vl, shard_cfg)
        ttnn.experimental.paged_update_cache(L["kc"], ks, update_idxs_tensor=cur_pos)
        ttnn.experimental.paged_update_cache(L["vc"], vs, update_idxs_tensor=cur_pos)
        att = ttnn.transformer.scaled_dot_product_attention_decode(qry, L["kc"], L["vc"], cur_pos_tensor=cur_pos, scale=scale, program_config=sdpa_cfg)
        att = ttnn.reshape(att, [1, 1, 1, nhp * hd]); gt = ttnn.reshape(gt, [1, 1, 1, nhp * hd])
        att = ttnn.mul(att, ttnn.sigmoid(gt))
        return AR(ttnn.linear(att, L["Wo"]))

    vd_p, nv_p = val_dim // TP, nv // TP  # deltanet zba per-chip slice sizes

    def deltanet(h, L):
        qkv = ttnn.linear(h, L["Wqkv"])  # fused q|k|v, one matmul
        # host-side conv1d (K taps elementwise) + SiLU. reader is pass-through.
        # (Only ~1.2ms in trace — not worth moving into the compute kernel.)
        dot = ttnn.add(ttnn.add(ttnn.mul(L["w0"], L["sA"]), ttnn.mul(L["w1"], L["sB"])),
                       ttnn.add(ttnn.mul(L["w2"], L["sC"]), ttnn.mul(L["w3"], qkv)))
        qkv = ttnn.silu(dot)
        zba = ttnn.linear(h, L["Wzba"])  # fused z|b|a, one matmul
        z = ttnn.slice(zba, [0, 0, 0, 0], [1, 1, 1, vd_p])
        b = ttnn.slice(zba, [0, 0, 0, vd_p], [1, 1, 1, vd_p + nv_p])
        a = ttnn.slice(zba, [0, 0, 0, vd_p + nv_p], [1, 1, 1, vd_p + 2 * nv_p])
        # decay/beta computed host-side (ttnn) instead of in the NCRISC reader kernel
        # (libm expf/logf overflow Wormhole's small NCRISC data region):
        #   beta = sigmoid(b);  decay = exp(-exp(A_log) * softplus(a + dt_bias))
        beta = ttnn.sigmoid(b)
        decay = ttnn.exp(ttnn.neg(ttnn.mul(ttnn.exp(L["A"]), ttnn.softplus(ttnn.add(a, L["dt"])))))
        o, _, _ = ttnn.experimental.deltanet_decode_full(qkv, z, beta, decay, L["cs"], L["rs"], L["conv"], L["A"], L["dt"], L["nw"],
            num_heads=nvp, num_k_heads=nkp, k_head_dim=dk, v_head_dim=dv, conv_dim=conv_dim_p // TP,
            conv_kernel_size=Kc, head_expand_ratio=ratio)
        return AR(ttnn.linear(o, L["Wo"]))

    Ip = I // TP
    def mlp(h, L):
        gu = ttnn.linear(h, L["gu"])  # fused gate|up, one matmul
        g = ttnn.silu(ttnn.slice(gu, [0, 0, 0, 0], [1, 1, 1, Ip]))
        u = ttnn.slice(gu, [0, 0, 0, Ip], [1, 1, 1, 2 * Ip])
        return AR(ttnn.linear(ttnn.mul(g, u), L["down"]))

    def step(x):
        for L in layers:
            r = x; h = ttnn.rms_norm(x, epsilon=1e-6, weight=L["in_norm"])
            h = attn(h, L) if L["type"] == "full_attention" else deltanet(h, L)
            x = ttnn.add(r, h)
            r = x; h = ttnn.rms_norm(x, epsilon=1e-6, weight=L["post_norm"])
            x = ttnn.add(r, mlp(h, L))
        x = ttnn.rms_norm(x, epsilon=1e-6, weight=final_norm)
        logits = ttnn.all_gather(ttnn.linear(x, lm_head), dim=3, cluster_axis=1, topology=ttnn.Topology.Linear)
        return logits

    try:
        x0 = fR(torch.randn(1, 1, 1, H) * 0.1)
        for _ in range(3):  # warmup / JIT
            step(x0)
        ttnn.synchronize_device(md)
        times = []
        for _ in range(args.steps):
            t0 = time.perf_counter()
            lg = step(x0)
            ttnn.synchronize_device(md)
            times.append((time.perf_counter() - t0) * 1000)
        med = statistics.median(times)
        print(f"\n[TP decode EAGER] {NL} layers, {TP} chips, {args.steps} steps", flush=True)
        print(f"  median {med:.1f} ms/tok ({1000/med:.2f} tok/s), min {min(times):.1f} ms", flush=True)

        # --- trace the step to get device-bound TP time (removes host dispatch) ---
        try:
            ttnn.synchronize_device(md)
            tid = ttnn.begin_trace_capture(md, cq_id=0)
            _ = step(x0)
            ttnn.end_trace_capture(md, tid, cq_id=0)
            ttnn.synchronize_device(md)
            tt = []
            for _ in range(args.steps):
                t0 = time.perf_counter()
                ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
                ttnn.synchronize_device(md)
                tt.append((time.perf_counter() - t0) * 1000)
            tmed = statistics.median(tt)
            print(f"[TP decode TRACE] median {tmed:.1f} ms/tok ({1000/tmed:.2f} tok/s), min {min(tt):.1f} ms", flush=True)
            ttnn.release_trace(md, tid)
        except Exception as e:
            print(f"[trace] skipped: {type(e).__name__}: {str(e)[:160]}", flush=True)

        print(f"  vs single-chip eager 172ms / trace 164ms (5.82-6.10 tok/s)", flush=True)
        print(f"  target: <=50ms/step = 20 tok/s", flush=True)
    finally:
        ttnn.close_mesh_device(md)


if __name__ == "__main__":
    main()
