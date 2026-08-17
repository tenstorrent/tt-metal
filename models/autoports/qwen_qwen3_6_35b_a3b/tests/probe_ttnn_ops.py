# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Device probe for the TTNN op behaviours the functional decoder depends on.

Not a pytest: it is a single-process script so each probe runs on one device open, and a
failure prints the exact op and shapes rather than aborting collection. Run with

    python models/autoports/qwen_qwen3_6_35b_a3b/tests/probe_ttnn_ops.py
"""

import traceback

import torch

import ttnn

RESULTS = []


def probe(name):
    def deco(fn):
        def run(device):
            try:
                detail = fn(device)
                RESULTS.append(("OK", name, detail))
            except Exception as exc:  # noqa: BLE001 - probe reports, never raises
                RESULTS.append(("FAIL", name, f"{type(exc).__name__}: {exc}"))
                traceback.print_exc()

        run.__name__ = fn.__name__
        return run

    return deco


def tt(x, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(x, dtype=dtype, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def sparse_pcfg(cores, m, n, k, in0_block_w=8, out_subblock_w=1):
    import math

    core_x, core_y = cores
    num_cores = core_x * core_y
    nt = math.ceil(n / 32)
    kt = math.ceil(k / 32)
    if kt % in0_block_w != 0:
        divs = [d for d in range(2, in0_block_w + 1) if kt % d == 0]
        in0_block_w = max(divs) if divs else kt
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(core_x, core_y),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=out_subblock_w,
        out_block_h=1,
        out_block_w=1,
        per_core_M=max(32, m) // 32,
        per_core_N=(nt + num_cores - 1) // num_cores,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )


def pcc(a, b):
    x = a.reshape(-1).double() - a.reshape(-1).double().mean()
    y = b.reshape(-1).double() - b.reshape(-1).double().mean()
    return float((x @ y) / (x.norm() * y.norm() + 1e-30))


# ---------------------------------------------------------------------------------------
@probe("slice non-tile-aligned on dim -2 (conv shift)")
def p_slice(device):
    x = torch.randn(1, 1, 64 + 3, 8192)
    t = tt(x, device)
    outs = []
    for j in range(4):
        s = ttnn.slice(t, [0, 0, j, 0], [1, 1, j + 64, 8192])
        outs.append(pcc(ttnn.to_torch(s), x[:, :, j : j + 64]))
    return f"pcc per shift {['%.6f' % p for p in outs]}"


@probe("broadcast mul [.,N,1] x [.,N,D]")
def p_bcast_mul(device):
    x = torch.randn(2, 4, 64, 128)
    s = torch.randn(2, 4, 64, 1)
    out = ttnn.to_torch(ttnn.mul(tt(x, device), tt(s, device)))
    return f"pcc {pcc(out, x * s):.6f}"


@probe("broadcast sub [.,C,1] - [.,1,C] (decay mask)")
def p_bcast_sub(device):
    g = torch.randn(8, 1, 64, 1)
    h = torch.randn(8, 1, 1, 64)
    out = ttnn.to_torch(ttnn.sub(tt(g, device), tt(h, device)))
    return f"shape {tuple(out.shape)} pcc {pcc(out, g - h):.6f}"


@probe("cumsum dim=-1 on 4D tile")
def p_cumsum(device):
    x = torch.randn(4, 8, 32, 64)
    out = ttnn.to_torch(ttnn.cumsum(tt(x, device, dtype=ttnn.float32), dim=-1))
    return f"pcc {pcc(out, x.cumsum(-1)):.6f}"


@probe("rms_norm with weight + epsilon")
def p_rms_norm(device):
    x = torch.randn(1, 2, 64, 128)
    w = torch.rand(128) + 0.5
    eps = 1e-6
    ref = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w
    out = ttnn.to_torch(ttnn.rms_norm(tt(x, device), weight=tt(w.reshape(1, 1, 1, -1), device), epsilon=eps))
    return f"pcc {pcc(out, ref):.6f}"


@probe("rms_norm as l2norm (eps/D, w=1/sqrt(D))")
def p_l2(device):
    d, eps = 128, 1e-6
    x = torch.randn(1, 4, 64, d)
    ref = x * torch.rsqrt(x.pow(2).sum(-1, keepdim=True) + eps)
    w = torch.full((1, 1, 1, d), d**-0.5)
    out = ttnn.to_torch(
        ttnn.rms_norm(tt(x, device, dtype=ttnn.float32), weight=tt(w, device, dtype=ttnn.float32), epsilon=eps / d)
    )
    return f"pcc {pcc(out, ref):.8f} maxabs {float((out - ref).abs().max()):.3e}"


@probe("rotary_embedding_hf on 64-wide partial rope (prefill)")
def p_rope_prefill(device):
    seq, nh, rd = 128, 16, 64
    q = torch.randn(1, nh, seq, rd)
    cos = torch.randn(1, 1, seq, rd)
    sin = torch.randn(1, 1, seq, rd)
    x1, x2 = q[..., : rd // 2], q[..., rd // 2 :]
    ref = q * cos + torch.cat([-x2, x1], dim=-1) * sin
    out = ttnn.to_torch(
        ttnn.experimental.rotary_embedding_hf(tt(q, device), tt(cos, device), tt(sin, device), is_decode_mode=False)
    )
    return f"pcc {pcc(out, ref):.6f}"


@probe("manual partial rope decode on [1,b,nh,256] interleaved")
def p_rope_decode(device):
    b, nh, hd, rd = 8, 16, 256, 64
    q = torch.randn(1, b, nh, hd)
    cos = torch.randn(1, b, 1, rd)
    sin = torch.randn(1, b, 1, rd)
    head, tail = q[..., :rd], q[..., rd:]
    x1, x2 = head[..., : rd // 2], head[..., rd // 2 :]
    ref = torch.cat([head * cos + torch.cat([-x2, x1], dim=-1) * sin, tail], dim=-1)

    tq = tt(q, device)
    tcos = ttnn.repeat(tt(cos, device), ttnn.Shape([1, 1, nh, 1]))
    tsin = ttnn.repeat(tt(sin, device), ttnn.Shape([1, 1, nh, 1]))
    t_head = ttnn.slice(tq, [0, 0, 0, 0], [1, b, nh, rd])
    t_tail = ttnn.slice(tq, [0, 0, 0, rd], [1, b, nh, hd])
    h1 = ttnn.slice(t_head, [0, 0, 0, 0], [1, b, nh, rd // 2])
    h2 = ttnn.slice(t_head, [0, 0, 0, rd // 2], [1, b, nh, rd])
    rot = ttnn.concat([ttnn.neg(h2), h1], dim=-1)
    rotated = ttnn.add(ttnn.mul(t_head, tcos), ttnn.mul(rot, tsin))
    out = ttnn.to_torch(ttnn.concat([rotated, t_tail], dim=-1))
    return f"pcc {pcc(out, ref):.6f}"


@probe("topk k=8 over 256 + scatter to dense")
def p_topk(device):
    tokens, e, k = 64, 256, 8
    logits = torch.randn(1, 1, tokens, e)
    t = tt(logits, device)
    vals, idx = ttnn.topk(t, k=k, dim=-1, sorted=True)
    dense = ttnn.scatter(ttnn.zeros_like(t), dim=3, index=idx, src=vals)
    ref_v, ref_i = torch.topk(logits, k, dim=-1)
    ref_dense = torch.zeros_like(logits).scatter_(-1, ref_i, ref_v)
    got = ttnn.to_torch(dense)
    nz = int((got != 0).sum())
    return f"vals pcc {pcc(ttnn.to_torch(vals), ref_v):.6f} dense pcc {pcc(got, ref_dense):.6f} nonzeros {nz} (want {tokens*k})"


@probe("nlp_create_qkv_heads 16/2 head_dim 256 (prefill)")
def p_create_heads(device):
    seq, hd, nh, nkv = 128, 256, 16, 2
    x = torch.randn(1, 1, seq, (nh + 2 * nkv) * hd)
    q, k, v = ttnn.experimental.nlp_create_qkv_heads(
        tt(x, device), num_heads=nh, num_kv_heads=nkv, transpose_k_heads=False, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    ref_q = x[..., : nh * hd].reshape(1, seq, nh, hd).permute(0, 2, 1, 3)
    ref_k = x[..., nh * hd : (nh + nkv) * hd].reshape(1, seq, nkv, hd).permute(0, 2, 1, 3)
    return (
        f"q {tuple(q.shape)} pcc {pcc(ttnn.to_torch(q), ref_q):.6f} "
        f"k {tuple(k.shape)} pcc {pcc(ttnn.to_torch(k), ref_k):.6f} v {tuple(v.shape)}"
    )


@probe("nlp_create_qkv_heads_decode 16/2 head_dim 256")
def p_create_heads_decode(device):
    b, hd, nh, nkv = 8, 256, 16, 2
    x = torch.randn(1, 1, b, (nh + 2 * nkv) * hd)
    q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
        tt(x, device),
        num_heads=nh,
        num_kv_heads=nkv,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1),
    )
    return f"q {tuple(q.shape)} k {tuple(k.shape)} v {tuple(v.shape)} qmem {q.memory_config().memory_layout}"


@probe("sparse_matmul mode (False,True) decode: a[1,B,1,K] b[1,E,K,N]")
def p_sparse_decode(device):
    b, e, k, n, topk = 4, 256, 2048, 512, 8
    a = torch.randn(1, b, 1, k)
    w = torch.randn(1, e, k, n) * 0.02
    mask = torch.zeros(1, b, 1, e)
    sel = torch.stack([torch.randperm(e)[:topk] for _ in range(b)])
    for i in range(b):
        mask[0, i, 0, sel[i]] = 1.0
    out = ttnn.sparse_matmul(
        tt(a, device),
        tt(w, device),
        sparsity=tt(mask, device, layout=ttnn.ROW_MAJOR_LAYOUT),
        nnz=b * topk,
        program_config=sparse_pcfg((8, 8), 1, n, k),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    got = ttnn.to_torch(out).reshape(b, e, n)
    ref = torch.zeros(b, e, n)
    for i in range(b):
        for j in sel[i]:
            ref[i, j] = a[0, i, 0] @ w[0, j]
    return f"out {tuple(out.shape)} pcc {pcc(got, ref):.6f}"


@probe("sparse_matmul mode (True,False) down: a[A,E,M,K] b[1,E,K,N] sparsity[1,1,A,E]")
def p_sparse_down(device):
    a_dim, e, m, k, n, topk = 4, 256, 32, 512, 2048, 8
    a = torch.randn(a_dim, e, m, k) * 0.05
    w = torch.randn(1, e, k, n) * 0.02
    mask = torch.zeros(1, 1, a_dim, e)
    sel = torch.stack([torch.randperm(e)[:topk] for _ in range(a_dim)])
    for i in range(a_dim):
        mask[0, 0, i, sel[i]] = 1.0
    out = ttnn.sparse_matmul(
        tt(a, device),
        tt(w, device),
        sparsity=tt(mask, device, layout=ttnn.ROW_MAJOR_LAYOUT),
        nnz=a_dim * topk,
        is_input_a_sparse=True,
        is_input_b_sparse=False,
        program_config=sparse_pcfg((8, 8), m, n, k),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    got = ttnn.to_torch(out)
    ref = torch.zeros(a_dim, e, m, n)
    for i in range(a_dim):
        for j in sel[i]:
            ref[i, j] = a[i, j] @ w[0, j]
    return f"out {tuple(out.shape)} pcc {pcc(got, ref):.6f}"


@probe("sparse_matmul mode (False,True) prefill tile-groups: a[1,G,32,K]")
def p_sparse_prefill(device):
    g, e, k, n, hit = 4, 256, 2048, 512, 40
    a = torch.randn(1, g, 32, k)
    w = torch.randn(1, e, k, n) * 0.02
    mask = torch.zeros(1, g, 1, e)
    sel = torch.stack([torch.randperm(e)[:hit] for _ in range(g)])
    for i in range(g):
        mask[0, i, 0, sel[i]] = 1.0
    out = ttnn.sparse_matmul(
        tt(a, device),
        tt(w, device),
        sparsity=tt(mask, device, layout=ttnn.ROW_MAJOR_LAYOUT),
        nnz=g * hit,
        program_config=sparse_pcfg((8, 8), 32, n, k),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    got = ttnn.to_torch(out).reshape(g, e, 32, n)
    ref = torch.zeros(g, e, 32, n)
    for i in range(g):
        for j in sel[i]:
            ref[i, j] = a[0, i] @ w[0, j]
    return f"out {tuple(out.shape)} pcc {pcc(got, ref):.6f}"


@probe("batched matmul [b,32,128,128] chains")
def p_batched_mm(device):
    a = torch.randn(2, 32, 128, 128) * 0.1
    b = torch.randn(2, 32, 128, 128) * 0.1
    out = ttnn.to_torch(ttnn.matmul(tt(a, device, dtype=ttnn.float32), tt(b, device, dtype=ttnn.float32)))
    return f"pcc {pcc(out, a @ b):.6f}"


@probe("in-place ops for persistent state (mul/add output_tensor, copy)")
def p_inplace(device):
    x = torch.randn(2, 32, 128, 128)
    y = torch.randn(2, 32, 128, 128)
    tx = tt(x, device, dtype=ttnn.float32)
    ty = tt(y, device, dtype=ttnn.float32)
    addr = tx.buffer_address()
    ttnn.mul(tx, 0.5, output_tensor=tx)
    ttnn.add(tx, ty, output_tensor=tx)
    same = tx.buffer_address() == addr
    ref = x * 0.5 + y
    got = ttnn.to_torch(tx)
    # ttnn.copy into an existing buffer
    tz = tt(torch.zeros_like(x), device, dtype=ttnn.float32)
    zaddr = tz.buffer_address()
    ttnn.copy(ty, tz)
    return (
        f"inplace pcc {pcc(got, ref):.6f} addr_stable={same} "
        f"copy pcc {pcc(ttnn.to_torch(tz), y):.6f} copy_addr_stable={tz.buffer_address() == zaddr}"
    )


@probe("paged_fill_cache + paged_sdpa_decode head_dim=256 nkv=2")
def p_paged(device):
    b, nkv, hd, nh = 2, 2, 256, 16
    block, seq = 64, 256
    num_blocks_per_seq = seq // block
    max_blocks = b * num_blocks_per_seq
    k = torch.randn(b, nkv, seq, hd)
    v = torch.randn(b, nkv, seq, hd)
    cache_k = tt(torch.zeros(max_blocks, nkv, block, hd), device)
    cache_v = tt(torch.zeros(max_blocks, nkv, block, hd), device)
    perm = torch.randperm(max_blocks).reshape(b, num_blocks_per_seq)
    page_table = ttnn.from_torch(perm.int(), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    for u in range(b):
        ttnn.experimental.paged_fill_cache(cache_k, tt(k[u : u + 1], device), page_table, batch_idx=u)
        ttnn.experimental.paged_fill_cache(cache_v, tt(v[u : u + 1], device), page_table, batch_idx=u)
    cur = torch.tensor([seq - 1, seq - 3], dtype=torch.int32)
    q = torch.randn(1, b, nh, hd)
    out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
        tt(q, device),
        cache_k,
        cache_v,
        page_table_tensor=page_table,
        cur_pos_tensor=ttnn.from_torch(cur, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device),
        scale=hd**-0.5,
    )
    got = ttnn.to_torch(out)  # [1, b, nh, hd]
    refs = []
    for u in range(b):
        n = int(cur[u]) + 1
        kk = k[u].repeat_interleave(nh // nkv, dim=0)[:, :n]
        vv = v[u].repeat_interleave(nh // nkv, dim=0)[:, :n]
        w = torch.softmax((q[0, u].unsqueeze(1) @ kk.transpose(-1, -2)) * hd**-0.5, dim=-1)
        refs.append((w @ vv).squeeze(1))
    ref = torch.stack(refs).unsqueeze(0)
    return f"out {tuple(out.shape)} pcc {pcc(got, ref):.6f}"


@probe("chunked_scaled_dot_product_attention with chunk_start_idx_tensor")
def p_chunked_sdpa(device):
    b, nkv, hd, nh = 1, 2, 256, 16
    block, seq, chunk = 64, 512, 128
    nbps = seq // block
    k = torch.randn(b, nkv, seq, hd)
    v = torch.randn(b, nkv, seq, hd)
    cache_k = tt(torch.zeros(nbps, nkv, block, hd), device)
    cache_v = tt(torch.zeros(nbps, nkv, block, hd), device)
    page_table = ttnn.from_torch(
        torch.arange(nbps).reshape(b, nbps).int(), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    ttnn.experimental.paged_fill_cache(cache_k, tt(k, device), page_table, batch_idx=0)
    ttnn.experimental.paged_fill_cache(cache_v, tt(v, device), page_table, batch_idx=0)
    q = torch.randn(b, nh, seq, hd)
    pcfg = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
        q_chunk_size=chunk,
        k_chunk_size=chunk,
        exp_approx_mode=False,
    )
    idx_t = ttnn.from_torch(
        torch.tensor([0], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    outs = []
    for start in range(0, seq, chunk):
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(torch.tensor([start], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT),
            idx_t,
        )
        o = ttnn.transformer.chunked_scaled_dot_product_attention(
            input_tensor_q=tt(q[:, :, start : start + chunk], device),
            input_tensor_k=cache_k,
            input_tensor_v=cache_v,
            page_table_tensor=page_table,
            chunk_start_idx_tensor=idx_t,
            program_config=pcfg,
            scale=hd**-0.5,
        )
        outs.append(ttnn.to_torch(o))
    got = torch.cat(outs, dim=2)
    kk = k.repeat_interleave(nh // nkv, dim=1)
    vv = v.repeat_interleave(nh // nkv, dim=1)
    ref = torch.nn.functional.scaled_dot_product_attention(q, kk, vv, is_causal=True, scale=hd**-0.5)
    return f"out {tuple(got.shape)} pcc {pcc(got, ref):.6f}"


@probe("paged_update_cache decode with page table (height-sharded input)")
def p_paged_update(device):
    b, nkv, hd = 2, 2, 256
    block, seq = 64, 256
    nbps = seq // block
    max_blocks = b * nbps
    cache = tt(torch.zeros(max_blocks, nkv, block, hd), device)
    perm = torch.randperm(max_blocks).reshape(b, nbps)
    page_table = ttnn.from_torch(perm.int(), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    upd = torch.randn(1, b, nkv, hd)
    pos = torch.tensor([5, 130], dtype=torch.int32)
    shard = ttnn.create_sharded_memory_config(
        shape=(32, hd),
        core_grid=ttnn.num_cores_to_corerangeset(b, device.compute_with_storage_grid_size(), row_wise=True),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    upd_tt = ttnn.to_memory_config(tt(upd, device), shard)
    ttnn.experimental.paged_update_cache(
        cache,
        upd_tt,
        update_idxs_tensor=ttnn.from_torch(pos, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device),
        page_table=page_table,
    )
    host = ttnn.to_torch(cache)
    ok = []
    for u in range(b):
        p = int(pos[u])
        blk = int(perm[u, p // block])
        ok.append(pcc(host[blk, :, p % block, :], upd[0, u]))
    return f"per-user write pcc {['%.6f' % x for x in ok]}"


@probe("embedding lookup for decode rope table")
def p_embedding(device):
    table = torch.randn(4096, 64)
    idx = torch.tensor([[0, 5, 4095, 1234]], dtype=torch.int32)
    out = ttnn.embedding(
        ttnn.from_torch(idx, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device),
        tt(table, device),
        layout=ttnn.TILE_LAYOUT,
    )
    got = ttnn.to_torch(out)
    return f"shape {tuple(got.shape)} pcc {pcc(got.reshape(-1, 64), table[idx[0].long()]):.6f}"


@probe("sum over expert dim + reshape 6D->4D")
def p_sum(device):
    x = torch.randn(1, 4, 1, 8, 1, 64)
    t = tt(x, device)
    r = ttnn.reshape(t, (4, 8, 1, 64))
    s = ttnn.sum(r, dim=1)
    return f"reshape {tuple(r.shape)} sum {tuple(s.shape)} pcc {pcc(ttnn.to_torch(s).reshape(4,1,64), x.sum(3).reshape(4,1,64)):.6f}"


def main():
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        for fn in [
            p_slice,
            p_bcast_mul,
            p_bcast_sub,
            p_cumsum,
            p_rms_norm,
            p_l2,
            p_rope_prefill,
            p_rope_decode,
            p_topk,
            p_create_heads,
            p_create_heads_decode,
            p_sparse_decode,
            p_sparse_down,
            p_sparse_prefill,
            p_batched_mm,
            p_inplace,
            p_paged,
            p_chunked_sdpa,
            p_paged_update,
            p_embedding,
            p_sum,
        ]:
            fn(device)
    finally:
        ttnn.close_mesh_device(device)
    print("\n" + "=" * 100)
    for status, name, detail in RESULTS:
        print(f"PROBE {status:4s} | {name:70s} | {detail}")
    print("=" * 100)
    print(f"PROBE SUMMARY {sum(1 for r in RESULTS if r[0] == 'OK')}/{len(RESULTS)} ok")


if __name__ == "__main__":
    main()
