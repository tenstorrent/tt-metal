# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""P0-C: M3 MSA ops at the cross-chunk cache-read shapes on a (2,4) sub-mesh (SP=2, TP=4).

Mirrors msa_sp_attention_cache_read -> msa_indexer_sparse (tt/attention/msa.py) after the high_bw_all_gather:
per chip
  q        [1, 16, rows, 128] bf16 (64 q heads / TP=4)      index_q [1, 1, rows, 128] bf16 TILE (4 index heads / TP)
  K, V     [1, 1, T, 128]  cache dtype bf8 TILE (1 kv head / TP), index_k [1, 1, T, 128] bf8
           (M3_INDEX_CACHE_BF16=1 -> bf16), T = seq_local * SP, the persistent gather buffer, block-cyclic
           slab order, valid natural prefix kv_len.
  chunk_local = rows, cached_len = kv_len - rows*SP (so the chunk just written ends at kv_len),
  chunk_start_idx = cached_len, cluster_axis = block_cyclic_sp_axis = 0, block_cyclic_chunk_local = rows,
  num_groups = 1 (local kv heads), block 128, top-16, scale 128**-0.5.
Grid points with kv_len < rows*SP (the chunk alone is longer than the context) are written as SKIP.

Ops (one CSV row each):
  indexer      indexer_score_msa (IndexerScoreProgramConfig(64, 1024, 0), kv_len bound)
  topk         topk_large_indices(block_scores, k=16, valid_length=kv_len/128)
  sparse_sdpa  sparse_sdpa_msa on the block ids (q already ROW_MAJOR)
  msa_chain    msa_indexer_sparse end to end (incl. its q untilize / out tilize)
  index_branch index_branch_forward: index_q/k proj (bf8 weights) -> split heads -> per-head RMSNorm ->
               indexed RoPE on the whole-cache block-cyclic cos/sin (kv_actual_global=cached_len)

  python3 bench_msa.py --dry-run
"""

import argparse
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_common as bc  # noqa: E402

COLUMNS = [
    "op",
    "rows",
    "kv_len",
    "worst_ms",
    "mean_ms",
    "min_ms",
    "flops",
    "bytes",
    "cores_used",
    "T_buffer",
    "cached_len",
] + bc.TIMING_COLUMNS

OPS = ["indexer", "topk", "sparse_sdpa", "msa_chain", "index_branch"]
HQ = bc.N_Q_HEADS // bc.TP  # 16 q heads per chip
N_IDX = bc.N_INDEX_HEADS // bc.TP  # 1 index head per chip


def extent(rows, kv_len, sp=bc.SP, block=bc.MSA_BLOCK):
    """(cached_len, T, n_rows) or None when infeasible; msa_cache_read_extent's math."""
    chunk_global = rows * sp
    cached_len = kv_len - chunk_global
    if cached_len < 0:
        return None
    end = cached_len + chunk_global
    kv = (end + block - 1) // block * block
    assert kv == kv_len, (rows, kv_len)
    full_slabs, rem = divmod(end, chunk_global)
    n_rows = full_slabs * rows + min(rem, rows)
    seq_local = math.ceil(n_rows / rows) * rows  # a cache capacity is a whole number of chunks
    return cached_len, seq_local * sp, n_rows


def cost(op, rows, kv_len, T, idx_bpe):
    """(flops, bytes) per chip. The indexer's flops count every (query, key<kv_len) pair (upper bound: the
    causal mask cuts the chunk's own triangle); sparse_sdpa K/V bytes count the 16 blocks per query token,
    which is how the kernel fetches them (one work item = one (kv group, query token))."""
    d, bs, k = bc.HEAD_DIM, bc.MSA_BLOCK, bc.MSA_TOPK_BLOCKS
    nb = T // bs
    bf8 = bc.BPE["bf8"]
    idx = dict(
        flops=2 * rows * N_IDX * kv_len * d,
        bytes=kv_len * d * idx_bpe + rows * N_IDX * d * 2 + rows * nb * 2,
    )
    tk = dict(flops=0, bytes=rows * nb * 2 + rows * k * 4)
    sd = dict(
        flops=2 * 2 * HQ * rows * k * bs * d,
        bytes=rows * HQ * d * 2 * 2 + rows * k * 4 + bc.N_KV_HEADS // bc.TP * rows * k * bs * d * 2 * bf8,
    )
    ib = dict(
        flops=2 * rows * bc.EMB * (N_IDX + 1) * d,
        bytes=rows * bc.EMB * 2 + bc.EMB * (N_IDX + 1) * d * bf8 + rows * (N_IDX + 1) * d * 2 * 4,
    )
    if op == "indexer":
        c = idx
    elif op == "topk":
        c = tk
    elif op == "sparse_sdpa":
        c = sd
    elif op == "msa_chain":
        c = {key: idx[key] + tk[key] + sd[key] for key in ("flops", "bytes")}
    else:
        c = ib
    return int(c["flops"]), int(c["bytes"])


def main():
    p = bc.add_common_args(argparse.ArgumentParser(description=__doc__), "msa.csv")
    p.add_argument("--rows", type=bc.int_list, default=[1024, 2048, 4096])
    p.add_argument("--kv-len", type=bc.int_list, default=[4096, 141312, 548864])
    p.add_argument("--ops", type=lambda s: s.split(","), default=OPS)
    args = p.parse_args()
    idx_bf16 = os.getenv("M3_INDEX_CACHE_BF16") == "1"
    idx_bpe = 2.0 if idx_bf16 else bc.BPE["bf8"]

    print(
        f"[bench_msa] sub-mesh {bc.SUBMESH}: {HQ} q heads / 1 kv head / {N_IDX} index head per chip, d {bc.HEAD_DIM}, "
        f"block {bc.MSA_BLOCK} top-{bc.MSA_TOPK_BLOCKS}, K/V bf8, index_k {'bf16' if idx_bf16 else 'bf8'}; "
        f"ops {args.ops}; warmup {args.warmup} repeats {args.repeats}"
    )
    points = []
    for kv_len in args.kv_len:
        for rows in args.rows:
            ext = extent(rows, kv_len)
            points.append((rows, kv_len, ext))
            if ext is None:
                print(f"  rows={rows:5d} kv_len={kv_len:7d}  SKIP (chunk {rows * bc.SP} > kv_len)")
                continue
            cached_len, T, n_rows = ext
            print(f"  rows={rows:5d} kv_len={kv_len:7d}  cached_len={cached_len:7d} T={T:7d} n_rows/chip={n_rows}")
            for op in args.ops:
                f, b = cost(op, rows, kv_len, T, idx_bpe)
                print(f"      {op:12s} GFLOP={f / 1e9:9.2f}  MB={b / 1e6:9.1f}")
    if args.dry_run:
        return 0

    bc.setup_env()
    import torch
    import ttnn

    from models.demos.minimax_m3.tt.attention.msa import index_branch_forward, msa_indexer_sparse
    from models.tt_transformers.tt.common import get_rot_transformation_mat

    galaxy, mesh = bc.open_submesh()
    out = bc.CsvOut(args.out, COLUMNS)
    rep = ttnn.ReplicateTensorToMesh(mesh)
    sp_rows = ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=(2, None))
    scale = bc.HEAD_DIM**-0.5
    idx_dtype = ttnn.bfloat16 if idx_bf16 else ttnn.bfloat8_b

    def dev(t, dtype, layout=ttnn.TILE_LAYOUT, mapper=rep):
        return ttnn.from_torch(
            t, device=mesh, dtype=dtype, layout=layout, mesh_mapper=mapper, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    def rnd(*shape):
        return torch.randn(*shape, dtype=torch.bfloat16) * 0.1

    try:
        torch.manual_seed(0)
        timer = bc.OpTimer(mesh)
        # Index-branch weights (per chip: index_q_proj column slice = 1 head, index_k_proj replicated).
        w = type("W", (), {})()
        w.index_q_proj = dev(rnd(1, 1, bc.EMB, N_IDX * bc.INDEX_DIM), ttnn.bfloat8_b)
        w.index_k_proj = dev(rnd(1, 1, bc.EMB, bc.INDEX_DIM), ttnn.bfloat8_b)
        w.index_q_norm = dev(
            torch.ones(1, 1, bc.INDEX_DIM // 32, 32, dtype=torch.bfloat16), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT
        )
        w.index_k_norm = dev(
            torch.ones(1, 1, bc.INDEX_DIM // 32, 32, dtype=torch.bfloat16), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT
        )
        trans_mat = dev(get_rot_transformation_mat(), ttnn.bfloat16)
        rotary_dim = bc.HEAD_DIM // 2

        kv_cache = {}  # T -> (k, v, index_k); one live set at a time
        for rows, kv_len, ext in points:
            if ext is None:
                for op in args.ops:
                    out.row(op=op, rows=rows, kv_len=kv_len, status=f"SKIP:chunk {rows * bc.SP} > kv_len")
                continue
            cached_len, T, _ = ext
            if T not in kv_cache:
                for tensors in kv_cache.values():
                    for x in tensors:
                        ttnn.deallocate(x)
                kv_cache.clear()
                kv_cache[T] = (
                    dev(rnd(1, 1, T, bc.HEAD_DIM), ttnn.bfloat8_b),
                    dev(rnd(1, 1, T, bc.HEAD_DIM), ttnn.bfloat8_b),
                    dev(rnd(1, 1, T, bc.INDEX_DIM), idx_dtype),
                )
            k, v, index_k = kv_cache[T]
            q_tile = dev(rnd(1, HQ, rows, bc.HEAD_DIM), ttnn.bfloat16)
            q_rm = ttnn.to_layout(q_tile, ttnn.ROW_MAJOR_LAYOUT)
            index_q = dev(rnd(1, N_IDX, rows, bc.INDEX_DIM), ttnn.bfloat16)

            common = dict(
                chunk_start_idx=cached_len,
                block_cyclic_sp_axis=0,
                block_cyclic_chunk_local=rows,
            )

            def run_indexer():
                return ttnn.experimental.indexer_score_msa(
                    index_q,
                    index_k,
                    scale=scale,
                    num_groups=1,
                    block_size=bc.MSA_BLOCK,
                    program_config=ttnn.IndexerScoreProgramConfig(
                        q_chunk_size=64, k_chunk_size=1024, head_group_size=0
                    ),
                    seq_shard_axes=[0],
                    kv_len=kv_len,
                    **common,
                )

            block_scores = run_indexer()
            ttnn.synchronize_device(mesh)

            def run_topk():
                return ttnn.experimental.topk_large_indices(
                    block_scores, k=bc.MSA_TOPK_BLOCKS, valid_length=kv_len // bc.MSA_BLOCK
                )

            block_ids = run_topk()

            def run_sparse():
                return ttnn.transformer.sparse_sdpa_msa(
                    q_rm, k, v, block_ids, scale=scale, block_size=bc.MSA_BLOCK, cluster_axis=0, **common
                )

            def run_chain():
                return msa_indexer_sparse(
                    index_q,
                    index_k,
                    q_tile,
                    k,
                    v,
                    scale=scale,
                    num_groups=1,
                    block_size=bc.MSA_BLOCK,
                    topk_blocks=bc.MSA_TOPK_BLOCKS,
                    device=mesh,
                    cluster_axis=0,
                    kv_len=kv_len,
                    **common,
                )

            hidden = rope = None
            if "index_branch" in args.ops:
                hidden = dev(rnd(1, 1, rows, bc.EMB), ttnn.bfloat16)
                # Whole-cache cos/sin, SP-sharded on rows (T/SP per chip), like TtPrefillRuntime._build_indexed_rope.
                ang = torch.rand(1, 1, T, rotary_dim) * 6.28
                rope = [
                    dev(torch.cos(ang).to(torch.bfloat16), ttnn.bfloat16, mapper=sp_rows),
                    dev(torch.sin(ang).to(torch.bfloat16), ttnn.bfloat16, mapper=sp_rows),
                ]

            def run_index_branch():
                return index_branch_forward(
                    hidden,
                    w,
                    rope,
                    trans_mat,
                    index_dim=bc.INDEX_DIM,
                    rms_norm_eps=1e-6,
                    kv_actual_global=cached_len,
                    cluster_axis=0,
                )

            fns = dict(
                indexer=run_indexer,
                topk=run_topk,
                sparse_sdpa=run_sparse,
                msa_chain=run_chain,
                index_branch=run_index_branch,
            )
            for op in args.ops:
                f, b = cost(op, rows, kv_len, T, idx_bpe)
                base = dict(op=op, rows=rows, kv_len=kv_len, flops=f, bytes=b, T_buffer=T, cached_len=cached_len)
                try:
                    res = timer.measure(fns[op], args.warmup, args.repeats)
                    out.row(**base, **res, status="OK")
                except Exception as e:
                    out.row(**base, status=f"ERROR:{type(e).__name__}:{str(e).splitlines()[0][:160] if str(e) else ''}")
            for x in [q_tile, q_rm, index_q, block_scores, block_ids, hidden] + (rope or []):
                if x is not None:
                    ttnn.deallocate(x)
    finally:
        out.close()
        bc.close_mesh(galaxy)
    print(f"[bench_msa] DONE -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
