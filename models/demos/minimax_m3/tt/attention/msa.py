# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MiniMax-M3 MSA (sparse) attention — the real model forward for the sparse layers (3-59).

Unlike the dense path (ring_joint, which reads the KV cache + gathers across SP *internally*),
``sparse_sdpa_msa`` is a pure dense-context kernel: it takes full-length K/V tensors and has no
cache-read. So the cache + cross-device gather live in THIS wrapper:

  each SP device holds a sequence shard of K / V / index_k (from the chunked-KV cache read).
  We AllGather those across the SP axis so every device materialises the full context, then:

    indexer_score_msa(index_q, index_k_full, chunk_start_idx=cached_len)  -> block scores
    topk_large_indices(topk_blocks)                                      -> block-ids   (the op
                                              already force-locals the current block, +inf)
    sparse_sdpa_msa(q, k_full, v_full, block-ids)                        -> attention out

SP sharding is CONTIGUOUS, no zigzag/balancing (per the op authors: chunked prefill needs no causal
load-balancing — MSA work per query is a fixed top-k, not the dense causal triangle).

``chunk_start_idx`` is the global position of query row 0. Because the indexer scores over the
gathered full context and the current chunk's queries all begin at ``cached_len``, this scalar is
uniform across SP devices (no per-device offset needed). Causality is encoded entirely by the block
selection; sparse_sdpa_msa applies no token mask.
"""

import os

import torch

import ttnn

from .operations import apply_qk_norm_per_head, apply_rope


def _dbg_msa_save(t, device, layer_idx, name):
    """DBG_MSA_BLOCKIDS=1: save a per-query MSA tensor (block_ids or block_scores), reassembled across SP
    rows on the query axis (col-0 device of each row), so we can diff chunked vs one-shot per global query
    position. -> /tmp/m3_msa/L{idx}_{name}.pt, shape [1, num_groups, S_global, last]."""
    if os.getenv("DBG_MSA_BLOCKIDS") != "1" or layer_idx is None:
        return
    try:
        import os as _os

        rows, cols = tuple(device.shape)
        dts = ttnn.get_device_tensors(t)
        shards = [ttnn.to_torch(dts[r * cols]) for r in range(rows)]  # col-0 of each SP row
        full = torch.cat(shards, dim=2)  # concat on the query (Sq) axis -> global order
        _os.makedirs("/tmp/m3_msa", exist_ok=True)
        torch.save(full, f"/tmp/m3_msa/L{layer_idx}_{name}.pt")
        from loguru import logger

        logger.info(f"[MSA L{layer_idx} {name}] saved {tuple(full.shape)}")
    except Exception as e:
        from loguru import logger

        logger.info(f"[MSA L{layer_idx} {name}] failed: {e}")


# M3 sparse_attention_config (configs/MiniMax-M3/config.json).
BLOCK_SIZE = 128
TOPK_BLOCKS = 16
NUM_INDEX_HEADS = 4  # sparse_num_index_heads (1 per GQA group; 1 per device at TP=4)
INDEX_DIM = 128  # sparse_index_dim


def _split_index_heads(t):
    """[1, 1, S, n*INDEX_DIM] -> [1, n, S, INDEX_DIM] (head-major split, like the main QKV split)."""
    s = t.shape[2]
    n = t.shape[-1] // INDEX_DIM
    t = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)
    t = ttnn.reshape(t, [1, s, n, INDEX_DIM])
    t = ttnn.permute(t, (0, 2, 1, 3))  # [1, n, S, INDEX_DIM]
    return ttnn.to_layout(t, ttnn.TILE_LAYOUT)


def index_branch_forward(hidden_states, weights, rope_mats, transformation_mat, *, rms_norm_eps):
    """The MSA index branch: pre-roped index_q (n index heads, 1/TP col) + index_k (single shared head).

    proj -> split heads -> per-head RMSNorm -> RoPE. index_q_proj is column-parallel (4 index heads ->
    1/TP col); index_k_proj is replicated (shared head). Per device this yields index_q [1, n_idx_local,
    S, INDEX_DIM] (n_idx_local=1 at TP=4) and index_k [1, 1, S, INDEX_DIM] -> the MSA indexer.

    VERIFIED 2026-06-26 against transformers-main MiniMaxM3VLIndexer source (not just a summary):
      * order proj -> q_norm/k_norm -> apply_rotary_pos_emb — matches.
      * rope on BOTH index_q AND index_k — matches.
      * rotary width: reference does apply_rotary_pos_emb(idx_q, idx_k, cos[..,:index_head_dim], ...);
        with head_dim=128, partial_rotary_factor=0.5 the model cos/sin are 64-wide, and index_head_dim=
        sparse_index_dim=128, so the slice yields the full 64 -> PARTIAL-64 (rotate first 64 of the
        128-wide index head), same as main attention. Our apply_rope(rope_mats=main 64-wide) matches.
      * norm: index_q_norm/index_k_norm gains ship in the checkpoint, applied per-head.
    """
    iq = _split_index_heads(ttnn.linear(hidden_states, weights.index_q_proj))  # [1, n_idx_local, S, IDX_DIM]
    iq = apply_qk_norm_per_head(iq, weights.index_q_norm, rms_norm_eps)
    iq = apply_rope(iq, rope_mats, transformation_mat, is_decode_mode=False)

    ik = _split_index_heads(ttnn.linear(hidden_states, weights.index_k_proj))  # [1, 1, S, IDX_DIM] (shared)
    ik = apply_qk_norm_per_head(ik, weights.index_k_norm, rms_norm_eps)
    ik = apply_rope(ik, rope_mats, transformation_mat, is_decode_mode=False)
    return iq, ik


def _cpu_fp32_indexer(index_q, index_k, ref, *, mode, chunk_start_idx, scale, cluster_axis, device):
    """Debug (option A): recompute the MSA indexer in HOST fp32 from the device's own bf16 index_q/index_k.
    mode="scores" (M3_CPU_INDEXER): return fp32 block_scores (bf16 tensor), device topk kept.
    mode="block_ids" (M3_CPU_TOPK): also do the top-k on host -> block_ids (uint32), bypassing the
        experimental topk_large_indices op. Sentinel 0xFFFFFFFF where the selected block scored -inf;
        top-k descending puts valid blocks as a contiguous prefix (matches sparse_sdpa_msa_reader).

    Isolates the indexer (score precision #6) and — in block_ids mode — the top-k op too. If KV-PCC
    recovers, the indexer/top-k was a driver; if flat, the decay is downstream (sdpa / MoE).

    Per device (r,c) at TP=4 holds ONE index group [1,1,s_local,dim] + the full (natural-order) index_k
    [1,1,T,dim]. Global query offset = chunk_start_idx + r*s_local (SP). Mirrors msa_block_selection.
    Reassembles [1,G,S,*] with dims=(2,1) (seq->rows, group->cols). ONE-SHOT/nocache only (index_k natural
    order; the cache-read path is block-cyclic)."""
    rows, cols = tuple(device.shape)
    iq_dts, ik_dts = ttnn.get_device_tensors(index_q), ttnn.get_device_tensors(index_k)
    iq0 = ttnn.to_torch(iq_dts[0]).float()
    s_local, dim = iq0.shape[2], iq0.shape[3]
    T = ttnn.to_torch(ik_dts[0]).float().shape[2]
    nblk = (T + BLOCK_SIZE - 1) // BLOCK_SIZE
    tpad = nblk * BLOCK_SIZE
    rs0 = ttnn.to_torch(ttnn.get_device_tensors(ref)[0])
    exp_last = nblk if mode == "scores" else TOPK_BLOCKS
    assert tuple(rs0.shape[-2:]) == (s_local, exp_last), f"cpu-indexer layout {tuple(rs0.shape)} != (s_local={s_local}, last={exp_last})"
    G, S = cols, rows * s_local
    last = nblk if mode == "scores" else TOPK_BLOCKS
    glob = torch.zeros(1, G, S, last, dtype=(torch.float32 if mode == "scores" else torch.int64))
    for d in range(rows * cols):
        r, c = d // cols, d % cols
        iq = ttnn.to_torch(iq_dts[d]).float()[0, 0]  # [s_local, dim]
        ik = ttnn.to_torch(ik_dts[d]).float()[0, 0]  # [T, dim]
        base = (chunk_start_idx or 0) + (r * s_local if cluster_axis is not None else 0)
        qpos = torch.arange(s_local) + base
        kpos = torch.arange(T)
        sc = scale * (iq @ ik.t())  # [s_local, T] fp32
        sc = sc.masked_fill(kpos[None, :] > qpos[:, None], float("-inf"))
        if tpad > T:
            sc = torch.cat([sc, sc.new_full((s_local, tpad - T), float("-inf"))], dim=1)
        bs = sc.view(s_local, nblk, BLOCK_SIZE).max(-1).values  # block max-pool [s_local, nblk]
        local = (qpos // BLOCK_SIZE).clamp(max=nblk - 1)
        bs[torch.arange(s_local), local] = float("inf")  # force-local current block
        if mode == "scores":
            glob[0, c, r * s_local : (r + 1) * s_local, :] = bs
        else:  # block_ids: top-k descending; -inf-scored slots -> sentinel (trail as a contiguous suffix)
            vals, idx = torch.topk(bs, TOPK_BLOCKS, dim=-1)  # [s_local, TOPK] descending
            idx = idx.to(torch.int64)
            idx[vals == float("-inf")] = 0xFFFFFFFF
            glob[0, c, r * s_local : (r + 1) * s_local, :] = idx
    mapper = ttnn.ShardTensor2dMesh(device, mesh_shape=(rows, cols), dims=(2, 1))
    out_dtype = ttnn.bfloat16 if mode == "scores" else ttnn.uint32
    return ttnn.from_torch(glob, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=out_dtype, mesh_mapper=mapper)


def _host_sdpa_glob(q, k, v, block_ids, *, chunk_start_idx, scale, cluster_axis, device, token_causal=True, mask_mode="causal"):
    """Host fp32 block-sparse attention from the device's own bf16 q/k/v + device block_ids (selection
    UNCHANGED). Returns the reassembled global output [1, Hq, S, hd] fp32 + the (rows,cols,Hql,s_local)
    shard geometry. ``token_causal`` matches the device op's causality (diag mask on/off) so an op-vs-host
    diff isolates numerics, not causality. Per device (r,c): Hq_local q heads, 1 kv head, query rows
    [r*s_local:...] at global base=chunk_start+r*s_local; placed into [1,Hq,S,hd] as dims=(2,1)."""
    rows, cols = tuple(device.shape)
    qd_, kd_, vd_, bd_ = (ttnn.get_device_tensors(t) for t in (q, k, v, block_ids))
    q0 = ttnn.to_torch(qd_[0]).float()
    Hql, s_local, hd = q0.shape[1], q0.shape[2], q0.shape[3]
    T = ttnn.to_torch(kd_[0]).float().shape[2]
    nblk = (T + BLOCK_SIZE - 1) // BLOCK_SIZE
    Hq, S, SENT = Hql * cols, rows * s_local, 0xFFFFFFFF
    glob = torch.zeros(1, Hq, S, hd, dtype=torch.float32)
    for d in range(rows * cols):
        r, c = d // cols, d % cols
        qd = ttnn.to_torch(qd_[d]).float()[0]  # [Hql, s_local, hd]
        kd = ttnn.to_torch(kd_[d]).float()[0, 0]  # [T, hd]
        vd = ttnn.to_torch(vd_[d]).float()[0, 0]  # [T, hd]
        bid = ttnn.to_torch(bd_[d])[0, 0].to(torch.int64)  # [s_local, TOPK]
        base = (chunk_start_idx or 0) + (r * s_local if cluster_axis is not None else 0)
        qpos = torch.arange(s_local) + base
        kpos = torch.arange(T)
        # block-selection mask [s_local, nblk] (sentinel -> dummy col), then expand to tokens (+ causal)
        bm = torch.zeros(s_local, nblk + 1, dtype=torch.bool)
        b = bid.clone()
        b[(b == SENT) | (b > nblk)] = nblk
        bm.scatter_(1, b, True)
        tokmask = bm[:, :nblk].repeat_interleave(BLOCK_SIZE, dim=1)[:, :T]  # [s_local, T]
        # mask_mode selects the causality applied on top of block selection:
        #   "block"    -> selected blocks only, no token causality (device diag=0)
        #   "causal"   -> + true token causality kpos<=qpos on ALL blocks (standard / golden)
        #   "faithful" -> mimic the DEVICE diag=1 op: attend all selected blocks fully, mask only the
        #                 DIAGONAL block's future tokens (block(k)==q//bs & k>q). Differs from "causal"
        #                 only on any SELECTED FUTURE block (block(k) > q//bs), which "causal" masks but
        #                 the device attends.
        if not token_causal or mask_mode == "block":
            mask = tokmask
        elif mask_mode == "faithful":
            block_of_k = (kpos // BLOCK_SIZE)[None, :]  # [1,T]
            db = (qpos // BLOCK_SIZE)[:, None]  # [s_local,1]
            diag_future = (block_of_k == db) & (kpos[None, :] > qpos[:, None])
            mask = tokmask & (~diag_future)
        else:  # "causal"
            mask = tokmask & (kpos[None, :] <= qpos[:, None])
        scores = torch.einsum("hid,td->hit", qd, kd) * scale  # [Hql, s_local, T]
        scores = scores.masked_fill(~mask[None], float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        glob[0, c * Hql : (c + 1) * Hql, r * s_local : (r + 1) * s_local, :] = torch.einsum("hit,td->hid", attn, vd)
    return glob, (rows, cols, Hql, s_local)


def _cpu_fp32_sdpa(q, k, v, block_ids, *, chunk_start_idx, scale, cluster_axis, device):
    """M3_CPU_SDPA=1 (debug, rung 2): replace the device sparse_sdpa_msa with HOST fp32 (fully token-causal).
    If KV-PCC recovers, the sdpa math is a driver; if flat, the decay is the MoE."""
    glob, (rows, cols, _, _) = _host_sdpa_glob(
        q, k, v, block_ids, chunk_start_idx=chunk_start_idx, scale=scale, cluster_axis=cluster_axis, device=device
    )
    mapper = ttnn.ShardTensor2dMesh(device, mesh_shape=(rows, cols), dims=(2, 1))
    return ttnn.from_torch(glob, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=mapper)


def _reassemble_msa_out(out_dev, rows, cols, Hql, s_local):
    """Device sparse_sdpa_msa output shards [1,Hql,s_local,hd] -> global [1,Hq,S,hd] fp32 (dims=(2,1)),
    matching _host_sdpa_glob's placement so the two can be PCC'd directly."""
    od_ = ttnn.get_device_tensors(out_dev)
    hd = ttnn.to_torch(od_[0]).shape[-1]
    glob = torch.zeros(1, Hql * cols, rows * s_local, hd, dtype=torch.float32)
    for d in range(rows * cols):
        r, c = d // cols, d % cols
        glob[0, c * Hql : (c + 1) * Hql, r * s_local : (r + 1) * s_local, :] = ttnn.to_torch(od_[d]).float()[0]
    return glob


def _dbg_op_vs_host(out_dev, q, k, v, block_ids, *, chunk_start_idx, scale, cluster_axis, device, layer_idx):
    """M3_MSA_DBG_OPDIFF=1: per-layer PCC of the DEVICE sparse_sdpa_msa output vs a HOST fp32 recompute on
    the SAME inputs + selection + causality. Isolates the kernel's own numerical error at real-model inputs
    (no KV-cache propagation). token_causal mirrors the live diag-mask setting so only numerics differ."""
    token_causal = os.getenv("M3_MSA_DIAG_MASK", "1") == "1"
    from loguru import logger

    def _pcc_t(dev, host):
        a, b = dev.flatten().double(), host.flatten().double()
        a, b = a - a.mean(), b - b.mean()
        den = (a.norm() * b.norm()).item()
        return 1.0 if den == 0 else (a @ b).item() / den

    # M3_MSA_DUMP_LAYER=N: find the worst SP-shard at layer N and save its exact real inputs so the diverging
    # op call can be replayed + DPRINT-dumped on a single device. Saved only for the real (low-PCC) pass.
    dump_layer = os.getenv("M3_MSA_DUMP_LAYER")
    if dump_layer is not None and int(dump_layer) == layer_idx:
        import os as _os

        rows, cols = tuple(device.shape)
        host_glob, (_, _, Hql, s_local) = _host_sdpa_glob(
            q, k, v, block_ids, chunk_start_idx=chunk_start_idx, scale=scale, cluster_axis=cluster_axis,
            device=device, token_causal=token_causal, mask_mode="causal",
        )
        od, qd, kd, vd, bd = (ttnn.get_device_tensors(t) for t in (out_dev, q, k, v, block_ids))
        worst_p, worst_d = 2.0, -1
        for dd in range(rows * cols):
            r, c = dd // cols, dd % cols
            dev_sh = ttnn.to_torch(od[dd]).float()[0]
            host_sh = host_glob[0, c * Hql : (c + 1) * Hql, r * s_local : (r + 1) * s_local, :]
            p = _pcc_t(dev_sh, host_sh)
            if p < worst_p:
                worst_p, worst_d = p, dd
        if worst_p < 0.95:  # real pass (compile/warmup pass is ~0.999); skip that one
            _os.makedirs("/tmp/m3_opdump", exist_ok=True)
            r, c = worst_d // cols, worst_d % cols
            torch.save(
                {
                    "q": ttnn.to_torch(qd[worst_d]), "k": ttnn.to_torch(kd[worst_d]),
                    "v": ttnn.to_torch(vd[worst_d]), "block_ids": ttnn.to_torch(bd[worst_d]),
                    "chunk_start_idx": (chunk_start_idx or 0) + (r * s_local if cluster_axis is not None else 0),
                    "scale": scale, "layer": layer_idx, "rank_r": r, "col_c": c, "shard_pcc": worst_p,
                    "dev_out": ttnn.to_torch(od[worst_d]),
                },
                f"/tmp/m3_opdump/L{layer_idx}_worst.pt",
            )
            logger.info(
                f"[msa-opdump L{layer_idx}] worst shard d={worst_d} (r={r},c={c}) PCC={worst_p:.5f} "
                f"chunk_start={(chunk_start_idx or 0)+r*s_local} -> /tmp/m3_opdump/L{layer_idx}_worst.pt"
            )

    def _pcc(dev, host):
        a, b = dev.flatten().double(), host.flatten().double()
        a, b = a - a.mean(), b - b.mean()
        denom = (a.norm() * b.norm()).item()
        return 1.0 if denom == 0 else (a @ b).item() / denom

    dev_glob = None
    # compare against both host causalities: standard kpos<=qpos vs device-faithful (diagonal-only mask)
    modes = ["causal", "faithful"] if token_causal else ["block"]
    out = []
    for m in modes:
        host_glob, geom = _host_sdpa_glob(
            q, k, v, block_ids, chunk_start_idx=chunk_start_idx, scale=scale, cluster_axis=cluster_axis,
            device=device, token_causal=token_causal, mask_mode=m,
        )
        if dev_glob is None:
            dev_glob = _reassemble_msa_out(out_dev, *geom)
        out.append(f"{m}={_pcc(dev_glob, host_glob):.5f}")
    logger.info(f"[msa-opdiff L{layer_idx}] device-op vs host-fp32  " + "  ".join(out))


def msa_indexer_sparse(
    index_q,
    index_k,
    q,
    k,
    v,
    *,
    chunk_start_idx,
    scale,
    num_groups,
    device,
    return_block_ids=False,
    cluster_axis=None,
    layer_idx=None,
    dbg_tag=None,
):
    """The MSA op chain over a FULL-context (already-gathered) K/V; index_q/q may stay SP-sharded.

    index_q [1, num_groups, Sq, INDEX_DIM]   index_k [1, 1, T, INDEX_DIM]   (1 shared index-k head)
    q       [1, Hq, Sq, head_dim]            k, v    [1, n_kv, T, head_dim]  (TILE layout)
    cluster_axis: when set, the merged op derives a PER-DEVICE causal chunk_start from the device's
      mesh coordinate along that axis -> chunk_start = chunk_start_idx + rank*Sq (Sq = q's S/sp rows),
      so q/index_q stay SP-sharded. None -> uniform chunk_start_idx (single-device / gathered query).
      (Replaces the old host-built per-device chunk_offset tile; mesh-coord approach, #47939.)
    -> out  [1, Hq, Sq, head_dim]
    """
    # DBG: dump the indexer's actual inputs to localize a chunked-vs-oneshot score divergence.
    if os.getenv("DBG_MSA_BLOCKIDS") == "1" and layer_idx is not None and dbg_tag is not None:
        _dbg_msa_save(index_q, device, layer_idx, f"{dbg_tag}_idxq")  # sharded on Sq -> global order
        try:  # index_k is the full gathered context, replicated across rows -> dev0 has it all
            import os as _os

            ik0 = ttnn.to_torch(ttnn.get_device_tensors(index_k)[0]).float()
            _os.makedirs("/tmp/m3_msa", exist_ok=True)
            torch.save(ik0, f"/tmp/m3_msa/L{layer_idx}_{dbg_tag}_idxk.pt")
        except Exception:
            pass

    # Block scores: scaled dot, causal -inf for future, group-sum, block-max-pool. bf16 row-major out.
    block_scores = ttnn.experimental.indexer_score_msa(
        index_q,
        index_k,
        chunk_start_idx=chunk_start_idx,
        scale=scale,
        num_groups=num_groups,
        block_size=BLOCK_SIZE,
        program_config=ttnn.IndexerScoreProgramConfig(q_chunk_size=64, k_chunk_size=1024, head_group_size=0),
        cluster_axis=cluster_axis,
    )

    # Debug (option A): M3_CPU_INDEXER -> host fp32 scores (device topk); M3_CPU_TOPK -> host fp32 scores
    # AND host top-k (bypasses topk_large_indices). Both keep the device sparse_sdpa_msa.
    cpu_topk = os.getenv("M3_CPU_TOPK") == "1"
    if os.getenv("M3_CPU_INDEXER") == "1" and not cpu_topk:
        block_scores = _cpu_fp32_indexer(
            index_q, index_k, block_scores, mode="scores",
            chunk_start_idx=chunk_start_idx, scale=scale, cluster_axis=cluster_axis, device=device,
        )

    # The op already force-locals the current block (+inf); the real MiniMax-M3 indexer forces ONLY the
    # local block (upstream minimax_m3_vl: index_local_blocks, no init/sink block), so nothing else is added.
    _dbg_msa_save(block_scores, device, layer_idx, f"{dbg_tag}_scores" if dbg_tag else None)

    # Top-k block ids (uint32 row-major) — the block selection that encodes causality.
    if cpu_topk:
        dev_bids = ttnn.experimental.topk_large_indices(block_scores, k=TOPK_BLOCKS)  # ref for shape/layout only
        block_ids = _cpu_fp32_indexer(
            index_q, index_k, dev_bids, mode="block_ids",
            chunk_start_idx=chunk_start_idx, scale=scale, cluster_axis=cluster_axis, device=device,
        )
    else:
        block_ids = ttnn.experimental.topk_large_indices(block_scores, k=TOPK_BLOCKS)
    _dbg_msa_save(block_ids, device, layer_idx, dbg_tag)

    # M3_CPU_SDPA=1 (debug, rung 2): recompute the attention in host fp32 (device block_ids kept).
    if os.getenv("M3_CPU_SDPA") == "1":
        out = ttnn.to_layout(
            _cpu_fp32_sdpa(q, k, v, block_ids, chunk_start_idx=chunk_start_idx, scale=scale, cluster_axis=cluster_axis, device=device),
            ttnn.TILE_LAYOUT,
        )
        return (out, block_ids) if return_block_ids else out

    # M3_SDPA_FP32DEST=1: fp32 DEST accumulation (HiFi4) on the device sparse_sdpa_msa. Legal for bf16 q
    # (only fp8 q *requires* it); tests whether the attention-math precision recovery we saw with host
    # fp32 is reachable on-device via a compute_kernel_config (one-liner) instead of a kernel change.
    _sdpa_kcfg = (
        ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
        )
        if os.getenv("M3_SDPA_FP32DEST") == "1"
        else None
    )
    # sparse_sdpa_msa: q + block-ids row-major, K/V tiled; expands blocks->tokens internally.
    out = ttnn.transformer.sparse_sdpa_msa(
        ttnn.to_layout(q, ttnn.ROW_MAJOR_LAYOUT),
        k,
        v,
        block_ids,
        scale=scale,
        block_size=BLOCK_SIZE,
        compute_kernel_config=_sdpa_kcfg,
        # PR#48700 (temp cherry-pick): activate the diagonal-block token-level causal mask. q is bf16
        # (fp8 q is rejected by the op). Under SP, cluster_axis=sp_axis derives per-device
        # chunk_start = chunk_start_idx + rank*Sq. chunk_start_idx=cached_len (0 for one-shot).
        # M3_MSA_DIAG_MASK=0 -> legacy block-only causality (A/B control to isolate the mask's effect).
        chunk_start_idx=(chunk_start_idx if os.getenv("M3_MSA_DIAG_MASK", "1") == "1" else None),
        cluster_axis=(cluster_axis if os.getenv("M3_MSA_DIAG_MASK", "1") == "1" else None),
    )
    # Per-layer device-op-vs-host-fp32 PCC on the SAME inputs (kernel numerical error, no cache propagation).
    if os.getenv("M3_MSA_DBG_OPDIFF") == "1" and layer_idx is not None:
        _dbg_op_vs_host(
            out, q, k, v, block_ids,
            chunk_start_idx=chunk_start_idx, scale=scale, cluster_axis=cluster_axis, device=device, layer_idx=layer_idx,
        )

    # sparse_sdpa_msa returns ROW_MAJOR; the model's concat_heads (prefill.py) needs TILE — match the
    # dense (ring_joint) output so the shared post-attention path works for MSA layers too.
    out = ttnn.to_layout(out, ttnn.TILE_LAYOUT)
    return (out, block_ids) if return_block_ids else out


def msa_sp_attention_nocache(
    q,
    k,
    v,
    index_q,
    index_k,
    *,
    mesh_config,
    ccl_manager,
    cached_len,
    s_local,
    scale,
    num_groups=1,
    return_block_ids=False,
    layer_idx=None,
):
    """Sharded-query MSA under SP: AllGather only the KEYS; q/index_q stay sharded (S/sp rows/device).

    Each device scores ONLY its own S/sp query rows against the gathered full context, with per-device
    causality from the op's native mesh-coord chunk_start (cluster_axis=sp_axis -> rank*s_local on top of
    chunk_start_idx=cached_len). Output stays SP-sharded [1, Hq, s_local, head_dim] — no replication, no
    reshard — which is what the SP residual stream needs. This is the deployed path (vs the gather-everything
    golden, which gathers the query too). index_q is the device's group's index head; q is its TP head-slice.
    """
    sp_axis = mesh_config.sp_axis
    device = ccl_manager.mesh_device
    k_full = mesh_config.allgather(k, ccl_manager, axis=sp_axis, dim=2)
    v_full = mesh_config.allgather(v, ccl_manager, axis=sp_axis, dim=2)
    index_k_full = mesh_config.allgather(index_k, ccl_manager, axis=sp_axis, dim=2)
    # Per-device causality via the merged op's native mesh-coord chunk_start (#47939): device r derives
    # chunk_start = cached_len + r*Sq (Sq = q's s_local rows) from its coordinate along cluster_axis=sp_axis.
    return msa_indexer_sparse(
        index_q,
        index_k_full,
        q,
        k_full,
        v_full,
        chunk_start_idx=cached_len,
        scale=scale,
        num_groups=num_groups,
        device=device,
        cluster_axis=sp_axis,
        layer_idx=layer_idx,
        dbg_tag="nocache",
    )


def _blockcyclic_to_natural(t, sp, n_chunks, chunk_local):
    """Reorder an AllGathered block-cyclic context [1, H, T, hd] to natural token order.

    ``update_padded_kv_cache`` stores chip r's slice as ``[chunk0_r, chunk1_r, ...]`` (chunk_local tokens
    each), so AllGather over the SP axis yields chip-major order — index ``(chip, chunk, c)``. Natural
    order is ``(chunk, chip, c)``. At chunk-aligned offsets that is exactly a transpose of the (chip, chunk)
    axes: reshape T -> [sp, n_chunks, chunk_local*hd], swap dims, reshape back. (Row-major for the middle
    transpose; the indexer/sparse re-tilize.)
    """
    H, T, hd = t.shape[1], t.shape[2], t.shape[3]
    t = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)
    t = ttnn.reshape(t, [H, sp, n_chunks, chunk_local * hd])
    t = ttnn.transpose(t, 1, 2)  # (chip, chunk) -> (chunk, chip)
    t = ttnn.reshape(t, [1, H, T, hd])
    return ttnn.to_layout(t, ttnn.TILE_LAYOUT)


def msa_sp_attention(
    q,
    k_acc,
    v_acc,
    index_q,
    index_k_acc,
    *,
    mesh_config,
    ccl_manager,
    cached_len,
    s_local,
    n_chunks,
    chunk_local,
    scale,
    num_groups=1,
    layer_idx=None,
):
    """Cross-chunk MSA: the CURRENT chunk's queries attend the ACCUMULATED context read from the
    block-cyclic SP cache (the multi-chunk read path; ``msa_sp_attention_nocache`` is its single-chunk,
    contiguous-context sibling).

    Args (per device, on the (sp, tp) mesh):
        q, index_q          CURRENT chunk's CONTIGUOUS SP shards: q [1, Hq_local, s_local, hd],
                            index_q [1, num_groups, s_local, hd]  (chip r owns chunk positions
                            [cached_len + r*s_local : ...]).
        k_acc, v_acc        ACCUMULATED context's BLOCK-CYCLIC SP shards (as the cache stores them):
                            [1, n_kv_local, n_chunks*chunk_local, hd].
        index_k_acc         accumulated index_k block-cyclic shard [1, 1, n_chunks*chunk_local, hd].
        cached_len          valid prefix length BEFORE the current chunk (= (n_chunks-1)*chunk_local*sp).
        n_chunks            total chunks now in the cache (incl. current); chunk_local = tokens/chip/chunk.

    AllGather K/V/index_k across SP -> full block-cyclic context -> reorder to NATURAL token order (so the
    indexer's block-pool + causal offset see true positions) -> indexer (per-device chunk_offset) ->
    sparse_sdpa. Returns the current chunk's SP-sharded attention out [1, Hq_local, s_local, hd].
    """
    sp_axis = mesh_config.sp_axis
    device = ccl_manager.mesh_device
    sp = device.shape[sp_axis]

    # KNOWN BUG (chunked-only): this device gather mis-orders the context (index_k PCC ~0.517 vs golden)
    # because to_memory_config does not delinearize the NdShard ("slab") cache to logical order, AND a
    # host-reorder workaround (from_torch) makes the indexer/sparse TensorAccessor read an invalid DRAM
    # bank (TT_FATAL "core 8-0"). PROPER FIX is a slab-aware kernel cache-read (see KERNEL TODO in msa
    # docstring / memory): make sparse_sdpa_msa + indexer_score_msa read the NdShard cache directly via
    # kv_cache_batch_idx/actual_isl like ring_joint. Until then the chunked MSA path is not correct.
    def gather_natural(t):
        t = ttnn.to_memory_config(t, ttnn.DRAM_MEMORY_CONFIG)
        full_bc = mesh_config.allgather(t, ccl_manager, axis=sp_axis, dim=2)
        if full_bc.dtype != ttnn.bfloat16:
            full_bc = ttnn.typecast(full_bc, ttnn.bfloat16)
        return _blockcyclic_to_natural(full_bc, sp, n_chunks, chunk_local)

    k_full = gather_natural(k_acc)
    v_full = gather_natural(v_acc)
    index_k_full = gather_natural(index_k_acc)
    # DEBUG (M3_DBG_REORDER=1): dump the reordered full-context index_k so it can be compared to the golden
    # natural-order index_k (with the Meta-RoPE `src` permutation) -> proves whether gather_natural is broken.
    if os.getenv("M3_DBG_REORDER") == "1" and layer_idx == int(os.getenv("M3_DBG_REORDER_LAYER", "4")) and n_chunks >= 2:
        import os as _os
        from loguru import logger as _lg

        _os.makedirs("/tmp/reorder_dbg", exist_ok=True)
        _ik = ttnn.to_torch(ttnn.get_device_tensors(index_k_full)[0])  # device0's full-context reorder [1,1,T,hd]
        # Also capture the pre-reorder AllGather (block-cyclic) so the exact block-cyclic->natural permutation
        # can be recovered offline (device-vs-device, no golden/RoPE-swizzle ambiguity).
        _bc = mesh_config.allgather(
            ttnn.to_memory_config(index_k_acc, ttnn.DRAM_MEMORY_CONFIG), ccl_manager, axis=sp_axis, dim=2)
        if _bc.dtype != ttnn.bfloat16:
            _bc = ttnn.typecast(_bc, ttnn.bfloat16)
        _bc_t = ttnn.to_torch(ttnn.get_device_tensors(_bc)[0])
        torch.save(
            {"gathernat": _ik, "full_bc": _bc_t, "n_chunks": n_chunks, "chunk_local": chunk_local, "sp": sp,
             "cached_len": cached_len, "layer": layer_idx},
            f"/tmp/reorder_dbg/L{layer_idx}_gathernat.pt",
        )
        _lg.info(f"[reorder-dbg] saved L{layer_idx} gather_natural index_k {tuple(_ik.shape)} -> /tmp/reorder_dbg")
    # Per-device causality via the merged op's native mesh-coord chunk_start (#47939): device r derives
    # chunk_start = cached_len + r*Sq (Sq = q's s_local rows) from its coordinate along cluster_axis=sp_axis.
    return msa_indexer_sparse(
        index_q,
        index_k_full,
        q,
        k_full,
        v_full,
        chunk_start_idx=cached_len,
        scale=scale,
        num_groups=num_groups,
        device=device,
        cluster_axis=sp_axis,
        layer_idx=layer_idx,
        dbg_tag="cacheread",
    )
