# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Torch reference for the Q-block-granular Sol-Attn (Q1-Q3), and the golden attention output for either granularity.

Background. Routing scores at (Q block 64 x K block 64) and then OR-reduces to the Q chunk before the kernel sees it
(`sol_prepass.torch_reference`: `m2 = exact.transpose @ qc_pool_t`). Every Q row of the chunk is then computed against
the union of what its TQ = q_chunk/64 Q blocks wanted, so Q-axis sparsity is discarded. Q1 removes that reduction.

This module is the device-free half of that work:

  qblock_mask()      the Q1 mask -- `exact` with the band/sink/full-block rules applied, no OR.
  chunk_mask()       the mask as built today, for side-by-side comparison.
  sol_attention()    golden output for EITHER granularity, including the pooled correction, so the same function
                     scores today's kernel and tomorrow's.
  mask_density()     s (per Q-block-K-block pair) vs s' (post-OR) -- the ratio Q1 is trying to recover.

Everything is fp32 torch and runs on a laptop. Shapes follow sol_prepass: q/k/v are [S, H, rows, D], one shard per
leading index; masks are [S, H, NQ, nbg] where NQ is geo.nb_pad for Q-block form or geo.nqc_pad for Q-chunk form.
"""

from __future__ import annotations

import torch

from .sol_prepass import LOG2E, VAR_EPS, SolGeometry, build_constants


def _moments(q_bar, kc, n_valid, *, tau, log2_scale, thresh_type="diag"):
    """Per-(head, Q block) threshold tau_i = mu_i + tau * sigma_i, in the kernel's base-2 logit units."""
    mean = kc.sum(dim=1, keepdim=True) / n_valid  # [H,1,D]
    q_raw_mean = (q_bar * mean[None]).sum(-1, keepdim=True)  # [S,H,nbp,1]
    if thresh_type == "diag":
        var = ((kc * kc).sum(dim=1, keepdim=True) / n_valid - mean * mean).clamp(min=0)
        q_var = (q_bar * q_bar * var[None]).sum(-1, keepdim=True)
    else:
        second = kc.transpose(1, 2) @ kc / n_valid  # [H,D,D] full covariance
        q_var = (((q_bar @ second[None]) * q_bar).sum(-1, keepdim=True) - q_raw_mean**2).clamp(min=0)
    return q_raw_mean * log2_scale + tau * torch.sqrt(q_var * log2_scale * log2_scale + VAR_EPS)


def _pool(x, pool_t):
    """[S,H,rows,D] -> per-shard block means. Returns q-form [S,H,nbp,D] and global form [H,nbg,D]."""
    S = x.shape[0]
    per_shard = [(x[s].float().transpose(1, 2) @ pool_t[s, 0]).transpose(1, 2) for s in range(S)]  # [H,nbp,D] each
    return torch.stack(per_shard), torch.cat(per_shard, dim=1)


def score_blocks(q, k, v, geo: SolGeometry, consts: dict, *, tau: float, scale: float, thresh_type: str = "diag"):
    """Shared front half of both granularities: pooled K/V, per-Q-block threshold, and the raw 64x64 selection.

    Returns `exact` [S,H,nb_pad,nbg] AFTER the band / sink / full-block rules, which is exactly the Q1 mask.
    """
    log2_scale = scale * LOG2E
    q_bar, _ = _pool(q, consts["pool_t"])  # [S,H,nbp,D]
    _, kc = _pool(k, consts["pool_t"])  # [H,nbg,D]
    _, vc = _pool(v, consts["pool_t"])  # [H,nbg,D]

    thr = _moments(q_bar, kc, consts["n_valid_blocks"], tau=tau, log2_scale=log2_scale, thresh_type=thresh_type)
    scores = (q_bar @ kc[None].transpose(2, 3)) * log2_scale  # [S,H,nbp,nbg]
    exact = (scores > thr).float() + consts["band"] + consts["sink_row"]
    exact = (exact > 0.5).float() * consts["full_row"]  # partial tail blocks are never exact; they stay pooled
    return {
        "exact": exact,
        "pooled_k": kc,
        "pooled_v": vc,
        "q_bar": q_bar,
        "threshold": thr,
        "scores_raw": scores,  # pre-forcing proxy scores, so conformance can re-derive the bare threshold rule
    }


def spec_constants(geo: SolGeometry, *, scale: float):
    """Constants with every non-paper forcing rule disabled, for the conformant path.

    Algorithm 1 defines S_i purely by the threshold. Our production constants additionally force a diagonal band,
    sink columns, dense prefix rows and the first chunk of every shard -- all conservative (they only ADD blocks),
    none of them in the paper. Conformance needs them off.

    `force_first_chunk=False` is the one with a kernel consequence: compute_streaming documents that "an iteration
    with zero computed chunks is not a supported state", so the device path needs to tolerate an empty ring
    iteration before this can be used there. It is free here in torch.

    Note band=-1, not 0: the rule is |q_block - k_block| <= band, so band=0 still forces the diagonal block. -1
    makes the predicate unsatisfiable, which is the only way to disable it.
    """
    return build_constants(
        geo, band=-1, sink_tokens=0, force_first_chunk=False, scale=scale, dense_prefix_queries=False
    )


def spec_mask(q, k, v, geo: SolGeometry, consts: dict, *, tau: float, scale: float):
    """Sol-Attn exactly as published: S_i = {j : s_ij > tau_i}, tau_i = mu_i + beta*sigma_i, per QUERY BLOCK.

    Requires consts from spec_constants() (no forcing) and uses the full d x d second moment, which is what the
    paper's O(d^2) auxiliary storage buys -- the diagonal form our device kernel implements is an approximation of
    sigma_i, so it selects a different set even at the same beta.
    """
    out = score_blocks(q, k, v, geo, consts, tau=tau, scale=scale, thresh_type="full")
    out["block_mask"] = out["exact"]
    return out


def qblock_mask(q, k, v, geo: SolGeometry, consts: dict, **kw):
    """Q1 alone: no OR, but keeping today's diagonal threshold and forcing rules. The intermediate step, not the spec."""
    out = score_blocks(q, k, v, geo, consts, **kw)
    out["block_mask"] = out["exact"]
    return out


def chunk_mask(q, k, v, geo: SolGeometry, consts: dict, **kw):
    """Today's mask: `exact` OR-reduced onto Q chunks. [S,H,nqc_pad,nbg]. Mirrors sol_prepass.torch_reference."""
    out = score_blocks(q, k, v, geo, consts, **kw)
    m2 = (out["exact"].transpose(2, 3) @ consts["qc_pool_t"]).transpose(2, 3)  # any Q block of the chunk
    out["block_mask"] = ((m2 + consts["force_row"]) > 0.5).float() * consts["full_row"]
    return out


def mask_density(exact, chunk, consts):
    """s vs s'. The gap between them is what Q1 recovers; under independence s' = 1-(1-s)^TQ."""
    valid = consts["full_row"].reshape(-1) > 0.5  # only full blocks are exact-eligible
    return {
        "s_qblock": exact[..., valid].mean().item(),
        "s_qchunk": chunk[..., valid].mean().item(),
        "ratio": chunk[..., valid].mean().item() / max(exact[..., valid].mean().item(), 1e-9),
    }


def build_pages(mask, geo: SolGeometry, *, page_blocks: int):
    """Q2's packing, as a reference. Returns pages[(s_q, h, q_unit)] = list of (ring_iter, [global block ids]).

    Two constraints shape this, and together they make the plan much cheaper than it first looks:

    1. A page may NOT span shards. The ring delivers shard r's K/V only during ring iteration r, so blocks from
       different shards are never co-resident. Packing is therefore within a ring iteration, across K-chunk
       boundaries -- which is where the win is anyway (today a page is confined to ONE chunk).

    2. Within a shard, blocks arrive in chunk order down the chain. So the kernel does not need a precomputed page
       layout at all: it accumulates kept blocks into a page buffer as they stream past, fires the matmul when the
       buffer reaches page_blocks, and flushes a partial page at the end of the shard. The "plan" is then just the
       Q-block mask that Q1 already produces -- no new addressing format, no (chunk, block) indirection.

    This function models exactly that streaming accumulation, so the kernel can be diffed against it.
    """
    S, H, NQ, _ = mask.shape
    nbp = geo.nb_pad
    out = {}
    for sq in range(S):
        for h in range(H):
            for u in range(NQ):
                pages, buf = [], []
                for r in range(S):  # ring iteration = source shard, in delivery order
                    for b in range(geo.nb):  # blocks of shard r, in chunk order
                        g = r * nbp + b
                        if mask[sq, h, u, g] > 0.5:
                            buf.append(g)
                            if len(buf) == page_blocks:
                                pages.append((r, buf))
                                buf = []
                    if buf:  # flush at the shard boundary: the next shard's blocks cannot share this page
                        pages.append((r, buf))
                        buf = []
                out[(sq, h, u)] = pages
    return out


def l1_budget(*, q_rows: int, k_keys: int, head_dim: int, v_shares_k: bool = False, apply_mask: bool = True):
    """Circular-buffer tile counts, mirroring ring_joint_sdpa_program_factory.cpp:1418-1431.

    Used to answer: how large can a Q2 page be once Q1 shrinks the compute unit? Every Sq-dependent CB scales with
    the Q rows, and the K/V buffers -- the ones a larger page grows -- do not. Dropping from a 256-row Q chunk to a
    64-row Q block therefore frees L1 that can be spent on a bigger page, which is what recovers the visit win.

    Tile COUNTS only; the formats differ per CB (qk is fp32 when fp32_dest_acc_en, mask is bf16), so convert with
    the real per-CB widths before trusting a byte figure.
    """
    sq, sk, dht = q_rows // 32, k_keys // 32, head_dim // 32
    cbs = {
        "q": sq * dht,
        "k": sk * dht * (3 if v_shares_k else 2),
        "v": sk * dht * 2,
        "mask": sq * sk if apply_mask else 0,
        "qk": sq * sk,
        "out_im": sq * dht,
        "out0": sq * dht,
        "statistics": sq,
        "scale": 1,
    }
    sq_dependent = sum(v for k, v in cbs.items() if k not in ("k", "v", "scale"))
    return {"cbs": cbs, "total": sum(cbs.values()), "sq_dependent": sq_dependent, "kv": cbs["k"] + cbs["v"]}


def max_page_blocks(*, q_rows: int, head_dim: int, budget_tiles: int, block: int = 64, **kw):
    """Largest page (in 64-key blocks) whose CBs fit a tile budget, for a given Q unit size.

    Call it twice -- once at today's Q chunk, once at a 64-row Q block -- to see what Q1 buys for Q2.
    """
    best = 0
    for nb in range(1, 65):
        if l1_budget(q_rows=q_rows, k_keys=nb * block, head_dim=head_dim, **kw)["total"] <= budget_tiles:
            best = nb
    return best


def page_stats(pages, *, page_blocks: int):
    """Visit count and mean page occupancy -- the two numbers Q2 is trying to move."""
    allp = [p for v in pages.values() for p in v]
    if not allp:
        return {"visits": 0, "fill": 0.0, "blocks": 0}
    blocks = sum(len(b) for _, b in allp)
    return {"visits": len(allp), "fill": blocks / (len(allp) * page_blocks), "blocks": blocks}


def _blockify(x, geo: SolGeometry):
    """[S,H,rows,D] -> [H, nbg, block, D] padded, plus per-global-block valid lengths [nbg]."""
    S, H, _, D = x.shape
    lengths = geo.block_lengths()  # [S, nb_pad]
    nbg = geo.n_blocks_pad
    out = x.new_zeros(H, nbg, geo.block, D)
    for s in range(S):
        for b in range(geo.nb):
            n = int(lengths[s, b])
            if n:
                out[:, s * geo.nb_pad + b, :n] = x[s, :, b * geo.block : b * geo.block + n].float()
    return out, lengths.reshape(-1)


def sol_attention(q, k, v, mask, pooled_k, pooled_v, geo: SolGeometry, *, scale: float):
    """Golden Sol-Attn output for a mask at EITHER granularity. Returns [S,H,rows,D].

    Kept blocks are attended exactly. Every other valid block contributes one pooled column with key K-bar and value
    V-bar, carrying a log(len) bias so its softmax weight equals the block's length -- the zeroth-order term, i.e.
    sum_j exp(q.k_j) v_j  ~=  len * exp(q.K-bar) * V-bar. This is the kernel's final out-of-loop inner_step.
    """
    S, H, rows, D = q.shape
    NQ = mask.shape[2]
    per_unit = geo.block if NQ == geo.nb_pad else geo.q_chunk  # 64 for Q-block form, q_chunk for Q-chunk form
    kb, lens = _blockify(k, geo)  # [H,nbg,block,D], [nbg]
    vb, _ = _blockify(v, geo)
    valid = lens > 0
    log_len = torch.where(valid, lens.float().clamp(min=1).log(), torch.zeros_like(lens.float()))  # [nbg]

    out = torch.zeros_like(q, dtype=torch.float32)
    for s in range(S):
        for u in range(-(-rows // per_unit)):  # the last unit may be partial (rows not a multiple of per_unit)
            rsl = slice(u * per_unit, (u + 1) * per_unit)
            qu = q[s, :, rsl].float()  # [H,m,D]
            keep = mask[s, :, u] > 0.5  # [H,nbg] -- per head, masks are not shared across heads

            # exact columns: [H, m, nbg, block], suppressed where the block is dropped, padded, or out of range
            le = torch.einsum("hmd,hgbd->hmgb", qu, kb) * scale
            ok = (keep[:, None, :, None] & valid[None, None, :, None]) & (
                torch.arange(geo.block, device=q.device)[None, None, None, :] < lens[None, None, :, None]
            )
            le = le.masked_fill(~ok, float("-inf"))

            # pooled columns: one per valid, dropped block
            lp = torch.einsum("hmd,hgd->hmg", qu, pooled_k) * scale + log_len[None, None, :]
            lp = lp.masked_fill(keep[:, None, :] | ~valid[None, None, :], float("-inf"))

            flat = torch.cat([le.reshape(H, per_unit, -1), lp], dim=-1)
            w = torch.softmax(flat, dim=-1)
            we, wp = w[..., : le[0, 0].numel()].reshape(le.shape), w[..., le[0, 0].numel() :]
            out[s, :, rsl] = torch.einsum("hmgb,hgbd->hmd", we, vb) + torch.einsum("hmg,hgd->hmd", wp, pooled_v)
    return out
