# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exact HOST reference of the fused SAMPLED accept + drafter-input build (stage S2 of work/notes/45, 2.2-2.3, 4.2).

Pure torch, float32 only (no float64): it mirrors the op sequence the device accept trace will run, so the device ops
have one target. No ttnn import. Its decisions equal the plugin's accept_sampled_support
(vllm_tt_plugin.spec_accept: _support_table -> _support_cut -> _walk_table) on the same candidates and the same uniforms
except where the float32 arithmetic flips a comparison (|p_d - u|, an inverse-CDF boundary or a top-p tail within about
1e-6; the plugin uses float64 uniforms and cumsums). tests/test_spec_fused_sampled_host.py proves it.

Layout: engine rows r = u*T + j, T = K + 1, R = B*T. The verify's topk trace gives, per row, the union of the TP shards'
per-shard top-M: union_vals bf16 [R, TP*M] and union_local_ids [R, TP*M] (block d at columns d*M .. d*M+M-1, ids local to
shard d, each block ordered (value desc, local id asc)).

Uniform table [B, K+2] fp32 per user (see predraw_uniforms): column 0 bonus, columns 1..K accept draws for draft columns
0..K-1, column K+1 correction.
"""

from typing import NamedTuple

import torch

from models.demos.blackhole.qwen36.tt.spec_fused import (
    AUX_ACTIVE,
    AUX_CONSTS,
    AUX_DECLINED,
    AUX_DELTA,
    AUX_KEEP,
    AUX_NUM_VALID,
    AUX_P0,
    CONST_ROPE_ROWS,
    CONST_SCRATCH,
    FUSED_AUX_ROWS,
    FusedGreedyDevice,
    aux_row_deltas,
    ridx_deltas,
)
from models.demos.blackhole.qwen36.tt.tp_common import pack_u32_rows, rope_row_index

UNION_PAD_WIDTH = 1024  # ttnn.topk multi-core needs width >= 1024 at one tile row, k <= 64
PAD_VALUE = -1e30
CERT_EPS = (
    1e-5  # top-p certificate margin: a row fails when q - p < CERT_EPS (fp32 error of Zc, Z and the division ~1e-6)
)
OVF_TIE = 1  # overflow word bit: a top-k tie group reached the last candidate
OVF_CERT = 2  # overflow word bit: the full-vocabulary top-p certificate failed (reference and the device variant body(cert=True))
STATS_LT_ROW = 32  # first row of the L_T half of the model's stats buffer fp32 [1,1,64,1] (rows 0..31 = L_raw)
_ONE_MINUS = float(torch.nextafter(torch.tensor(1.0, dtype=torch.float32), torch.tensor(0.0, dtype=torch.float32)))


class FusedSampledOut(NamedTuple):
    committed: torch.Tensor  # [B, T] int32, -1 past count
    counts: torch.Tensor  # [B] int32 in [1, T]
    overflow: (
        torch.Tensor
    )  # [B] int32 word: OVF_TIE (1) = an examined row hit the top-k tie rule, OVF_CERT (2) = the top-p certificate failed -> redo the step on the host
    pack: torch.Tensor  # [2K+5, W] int32 (uint32 words) = _dr_pack_host's pack
    rs_pt: torch.Tensor  # [R, nb] int32
    mtp_pt: torch.Tensor  # [B, nb] int32
    declined: torch.Tensor  # [B] bool
    num_valid_out: torch.Tensor  # [B] int32
    drafter_runs: bool
    # audit fields (the device audit mode writes prob and p_d)
    prob: torch.Tensor  # [R, M] fp32 renormalised probabilities of the candidates
    cand_ids: torch.Tensor  # [R, M] int64 global ids in (value desc, id asc) order
    p_d: torch.Tensor  # [B, K] fp32 prob mass of each draft token (0 when not a candidate / dropped)
    cum: torch.Tensor  # [R, M] fp32 reverse cumsum of the top-p softmax (top-p boundary audits)
    cert_q: torch.Tensor = (
        None  # [R] fp32 candidate mass fraction q = Zc / Z of certificate rows (1.0 elsewhere); None without full_norm
    )


def union_topk_stable(union_vals, union_local_ids, per_shard, M):
    """Step 1: global top-M of the union under (value desc, global id asc), as the device does it: gid = local + (col // M)
    * per_shard, the union padded to UNION_PAD_WIDTH with PAD_VALUE, then a STABLE descending top-M. Union position order is
    shard-block order then per-shard rank, i.e. already (shard asc, value desc, local id asc), so stability yields global
    id ascending among equal values. Returns (values fp32 [R, M], gids int64 [R, M])."""
    vals = torch.as_tensor(union_vals).to(torch.float32)
    R, W = vals.shape
    if W % M or W > UNION_PAD_WIDTH:
        raise ValueError(f"union width {W} must be a multiple of M={M} and <= {UNION_PAD_WIDTH}")
    local = torch.as_tensor(union_local_ids).to(torch.int64)
    if local.shape != vals.shape:
        raise ValueError(f"union ids {tuple(local.shape)} != vals {tuple(vals.shape)}")
    gid = local + (torch.arange(W, dtype=torch.int64) // M) * per_shard
    pad = UNION_PAD_WIDTH - W
    vals = torch.cat([vals, torch.full((R, pad), PAD_VALUE, dtype=torch.float32)], dim=1)
    gid = torch.cat([gid, torch.zeros((R, pad), dtype=torch.int64)], dim=1)
    order = torch.sort(vals, dim=1, descending=True, stable=True).indices[:, :M]
    return vals.gather(1, order), gid.gather(1, order)


def _inverse_cdf_f32(w, u):
    """_inverse_cdf in float32: w [N, M] >= 0, u [N] -> column index [N] (scale free, clamped to the last positive weight)."""
    cdf = w.cumsum(dim=1)
    thr = (u * cdf[:, -1]).unsqueeze(1)
    pick = (cdf <= thr).sum(dim=1)
    last = w.shape[1] - 1 - (w > 0).to(torch.int8).flip(1).argmax(dim=1)
    return torch.minimum(pick, last), cdf


def fused_drafter_reference(committed, counts, aux, pt, *, B, K):
    """Steps 2-4 of spec_fused.fused_greedy_reference (propose decline, _dr_pack_host, _dr_rs_table) from a given
    ``committed`` [B, T] and ``counts`` [B], bit for bit the same code path. Returns
    (pack, rs_pt, mtp_pt, declined, num_valid_out, drafter_runs)."""
    T = K + 1
    aux = torch.as_tensor(aux).to(torch.int64) & 0xFFFFFFFF
    dec0, keep, p0 = aux[AUX_DECLINED, :B], aux[AUX_KEEP, :B], aux[AUX_P0, :B]
    scratch, rope_rows = (int(aux[AUX_CONSTS, i]) for i in (CONST_SCRATCH, CONST_ROPE_ROWS))
    delta = ridx_deltas(aux_row_deltas(aux, B), T, K)
    committed = torch.as_tensor(committed).to(torch.int64)
    m = torch.as_tensor(counts).to(torch.int64) - 1
    anchor_pos = p0 + m
    declined = dec0.bool()
    m_kv = m + (declined & keep.bool()).to(torch.int64)
    pending = torch.where(declined, torch.zeros_like(m), committed.gather(1, m[:, None]).reshape(-1))
    p = torch.where(declined, torch.zeros_like(p0), anchor_pos)
    real = torch.arange(T)[None, :] < m_kv[:, None]
    zero = torch.zeros((), dtype=torch.int64)
    rtok = torch.where(real, committed, zero).reshape(-1)
    rpos = torch.where(real, (p0[:, None] + torch.arange(T)[None, :]), zero).reshape(-1)
    steps = p[None, :] + torch.arange(K)[:, None]
    ridx = rope_row_index(torch.cat([rpos, steps.reshape(-1)]), delta, rope_rows)
    R = B * T
    anchor_rows = torch.arange(B) * T + m
    pack = pack_u32_rows([pending, *steps, rtok, rpos, anchor_rows, ridx[:R], *ridx[R:].reshape(K, B).to(torch.int64)])
    pt = torch.as_tensor(pt).to(torch.int32)
    pt_rep = pt.repeat_interleave(T, dim=0)
    rs_pt = torch.where(real.reshape(-1, 1), pt_rep, torch.full_like(pt_rep, scratch))
    mtp_pt = torch.where(declined[:, None], torch.full_like(pt, scratch), pt)
    num_valid_out = torch.where(declined, 0, K).to(torch.int32)
    return pack, rs_pt.to(torch.int32), mtp_pt.to(torch.int32), declined, num_valid_out, not bool(declined.all())


def certificate_cut(zc, full_norm, top_p, eps=CERT_EPS):
    """Top-p certificate of a full-vocabulary row (work/research_1026/67 section 5.1). ``zc`` = sum over the first M candidates
    of exp(v_i - v_0) (v = logit / T), ``full_norm`` = the same sum over the whole vocabulary (L_T of the stats trace), ``top_p``.
    q = zc / max(full_norm, zc) in (0, 1]. When q >= p + eps every position past the candidates is dropped by the nucleus rule
    and, inside the candidates, position i is dropped iff S_i / zc <= 1 - p / q (S_i = candidate mass from i on). Returns
    (q, omp_eff = 1 - p / q, fail = q - p < eps); a failing row can only be a false "fail", never a false "pass"."""
    zc = torch.as_tensor(zc).to(torch.float32)
    q = zc / torch.maximum(torch.as_tensor(full_norm).to(torch.float32), zc)
    p = torch.as_tensor(top_p).to(torch.float32)
    return q, 1.0 - p / q, (q - p) < eps


def fused_sampled_reference(
    union_vals,
    union_local_ids,
    tok_buf,
    ids,
    aux,
    pt,
    *,
    B,
    K,
    per_shard,
    M,
    greedy,
    temperature,
    top_k,
    top_p,
    uniforms,
    num_valid=None,
    full_norm=None,
):
    """Fused sampled (and greedy-mixed) accept + drafter inputs, float32, the device algorithm.

    union_vals bf16 [R, TP*M], union_local_ids int [R, TP*M]: the topk trace readback. ``tok_buf`` [R] (row u*T+j = token at
    column j; the draft of row u*T+j is tok_buf[u*T+j+1]). ``ids`` [R] argmax ids (greedy rows commit them like S1).
    ``aux``: spec_fused.fused_accept_aux_host output (num_valid, active, decline words, p0, consts). ``pt`` [B, nb].
    greedy [B] bool, temperature [B] fp32, top_k [B] int (1..M-1 for a sampled row), top_p [B] fp32 (1.0 = off),
    uniforms [B, K+2] fp32. ``num_valid`` [B] defaults to aux row 0. An inactive row (aux active = 0) takes the greedy path.
    ``full_norm`` [R] fp32 (optional): L_T = sum over the WHOLE vocabulary of exp((x - max) / T) per row (the stats trace). With it a
    sampled row with top_k = 0 (full vocabulary) and top_p < 1 is a certificate row: keep all M candidates, nucleus cut
    ``1 - p / q`` (certificate_cut); overflow bit OVF_CERT when the certificate fails, never OVF_TIE for it. Without it top_k = 0
    stays out of range as before.
    A user whose ``overflow`` is not 0 has meaningless committed/counts/drafter fields: redo the step on the host."""
    T, R = K + 1, B * (K + 1)
    aux64 = torch.as_tensor(aux).to(torch.int64) & 0xFFFFFFFF
    nv = aux64[AUX_NUM_VALID, :B] if num_valid is None else torch.as_tensor(num_valid).reshape(B).to(torch.int64)
    active = aux64[AUX_ACTIVE, :B].bool()
    greedy = torch.as_tensor(greedy).reshape(B).bool() | ~active
    temp = torch.as_tensor(temperature).reshape(B).to(torch.float32)
    temp = torch.where(greedy | (temp <= 0), torch.ones_like(temp), temp)
    kk = torch.as_tensor(top_k).reshape(B).to(torch.int64)
    pp = torch.as_tensor(top_p).reshape(B).to(torch.float32)
    uni = torch.as_tensor(uniforms).reshape(B, K + 2).to(torch.float32)
    cert = ~greedy & (kk <= 0) if full_norm is not None else torch.zeros_like(greedy)
    bad = ~greedy & ~cert & ((kk < 1) | (kk > M - 1))
    if bool(bad.any()):
        raise ValueError(f"rows {bad.nonzero().reshape(-1).tolist()}: top_k outside [1, {M - 1}] cannot run fused")
    kk = torch.where(greedy, torch.ones_like(kk), kk)
    kk = torch.where(cert, torch.full_like(kk, M), kk)  # certificate rows keep every candidate (kth = v[M - 1])
    toks = torch.as_tensor(tok_buf).reshape(-1)[:R].to(torch.int64).reshape(B, T)
    drafts = toks[:, 1:]  # [B, K]
    am = torch.as_tensor(ids).reshape(-1)[:R].to(torch.int64).reshape(B, T)

    # 1. global top-M of the union
    vals, gids = union_topk_stable(union_vals, union_local_ids, per_shard, M)  # [R, M]
    # 2. temperature, top-k cut, overflow flag
    v = vals / temp.repeat_interleave(T)[:, None]
    kth = v.gather(1, (kk - 1).repeat_interleave(T)[:, None])
    keep = v >= kth
    neg = float("-inf")
    overflow_row = (v[:, M - 1 :] >= kth) & (kth > neg)  # [R, 1]
    overflow_row = overflow_row.reshape(B, T) & ~cert[:, None]
    v = torch.where(keep, v, torch.full_like(v, neg))
    # 3. top-p after the top-k cut (softmax over kept, reverse cumsum, drop tail mass <= 1 - p, never column 0)
    e = torch.exp(v - v[:, :1])
    p0 = e / e.sum(dim=1, keepdim=True)
    cum = p0.flip(1).cumsum(dim=1).flip(1)
    omp = (1.0 - pp).repeat_interleave(T)
    cert_q = None
    cert_fail = torch.zeros(B, T, dtype=torch.bool)
    if full_norm is not None:
        fnorm = torch.as_tensor(full_norm).reshape(-1)[:R].to(torch.float32)
        q, omp_c, fail_c = certificate_cut(e.sum(dim=1), fnorm, pp.repeat_interleave(T))
        cr = cert.repeat_interleave(T)
        omp = torch.where(cr, omp_c, omp)
        cert_q = torch.where(cr, q, torch.ones_like(q))
        cert_fail = (fail_c & cr).reshape(B, T)
    drop = cum <= omp[:, None]
    drop[:, 0] = False
    e = torch.where(drop, torch.zeros_like(e), e)
    prob = e / e.sum(dim=1, keepdim=True)  # [R, M]
    # 4. draft mass and the accept walk
    pr = prob.reshape(B, T, M)[:, :K]
    hit = gids.reshape(B, T, M)[:, :K] == drafts[:, :, None]
    p_d = (pr * hit).sum(dim=2)  # [B, K]
    jj = torch.arange(K)[None, :]
    valid = jj < nv[:, None]
    accept = (p_d > 0) & (p_d >= uni[:, 1 : K + 1]) & valid
    m = accept.to(torch.int64).cumprod(dim=1).sum(dim=1)  # [B], first rejection or nv
    # 5. correction (m < nv, draft removed) or bonus (m == nv) at row u*T + m
    ur = torch.arange(B)
    prob_m = prob.reshape(B, T, M)[ur, m]  # [B, M]
    gid_m = gids.reshape(B, T, M)[ur, m]
    d_m = drafts[ur, m.clamp(max=K - 1)]
    is_corr = m < nv
    w = torch.where(is_corr[:, None] & (gid_m == d_m[:, None]), torch.zeros_like(prob_m), prob_m)
    u_pick = torch.where(is_corr, uni[:, K + 1], uni[:, 0])
    pick, _ = _inverse_cdf_f32(w, u_pick)
    tok_pick = gid_m.gather(1, pick[:, None]).reshape(-1)
    # 6. committed
    cols = torch.arange(T)[None, :]
    drafts_pad = torch.cat([drafts, torch.full((B, 1), -1, dtype=torch.int64)], dim=1)
    s_committed = torch.where(cols < m[:, None], drafts_pad, torch.full_like(drafts_pad, -1))
    s_committed = torch.where(cols == m[:, None], tok_pick[:, None], s_committed)
    s_counts = m + 1
    # greedy rows: S1's walk on the argmax ids
    rejected = valid & (am[:, :K] != drafts)
    first_rej = rejected.to(torch.int8).argmax(dim=1)
    g_counts = torch.where(rejected.any(dim=1), first_rej + 1, nv + 1)
    g_committed = torch.where(cols < g_counts[:, None], am, torch.full_like(am, -1))
    counts = torch.where(greedy, g_counts, s_counts)
    committed = torch.where(greedy[:, None], g_committed, s_committed)
    # 7. overflow word: rows the walk examined are columns 0..m of the SAMPLED walk
    examined = cols <= m[:, None]
    overflow = ((overflow_row & examined).any(dim=1) & ~greedy).to(torch.int32) * OVF_TIE
    overflow = overflow + ((cert_fail & examined).any(dim=1) & ~greedy).to(torch.int32) * OVF_CERT
    # 8. drafter inputs
    pack, rs_pt, mtp_pt, declined, nv_out, runs = fused_drafter_reference(committed, counts, aux, pt, B=B, K=K)
    return FusedSampledOut(
        committed.to(torch.int32),
        counts.to(torch.int32),
        overflow,
        pack,
        rs_pt,
        mtp_pt,
        declined,
        nv_out,
        runs,
        prob,
        gids,
        p_d,
        cum,
        cert_q,
    )


def predraw_uniforms(generators, rows, K, num_valid, seeded_mask=None, *, draw_global=True):
    """The [B=rows, K+2] fp32 uniform table (column 0 bonus, 1..K accepts, K+1 correction) consumed in the SAME order as
    spec_accept._walk_table / _uniform, so a seeded request stays reproducible and the global stream stays aligned:
      1. global torch.rand((rows, 2), float64): col 0 bonus, col 1 correction (every row, greedy and padding too);
      2. seeded rows with num_valid > 0: bonus from the row's own generator (overwrites col 0), in row order;
      3. global torch.rand((rows, K), float64): accepts (every row);
      4. seeded rows with num_valid > 0: num_valid accept draws from the own generator, then the correction draw
         (overwrites col 1), in row order.
    So a seeded row draws bonus, then num_valid accepts, then the correction. The host walk draws the seeded correction for
    EVERY seeded row with num_valid > 0, also when no rejection happens (the draw is unused then), so predraw consumes the
    same count: no order difference there. Documented differences:
      * float64 draws are rounded to float32 (and clamped below 1.0): a decision flips only if |p - u| < ~6e-8.
      * a seeded row with num_valid == 0 samples its bonus by a full-vocabulary exponential race (spec_accept.py:1314-1317)
        that needs the whole vocab noise and the candidates, which the device cannot do: this raises ValueError. Route such a
        step to the host walk (or unseed the row).
      * ``draw_global=False`` skips the global draws: the host walk draws nothing when every row is greedy.
    ``generators``: dict row -> torch.Generator; ``seeded_mask`` [rows] bool (default: the keys of generators)."""
    nv = torch.as_tensor(num_valid).reshape(-1).to(torch.int64).tolist()
    if len(nv) != rows:
        raise ValueError(f"num_valid has {len(nv)} entries for {rows} rows")
    seeded = set(generators) if seeded_mask is None else {r for r in range(rows) if bool(seeded_mask[r])}
    missing = [r for r in seeded if r not in generators]
    if missing:
        raise ValueError(f"seeded rows {missing} have no generator")
    draftless = sorted(r for r in seeded if nv[r] == 0)
    if draftless:
        raise ValueError(
            f"seeded rows {draftless} carry no drafts: their bonus is a full-vocabulary exponential race the device cannot run"
        )
    draws = torch.zeros((rows, 2), dtype=torch.float64)
    acc = torch.zeros((rows, K), dtype=torch.float64)
    if draw_global:
        draws = torch.rand((rows, 2), dtype=torch.float64)
    for r in sorted(seeded):
        draws[r, 0] = torch.rand((1,), dtype=torch.float64, generator=generators[r])[0]
    if draw_global:
        acc = torch.rand((rows, K), dtype=torch.float64)
    for r in sorted(seeded):
        acc[r, : nv[r]] = torch.rand((nv[r],), dtype=torch.float64, generator=generators[r])
    for r in sorted(seeded):
        draws[r, 1] = torch.rand((1,), dtype=torch.float64, generator=generators[r])[0]
    out = torch.cat([draws[:, :1], acc, draws[:, 1:2]], dim=1).to(torch.float32)
    return out.clamp(max=_ONE_MINUS)


SAMP_PARAM_ROWS = (
    4  # samp_in rows 0..3: temperature, top_k, top_p, greedy (each [B] in words 0..B-1); rows 4.. = uniforms
)


def pack_samp_host(temperature, top_k, top_p, greedy, uniforms, W):
    """The fp32 [4 + B, W] samp_in staging tensor: rows temperature/top_k/top_p/greedy (words 0..B-1), then one row per
    user holding its K+2 uniforms (column 0 bonus, 1..K accepts, K+1 correction)."""
    B = int(torch.as_tensor(temperature).numel())
    uni = torch.as_tensor(uniforms).reshape(B, -1).to(torch.float32)
    if W < max(B, uni.shape[1]):
        raise ValueError(f"W={W} too small for B={B}, uniforms {tuple(uni.shape)}")
    out = torch.zeros(SAMP_PARAM_ROWS + B, W, dtype=torch.float32)
    for i, x in enumerate((temperature, top_k, top_p, greedy)):
        out[i, :B] = torch.as_tensor(x).reshape(B).to(torch.float32)
    out[SAMP_PARAM_ROWS:, : uni.shape[1]] = uni
    return out


class FusedSampledDevice(FusedGreedyDevice):
    """Device twin of fused_sampled_reference + the S1 drafter-input build (stage S2). ``allocate`` creates the persistent
    buffers (replicated over the mesh); ``body`` is trace-capturable and writes pack_out / rs_pt_out / result_out IN PLACE.

    Inputs : union_vals_in bf16 [1,1,32,TP*M] and union_ids_in uint32 [1,1,32,TP*M] (TILE; the model binds the topk trace's
             _vfy_topk_vals_out / _vfy_topk_ids_out: block d at columns d*M.., ids local to shard d; rows >= R are ignored),
             ids_in uint32 RM [1,1,1,32] (argmax ids, greedy rows), tok_in uint32 RM [1,R], aux_in int32 RM [7,W] (S1),
             samp_in fp32 RM [4+B, W] (pack_samp_host), pt_rows_in int32 RM [R,nb].
    Outputs: pack_out uint32 RM [2K+5,W], rs_pt_out int32 RM [R,nb] (as S1, computed from the sampled committed/counts),
             result_out int32 RM [3,W]: row 0 committed flattened r = u*T+j (-1 past the count), row 1 counts[B],
             row 2 overflow word [B] (OVF_TIE 1 / OVF_CERT 2; non-zero = redo on the host). Zero past their length.
    All math is fp32 TILE (token ids and positions < 2^24 are exact); exp is the accurate (non fast) mode.
    Every per-row quantity is kept per ENGINE ROW ([R, .]); the per-user pick is a row-select (j == m) + reshape [B, T] + sum.
    """

    @classmethod
    def allocate(
        cls,
        mesh,
        B,
        K,
        nb,
        W,
        TP,
        M,
        per_shard,
        union_vals_in=None,
        union_ids_in=None,
        ids_in=None,
        tok_in=None,
        pack_out=None,
        rs_pt_out=None,
        aux_in=None,
        pt_rows_in=None,
        stats_in=None,
    ):
        """``aux_in`` / ``pt_rows_in``: the S1 device's staging buffers to share (the engine stages them once per step for
        whichever accept trace runs); default: own buffers. ``stats_in``: the model's stats buffer (fp32 TILE [1,1,64,1], L_T in
        rows 32..) for the certificate variant body(cert=True); None until bind_stats."""
        import ttnn

        self = cls()
        self.mesh, self.B, self.K, self.T, self.R, self.nb, self.W = mesh, B, K, K + 1, B * (K + 1), nb, W
        self.TP, self.M, self.per_shard = TP, M, per_shard
        T, R = self.T, self.R
        TPM = TP * M
        assert R <= 32 and B <= W and K + 2 <= W and TPM <= UNION_PAD_WIDTH and M <= 64 and TPM % M == 0
        assert (TP - 1) * per_shard + per_shard < (1 << 24), "global ids must stay exact in fp32"
        RM, TL = ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT
        u32, i32, f32, bf16 = ttnn.uint32, ttnn.int32, ttnn.float32, ttnn.bfloat16
        z = lambda *s, dt=torch.int32: torch.zeros(*s, dtype=dt)
        self.ids_in = self.ids_standin = None
        self.union_vals_in = union_vals_in
        self.union_ids_in = union_ids_in
        for name, t, dt in (("union_vals_in", union_vals_in, bf16), ("union_ids_in", union_ids_in, u32)):
            if t is not None:
                got = (tuple(int(d) for d in t.shape), t.dtype)
                assert got == ((1, 1, 32, TPM), dt), f"bound {name}: {got}"
        if union_vals_in is None:
            self.union_vals_in = cls._dev(mesh, z(1, 1, 32, TPM, dt=torch.bfloat16), bf16, TL)
        if union_ids_in is None:
            self.union_ids_in = cls._dev(mesh, z(1, 1, 32, TPM), u32, TL)
        if ids_in is not None:
            self.bind_ids(ids_in)
        else:
            self.ids_in = self.ids_standin = cls._dev(mesh, z(1, 1, 1, 32), u32, RM)
        self.tok_in, self.pack_out, self.rs_pt_out = tok_in, pack_out, rs_pt_out
        for name, t, shape, dt in (
            ("tok_in", tok_in, (1, R), u32),
            ("pack_out", pack_out, (2 * K + 5, W), u32),
            ("rs_pt_out", rs_pt_out, (R, nb), i32),
        ):
            if t is not None:
                cls._check_bound(name, t, shape, dt, RM)
        if tok_in is None:
            self.tok_in = cls._dev(mesh, z(1, R), u32, RM)
        self.aux_in = aux_in if aux_in is not None else cls._dev(mesh, z(FUSED_AUX_ROWS, W), i32, RM)
        self.samp_in = cls._dev(mesh, z(SAMP_PARAM_ROWS + B, W, dt=torch.float32), f32, RM)
        self.pt_rows_in = pt_rows_in if pt_rows_in is not None else cls._dev(mesh, z(R, nb), i32, RM)
        if pack_out is None:
            self.pack_out = cls._dev(mesh, z(2 * K + 5, W), u32, RM)
        if rs_pt_out is None:
            self.rs_pt_out = cls._dev(mesh, z(R, nb), i32, RM)
        self.result_out = cls._dev(mesh, z(3, W), i32, RM)
        f = lambda t: cls._dev(mesh, t.to(torch.float32), f32, TL)
        self.c_iota_t = f(torch.arange(T).reshape(1, T))
        self.c_iota_k = f(torch.arange(K).reshape(1, K))
        self.c_iota_m = f(torch.arange(M).reshape(1, M))
        self.c_notfirst = f((torch.arange(M) > 0).reshape(1, M))
        self.c_ut = f((torch.arange(B) * T).reshape(B, 1))
        self.c_jr = f((torch.arange(R) % T).reshape(R, 1))  # column j of engine row r
        self.c_ones_t = f(torch.ones(1, T))
        self.c_one = f(torch.ones(1, 1))
        self.c_zero = f(torch.zeros(1, 1))
        self.c_zero_b = f(torch.zeros(B, 1))
        self.c_eps = f(torch.full((1, 1), CERT_EPS))
        self.stats_in = None
        if stats_in is not None:
            self.bind_stats(stats_in)
        self.c_off = cls._dev(
            mesh, (torch.arange(TPM) // M * per_shard).reshape(1, TPM).repeat(32, 1).to(torch.int32), u32, TL
        )
        pad = UNION_PAD_WIDTH - TPM
        self.c_pad_v = cls._dev(mesh, torch.full((32, pad), PAD_VALUE, dtype=torch.bfloat16), bf16, TL) if pad else None
        self.c_pad_i = cls._dev(mesh, z(32, pad), u32, TL) if pad else None
        return self

    def bind_union(self, vals, ids):
        """Point the union inputs at the topk trace outputs (they exist only after that capture)."""
        import ttnn

        TPM = self.TP * self.M
        for name, t, dt in (("union_vals_in", vals, ttnn.bfloat16), ("union_ids_in", ids, ttnn.uint32)):
            got = (tuple(int(d) for d in t.shape), t.dtype)
            assert got == ((1, 1, 32, TPM), dt), f"bound {name}: {got}"
        self.union_vals_in, self.union_ids_in = vals, ids

    def bind_stats(self, stats):
        """Point the certificate variant at the stats trace's output buffer (it exists after prepare_verify_stats)."""
        import ttnn

        got = (tuple(int(d) for d in stats.shape), stats.dtype)
        assert got == ((1, 1, 2 * STATS_LT_ROW, 1), ttnn.float32), f"bound stats_in: {got}"
        self.stats_in = stats

    def stage_union(self, union_vals, union_local_ids):
        """Standalone use: host -> device copy of the union (torch [R, TP*M]; rows R..31 zero)."""
        import ttnn

        TPM = self.TP * self.M
        v = torch.zeros(1, 1, 32, TPM, dtype=torch.bfloat16)
        v[0, 0, : self.R] = torch.as_tensor(union_vals).to(torch.bfloat16)
        i = torch.zeros(1, 1, 32, TPM, dtype=torch.int32)
        i[0, 0, : self.R] = torch.as_tensor(union_local_ids).to(torch.int32)
        for dst, t, dt in ((self.union_vals_in, v, ttnn.bfloat16), (self.union_ids_in, i, ttnn.uint32)):
            host = ttnn.from_torch(
                t, dtype=dt, layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh)
            )
            ttnn.copy_host_to_device_tensor(host, dst)

    def stage_samp(self, temperature, top_k, top_p, greedy, uniforms):
        import ttnn

        t = pack_samp_host(temperature, top_k, top_p, greedy, uniforms, self.W)
        host = ttnn.from_torch(
            t, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh)
        )
        ttnn.copy_host_to_device_tensor(host, self.samp_in)

    def stage_sampled(
        self,
        union_vals,
        union_local_ids,
        ids_words,
        tok_words,
        aux,
        pt_rows,
        temperature,
        top_k,
        top_p,
        greedy,
        uniforms,
    ):
        """Standalone use: every per-step input (the four S1 inputs + union + samp)."""
        self.stage(ids_words, tok_words, aux, pt_rows)
        self.stage_union(union_vals, union_local_ids)
        self.stage_samp(temperature, top_k, top_p, greedy, uniforms)

    def body(self, cert=False):
        """The trace-capturable sequence. No host round trip, no data-dependent shape, outputs written with ttnn.copy.
        ``cert=True`` (phase 2, work/research_1026/67 section 5.2): the certificate variant. A sampled row with top_k = 0 keeps all
        M candidates, takes L_T from ``stats_in`` and cuts the nucleus at 1 - p / q (q = Zc / max(L_T, Zc)); the overflow word gets
        bit 2 (OVF_CERT) when q - p < CERT_EPS in an examined column. Every other op and every non-certificate row is the plain
        body's (a certificate mask of 0 adds exactly 0): the plain body is NOT changed, ``cert=False`` records the same ops as before.
        """
        import ttnn

        B, K, T, R, W, nb, M = self.B, self.K, self.T, self.R, self.W, self.nb, self.M
        TPM = self.TP * M
        TL, RM, f32, u32 = ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT, ttnn.float32, ttnn.uint32
        tl = lambda x: ttnn.to_layout(x, TL)
        fl = lambda x: ttnn.typecast(x, f32)
        cat = lambda xs, d: ttnn.concat(xs, dim=d)
        mul, add, sub, rsub = ttnn.multiply, ttnn.add, ttnn.subtract, ttnn.rsub
        sumk = lambda x: ttnn.sum(x, dim=1, keepdim=True)
        expand = lambda x: ttnn.reshape(mul(x, self.c_ones_t), (R, 1))  # [B,1] -> per engine row [R,1]
        per_user = lambda x: ttnn.reshape(x, (B, T))  # [R,1] -> [B,T]
        # ---- inputs -> fp32 TILE
        ids = fl(tl(ttnn.reshape(ttnn.slice(self.ids_in, [0, 0, 0, 0], [1, 1, 1, R]), (B, T))))  # [B,T]
        tok = fl(tl(ttnn.reshape(self.tok_in, (B, T))))  # [B,T]
        aux = fl(tl(self.aux_in))  # [7,W]
        auxt = ttnn.transpose(aux, 0, 1)
        col = lambda i: ttnn.slice(auxt, [0, i], [B, i + 1])
        nv, act, dec, keep, p0 = col(AUX_NUM_VALID), col(AUX_ACTIVE), col(AUX_DECLINED), col(AUX_KEEP), col(AUX_P0)
        c = lambda i: ttnn.slice(aux, [AUX_CONSTS, i], [AUX_CONSTS + 1, i + 1])
        scratch, rows = c(CONST_SCRATCH), c(CONST_ROPE_ROWS)
        delta = col(AUX_DELTA)  # [B,1] per-row rope delta
        rows_m1 = sub(rows, self.c_one)
        samp = fl(tl(self.samp_in))  # [4+B,W]
        sampt = ttnn.transpose(samp, 0, 1)  # [W,4+B]
        scol = lambda i: ttnn.slice(sampt, [0, i], [B, i + 1])
        temp, topk_u, topp = scol(0), scol(1), scol(2)
        uni = ttnn.slice(samp, [SAMP_PARAM_ROWS, 0], [SAMP_PARAM_ROWS + B, K + 2])  # [B,K+2]
        # ---- effective per-user params (a greedy / inactive row runs the sampled math with temp 1, top_k 1; result unused)
        g = ttnn.maximum(scol(3), rsub(act, 1.0))  # [B,1] 0/1
        bad = ttnn.maximum(g, fl(ttnn.le(temp, self.c_zero)))
        temp_r = expand(add(mul(temp, rsub(bad, 1.0)), bad))  # [R,1]
        kidx_r = expand(sub(add(mul(topk_u, rsub(g, 1.0)), g), 1.0))  # top_k - 1
        omp_r = expand(rsub(topp, 1.0))  # 1 - top_p
        if cert:
            assert self.stats_in is not None, "body(cert=True) needs bind_stats first"
            lt = ttnn.slice(
                ttnn.reshape(self.stats_in, (2 * STATS_LT_ROW, 1)), [STATS_LT_ROW, 0], [STATS_LT_ROW + R, 1]
            )  # L_T [R,1]
            cert_r = expand(mul(fl(ttnn.le(topk_u, self.c_zero)), rsub(g, 1.0)))  # [R,1] sampled row with top_k = 0
            topp_r = expand(topp)
            kidx_r = add(kidx_r, mul(cert_r, rsub(kidx_r, float(M - 1))))  # kth = v[M - 1]: keep every candidate
        # ---- 1. union -> global top-M (value desc, gid asc): gid = local + (col // M) * per_shard
        uv = ttnn.reshape(tl(self.union_vals_in), (32, TPM))
        uvp = cat([uv, self.c_pad_v], 1) if self.c_pad_v is not None else uv
        gu = add(ttnn.reshape(tl(self.union_ids_in), (32, TPM)), self.c_off)  # uint32
        gup = cat([gu, self.c_pad_i], 1) if self.c_pad_i is not None else gu
        tv, ti = ttnn.topk(uvp, k=M, dim=-1, largest=True, sorted=True, stable=True)  # [32,M] bf16, uint16
        gids = fl(ttnn.gather(gup, 1, ttnn.typecast(ti, u32)))  # [32,M]
        vals = ttnn.slice(fl(tv), [0, 0], [R, M])
        gids = ttnn.slice(gids, [0, 0], [R, M])
        # ---- 2. temperature, top-k cut, overflow flag
        v = ttnn.divide(vals, temp_r)
        kth = sumk(mul(v, fl(ttnn.eq(self.c_iota_m, kidx_r))))  # v at column top_k - 1 (one-hot sum)
        keepm = fl(ttnn.ge(v, kth))  # [R,M]
        ovf_row = fl(ttnn.ge(ttnn.slice(v, [0, M - 1], [R, M]), kth))  # [R,1]
        # ---- 3. top-p after the top-k cut (reverse cumsum of the softmax, drop tail mass <= 1 - p, never column 0)
        e = mul(ttnn.exp(sub(v, ttnn.slice(v, [0, 0], [R, 1])), fast_and_approximate_mode=False), keepm)
        if cert:
            zc = sumk(e)
            q = ttnn.divide(
                zc, ttnn.maximum(lt, zc)
            )  # candidate mass fraction, (0, 1] and finite for any stats content
            omp_r = add(omp_r, mul(cert_r, sub(rsub(ttnn.divide(topp_r, q), 1.0), omp_r)))
            fail_row = mul(cert_r, fl(ttnn.lt(sub(q, topp_r), self.c_eps)))  # [R,1]
            ovf_row = mul(ovf_row, rsub(cert_r, 1.0))  # no top-k tie rule on a certificate row
        p0m = ttnn.divide(e, sumk(e))
        cum = ttnn.cumsum(p0m, dim=1, reverse_order=True)
        drop = mul(fl(ttnn.le(cum, omp_r)), self.c_notfirst)
        e2 = mul(e, rsub(drop, 1.0))
        prob = ttnn.divide(e2, sumk(e2))  # [R,M]
        # ---- 4. draft mass p_d and the accept walk (draft of row r = tok flat [r + 1]; last row of a user unused)
        tok_flat = ttnn.reshape(tok, (1, R))
        d_row = ttnn.transpose(cat([ttnn.slice(tok_flat, [0, 1], [1, R]), self.c_zero], 1), 0, 1)  # [R,1]
        hit = fl(ttnn.eq(gids, d_row))  # [R,M]
        p_d = ttnn.slice(per_user(sumk(mul(prob, hit))), [0, 0], [B, K])  # [B,K]
        valid = fl(ttnn.lt(self.c_iota_k, nv))
        acc_s = mul(mul(fl(ttnn.gt(p_d, self.c_zero)), fl(ttnn.ge(p_d, ttnn.slice(uni, [0, 1], [B, K + 1])))), valid)
        m_s = sumk(ttnn.cumprod(acc_s, dim=1))  # [B,1]
        # ---- 5. correction (m < nv: draft removed) or bonus (m == nv) at row u*T + m, inverse CDF
        is_corr = fl(ttnn.lt(m_s, nv))  # [B,1]
        corr_r, m_r = expand(is_corr), expand(m_s)
        u_pick = add(
            mul(ttnn.slice(uni, [0, 0], [B, 1]), rsub(is_corr, 1.0)),
            mul(ttnn.slice(uni, [0, K + 1], [B, K + 2]), is_corr),
        )
        w = mul(prob, rsub(mul(corr_r, hit), 1.0))  # [R,M] draft removed on a correction row
        cdf = ttnn.cumsum(w, dim=1)
        thr = mul(expand(u_pick), ttnn.slice(cdf, [0, M - 1], [R, M]))
        pick = sumk(fl(ttnn.le(cdf, thr)))  # [R,1] first bucket whose cdf > thr
        pos_rc = ttnn.cumsum(fl(ttnn.gt(w, self.c_zero)), dim=1, reverse_order=True)
        last = ttnn.relu(sub(sumk(fl(ttnn.ge(pos_rc, self.c_one))), 1.0))  # last positive-weight column
        pick = ttnn.minimum(pick, last)
        tok_pick_r = sumk(mul(gids, fl(ttnn.eq(self.c_iota_m, pick))))  # [R,1]
        rowsel = fl(ttnn.eq(self.c_jr, m_r))  # [R,1] the row j == m of its user
        tok_pick = sumk(per_user(mul(tok_pick_r, rowsel)))  # [B,1]
        # ---- 6. committed (sampled) and greedy rows exactly as S1
        drafts_pad = cat([ttnn.slice(tok, [0, 1], [B, T]), self.c_zero_b], 1)  # [B,T]
        s_comm = sub(
            add(
                mul(add(drafts_pad, 1.0), fl(ttnn.lt(self.c_iota_t, m_s))),
                mul(add(tok_pick, 1.0), fl(ttnn.eq(self.c_iota_t, m_s))),
            ),
            1.0,
        )
        eq_g = fl(ttnn.eq(ttnn.slice(ids, [0, 0], [B, K]), ttnn.slice(tok, [0, 1], [B, T])))
        m_g = sumk(ttnn.cumprod(mul(eq_g, valid), dim=1))
        g_comm = sub(mul(add(ids, 1.0), fl(ttnn.le(self.c_iota_t, m_g))), 1.0)
        ng = rsub(g, 1.0)
        m = add(mul(m_g, g), mul(m_s, ng))
        committed = add(mul(g_comm, g), mul(s_comm, ng))  # [B,T]
        cnt = add(m, 1.0)
        # ---- 7. overflow word: rows 0..m of a sampled user
        examined = fl(ttnn.le(self.c_jr, m_r))
        ovf = ttnn.gt(sumk(per_user(mul(ovf_row, examined))), self.c_zero_b)
        ovf = mul(fl(ovf), ng)  # [B,1]
        if cert:  # bit 2: the certificate failed in an examined column
            ovf_c = ttnn.gt(sumk(per_user(mul(fail_row, examined))), self.c_zero_b)
            ovf = add(ovf, mul(mul(fl(ovf_c), ng), float(OVF_CERT)))
        # ---- 8. drafter inputs (S1 tail; pending = committed[m])
        one_hot = fl(ttnn.eq(self.c_iota_t, m))
        ndec = rsub(dec, 1.0)
        pending = mul(sumk(mul(one_hot, committed)), ndec)
        p = mul(add(p0, m), ndec)
        anchor = add(self.c_ut, m)
        real = fl(ttnn.lt(self.c_iota_t, add(m, mul(dec, keep))))  # [B,T]
        rtok = mul(committed, real)
        rpos = mul(add(p0, self.c_iota_t), real)
        steps = add(p, self.c_iota_k)
        ridx = lambda x: ttnn.minimum(ttnn.relu(add(x, delta)), rows_m1)
        padw = lambda x, n: ttnn.pad(x, [(0, 0), (0, W - n)], value=0.0)
        g1 = padw(ttnn.transpose(cat([pending, steps], 1), 0, 1), B)
        g2 = padw(ttnn.reshape(cat([rtok, rpos], 0), (2, R)), R)
        g3 = padw(ttnn.transpose(anchor, 0, 1), B)
        g4 = padw(ttnn.reshape(ridx(rpos), (1, R)), R)
        g5 = padw(ttnn.transpose(ridx(steps), 0, 1), B)
        pack = cat([g1, g2, g3, g4, g5], 0)
        ttnn.copy(ttnn.to_layout(ttnn.typecast(pack, ttnn.uint32), RM), self.pack_out)
        res = cat(
            [
                padw(ttnn.reshape(committed, (1, R)), R),
                padw(ttnn.transpose(cnt, 0, 1), B),
                padw(ttnn.transpose(ovf, 0, 1), B),
            ],
            0,
        )
        ttnn.copy(ttnn.to_layout(ttnn.typecast(res, ttnn.int32), RM), self.result_out)
        pt = fl(tl(self.pt_rows_in))
        real_col = ttnn.reshape(real, (R, 1))
        rs = add(mul(sub(pt, scratch), real_col), scratch)
        ttnn.copy(ttnn.to_layout(ttnn.typecast(rs, ttnn.int32), RM), self.rs_pt_out)


# --------------------------------------------------------------------------- #
# Overflow redo (host, full logits, float64) and the audit's boundary classification
# --------------------------------------------------------------------------- #
ROW_DIST_PREFIX = (
    4096  # candidates the fast row distribution sorts (the rest is bounded by a full-vocabulary normaliser)
)


def _row_distribution_full(logit_row, temp, k, p):
    """One engine row's sampling distribution on the FULL logits, float64: temperature, top-k with ties kept, top-p (tail mass
    <= 1 - p dropped, never the first). Returns (ids int64 [V'], prob float64 [V']) in (value desc, id asc) order; V' is the
    vocabulary (entries cut by top-k / top-p have probability 0). A top-k <= 0 or >= V means "off". Reference
    implementation: a float64 sort of the whole vocabulary (about 35 ms per row); _row_distribution is the fast twin."""
    v = torch.as_tensor(logit_row).to(torch.float64) / float(temp)
    order = torch.sort(v, descending=True, stable=True).indices  # ties: ascending id (stable on the id-ordered row)
    vs = v[order]
    k = vs.numel() if int(k) <= 0 else min(int(k), vs.numel())
    keep = vs >= vs[k - 1]
    vs = torch.where(keep, vs, torch.full_like(vs, float("-inf")))
    e = torch.exp(vs - vs[0])
    p0 = e / e.sum()
    if float(p) < 1.0:
        cum = p0.flip(0).cumsum(0).flip(0)
        drop = cum <= (1.0 - float(p))
        drop[0] = False
        e = torch.where(drop, torch.zeros_like(e), e)
    return order, e / e.sum()


_TAIL_SKIP = 1.5  # go straight to the full sort when the estimated tail mass past the prefix is above this x (1 - p)


def _tail_beyond_prefix_estimate(x, temp, prefix):
    """Cheap (about 1 ms) estimate of the probability mass of the entries below the ``prefix``-th largest value: the normaliser is the
    exact float32 exp-sum, the threshold and the numerator come from a 1/16 strided sample (the mass past the prefix is spread over
    many entries, so the sample is good for the flat rows this is for). It only CHOOSES the path of _row_distribution (a flat row
    would fail the exact tail check after a 6 ms attempt); the exact check stays, so a wrong estimate costs time, never exactness.
    """
    x32 = x.to(torch.float32)
    e = torch.exp((x32 - x32.max()) / float(temp))
    xs, es = x32[::16], e[::16]
    thr = torch.kthvalue(xs, max(1, xs.numel() - prefix // 16 + 1)).values  # about the prefix-th largest value
    return float(
        16.0 * torch.where(xs < thr, es, torch.zeros_like(es)).sum(dtype=torch.float64) / e.sum(dtype=torch.float64)
    )


def _row_distribution(logit_row, temp, k, p, prefix=ROW_DIST_PREFIX):
    """_row_distribution_full without the full sort (about 5 ms instead of 35 ms per row): sort only the entries >= the
    ``prefix``-th largest value (ties at the threshold stay together, so the order is still (value desc, id asc)). Exact
    cases: (a) top-k active and its ties lie inside the prefix: nothing beyond the prefix is kept; (b) top-k off and top-p < 1
    and the mass beyond the prefix, from a full-vocabulary float64 normaliser, is <= 1 - p: every position beyond the
    prefix is dropped by the nucleus rule, and inside it the cut uses the full normaliser. Anything else (a prefix too
    short, p = 1 without a small top-k, a huge tie group) falls back to _row_distribution_full. Returns the TRUNCATED
    (ids int64 [n], prob float64 [n]): the dropped tail has probability 0 and is not listed, which does not change an
    inverse CDF, a hit test or a draft-removal. Equal to _row_distribution_full's first n entries (ties and all)."""
    x = torch.as_tensor(logit_row)
    V = x.numel()
    k = V if int(k) <= 0 else min(int(k), V)
    if V <= 2 * prefix or (k >= V and float(p) >= 1.0) or (k < V and k > prefix):
        return _row_distribution_full(logit_row, temp, k, p)
    if k >= V and _tail_beyond_prefix_estimate(x, temp, prefix) > _TAIL_SKIP * (1.0 - float(p)):
        return _row_distribution_full(
            logit_row, temp, k, p
        )  # a flat row: the nucleus reaches far past the prefix (bench: s3_redo_bench)
    v = x.to(torch.float64) / float(temp)
    thr = torch.topk(v, prefix).values[-1]
    ids = (v >= thr).nonzero().reshape(-1)  # ascending ids
    if ids.numel() > 2 * prefix:  # a huge tie group at the threshold
        return _row_distribution_full(logit_row, temp, k, p)
    vs0 = v[ids]
    order = torch.sort(vs0, descending=True, stable=True).indices
    ids, vs = ids[order], vs0[order]
    n = vs.numel()
    if k < V:
        if k > n:
            return _row_distribution_full(logit_row, temp, k, p)
        keep = vs >= vs[k - 1]
        vs = torch.where(keep, vs, torch.full_like(vs, float("-inf")))
        e = torch.exp(vs - vs[0])
        z = e.sum()
        tail = 0.0
    else:
        e = torch.exp(vs - vs[0])
        z = torch.exp(v - vs[0]).sum()  # full-vocabulary normaliser
        tail = float((z - e.sum()) / z)
        if tail > 1.0 - float(p):  # the nucleus may reach beyond the prefix
            return _row_distribution_full(logit_row, temp, k, p)
    p0 = e / z
    if float(p) < 1.0:
        cum = p0.flip(0).cumsum(0).flip(0) + tail
        drop = cum <= (1.0 - float(p))
        drop[0] = False
        e = torch.where(drop, torch.zeros_like(e), e)
    return ids, e / e.sum()


def _inverse_cdf_f64(w, u):
    """The plugin's / device's inverse CDF on weights w [N] >= 0 (scale free): first bucket whose cdf > u * total, clamped to
    the last positive weight."""
    cdf = w.cumsum(0)
    pick = int((cdf <= float(u) * float(cdf[-1])).sum())
    last = int((w > 0).nonzero().max())
    return min(pick, last)


def host_redo_walk(full_logits, tok_buf, params, uniforms, *, B, K, num_valid, users=None):
    """The exact accept walk of one verify step on the FULL logits, for the step the device flagged ``overflow`` (a top-k tie
    group reached the last union candidate, so the candidates cannot decide it). Same decisions as accept_sampled_support's
    reference path with the same ``uniforms`` [B, K+2] (column 0 bonus, 1..K accepts, K+1 correction): float64 arithmetic,
    top-k with ties kept, top-p, draft accepted iff its probability is > 0 and >= its uniform, correction from the row without
    the draft, bonus from row nv, inverse CDF in (value desc, id asc) order. Greedy rows (params["greedy"]) walk the argmax
    (lowest id among ties), as the device S1 / plugin greedy walk.

    full_logits [R, V] (rows u*T+j, any float dtype), tok_buf [R] or [B, T] (column j token; drafts = columns 1..K), params:
    dict temperature [B] / top_k [B] / top_p [B] / greedy [B] bool, num_valid [B]. Only the rows the walk examines are
    evaluated. ``users`` (optional iterable of user indices): redo only those users; the other users come back as
    committed = -1 and counts = 1 (meaningless: merge_redo keeps the device result for them). A top_k <= 0 or >= V means off.
    Returns (committed [B, T] int32, -1 past the count; counts [B] int32 in [1, T])."""
    T, R = K + 1, B * (K + 1)
    logits = torch.as_tensor(full_logits)
    assert logits.shape[0] >= R, (tuple(logits.shape), R)
    toks = torch.as_tensor(tok_buf).reshape(-1)[:R].to(torch.int64).reshape(B, T)
    nv = torch.as_tensor(num_valid).reshape(B).to(torch.int64).tolist()
    greedy = torch.as_tensor(params["greedy"]).reshape(B).bool().tolist()
    temp = torch.as_tensor(params["temperature"]).reshape(B).to(torch.float64).tolist()
    topk = torch.as_tensor(params["top_k"]).reshape(B).to(torch.int64).tolist()
    topp = torch.as_tensor(params["top_p"]).reshape(B).to(torch.float64).tolist()
    uni = torch.as_tensor(uniforms).reshape(B, K + 2).to(torch.float64)
    committed = torch.full((B, T), -1, dtype=torch.int32)
    counts = torch.ones(B, dtype=torch.int32)

    def argmax_row(r):
        row = logits[r].to(torch.float32)
        return int((row == row.max()).nonzero()[0])

    todo = range(B) if users is None else sorted({int(u) for u in users})
    for u in todo:
        n = nv[u]
        if greedy[u] or temp[u] <= 0:
            m = 0
            while m < n and argmax_row(u * T + m) == int(toks[u, m + 1]):
                m += 1
            for j in range(m + 1):
                committed[u, j] = argmax_row(u * T + j)
            counts[u] = m + 1
            continue
        m, rejected, dist = 0, False, None
        while m < n:
            order, prob = _row_distribution(logits[u * T + m], temp[u], topk[u], topp[u])
            hit = (order == int(toks[u, m + 1])).nonzero()
            p_d = float(prob[int(hit[0])]) if hit.numel() else 0.0
            if p_d > 0 and p_d >= float(uni[u, 1 + m]):
                committed[u, m] = int(toks[u, m + 1])
                m += 1
                continue
            rejected, dist = True, (order, prob)
            break
        if dist is None:  # every draft accepted (or none): bonus from row n
            dist = _row_distribution(logits[u * T + m], temp[u], topk[u], topp[u])
        order, prob = dist
        w, uc = prob.clone(), float(uni[u, K + 1] if rejected else uni[u, 0])
        if rejected:
            w[order == int(toks[u, m + 1])] = 0.0
        committed[u, m] = int(order[_inverse_cdf_f64(w, uc)])
        counts[u] = m + 1
    return committed, counts


def merge_redo(dev_committed, dev_counts, redo_committed, redo_counts, users):
    """Per-user merge of a subset redo into the device result: users in ``users`` take the host redo, every other user keeps
    the device committed row and count (a user's result does not depend on the other users). Returns (committed, counts)
    as new int32 tensors."""
    committed = torch.as_tensor(dev_committed).to(torch.int32).clone()
    counts = torch.as_tensor(dev_counts).to(torch.int32).clone()
    idx = torch.as_tensor(sorted({int(u) for u in users}), dtype=torch.int64)
    if idx.numel():
        committed[idx] = torch.as_tensor(redo_committed).to(torch.int32)[idx]
        counts[idx] = torch.as_tensor(redo_counts).to(torch.int32)[idx]
    return committed, counts


def audit_boundary_users(ref, uniforms, committed, counts, drafts, num_valid, *, B, K, M, eps=1e-5, top_p=None):
    """Users whose device result differs from ``ref`` (fused_sampled_reference) but only by a float32 boundary: an examined draft
    probability within ``eps`` of its uniform (accept test), or the inverse-CDF threshold within ``eps`` (relative to the total)
    of a bucket edge of the corrected / bonus row. Returns {user: reason} for the explained ones; a differing user that is not in
    it is a real mismatch. ``drafts`` [B, K], ``num_valid`` [B]. With ``top_p`` [B] and a reference built with ``full_norm``
    (ref.cert_q), a certificate row of the user within ``eps`` of the certificate margin (|q - p| < eps) or of the nucleus
    cut (|cum - (1 - p / q)| < eps) in an examined column is the explained reason "cert"."""
    committed, counts = torch.as_tensor(committed).to(torch.int64), torch.as_tensor(counts).to(torch.int64)
    rc, rn = ref.committed.to(torch.int64), ref.counts.to(torch.int64)
    uni = torch.as_tensor(uniforms).reshape(B, K + 2).to(torch.float64)
    drafts = torch.as_tensor(drafts).reshape(B, K).to(torch.int64)
    nv = torch.as_tensor(num_valid).reshape(B).to(torch.int64)
    T = K + 1
    out = {}
    for u in range(B):
        if bool(torch.equal(rc[u], committed[u])) and int(rn[u]) == int(counts[u]):
            continue
        first_split = min(
            int(counts[u]), int(rn[u])
        )  # accept tests 0 .. first_split - 1 are shared; the split is at column first_split - 1
        tests = range(min(first_split, int(nv[u])))
        if any(abs(float(ref.p_d[u, j]) - float(uni[u, 1 + j])) < eps for j in tests):
            out[u] = "accept"
            continue
        if ref.cert_q is not None and top_p is not None:
            pu = float(torch.as_tensor(top_p).reshape(B)[u])
            reach = min(T, max(int(counts[u]), int(rn[u])))
            for c in range(reach):
                q = float(ref.cert_q[u * T + c])
                if q >= 1.0:
                    continue
                cut = 1.0 - pu / q
                if abs(q - pu) < eps or bool(((ref.cum[u * T + c].double() - cut).abs() < eps).any()):
                    out[u] = "cert"
                    break
            if u in out:
                continue
        if int(rn[u]) == int(counts[u]):  # same m: only the pick differs
            m = int(rn[u]) - 1
            w = ref.prob.reshape(B, T, M)[u, m].to(torch.float64).clone()
            corr = m < int(nv[u])
            if corr:
                w[ref.cand_ids.reshape(B, T, M)[u, m] == int(drafts[u, m])] = 0.0
            cdf = w.cumsum(0)
            thr = float(uni[u, K + 1 if corr else 0]) * float(cdf[-1])
            if bool((((cdf - thr).abs()) < eps * float(cdf[-1])).any()):
                out[u] = "pick"
    return out
