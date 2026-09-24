# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""mHC (manifold-constrained hyper-connections) as the exact op sequence the ttnn module runs, in torch.

This is the algebra lock for ``tt/v4/hyper_connection.py``: every step here has a one-to-one ttnn counterpart
(matmul / add / multiply / sigmoid / exp / rsqrt / reciprocal / column slice), so a host test against the reference
``DeepseekV4HyperConnection`` (REF ``modeling_deepseek_v4.py:877-953``) proves the decomposition before any device
time, and the device test then only has to prove ttnn's numerics.

Decomposition (per site; ``H = hc_mult = 4``, ``D = hidden``, streams are a LIST of ``H`` tensors ``[S, D]`` --
on device each is ``[1, 1, S_l, D_l]``, TP-sharded on D):

1. fold the unweighted RMSNorm past the linear: ``F.linear(x * rstd, fn) = rstd * (x @ fn^T)``, rstd a per-token
   scalar. Nothing ``[S, H*D]``-wide is ever materialised.
2. ``mix_partial = sum_h x_h @ FN_h`` with ``FN_h = fn[:, h*D:(h+1)*D]^T`` zero-padded to 32 columns (24 real);
   ``ss_partial = sum_h (x_h * x_h) @ ONES_h`` where ``ONES_h[:, 24] = 1`` puts the per-token sum of squares into
   column 24 of the same 32-wide row. Both are TP-partial sums over the local D slice.
3. one TP all-reduce of the ``[S, 32]`` row (device: all-gather on dim 1 + fast_reduce_nc).
4. ``ms = part @ MS`` broadcasts column 24 / (H*D) to every column; ``rstd = rsqrt(ms + eps)``; ``mix = part * rstd``.
5. affine per column: ``aff = mix * SV + BV`` with ``SV = [s0]*4 + [s1]*4 + [s2]*16 + [0]*8`` and the matching
   ``base`` layout; ``sig = sigmoid(aff)``.
6. ``pre = sig + eps`` (cols 0..3), ``post = 2 * sig`` (cols 4..7), ``comb``: ``E = exp(aff) * COMB_MASK`` (cols
   8..23, zero elsewhere), row softmax = ``E / (E @ R)`` where ``R[(i,j),(i,j')] = 1`` sums each 4-row; ``+ eps``
   (on the 16 comb columns); Sinkhorn: column normalise ``X / (X @ C + eps)``, then 19 x (row ``X / (X @ R + eps)``,
   column). ``R``/``C`` are constant 32x32 0/1 matrices; the padded columns stay exactly 0 (0 / eps).
7. ``collapsed = sum_h pre[:, h] * x_h`` (column broadcast).
8. mix site: ``out_k = post[:, 4+k] * y + sum_j comb[:, 8 + j*4 + k] * x_j`` -- comb consumed TRANSPOSED
   (``comb[j, k]`` weights stream j into output k), as the reference does.

Column layout of the 32-wide row: ``[pre 0..3 | post 4..7 | comb 8..23 (i*4+j) | ss 24 | pad 25..31]``.
"""

from __future__ import annotations

import torch

HC = 4
ROW = 32  # padded mix row width (24 real logits + the sum-of-squares column + the ones column + pad)
PRE0, POST0, COMB0, SS_COL, ONE_COL = 0, 4, 8, 24, 25
EXP_CLAMP = 60.0  # exp() guard for the comb logits (the reference's softmax subtracts the row max; we do not)


def comb_index(i: int, j: int) -> int:
    return COMB0 + i * HC + j


def row_sum_matrix() -> torch.Tensor:
    """R: (X @ R)[:, c] = sum over the 4 comb columns sharing c's row i (zero outside the comb block)."""
    R = torch.zeros(ROW, ROW)
    for i in range(HC):
        for j in range(HC):
            for j2 in range(HC):
                R[comb_index(i, j), comb_index(i, j2)] = 1.0
    return R


def col_sum_matrix() -> torch.Tensor:
    """C: (X @ C)[:, c] = sum over the 4 comb columns sharing c's column j."""
    C = torch.zeros(ROW, ROW)
    for i in range(HC):
        for j in range(HC):
            for i2 in range(HC):
                C[comb_index(i, j), comb_index(i2, j)] = 1.0
    return C


def _augment(Msum: torch.Tensor, eps: float) -> torch.Tensor:
    """The ones-column trick: X keeps a constant 1 in ONE_COL, so ``X @ M_aug`` yields ``sum + eps`` on the comb
    columns (row ONE_COL carries eps there) and exactly 1 on every other column (row ONE_COL carries 1 there),
    making ``X / (X @ M_aug)`` a no-op on the padded columns and an eps-guarded normalisation on the comb block --
    one matmul + one divide per Sinkhorn step, no separate add."""
    M = Msum.clone()
    for c in range(ROW):
        in_comb = COMB0 <= c < COMB0 + HC * HC
        M[ONE_COL, c] = eps if in_comb else 1.0
    return M


def row_sum_matrix_aug(eps: float) -> torch.Tensor:
    return _augment(row_sum_matrix(), eps)


def col_sum_matrix_aug(eps: float) -> torch.Tensor:
    return _augment(col_sum_matrix(), eps)


def mean_square_matrix(hidden: int) -> torch.Tensor:
    """MS: (part @ MS)[:, c] = part[:, SS_COL] / (H*D) for every column c."""
    MS = torch.zeros(ROW, ROW)
    MS[SS_COL, :] = 1.0 / float(HC * hidden)
    return MS


def comb_mask() -> torch.Tensor:
    m = torch.zeros(ROW)
    m[COMB0 : COMB0 + HC * HC] = 1.0
    return m


def one_col() -> torch.Tensor:
    m = torch.zeros(ROW)
    m[ONE_COL] = 1.0
    return m


def prep_hyper_connection(fn: torch.Tensor, base: torch.Tensor, scale: torch.Tensor, hidden: int) -> dict:
    """Constant tensors for one HyperConnection site. ``fn [24, H*D]`` (row r = logit r; columns h*D + d),
    ``base [24]``, ``scale [3]``. Returns fp32 host tensors:
      FN   [H, D, 32]  fn^T per stream, padded (col r < 24 = logit r)
      ONES [H, D, 32]  column SS_COL = 1
      SV, BV [32]      per-column scale / base (0 outside the 24 logits)
      R, C, MS [32, 32], COMB_MASK [32]
    """
    fn = fn.float()
    mix = (2 + HC) * HC
    assert tuple(fn.shape) == (mix, HC * hidden), fn.shape
    FN = torch.zeros(HC, hidden, ROW)
    for h in range(HC):
        FN[h, :, :mix] = fn[:, h * hidden : (h + 1) * hidden].T
    ONES = torch.zeros(HC, hidden, ROW)
    ONES[:, :, SS_COL] = 1.0
    SV = torch.zeros(ROW)
    BV = torch.zeros(ROW)
    s = scale.float()
    b = base.float()
    SV[PRE0 : PRE0 + HC] = s[0]
    SV[POST0 : POST0 + HC] = s[1]
    SV[COMB0 : COMB0 + HC * HC] = s[2]
    BV[:mix] = b
    return {
        "FN": FN,
        "ONES": ONES,
        "SV": SV,
        "BV": BV,
        "R_SOFT": row_sum_matrix_aug(0.0),  # exact row softmax denominators (no eps) on the comb block
        "MS": mean_square_matrix(hidden),
        "COMB_MASK": comb_mask(),
        "ONE_COL": one_col(),
    }


def sinkhorn_constants(eps: float) -> dict:
    """The eps-augmented sum matrices and the eps-on-comb-columns vector, per hc_eps."""
    return {"R_AUG": row_sum_matrix_aug(eps), "C_AUG": col_sum_matrix_aug(eps), "EPS_COMB": comb_mask() * eps}


def prep_hyper_head(hc_fn: torch.Tensor, hc_base: torch.Tensor, hc_scale: torch.Tensor, hidden: int) -> dict:
    """The HyperHead is the pre-branch alone: ``hc_fn [H, H*D]``, ``hc_base [H]``, ``hc_scale [1]``."""
    hc_fn = hc_fn.float()
    assert tuple(hc_fn.shape) == (HC, HC * hidden), hc_fn.shape
    FN = torch.zeros(HC, hidden, ROW)
    for h in range(HC):
        FN[h, :, :HC] = hc_fn[:, h * hidden : (h + 1) * hidden].T
    ONES = torch.zeros(HC, hidden, ROW)
    ONES[:, :, SS_COL] = 1.0
    SV = torch.zeros(ROW)
    BV = torch.zeros(ROW)
    SV[:HC] = hc_scale.float()[0]
    BV[:HC] = hc_base.float()
    return {"FN": FN, "ONES": ONES, "SV": SV, "BV": BV, "MS": mean_square_matrix(hidden)}


def _tp_shards(x: torch.Tensor, tp: int) -> list[torch.Tensor]:
    d = x.shape[-1]
    assert d % tp == 0
    return list(x.float().split(d // tp, dim=-1))


def mix_row(streams: list[torch.Tensor], W: dict, *, tp: int = 1) -> torch.Tensor:
    """Steps 2-4: the all-reduced, norm-folded ``[S, 32]`` logit row (column SS_COL holds the raw sum of squares
    times rstd -- unused afterwards). ``tp`` emulates TP partial sums over D slices."""
    assert len(streams) == HC
    S = streams[0].shape[0]
    part = torch.zeros(S, ROW)
    for h, x in enumerate(streams):
        for t, xs in enumerate(_tp_shards(x, tp)):
            dl = xs.shape[-1]
            FN_t = W["FN"][h, t * dl : (t + 1) * dl, :]
            ONES_t = W["ONES"][h, t * dl : (t + 1) * dl, :]
            part = part + xs @ FN_t + (xs * xs) @ ONES_t
    ms = part @ W["MS"]
    rstd = torch.rsqrt(ms + W.get("eps", 1e-6))
    return part * rstd


def pre_site(streams: list[torch.Tensor], W: dict, *, eps: float = 1e-6, sinkhorn_iters: int = 20, tp: int = 1):
    """Steps 1-7. Returns ``(pre [S,32], post [S,32], comb [S,32], collapsed [S, D])`` (fp32); ``pre``/``post``
    are meaningful in their own columns only, ``comb`` in columns 8..23 (i*4+j), zeros elsewhere."""
    W = dict(W)
    W["eps"] = W.get("rms_eps", 1e-6)
    mix = mix_row(streams, W, tp=tp)
    aff = mix * W["SV"] + W["BV"]
    sig = torch.sigmoid(aff)
    pre = sig + eps
    post = 2.0 * sig
    K = sinkhorn_constants(eps)
    # comb block: exp on the 16 comb columns, a constant 1 in ONE_COL, 0 elsewhere
    E = torch.exp(torch.clamp(aff, -EXP_CLAMP, EXP_CLAMP)) * W["COMB_MASK"] + W["ONE_COL"]
    X = E / (E @ W["R_SOFT"])  # exact row softmax (denominator = row sum on the comb block, 1 elsewhere)
    X = X + K["EPS_COMB"]
    X = X / (X @ K["C_AUG"])  # column normalise with + eps folded in via the ones column
    for _ in range(sinkhorn_iters - 1):
        X = X / (X @ K["R_AUG"])
        X = X / (X @ K["C_AUG"])
    collapsed = torch.zeros_like(streams[0], dtype=torch.float32)
    for h, x in enumerate(streams):
        collapsed = collapsed + pre[:, PRE0 + h : PRE0 + h + 1] * x.float()
    return pre, post, X, collapsed


def mix_site(
    streams: list[torch.Tensor], y: torch.Tensor, post: torch.Tensor, comb: torch.Tensor
) -> list[torch.Tensor]:
    """Step 8: ``out_k = post_k * y + sum_j comb[j, k] * x_j`` (comb transposed, as the reference)."""
    out = []
    for k in range(HC):
        acc = post[:, POST0 + k : POST0 + k + 1] * y.float()
        for j, x in enumerate(streams):
            c = comb_index(j, k)
            acc = acc + comb[:, c : c + 1] * x.float()
        out.append(acc)
    return out


def hyper_head(streams: list[torch.Tensor], W: dict, *, eps: float = 1e-6, tp: int = 1) -> torch.Tensor:
    W = dict(W)
    W["eps"] = W.get("rms_eps", 1e-6)
    mix = mix_row(streams, W, tp=tp)
    pre = torch.sigmoid(mix * W["SV"] + W["BV"]) + eps
    out = torch.zeros_like(streams[0], dtype=torch.float32)
    for h, x in enumerate(streams):
        out = out + pre[:, h : h + 1] * x.float()
    return out
