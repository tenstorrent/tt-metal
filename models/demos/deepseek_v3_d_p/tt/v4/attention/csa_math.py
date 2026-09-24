# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The V4 two-series (CSA) compressor as the exact op sequence the ttnn module runs, in torch -- the algebra lock
for ``tt/v4/attention/csa.py`` (same role as ``tt/v4/mhc_math.py`` for the hyper-connections).

Reference (``modeling_deepseek_v4.py`` DeepseekV4CSACompressor / DeepseekV4Indexer): every token projects to
``[Ca | Cb]`` (2W wide) for kv and gate; ``gate += position_bias[slot]`` (slot = token % 4, bias 2W wide); entry
``w`` is the softmax-gated (over 8 slots, fp32) combination of window ``w-1``'s 4 Ca rows and window ``w``'s 4 Cb
rows; window 0 takes the PREVIOUS CHUNK's last-window Ca rows (zero kv / -inf gate on the very first chunk); then
RMSNorm(W) and RoPE("compress") at token position ``4w + first_window_position``.

Device formulation (no 4-row tensors, no softmax over a padded slot axis):
  * SP all-gather kv/gate so every chip holds the whole chunk (5120 x 2W) and computes every entry (replicated);
  * one per-channel constant ``M`` (max over the chunk's gates AND the prior block, both series) is subtracted
    before ``exp`` -- softmax is invariant to it and it keeps exp in range;
  * window sums are matmuls with the constant 0/1 matrix ``G [n_win, S]`` (``G[w, s] = 1 iff s // 4 == w``);
  * the Ca series is shifted one window with the constant ``Sh [n_win, n_win]`` (``Sh[w, w-1] = 1``) and window 0
    takes the prior block's last-4-row sums (``Sel [32, 32]``: row 0 = ones on rows 28..31; the prior block is
    the previous chunk's last 32 REAL Ca rows, tile-aligned) through ``P0 [n_win, 32]`` (one-hot at [0, 0]);
  * entry = (num_a + num_b) / (den_a + den_b), then RMSNorm and rope.
The prior kept for the next chunk: the last 32 real rows of this chunk's Ca kv and (biased) Ca gate.
"""

from __future__ import annotations

import torch

RATE = 4
TILE = 32
NEG = -1e9  # a finite stand-in for the reference's -inf gate on the very first chunk (exp underflows to exactly 0)


def group_sum_matrix(S: int, rate: int = RATE) -> torch.Tensor:
    G = torch.zeros(S // rate, S)
    for w in range(S // rate):
        G[w, w * rate : (w + 1) * rate] = 1.0
    return G


def shift_matrix(n: int) -> torch.Tensor:
    Sh = torch.zeros(n, n)
    for w in range(1, n):
        Sh[w, w - 1] = 1.0
    return Sh


def prior_select_matrix(rate: int = RATE) -> torch.Tensor:
    """[32, 32] with row 0 = ones on the last ``rate`` rows: (Sel @ X)[0] sums the prior block's last window."""
    Sel = torch.zeros(TILE, TILE)
    Sel[0, TILE - rate :] = 1.0
    return Sel


def first_window_matrix(n: int) -> torch.Tensor:
    P0 = torch.zeros(n, TILE)
    P0[0, 0] = 1.0
    return P0


def bias_rows(position_bias: torch.Tensor, S: int) -> torch.Tensor:
    """[S, 2W]: row s carries position_bias[s % rate]."""
    return position_bias.float().repeat(S // position_bias.shape[0], 1)


def empty_prior(W: int) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.zeros(TILE, W), torch.full((TILE, W), NEG)


def pool_entries(kv: torch.Tensor, gate: torch.Tensor, position_bias: torch.Tensor, prior: tuple, *, rate: int = RATE):
    """kv/gate ``[S, 2W]`` (fp32, whole chunk), ``prior = (kv_a_last32 [32, W], gate_a_last32 [32, W])`` (gate already
    biased). Returns ``(entries_unnormed [S/rate, W], new_prior)`` -- before RMSNorm/rope."""
    S, W2 = kv.shape
    W = W2 // 2
    n = S // rate
    gate = gate.float() + bias_rows(position_bias, S)
    kv = kv.float()
    kv_a, kv_b = kv[:, :W], kv[:, W:]
    g_a, g_b = gate[:, :W], gate[:, W:]
    pk, pg = prior
    M = torch.maximum(torch.maximum(g_a.max(0).values, g_b.max(0).values), pg.max(0).values)  # [W]
    E_a, E_b, E_p = torch.exp(g_a - M), torch.exp(g_b - M), torch.exp(pg - M)
    G, Sh, Sel, P0 = group_sum_matrix(S, rate), shift_matrix(n), prior_select_matrix(rate), first_window_matrix(n)
    den_b, num_b = G @ E_b, G @ (E_b * kv_b)
    den_a_win, num_a_win = G @ E_a, G @ (E_a * kv_a)
    den_a = Sh @ den_a_win + P0 @ (Sel @ E_p)
    num_a = Sh @ num_a_win + P0 @ (Sel @ (E_p * pk))
    entries = (num_a + num_b) / (den_a + den_b)
    return entries, (kv_a[S - TILE :], g_a[S - TILE :])


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    var = x.float().pow(2).mean(-1, keepdim=True)
    return weight.float() * (x.float() * torch.rsqrt(var + eps))


def rope_entries(entries: torch.Tensor, rotary_emb, first_window_position: int, rate: int = RATE) -> torch.Tensor:
    """RoPE("compress") at token positions ``4w + first_window_position`` on the trailing rope dims."""
    from models.demos.deepseek_v3_d_p.reference.deepseek_v4.modeling_deepseek_v4 import apply_rotary_pos_emb

    n = entries.shape[0]
    pos = (torch.arange(n) * rate + first_window_position).unsqueeze(0)
    cos, sin = rotary_emb(entries, position_ids=pos, layer_type="compress")
    return apply_rotary_pos_emb(entries.view(1, 1, n, -1), cos, sin)[0, 0]


def compress_chunk(
    kv, gate, position_bias, kv_norm_weight, eps, rotary_emb, prior, first_window_position, *, rate=RATE
):
    entries, new_prior = pool_entries(kv, gate, position_bias, prior, rate=rate)
    return rope_entries(rms_norm(entries, kv_norm_weight, eps), rotary_emb, first_window_position, rate), new_prior


def indexer_scores(q_heads: torch.Tensor, keys: torch.Tensor, weights: torch.Tensor, head_dim: int, n_heads: int):
    """``q_heads [H, S, Dh]`` (roped), ``keys [T, Dh]``, ``weights [S, H]`` (weights_proj output) ->
    ``scores [S, T] = sum_h relu(q_h . k) * Dh^-0.5 * w_h * H^-0.5`` (the reference scorer)."""
    s = torch.relu(q_heads.float() @ keys.float().T) * head_dim**-0.5  # [H, S, T]
    w = weights.float() * n_heads**-0.5  # [S, H]
    return (s * w.T.unsqueeze(-1)).sum(0)


def selection_mask(scores_masked: torch.Tensor, k: int) -> torch.Tensor:
    """Additive 0/-inf mask selecting each row's top-k of the (already causally -inf-masked) scores by threshold:
    theta = k-th largest; keep scores >= theta. Rows with fewer than k valid entries get theta = -inf and keep
    everything (the causal mask, added afterwards, removes the invalid entries again)."""
    theta = torch.topk(scores_masked, k, dim=-1).values.min(-1, keepdim=True).values
    return torch.log((scores_masked >= theta).float())
