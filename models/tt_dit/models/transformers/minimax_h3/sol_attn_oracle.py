# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Host oracle for Sol-Attn routing (S2 of `pipelines/minimax_h3/PLAN_SOL.md`).

A pure-torch transcription of the vendored Triton reference in
`sol_attn_reference/`, which cannot run here (no triton, no CUDA). Everything
below is fp32 and computed in one shot rather than in 32-block groups: the
kernel's grouping changes only the online-softmax rounding, not the result.

Sol-Attn differs from the VSA this model ships in two ways that matter for the
port, and both are visible in `sol_attention` below:

* **The threshold is query-dependent and analytic.** Per (query block, head),
  `mu + tau * sigma` where `mu` and `sigma` come from the *population statistics*
  of the pooled keys, not from a top-k over realised scores. So the number of
  selected blocks is data-dependent -- there is no `k`.
* **Unselected blocks are approximated, not dropped.** They enter the same
  softmax as one pooled token carrying `kc` (the block's mean key) and `vc`
  (its summed value), length-weighted in the denominator. VSA discards them and
  reinstates a learned-gated coarse output instead.

Scores are held in base 2 (`scale * log2(e)`) exactly as the kernel does, so the
threshold comparison happens in the same units the kernel compares in.

Two limits worth knowing before reaching for this:

* `block_summaries` / `query_centroids` / `routing_threshold` / `route` cost
  O(n_blocks^2) and run at H3's production geometry (1808 blocks at 15 s).
  `sol_attention` materialises a `T x T` logit tensor and does not -- use it on
  small shapes to check the arithmetic, and the routing functions alone to ask
  what the thresholds and densities do at real sizes.
* The kernel keeps `kc`/`vc` in bf16; this stays fp32 throughout. That is the
  point of an oracle, but it means a PCC against the kernel is bounded by bf16
  summary rounding, not by this file.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

SOL_BLOCK_SIZE = 64
SOL_DEFAULT_TAU = 1.0  # the validated H3 policy; Sol's own notes record that per-shape calibration
# returned an empty route set on this model's shape and silently fell back to dense
_LOG2_E = 1.4426950408889634
_VAR_EPS = 1.0e-6


def _block_lengths(tokens: int, block: int, device) -> torch.Tensor:
    """Valid token count per block; only the tail block is short."""
    n_blocks = (tokens + block - 1) // block
    starts = torch.arange(n_blocks, device=device) * block
    return (tokens - starts).clamp(max=block).to(torch.float32)


def block_summaries(
    k: torch.Tensor, v: torch.Tensor, *, block: int = SOL_BLOCK_SIZE
) -> tuple[torch.Tensor, torch.Tensor]:
    """[B, H, T, D] K/V -> (kc, vc) at [B, H, n_blocks, D].

    `kc` is the block **mean** and `vc` the block **sum**: the pair reconstructs a
    mean value when the approximate branch divides by the length it accumulates.
    """
    b, h, tokens, d = k.shape
    n_blocks = (tokens + block - 1) // block
    pad = n_blocks * block - tokens
    if pad:
        k = torch.nn.functional.pad(k, (0, 0, 0, pad))
        v = torch.nn.functional.pad(v, (0, 0, 0, pad))
    lengths = _block_lengths(tokens, block, k.device).view(1, 1, n_blocks, 1)
    kc = k.float().view(b, h, n_blocks, block, d).sum(dim=3) / lengths
    vc = v.float().view(b, h, n_blocks, block, d).sum(dim=3)
    return kc, vc


def query_centroids(q: torch.Tensor, *, block: int = SOL_BLOCK_SIZE) -> torch.Tensor:
    """[B, H, T, D] Q -> per-query-block centroid at [B, H, n_q_blocks, D]."""
    b, h, tokens, d = q.shape
    n_blocks = (tokens + block - 1) // block
    pad = n_blocks * block - tokens
    if pad:
        q = torch.nn.functional.pad(q, (0, 0, 0, pad))
    lengths = _block_lengths(tokens, block, q.device).view(1, 1, n_blocks, 1)
    return q.float().view(b, h, n_blocks, block, d).sum(dim=3) / lengths


def routing_threshold(
    q_bar: torch.Tensor,
    kc: torch.Tensor,
    *,
    tau: float = SOL_DEFAULT_TAU,
    scale: float,
    thresh_type: str = "diag",
) -> torch.Tensor:
    """Per-(query block, head) threshold at [B, H, n_q_blocks], in base-2 score units.

    `diag` keeps only the per-dimension variance of the pooled keys; `exact` keeps
    the full `D x D` second moment. They agree when the pooled-key dimensions are
    uncorrelated, and `exact` is the larger threshold otherwise.
    """
    log2_scale = scale * _LOG2_E
    kc_mean = kc.mean(dim=2)  # [B, H, D]
    raw_mean = (q_bar * kc_mean.unsqueeze(2)).sum(dim=-1)  # [B, H, n_q]

    if thresh_type == "exact":
        second_moment = kc.transpose(-1, -2) @ kc / kc.shape[2]  # [B, H, D, D]
        raw_second = ((q_bar @ second_moment) * q_bar).sum(dim=-1)
        variance = (raw_second - raw_mean * raw_mean).clamp(min=0.0) * (log2_scale**2)
    elif thresh_type == "diag":
        kc_var = (kc * kc).mean(dim=2) - kc_mean * kc_mean  # [B, H, D]
        variance = (q_bar * q_bar * kc_var.unsqueeze(2).clamp(min=0.0)).sum(dim=-1) * (log2_scale**2)
    else:
        raise ValueError(f"thresh_type must be 'diag' or 'exact', got {thresh_type!r}")

    return raw_mean * log2_scale + tau * torch.sqrt(variance + _VAR_EPS)


def sink_block_range(tokens: int, sink_start: int | None, sink_tokens: int) -> tuple[int, int]:
    """Half-open block range every query attends exactly, at block granularity."""
    if not sink_tokens:
        return 0, 0
    start = tokens - sink_tokens if sink_start is None else sink_start
    return start // SOL_BLOCK_SIZE, (start + sink_tokens + SOL_BLOCK_SIZE - 1) // SOL_BLOCK_SIZE


def route(
    q_bar: torch.Tensor,
    kc: torch.Tensor,
    threshold: torch.Tensor,
    *,
    scale: float,
    tokens: int,
    sink_start: int | None = None,
    sink_tokens: int = 0,
) -> torch.Tensor:
    """[B, H, n_q_blocks, n_kv_blocks] bool: which blocks are attended exactly.

    A block is exact when its pooled score clears the threshold, when it is within
    one block of the query (the local window), or when it lies in the sink range.
    The kernel takes the column mean of the full query block's scores, which is
    the centroid's score by linearity, so `q_bar` serves directly here.
    """
    log2_scale = scale * _LOG2_E
    pooled = (q_bar @ kc.transpose(-1, -2)) * log2_scale  # [B, H, n_q, n_kv]
    exact = pooled > threshold.unsqueeze(-1)

    n_q, n_kv = pooled.shape[-2], pooled.shape[-1]
    qi = torch.arange(n_q, device=pooled.device).view(n_q, 1)
    ki = torch.arange(n_kv, device=pooled.device).view(1, n_kv)
    exact = exact | ((qi - ki).abs() <= 1)

    lo, hi = sink_block_range(tokens, sink_start, sink_tokens)
    if hi > lo:
        exact = exact | ((ki >= lo) & (ki < hi))
    return exact


def _bf16(x: torch.Tensor) -> torch.Tensor:
    """Round through bf16 and return to fp32, standing in for a device pack."""
    return x.to(torch.bfloat16).float()


@dataclass
class SolAttnResult:
    out: torch.Tensor  # [B, H, T, D]
    exact: torch.Tensor  # [B, H, n_q_blocks, n_kv_blocks] bool
    density: float  # fraction of (query block, kv block) pairs attended exactly


def sol_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    tau: float = SOL_DEFAULT_TAU,
    scale: float | None = None,
    thresh_type: str = "diag",
    sink_start: int | None = None,
    sink_tokens: int = 0,
    block: int = SOL_BLOCK_SIZE,
    emulate_bf16: bool = False,
) -> SolAttnResult:
    """Sol-Attn over [B, H, T, D] Q/K/V. `tau -> -inf` reduces to dense attention.

    `emulate_bf16` rounds at the points a device kernel packs -- the block summaries, the QK logits
    and both probability blocks -- while accumulating in fp32, as DEST does.

    It is a **lower bound on the format loss**, not a faithful device model: this computes the output
    in one shot, so it does not model the per-visit O accumulation or the online-softmax rescale
    chain that a streaming kernel walks hundreds of times per row. Treat a device result that beats
    this reference tightly as evidence the kernel is right, and a device result that misses it as
    needing the gap attributed before blaming the kernel.
    """
    b, h, tokens, d = q.shape
    scale = 1.0 / math.sqrt(d) if scale is None else scale
    log2_scale = scale * _LOG2_E

    kc, vc = block_summaries(k, v, block=block)
    if emulate_bf16:
        kc, vc = _bf16(kc), _bf16(vc)
    q_bar = query_centroids(q, block=block)
    threshold = routing_threshold(q_bar, kc, tau=tau, scale=scale, thresh_type=thresh_type)
    exact = route(q_bar, kc, threshold, scale=scale, tokens=tokens, sink_start=sink_start, sink_tokens=sink_tokens)

    n_q, n_kv = exact.shape[-2], exact.shape[-1]
    lengths = _block_lengths(tokens, block, q.device)  # [n_kv]
    qf, kf, vf = q.float(), k.float(), v.float()

    # Per-token exactness, so the exact branch is one masked softmax over real tokens
    token_exact = exact.repeat_interleave(block, dim=3)[..., :tokens]  # [B, H, n_q, T]
    dense_logits = (qf @ kf.transpose(-1, -2)) * log2_scale  # [B, H, T, T]
    pooled_logits = (qf @ kc.transpose(-1, -2)) * log2_scale  # [B, H, T, n_kv]
    if emulate_bf16:  # the QK pack, which dominates the format loss over hundreds of visits
        dense_logits, pooled_logits = _bf16(dense_logits), _bf16(pooled_logits)

    q_block_of_row = torch.arange(tokens, device=q.device) // block
    exact_rowmask = token_exact.gather(
        2, q_block_of_row.view(1, 1, tokens, 1).expand(b, h, tokens, tokens)
    )  # [B, H, T, T]
    approx_blockmask = ~exact.gather(
        2, q_block_of_row.view(1, 1, tokens, 1).expand(b, h, tokens, n_kv)
    )  # [B, H, T, n_kv]

    neg_inf = torch.finfo(torch.float32).min
    dense_logits = dense_logits.masked_fill(~exact_rowmask, neg_inf)
    pooled_logits = pooled_logits.masked_fill(~approx_blockmask, neg_inf)

    # One shared max across both branches: they share a denominator, so they must
    # share the shift that keeps it finite
    row_max = torch.maximum(dense_logits.max(dim=-1).values, pooled_logits.max(dim=-1).values)
    row_max = row_max.unsqueeze(-1)
    p_exact = torch.exp2(dense_logits - row_max).masked_fill(~exact_rowmask, 0.0)
    p_approx = torch.exp2(pooled_logits - row_max).masked_fill(~approx_blockmask, 0.0)
    if emulate_bf16:
        p_exact, p_approx = _bf16(p_exact), _bf16(p_approx)

    numerator = p_exact @ vf + p_approx @ vc
    denominator = p_exact.sum(dim=-1) + (p_approx * lengths.view(1, 1, 1, n_kv)).sum(dim=-1)
    out = numerator / denominator.unsqueeze(-1)

    valid_pairs = float(n_q * n_kv)
    return SolAttnResult(out=out, exact=exact, density=float(exact.float().sum()) / (b * h * valid_pairs))
