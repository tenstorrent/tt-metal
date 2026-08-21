# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Scoring and reporting layer for ``test_teacher_forcing_e2e.py``.

Everything here turns the two logit row-sets a teacher-forced run produces (TT
and the HF reference) into rates, confidence splits, logit PCCs and the log
output that explains them. None of it drives the device, so it lives beside the
test rather than inside it — the test file keeps the run harness, the floor
table, and the assertions.

Not named ``*_common.py`` like ``decoder_pcc_common``/``tracy_prefill_common``:
those are shared by several test files, whereas this is one test module's report
layer. It is still un-prefixed so pytest does not collect it.

The private names are imported by ``test_teacher_forcing_e2e`` unchanged; they
kept their leading underscore so this stayed a pure move.
"""

from __future__ import annotations

import math
import os

import torch
from loguru import logger

from models.common.utility_functions import comp_pcc
from models.demos.gemma4.demo.text_demo_v2 import _host_sample

# Two-sided confidence level for the Wilson interval printed beside every rate.
_CI_Z = float(os.getenv("GEMMA4_TF_CI_Z", "1.96"))
# Matches TokenAccuracy's top-5 in models/tt_transformers/demo/simple_text_demo.py.
_TOP_K = int(os.getenv("GEMMA4_TF_TOP_K", "5"))
# Per-step token table printed alongside the decoded text; 0 prints text only.
_PRINT_STEPS = int(os.getenv("GEMMA4_TF_PRINT_STEPS", "16"))
# A top-1 flip is only a defect at a CONFIDENT token — one the REFERENCE itself
# decided by more than this logit margin, i.e. HF's own top-1 minus its own
# top-2. Smaller margins are near-ties that bf16 activations and bfp8 weights are
# expected to reorder. Raw match rate alone cannot tell the two apart.
#
# This used to be applied to ``hf[hf_top1] - hf[tt_top1]`` instead, which is a
# monotone function of how far down HF's ordering TT landed, not of HF's
# confidence. Gemma4 softcaps logits to +/-30 over a 262k vocab, so on the
# default corpus the gap from HF's top-1 to its own rank-5 token already exceeds
# 5.0 at roughly three quarters of positions: picking HF's 5th choice scored as a
# "confident divergence" however flat the row was, and the count tracked
# "outside HF top-5" because both were thresholds on the same quantity.
# _log_confidence_split still prints that old number, labelled, so runs stay
# comparable with earlier logs.
_CONFIDENT_GAP = float(os.getenv("GEMMA4_TF_CONFIDENT_GAP", "5.0"))
# Width of the decision-relevant logit comparison. Full-vocab PCC spreads its
# weight over 262k mostly-irrelevant entries and is further compressed by the
# tanh final_logit_softcapping (30.0), so it is a weak discriminator of sampling
# behaviour; KL and the max |delta logit| over HF's top-K are not.
_LOGIT_CMP_K = int(os.getenv("GEMMA4_TF_LOGIT_CMP_K", "32"))
# Number of equal-width position bins in the drift report.
_TREND_BINS = int(os.getenv("GEMMA4_TF_TREND_BINS", "4"))
# Token selection uses the demo's sampler (text_demo_v2._host_sample) with the
# demo's defaults, applied to TT and HF logits alike, so "top-1" on either side
# is the token text_demo_v2 would emit rather than a bare torch argmax. At
# temperature 0 _host_sample is argmax and top_p is never consulted.
_TEMPERATURE = float(os.getenv("GEMMA4_TF_TEMPERATURE", "0"))
_TOP_P = float(os.getenv("GEMMA4_TF_TOP_P", "0.08"))


def _greedy_tokens(rows):
    """Pick one token per row with the demo's sampler.

    ``_host_sample`` treats dim 0 as batch, so a whole [N, vocab] block samples
    in one call; at ``_TEMPERATURE == 0`` it reduces to argmax. Used for TT and
    HF alike so neither side gets a different selection rule.
    """
    return _host_sample(rows, _TEMPERATURE, _TOP_P).reshape(-1)


def _vs_truth(rows, truth):
    """(top-1, top-k) hit rate of ``rows`` against the ground-truth next tokens."""
    if rows.shape[0] == 0:
        return (0.0, 0.0)
    top1 = _greedy_tokens(rows)
    topk = rows.topk(_TOP_K, dim=-1).indices
    return (
        (top1 == truth).float().mean().item(),
        (topk == truth.unsqueeze(-1)).any(dim=-1).float().mean().item(),
    )


def _rates(tt_rows, hf_rows):
    """TT-vs-HF top-1 equality and top-k containment, per prediction.

    Both top-1 columns come from ``_greedy_tokens``, i.e. the demo's sampler on
    the same settings, so the comparison isolates the logits rather than mixing
    in two different selection rules.
    """
    tt_top1 = _greedy_tokens(tt_rows)
    hf_top1 = _greedy_tokens(hf_rows)
    hf_topk = hf_rows.topk(_TOP_K, dim=-1).indices
    return {
        "tt_top1": tt_top1,
        "hf_top1": hf_top1,
        "hf_topk": hf_topk,
        "agree_top1": tt_top1 == hf_top1,
        "agree_topk": (hf_topk == tt_top1.unsqueeze(-1)).any(dim=-1),
    }


def _log_segment(label, r, tt_rows, hf_rows, truth, sl):
    """One report row: TT-vs-HF and both vs-truth rates over a slice of predictions."""
    agree_top1 = r["agree_top1"][sl]
    if agree_top1.numel() == 0:
        return
    agree_topk = r["agree_topk"][sl]
    tt1, ttk = _vs_truth(tt_rows[sl], truth[sl])
    hf1, hfk = _vs_truth(hf_rows[sl], truth[sl])
    logger.info(
        "  {:<8} n={:<5} TT-vs-HF top1={:>7.2%} top{}={:>7.2%}  |  "
        "TT-vs-truth top1={:>7.2%} top{}={:>7.2%}  |  HF-vs-truth top1={:>7.2%} top{}={:>7.2%}",
        label,
        agree_top1.numel(),
        agree_top1.float().mean().item(),
        _TOP_K,
        agree_topk.float().mean().item(),
        tt1,
        _TOP_K,
        ttk,
        hf1,
        _TOP_K,
        hfk,
    )


def _log_generated_text(tokenizer, tt_top1, hf_top1, truth, *, per_step_limit=None):
    """Print the teacher-forced predictions as text, from TT and from HF alike.

    Every prediction is made from ground-truth context, so these strings are not
    a free-running generation: they are the token each side would have emitted
    at each position of the reference text. Concatenated they read as the
    teacher-forced continuation, which is what makes TT and HF directly
    comparable line by line. ``per_step_limit`` (env ``GEMMA4_TF_PRINT_STEPS``,
    0 to disable) also prints the first steps token by token.
    """

    def _decode(ids):
        return tokenizer.decode([int(t) for t in ids])

    logger.info("-" * 78)
    logger.info("Teacher-forced output (each token predicted from ground-truth context):")
    logger.info("  reference (HF) : {!r}", _decode(hf_top1))
    logger.info("  device    (TT) : {!r}", _decode(tt_top1))
    logger.info("  ground truth   : {!r}", _decode(truth))

    limit = _PRINT_STEPS if per_step_limit is None else per_step_limit
    if limit <= 0:
        return
    n = min(limit, len(tt_top1))
    logger.info("  first {} predictions, token by token:", n)
    logger.info("      {:<11} {:<16} {:<16} {:<16} {}", "step", "truth", "HF", "TT", "match")
    for i in range(n):
        where = "prefill" if i == 0 else f"decode[{i - 1}]"
        logger.info(
            "      {:<11} {!r:<16} {!r:<16} {!r:<16} {}",
            where,
            _decode([truth[i]]),
            _decode([hf_top1[i]]),
            _decode([tt_top1[i]]),
            "ok" if int(tt_top1[i]) == int(hf_top1[i]) else "MISMATCH",
        )
    if len(tt_top1) > n:
        logger.info("      ... {} more predictions omitted", len(tt_top1) - n)


def _wilson(successes, n, z=_CI_Z):
    """Wilson score interval for a binomial proportion.

    Printed beside every rate because the assertion compares a rate against a
    fixed floor, and what the test can actually resolve is set by ``n``: at the
    default 501 predictions a 95% interval is roughly +/-4 points wide, so a rate
    within ~2 points of a floor is neither a pass nor a fail — it is an
    unresolved measurement, and reading it as a defect sends the next session
    chasing noise. At the older 129-prediction cases the interval is +/-7 points,
    which is wider than every A/B difference those runs were used to judge.

    Teacher-forced positions are correlated (neighbouring tokens share context),
    so the true interval is wider than this independent-Bernoulli one. Treat the
    printed span as a lower bound on the uncertainty.
    """
    if n <= 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _reference_confidence(hf_rows):
    """The reference's OWN per-position confidence: ``(margin, entropy)``.

    ``margin`` is HF's top-1 minus HF's top-2 logit; ``entropy`` is the entropy of
    HF's own distribution. Neither involves TT.

    This is the control the position-trend report needs. A falling TT-vs-HF rate
    has two unrelated causes — device state accumulating with position, or the
    reference simply becoming less decisive over that stretch of text — and they
    are indistinguishable from the agreement rate alone. On the default corpus
    the second effect is large: between the first and last position bin the
    median margin more than halves and the entropy roughly triples, because the
    text moves from a chapter list into prose the model is genuinely unsure
    about. A trend line read without these columns attributes that to the KV
    cache.
    """
    top2 = hf_rows.topk(2, dim=-1).values
    margin = top2[:, 0] - top2[:, 1]
    logprob = hf_rows.log_softmax(dim=-1)
    entropy = -(logprob.exp() * logprob).sum(dim=-1)
    return margin, entropy


def _tt_rank_under_hf(hf_rows, tt_top1):
    """1-based rank of TT's pick in HF's own ordering, per prediction.

    Rank 1 means the two agree. This is the severity of a flip — how far down the
    reference's ordering TT landed — and is reported separately from whether the
    reference was confident, because the two answer different questions and were
    previously collapsed into one number.
    """
    picked = hf_rows.gather(-1, tt_top1.unsqueeze(-1))
    return (hf_rows > picked).sum(dim=-1) + 1


def _confidence_gaps(hf_rows, tt_top1, hf_top1):
    """SUPERSEDED metric: HF's logit at its own pick minus its logit at TT's pick.

    ``hf_rows[i, hf_top1[i]] - hf_rows[i, tt_top1[i]]`` — zero where the two
    agree. Kept, and still printed by ``_log_confidence_split``, so numbers stay
    comparable with logs from before the classifier changed.

    Do not gate on it. It measures how far down HF's ordering TT landed, not how
    confident HF was, and those diverge badly on a softcapped 262k-wide
    distribution: see the ``_CONFIDENT_GAP`` comment and ``_tt_rank_under_hf``.
    Use ``_reference_confidence`` for confidence and ``_tt_rank_under_hf`` for
    severity.
    """
    idx = torch.stack([hf_top1, tt_top1], dim=-1).long()
    picked = hf_rows.gather(-1, idx)
    return picked[:, 0] - picked[:, 1]


def _decision_metrics(tt_rows, hf_rows):
    """Per-step KL(HF || TT) over the full row, and max |delta logit| on HF's top-K.

    Both are read on the distribution that sampling actually consults, so they
    move with real behaviour rather than with the saturated tail that dominates
    a 262k-wide PCC.
    """
    hf_logprob = torch.log_softmax(hf_rows, dim=-1)
    tt_logprob = torch.log_softmax(tt_rows, dim=-1)
    kl = (hf_logprob.exp() * (hf_logprob - tt_logprob)).sum(dim=-1)

    k = min(_LOGIT_CMP_K, hf_rows.shape[-1])
    idx = hf_rows.topk(k, dim=-1).indices
    max_delta = (hf_rows.gather(-1, idx) - tt_rows.gather(-1, idx)).abs().amax(dim=-1)
    return kl, max_delta


def _log_confidence_split(r, hf_rows, truth, tokenizer):
    """Split top-1 flips by the REFERENCE's own confidence; report severity apart.

    Two independent questions, previously collapsed into one number:

      * **Should this flip have happened?** Answered by HF's own top-1 minus
        top-2 margin. A flip at a position the reference itself decided by 0.3
        logits is a near-tie that no dtype or fidelity knob will remove; a flip
        where the reference was decisive is a divergence worth chasing.
      * **How badly did TT miss?** Answered by the rank of TT's pick in HF's
        ordering. That is severity, not diagnosis — a rank-8 pick in a flat row
        is still a near-tie.

    Gating on ``hf[hf_top1] - hf[tt_top1]`` conflated them, and on this corpus the
    conflation dominates: the gap to HF's own rank-5 token exceeds 5.0 logits at
    roughly three quarters of positions, so "confident" counted rank rather than
    confidence and moved in lockstep with "outside HF top-5". Both the new split
    and the old count are printed.

    Returns the number of flips where the reference was confident.
    """
    ref_margin, _ = _reference_confidence(hf_rows)
    reach = _confidence_gaps(hf_rows, r["tt_top1"], r["hf_top1"])
    rank = _tt_rank_under_hf(hf_rows, r["tt_top1"])
    flipped = ~r["agree_top1"]
    n_flip = int(flipped.sum())
    n_total = r["agree_top1"].numel()

    confident = flipped & (ref_margin > _CONFIDENT_GAP)
    n_confident = int(confident.sum())

    logger.info("-" * 78)
    logger.info("Flip classification (reference's own top1-top2 margin > {:.2f} logits):", _CONFIDENT_GAP)
    logger.info("  top-1 flips        : {}/{}", n_flip, n_total)
    logger.info(
        "      near-tie       : {:<5} (reference margin <= {:.2f} — two numeric paths reordering near-equal "
        "logits; not fixable by dtype or fidelity)",
        n_flip - n_confident,
        _CONFIDENT_GAP,
    )
    logger.info(
        "      confident      : {:<5} (reference was decisive — chase these)",
        n_confident,
    )
    if n_flip:
        logger.info(
            "  reference margin on flips : mean={:.3f} median={:.3f} max={:.3f}",
            ref_margin[flipped].mean().item(),
            ref_margin[flipped].median().item(),
            ref_margin[flipped].max().item(),
        )
        logger.info(
            "  rank of TT's pick under HF (severity, not confidence): median={} p90={} max={}",
            int(rank[flipped].median()),
            int(rank[flipped].float().quantile(0.9)),
            int(rank[flipped].max()),
        )
    outside = int((~r["agree_topk"]).sum())
    logger.info("  outside HF top-{}   : {:<5} (TT's pick below the reference's {}th choice)", _TOP_K, outside, _TOP_K)

    # The pre-change number, so a run can still be lined up against older logs.
    n_old = int((flipped & (reach > _CONFIDENT_GAP)).sum())
    logger.info(
        "  superseded metric  : {:<5} flips with hf[hf_top1]-hf[tt_top1] > {:.2f} — reported for continuity with "
        "logs before the classifier changed; it tracks rank, not confidence",
        n_old,
        _CONFIDENT_GAP,
    )

    idx = confident.nonzero(as_tuple=True)[0].tolist()
    for i in idx[:8]:
        where = "prefill" if i == 0 else f"decode[{i - 1}]"
        logger.info(
            "      {:<11} ref_margin={:>7.3f} tt_rank={:<5} truth={!r:<14} HF={!r:<14} TT={!r}",
            where,
            ref_margin[i].item(),
            int(rank[i]),
            tokenizer.decode([int(truth[i])]),
            tokenizer.decode([int(r["hf_top1"][i])]),
            tokenizer.decode([int(r["tt_top1"][i])]),
        )
    if len(idx) > 8:
        logger.info("      ... {} more confident flips", len(idx) - 8)
    return n_confident


def _log_decision_metrics(tt_rows, hf_rows):
    """Report KL and top-K logit distance beside the PCC numbers."""
    kl, max_delta = _decision_metrics(tt_rows, hf_rows)
    logger.info("-" * 78)
    logger.info("Decision-relevant logit distance (top-{} window):", _LOGIT_CMP_K)
    logger.info("  KL(HF||TT)        mean={:.6f} max={:.6f}", kl.mean().item(), kl.max().item())
    logger.info(
        "  max|dlogit| top-{} mean={:.4f} max={:.4f}", _LOGIT_CMP_K, max_delta.mean().item(), max_delta.max().item()
    )


def _log_position_trend(r, pccs, hf_rows, truth):
    """Agreement and PCC per position bin, BESIDE the reference's own confidence.

    A falling agreement rate is often read as state accumulating with position —
    the KV cache, the page table, position handling. It only supports that
    reading if the reference's own difficulty is flat across the same bins, and
    on this corpus it is not: the right-hand columns show HF's own next-token
    accuracy, its median top-1/top-2 margin and its entropy per bin. When the
    reference's margin halves over the same span, a falling agreement rate is the
    expected consequence of comparing two numeric paths on near-ties, with no
    device drift involved.

    So read the two halves together:

      * agreement falls, reference margin flat  → positional drift; look at KV
        writes, page table, position handling.
      * agreement falls, reference margin falls → text difficulty; the trend line
        says nothing about the device. Compare bins at similar margin instead, or
        change corpus.
      * agreement flat                          → per-step numerics, which is
        what the dtype and fidelity knobs move.
    """
    n = r["agree_top1"].numel()
    bins = max(1, min(_TREND_BINS, n))
    width = (n + bins - 1) // bins
    margin, entropy = _reference_confidence(hf_rows)
    hf_hit = r["hf_top1"] == truth

    logger.info("-" * 78)
    logger.info("Position trend ({} bins over {} predictions):", bins, n)
    logger.info(
        "{}",
        f"  {'steps':<12}{'top1':>8}{'top' + str(_TOP_K):>8}{'PCC mean':>11}{'PCC min':>10}"
        f"   |{'HF-truth':>10}{'margin med':>12}{'m<1':>7}{'entropy':>9}",
    )
    for b in range(bins):
        lo, hi = b * width, min((b + 1) * width, n)
        if lo >= hi:
            continue
        seg_pcc = pccs[lo:hi]
        seg_margin = margin[lo:hi]
        logger.info(
            "{}",
            f"  {f'{lo}-{hi - 1}':<12}"
            f"{r['agree_top1'][lo:hi].float().mean().item():>7.2%} "
            f"{r['agree_topk'][lo:hi].float().mean().item():>7.2%} "
            f"{sum(seg_pcc) / len(seg_pcc):>10.6f}"
            f"{min(seg_pcc):>10.6f}"
            f"   |{hf_hit[lo:hi].float().mean().item():>9.2%} "
            f"{seg_margin.median().item():>11.3f}"
            f"{(seg_margin < 1.0).float().mean().item():>7.2f}"
            f"{entropy[lo:hi].mean().item():>9.3f}",
        )
    logger.info(
        "  right of the bar is the reference alone (no TT): its own accuracy, its own top1-top2 margin, the "
        "fraction of positions it decided by under 1 logit, and its entropy."
    )
    logger.info(
        "  A left-hand fall that tracks a right-hand fall is text difficulty, not positional drift — see the "
        "docstring."
    )


def _log_resolution(r, min_top1, min_top5, provenance, inherited):
    """Print each rate with its confidence interval next to the floor it is gated on.

    The point of this block is to make an unresolved measurement look unresolved.
    A rate whose interval straddles the floor is neither a pass nor a fail, and
    reporting it as a failure is what sends the next session hunting a defect that
    the sample size cannot establish.
    """
    n = r["agree_top1"].numel()
    top1_hits = int(r["agree_top1"].sum())
    topk_hits = int(r["agree_topk"].sum())
    t1_lo, t1_hi = _wilson(top1_hits, n)
    tk_lo, tk_hi = _wilson(topk_hits, n)

    logger.info("-" * 78)
    logger.info("Resolution (Wilson interval, z={:.2f}, n={}):", _CI_Z, n)
    logger.info(
        "  top-1  {:>7.2%}  CI [{:>7.2%}, {:>7.2%}]  floor {:>7.2%}  {}",
        top1_hits / n,
        t1_lo,
        t1_hi,
        min_top1,
        (
            "RESOLVED pass"
            if t1_lo >= min_top1
            else ("RESOLVED fail" if t1_hi < min_top1 else "UNRESOLVED — CI straddles the floor")
        ),
    )
    logger.info(
        "  top-{}  {:>7.2%}  CI [{:>7.2%}, {:>7.2%}]  floor {:>7.2%}  {}",
        _TOP_K,
        topk_hits / n,
        tk_lo,
        tk_hi,
        min_top5,
        (
            "RESOLVED pass"
            if tk_lo >= min_top5
            else ("RESOLVED fail" if tk_hi < min_top5 else "UNRESOLVED — CI straddles the floor")
        ),
    )
    logger.info("  one prediction is {:.2f} percentage points at this n", 100.0 / n)
    logger.info("  floors from : {}", provenance)
    if inherited:
        logger.warning(
            "  This length case has no measured floor of its own — it is being gated on numbers from a "
            "different length. Add a _MEASURED_FLOORS row after remeasuring on T3K; do not edit an existing "
            "row to make this case pass."
        )
    logger.info(
        "  Positions are correlated through the text, so the true interval is wider than this "
        "independent-Bernoulli one."
    )


def _logit_pccs(tt_rows, hf_rows):
    """Per-step logit PCC (TT vs HF) over the full vocab row."""
    pccs = []
    for i in range(tt_rows.shape[0]):
        _, pcc = comp_pcc(hf_rows[i], tt_rows[i], pcc=0.0)
        pccs.append(float(pcc))
    return pccs


def _log_pcc_segment(label, pccs, sl):
    """Summarize min/mean logit PCC over a slice of teacher-forced steps."""
    seg = pccs[sl]
    if not seg:
        return
    logger.info(
        "  {:<8} n={:<5} logit_PCC min={:.6f} mean={:.6f}",
        label,
        len(seg),
        min(seg),
        sum(seg) / len(seg),
    )


def _log_per_step_pcc(pccs, *, limit=8):
    """Print per-step PCC and highlight the worst step."""
    logger.info("  per-step logit PCC (TT vs HF):")
    for i, pcc in enumerate(pccs[:limit]):
        where = "prefill" if i == 0 else f"decode[{i - 1}]"
        logger.info("      {:<11} PCC={:.6f}", where, pcc)
    if len(pccs) > limit:
        logger.info("      ... {} more steps omitted", len(pccs) - limit)
    worst_i, worst_pcc = min(enumerate(pccs), key=lambda x: x[1])
    worst_where = "prefill" if worst_i == 0 else f"decode[{worst_i - 1}]"
    logger.info("  worst step {:<11} PCC={:.6f}", worst_where, worst_pcc)
