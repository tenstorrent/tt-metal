# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end teacher-forced evaluation for Qwen3.5-9B / Qwen3.6-27B.

Drives the real generation path — ``prefill_paged`` followed by the demo's decode
chain (``prepare_inputs_decode`` → ``ttnn_decode_forward`` →
``process_output_decode``) — but replaces the sampled token at every decode step
with the ground-truth token. Because the model is re-anchored to truth each step, a
single bad prediction cannot derail the rest of the sequence, so every step is an
independent measurement instead of one diverged sample. This is the teacher-forcing
method ``TokenAccuracy`` implements for tt_transformers
(``models/tt_transformers/demo/simple_text_demo.py``).

Where this sits relative to the other qwen tests: ``tests/unit/test_prefill.py`` and
``tests/unit/test_decode.py`` gate a 128-token prompt and 5 decode steps against HF.
Five steps cannot show accuracy *drift* — error that accumulates through the paged KV
and the GDN recurrent/conv state as the sequence grows. This test runs hundreds of
steps and reports the trend, which is the failure mode a short test structurally
cannot see.

Ground truth defaults to *A Tale of Two Cities* (the corpus
``models/tt_transformers/tests/generate_reference_hf.py`` uses). Override with
``QWEN36_TF_TEXT_FILE``.

Predictions scored, for ``prefill_len`` prompt tokens and ``max_new_tokens`` steps:

  * **prefill** — 1 prediction. Prefill over tokens ``0..prefill_len-1`` predicts
    token ``prefill_len``.
  * **decode**  — ``max_new_tokens`` predictions. Step ``j`` is fed ground-truth
    token ``prefill_len+j`` at position ``prefill_len+j`` and predicts
    ``prefill_len+j+1``.
  * **e2e**     — the ``max_new_tokens+1`` combined.

Each is reported top-1 and top-5 against two references:

  * **vs HF** — TT's token equals HF's token (top-1), and TT's token is inside HF's
    top-5. This is the device-correctness signal and what the accuracy test asserts.
  * **vs truth** — the token actually next in the text, for TT and HF alike.
    Informational: it measures the checkpoint, not the device.

Reading the pair: top-1 falling while top-5 holds means TT is reordering near-ties,
the ordinary consequence of bfp8 weights. Top-5 falling too means TT is emitting
tokens the reference does not rank at all — a real divergence, not rounding.

Both sides pick tokens by argmax, which is what the qwen demo does at its default
``QWEN35_TEMP=0`` (its sampler is an inner closure with repetition-penalty knobs that
do not apply under teacher forcing, so it is not imported here).

Diagnostics printed beyond the raw rates, because a sub-100% rate has two unrelated
causes that want opposite fixes:

  * **flip classification** — every top-1 flip scored by the *reference's own* top-1
    minus top-2 margin. Flips inside ``QWEN36_TF_CONFIDENT_GAP`` (default 5.0 logits)
    are positions the reference itself barely decided, which two numeric paths are
    expected to reorder; only confident flips indicate a defect. The rank of TT's
    pick under HF is printed separately, as severity.
  * **position trend** — accuracy and PCC per position bin, beside the reference's
    own accuracy, margin and entropy over the same bins. Agreement falling while the
    reference's margin stays flat means state accumulating with position (paged KV,
    GDN recurrent/conv carry, page table); both falling together means the text got
    harder and the trend says nothing about the device; flat means per-step numerics.
  * **resolution** — each rate with its Wilson interval next to the floor it is gated
    on. A rate whose interval straddles the floor is an unresolved measurement, not a
    defect. Floors are keyed per length case so changing the lengths cannot silently
    inherit a floor calibrated elsewhere.
  * **decision-relevant distance** — KL(HF||TT) and max |delta logit| over HF's
    top-``QWEN36_TF_LOGIT_CMP_K``. Full-vocab PCC spreads its weight over 248320
    mostly-irrelevant entries, so it tracks sampling behaviour poorly.

Both tests also print the teacher-forced output itself — TT's token sequence and
HF's, decoded and shown beside the ground truth. Every prediction is anchored to
truth, so the three strings align position by position.
``QWEN36_TF_PRINT_STEPS`` (default 16, 0 to disable) also prints the first steps as a
truth / HF / TT token table.

Measured (Wormhole, bf16 HF reference, prefill_128 / max_new_tokens_128)
------------------------------------------------------------------------
* **9B / N300 (TP=2), 32 layers** — top-1 90.70%, top-5 96.12%, logit PCC mean
  0.9674, worst step 0.5763. Agreement declines only in the bins where the
  reference's OWN margin declines (5.1 → 1.7 logits, entropy tripling), i.e. text
  difficulty, not device drift.
* **27B / T3K (TP=8), 64 layers** — top-1 75.97%, top-5 84.50%, logit PCC mean
  0.8403, worst step 0.3957, and a drift signature the 9B does not show:
  agreement falls 96.97% → 43.33% across the four position bins while the
  reference's own margin RISES (7.75 → 9.75) and its entropy falls to 0.05. 20 of
  31 top-1 flips are at positions the reference decided by a median 8.4 logits,
  TT's pick ranking as far down as 751. ``tests/unit/test_decode.py`` (5 steps)
  passes on the same checkpoint at min PCC 0.9595 with every argmax matching, so
  this is accumulation over ~130 decode steps rather than per-step numerics —
  exactly what a short test cannot see. Two caveats before calling it a device
  defect: it was measured at TP=8 on Wormhole T3K, while the 27B's validated
  configuration is P150x4 (TP=4, Blackhole), and it has not yet been bisected to
  the paged KV vs the GDN recurrent/conv carry. The floors in ``_MEASURED_FLOORS``
  record this state so a change is visible; they are not a statement that it is
  acceptable.

A single step's full-vocab PCC over 248320 logits is a coarse instrument — read
the mean, the position trend and the flip classification, not one worst step.

Tests in this module:

  * ``test_teacher_forcing_e2e``        — top-1 / top-5 token accuracy.
  * ``test_teacher_forcing_logits_pcc`` — full-vocab logit PCC at every step.

Run (use ``--timeout=0``; both models, one test — ``HF_MODEL`` picks the checkpoint
and ``MESH_DEVICE`` the mesh, same as the demo):

  # 9B
  HF_MODEL=Qwen/Qwen3.5-9B MESH_DEVICE=P150 \
    pytest models/demos/blackhole/qwen36/tests/e2e/test_teacher_forcing_e2e.py -sv --timeout=0

  # 27B
  HF_MODEL=Qwen/Qwen3.6-27B MESH_DEVICE=P150x4 \
    pytest models/demos/blackhole/qwen36/tests/e2e/test_teacher_forcing_e2e.py -sv --timeout=0

  # accuracy only — select by node id, NOT ``-k``: the module filename is itself a
  # keyword, so ``-k test_teacher_forcing_e2e`` also matches the PCC test.
  pytest models/demos/blackhole/qwen36/tests/e2e/test_teacher_forcing_e2e.py::test_teacher_forcing_e2e -sv

  # longer horizon (gemma4's default case)
  QWEN36_TF_PREFILL_LEN=512 QWEN36_TF_MAX_NEW_TOKENS=500 pytest ... -sv --timeout=0
"""

from __future__ import annotations

import bz2
import math
import os
from pathlib import Path

import pytest
import torch
from loguru import logger

from models.common.utility_functions import comp_pcc
from models.demos.blackhole.qwen36.tests.unit.full_depth_pcc_common import (
    BLOCK_SIZE,
    allocate_paged_kv,
    build_full_depth_model,
    parametrize_full_depth,
    tt_decode_logits,
    tt_prefill_logits,
)

# Prefill length + decode steps; total tokens consumed is prefill_len +
# max_new_tokens + 1 (the last decode step still needs a ground-truth target).
# Overridable so a longer horizon can be run without editing the file — the id is
# built from the effective values, so the floor lookup can never drift from the
# lengths actually run.
_PREFILL_LEN = int(os.getenv("QWEN36_TF_PREFILL_LEN", "128"))
_MAX_NEW_TOKENS = int(os.getenv("QWEN36_TF_MAX_NEW_TOKENS", "128"))
_TF_LENGTHS = [
    pytest.param(_PREFILL_LEN, _MAX_NEW_TOKENS, id=f"prefill_{_PREFILL_LEN}-max_new_tokens_{_MAX_NEW_TOKENS}"),
]

# Floors for the assertions — REGRESSION DETECTORS at the model, mesh and length they
# were measured at, not correctness targets. Keyed by ``(model_key, case_id)`` so that
# changing either the checkpoint or the lengths cannot silently inherit numbers
# calibrated elsewhere; an unlisted combination falls back to ``_FALLBACK_FLOORS`` and
# the run logs a warning saying so.
#
# Each rate floor is the WILSON LOWER BOUND of its measurement at this n, not the
# measurement itself: a floor inside the interval makes every healthy run print
# "UNRESOLVED — CI straddles the floor", which teaches the reader to skip the block that
# exists to catch exactly that. At the lower bound a good run reads RESOLVED pass, and a
# regression of more than the sample size can absorb is what flips it. Raising n (more
# decode steps) narrows the interval and would let the floors move up.
#
# Why per model, when pcc_thresholds.json is deliberately flat and function-keyed: the
# two checkpoints do not behave the same over a long teacher-forced run (see the 27B
# row), and one shared number would either bless the weaker configuration or fail the
# stronger one. These floors also need provenance text that a flat JSON cannot carry,
# which is why they live here rather than in that table.
_MEASURED_FLOORS = {
    ("32L-4096", "prefill_128-max_new_tokens_128"): (
        0.84,
        0.91,
        0.55,
        "Qwen3.5-9B WH N300 1x2 (measured top1 90.70%, top5 96.12%, worst step PCC 0.5763); "
        "agreement declines only where the reference's own margin declines — no positional drift",
    ),
    ("64L-5120", "prefill_128-max_new_tokens_128"): (
        0.87,
        0.93,
        0.75,
        "Qwen3.6-27B WH T3K 1x8 (measured top1 93.02%, top5 97.67%, worst step PCC 0.8768). "
        "RESOLVED: the earlier 75.97%/84.50%/0.3957 state was a MISSING PAGED KV-CACHE WRITE in "
        "TPAttention.forward_decode -- the `not wh_9b_n300` branch prepared k_sh/v_sh and never "
        "called paged_update_cache, so decode attended over a cache holding only the prefill "
        "tokens with every generated token's slot left at zero. That is why agreement fell "
        "96.97% -> 43.33% across position bins (each step lost more history) while the reference "
        "grew more decisive, and why the 9B was unaffected (it takes the wh_9b_n300 branch, which "
        "does write). Fixing it removed the position trend entirely (bins now 93.94 / 93.94 / "
        "87.88 / 96.67%, the last bin the best) and the 27B now exceeds the 9B's 90.70%. "
        "TT-vs-truth 89.84% now sits just under HF-vs-truth 93.75%, i.e. the residual gap is "
        "ordinary numerics, not a structural defect. Do not edit these floors down if agreement "
        "worsens -- a drop means a regression",
    ),
}
_FALLBACK_FLOORS = (0.75, 0.92, 0.55, "NOT measured for this model/length — inherited default")

# Env overrides win over the table. ``None`` means "use the table".
_MIN_TOP1_ENV = os.getenv("QWEN36_TF_MIN_TOP1")
_MIN_TOP5_ENV = os.getenv("QWEN36_TF_MIN_TOP5")
_MIN_LOGIT_PCC_ENV = os.getenv("QWEN36_TF_MIN_LOGIT_PCC")

# Two-sided confidence level for the Wilson interval printed beside every rate.
_CI_Z = float(os.getenv("QWEN36_TF_CI_Z", "1.96"))

# Matches TokenAccuracy's top-5 in models/tt_transformers/demo/simple_text_demo.py.
_TOP_K = int(os.getenv("QWEN36_TF_TOP_K", "5"))

# Per-step token table printed alongside the decoded text; 0 prints text only.
_PRINT_STEPS = int(os.getenv("QWEN36_TF_PRINT_STEPS", "16"))

# A top-1 flip is only a defect at a CONFIDENT token — one the REFERENCE itself
# decided by more than this logit margin (its own top-1 minus its own top-2). Smaller
# margins are near-ties that bf16 activations and bfp8 weights are expected to
# reorder, and a raw match rate cannot tell the two apart.
_CONFIDENT_GAP = float(os.getenv("QWEN36_TF_CONFIDENT_GAP", "5.0"))

# Width of the decision-relevant logit comparison. Full-vocab PCC spreads its weight
# over 248320 mostly-irrelevant entries, so it is a weak discriminator of sampling
# behaviour; KL and max |delta logit| over HF's top-K are not.
_LOGIT_CMP_K = int(os.getenv("QWEN36_TF_LOGIT_CMP_K", "32"))

# Number of equal-width position bins in the drift report.
_TREND_BINS = int(os.getenv("QWEN36_TF_TREND_BINS", "4"))

_TALE_OF_TWO_CITIES = Path(__file__).resolve().parents[5] / "tt_transformers" / "tests" / "tale-of-two-cities.txt.bz2"

# Used only when the Tale corpus is unavailable (e.g. sparse checkout).
_FALLBACK_TEXT = """The history of computing hardware begins with mechanical devices built to
automate arithmetic. Charles Babbage designed the Analytical Engine in the nineteenth century,
and Ada Lovelace wrote what is now regarded as the first algorithm intended for a machine."""


def _load_text():
    """Teacher-forcing text: ``QWEN36_TF_TEXT_FILE`` → Tale of Two Cities → fallback."""
    path = os.getenv("QWEN36_TF_TEXT_FILE")
    if path:
        with open(path) as f:
            return f.read()
    if _TALE_OF_TWO_CITIES.is_file():
        with bz2.open(_TALE_OF_TWO_CITIES, "rt", encoding="utf-8") as f:
            return f.read()
    logger.warning("Tale of Two Cities corpus not found at {}; using fallback text", _TALE_OF_TWO_CITIES)
    return _FALLBACK_TEXT


def _build_tokens(tokenizer, total_len):
    """Ground-truth sequence of exactly ``total_len`` tokens, repeating text if short.

    Repetition is harmless under teacher forcing: every position is re-anchored to
    truth, so predictions stay independent, and TT and HF see identical input either
    way.
    """
    ids = tokenizer.encode(_load_text(), add_special_tokens=True)
    if len(ids) < total_len:
        body = tokenizer.encode(_load_text(), add_special_tokens=False)
        while len(ids) < total_len:
            ids.extend(body)
    return torch.tensor(ids[:total_len], dtype=torch.long).unsqueeze(0)


# ── the two sides ─────────────────────────────────────────────────────────


def _run_teacher_forced_tt(model, page_table, tokens, prefill_len, max_new_tokens):
    """Prefill the prompt, then decode ``max_new_tokens`` steps feeding truth back.

    Returns ``[1 + max_new_tokens, vocab]``: the prefill row followed by one row per
    decode step, in order. Feeding the ground-truth token rather than TT's own pick is
    the teacher forcing; what still accumulates across steps is device state — the
    paged KV and the GDN recurrent/conv carry — which is the point of running hundreds
    of steps rather than five.
    """
    rows = [tt_prefill_logits(model, tokens[:, :prefill_len], page_table)]
    for j in range(max_new_tokens):
        forced = int(tokens[0, prefill_len + j])
        rows.append(tt_decode_logits(model, forced, prefill_len + j, page_table))
        if (j + 1) % 64 == 0:
            logger.info("  teacher-forced decode {}/{} steps", j + 1, max_new_tokens)
    return torch.stack(rows, dim=0)


def _hf_reference_rows(ckpt_dir, tokens, prefill_len, max_new_tokens):
    """HF logits rows aligned to the TT predictions.

    TT prediction ``i`` predicts token ``prefill_len + i``, which HF produces at row
    ``prefill_len + i - 1``. ONE HF forward covers every row, because HF prefill is
    itself teacher-forced by causal masking — no decode loop needed on the reference
    side, which is what keeps a 500-step run affordable on CPU.
    """
    from transformers.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    end = prefill_len + max_new_tokens
    ref_dtype = getattr(torch, os.environ.get("QWEN36_FULL_DEPTH_REF_DTYPE", "bfloat16"))
    logger.info("Loading HF reference ({}) from {} ...", ref_dtype, ckpt_dir)
    text_config = Qwen3_5TextConfig.from_pretrained(ckpt_dir)
    hf_model, loading_info = Qwen3_5ForCausalLM.from_pretrained(
        ckpt_dir, config=text_config, dtype=ref_dtype, output_loading_info=True
    )
    # A composite 3.6 VLM checkpoint carries visual.*/mtp.* the text-only class does
    # not want (unexpected keys are fine); a MISSING key means a weight stayed at its
    # random init and every rate below would be measured against noise.
    assert not loading_info["missing_keys"], f"HF reference has uninitialized weights: {loading_info['missing_keys']}"
    hf_model.eval()
    try:
        with torch.no_grad():
            out = hf_model(tokens[:, :end].long())
        return out.logits[0, prefill_len - 1 : end, :].float()
    finally:
        del hf_model


# ── scoring ───────────────────────────────────────────────────────────────


def _greedy_tokens(rows):
    """One token per row, argmax — the qwen demo's default (``QWEN35_TEMP=0``).

    Used for TT and HF alike so the comparison isolates the logits rather than mixing
    in two different selection rules.
    """
    return rows.argmax(dim=-1).reshape(-1)


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
    """TT-vs-HF top-1 equality and top-k containment, per prediction."""
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


def _logit_pccs(tt_rows, hf_rows):
    """Per-step full-vocab logit PCC (TT vs HF)."""
    return [float(comp_pcc(hf_rows[i], tt_rows[i], pcc=0.0)[1]) for i in range(tt_rows.shape[0])]


def _wilson(successes, n, z=_CI_Z):
    """Wilson score interval for a binomial proportion.

    Printed beside every rate because the assertion compares a rate against a fixed
    floor, and what the test can actually resolve is set by ``n``: at 129 predictions
    a 95% interval is roughly ±7 points wide, so a rate within a few points of a floor
    is neither a pass nor a fail — it is an unresolved measurement, and reading it as
    a defect sends the next session chasing noise. Teacher-forced positions are
    correlated (neighbouring tokens share context), so the true interval is wider than
    this independent-Bernoulli one; treat the printed span as a lower bound.
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

    ``margin`` is HF's top-1 minus HF's top-2 logit; ``entropy`` is the entropy of HF's
    own distribution. Neither involves TT. This is the control the position-trend
    report needs: a falling TT-vs-HF rate has two unrelated causes — device state
    accumulating with position, or the reference simply becoming less decisive over
    that stretch of text — and they are indistinguishable from the agreement rate
    alone.
    """
    top2 = hf_rows.topk(2, dim=-1).values
    logprob = hf_rows.log_softmax(dim=-1)
    return top2[:, 0] - top2[:, 1], -(logprob.exp() * logprob).sum(dim=-1)


def _tt_rank_under_hf(hf_rows, tt_top1):
    """1-based rank of TT's pick in HF's own ordering (rank 1 = agreement).

    Severity of a flip — how far down the reference's ordering TT landed — reported
    separately from whether the reference was confident, because the two answer
    different questions.
    """
    picked = hf_rows.gather(-1, tt_top1.unsqueeze(-1))
    return (hf_rows > picked).sum(dim=-1) + 1


def _decision_metrics(tt_rows, hf_rows):
    """Per-step KL(HF || TT) over the full row, and max |delta logit| on HF's top-K."""
    hf_logprob = torch.log_softmax(hf_rows, dim=-1)
    tt_logprob = torch.log_softmax(tt_rows, dim=-1)
    kl = (hf_logprob.exp() * (hf_logprob - tt_logprob)).sum(dim=-1)
    k = min(_LOGIT_CMP_K, hf_rows.shape[-1])
    idx = hf_rows.topk(k, dim=-1).indices
    return kl, (hf_rows.gather(-1, idx) - tt_rows.gather(-1, idx)).abs().amax(dim=-1)


def _model_key(model_args):
    """Stable key for the checkpoint under test: ``"<layers>L-<dim>"``.

    Derived from the config rather than from ``HF_MODEL`` because that variable is
    rewritten to the resolved snapshot path, whose basename is an opaque hash — the
    same reason ``pcc_thresholds.json`` is keyed by test rather than by model name.
    """
    return f"{model_args.n_layers}L-{model_args.dim}"


def _resolve_floors(model_key, case_id):
    """``(min_top1, min_top5, min_step_pcc, provenance, inherited)`` for one run.

    ``inherited`` is True when this (model, length) has no row of its own in
    ``_MEASURED_FLOORS`` and is therefore gated on numbers measured elsewhere — logged
    as a warning rather than passing silently.
    """
    row = _MEASURED_FLOORS.get((model_key, case_id))
    inherited = row is None
    top1, top5, step_pcc, provenance = row if row is not None else _FALLBACK_FLOORS
    overrides = []
    if _MIN_TOP1_ENV is not None:
        top1 = float(_MIN_TOP1_ENV)
        overrides.append("QWEN36_TF_MIN_TOP1")
    if _MIN_TOP5_ENV is not None:
        top5 = float(_MIN_TOP5_ENV)
        overrides.append("QWEN36_TF_MIN_TOP5")
    if _MIN_LOGIT_PCC_ENV is not None:
        step_pcc = float(_MIN_LOGIT_PCC_ENV)
        overrides.append("QWEN36_TF_MIN_LOGIT_PCC")
    if overrides:
        provenance, inherited = f"env override ({', '.join(overrides)})", False
    return top1, top5, step_pcc, provenance, inherited


# ── reporting ─────────────────────────────────────────────────────────────


def _where(i):
    return "prefill" if i == 0 else f"decode[{i - 1}]"


def _log_segment(label, r, tt_rows, hf_rows, truth, sl):
    """One report row: TT-vs-HF and both vs-truth rates over a slice of predictions."""
    agree_top1 = r["agree_top1"][sl]
    if agree_top1.numel() == 0:
        return
    tt1, ttk = _vs_truth(tt_rows[sl], truth[sl])
    hf1, hfk = _vs_truth(hf_rows[sl], truth[sl])
    logger.info(
        "  {:<8} n={:<5} TT-vs-HF top1={:>7.2%} top{}={:>7.2%}  |  "
        "TT-vs-truth top1={:>7.2%} top{}={:>7.2%}  |  HF-vs-truth top1={:>7.2%} top{}={:>7.2%}",
        label,
        agree_top1.numel(),
        agree_top1.float().mean().item(),
        _TOP_K,
        r["agree_topk"][sl].float().mean().item(),
        tt1,
        _TOP_K,
        ttk,
        hf1,
        _TOP_K,
        hfk,
    )


def _log_pcc_segment(label, pccs, sl):
    seg = pccs[sl]
    if seg:
        logger.info("  {:<8} n={:<5} logit_PCC min={:.6f} mean={:.6f}", label, len(seg), min(seg), sum(seg) / len(seg))


def _log_per_step_pcc(pccs, *, limit=8):
    logger.info("  per-step logit PCC (TT vs HF):")
    for i, pcc in enumerate(pccs[:limit]):
        logger.info("      {:<11} PCC={:.6f}", _where(i), pcc)
    if len(pccs) > limit:
        logger.info("      ... {} more steps omitted", len(pccs) - limit)
    worst_i, worst_pcc = min(enumerate(pccs), key=lambda x: x[1])
    logger.info("  worst step {:<11} PCC={:.6f}", _where(worst_i), worst_pcc)


def _log_decision_metrics(tt_rows, hf_rows):
    kl, max_delta = _decision_metrics(tt_rows, hf_rows)
    logger.info("-" * 78)
    logger.info("Decision-relevant logit distance (top-{} window):", _LOGIT_CMP_K)
    logger.info("  KL(HF||TT)        mean={:.6f} max={:.6f}", kl.mean().item(), kl.max().item())
    logger.info(
        "  max|dlogit| top-{} mean={:.4f} max={:.4f}", _LOGIT_CMP_K, max_delta.mean().item(), max_delta.max().item()
    )


def _log_position_trend(r, pccs, hf_rows, truth):
    """Agreement and PCC per position bin, BESIDE the reference's own confidence.

    Read the two halves together:
      * agreement falls, reference margin flat  → positional drift; look at the paged
        KV writes, the GDN state carry, position handling.
      * agreement falls, reference margin falls → text difficulty; the trend says
        nothing about the device.
      * agreement flat                          → per-step numerics (dtype/fidelity).
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
        seg_pcc, seg_margin = pccs[lo:hi], margin[lo:hi]
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


def _log_confidence_split(r, hf_rows, truth, tokenizer):
    """Split top-1 flips by the REFERENCE's own confidence; report severity apart.

    Two independent questions: *should this flip have happened* (HF's own top1-top2
    margin — a flip the reference decided by 0.3 logits is a near-tie no dtype knob
    will remove) and *how badly did TT miss* (the rank of TT's pick under HF).
    """
    ref_margin, _ = _reference_confidence(hf_rows)
    rank = _tt_rank_under_hf(hf_rows, r["tt_top1"])
    flipped = ~r["agree_top1"]
    n_flip, n_total = int(flipped.sum()), r["agree_top1"].numel()
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
    logger.info("      confident      : {:<5} (reference was decisive — chase these)", n_confident)
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
    logger.info(
        "  outside HF top-{}   : {:<5} (TT's pick below the reference's {}th choice)",
        _TOP_K,
        int((~r["agree_topk"]).sum()),
        _TOP_K,
    )
    idx = confident.nonzero(as_tuple=True)[0].tolist()
    for i in idx[:8]:
        logger.info(
            "      {:<11} ref_margin={:>7.3f} tt_rank={:<5} truth={!r:<14} HF={!r:<14} TT={!r}",
            _where(i),
            ref_margin[i].item(),
            int(rank[i]),
            tokenizer.decode([int(truth[i])]),
            tokenizer.decode([int(r["hf_top1"][i])]),
            tokenizer.decode([int(r["tt_top1"][i])]),
        )
    if len(idx) > 8:
        logger.info("      ... {} more confident flips", len(idx) - 8)
    return n_confident


def _log_resolution(r, min_top1, min_top5, provenance, inherited):
    """Print each rate with its confidence interval next to the floor it is gated on.

    The point of this block is to make an unresolved measurement look unresolved: a
    rate whose interval straddles the floor is neither a pass nor a fail.
    """
    n = r["agree_top1"].numel()
    top1_hits, topk_hits = int(r["agree_top1"].sum()), int(r["agree_topk"].sum())
    t1_lo, t1_hi = _wilson(top1_hits, n)
    tk_lo, tk_hi = _wilson(topk_hits, n)

    def verdict(lo, hi, floor):
        if lo >= floor:
            return "RESOLVED pass"
        return "RESOLVED fail" if hi < floor else "UNRESOLVED — CI straddles the floor"

    logger.info("-" * 78)
    logger.info("Resolution (Wilson interval, z={:.2f}, n={}):", _CI_Z, n)
    logger.info(
        "  top-1  {:>7.2%}  CI [{:>7.2%}, {:>7.2%}]  floor {:>7.2%}  {}",
        top1_hits / n,
        t1_lo,
        t1_hi,
        min_top1,
        verdict(t1_lo, t1_hi, min_top1),
    )
    logger.info(
        "  top-{}  {:>7.2%}  CI [{:>7.2%}, {:>7.2%}]  floor {:>7.2%}  {}",
        _TOP_K,
        topk_hits / n,
        tk_lo,
        tk_hi,
        min_top5,
        verdict(tk_lo, tk_hi, min_top5),
    )
    logger.info("  one prediction is {:.2f} percentage points at this n", 100.0 / n)
    logger.info("  floors from : {}", provenance)
    if inherited:
        logger.warning(
            "  This length case has no measured floor of its own — it is being gated on numbers from a "
            "different length. Add a _MEASURED_FLOORS row after remeasuring; do not edit an existing row to "
            "make this case pass."
        )
    logger.info(
        "  Positions are correlated through the text, so the true interval is wider than this "
        "independent-Bernoulli one."
    )


def _log_generated_text(tokenizer, tt_top1, hf_top1, truth):
    """Print the teacher-forced predictions as text, from TT and from HF alike.

    Every prediction is made from ground-truth context, so these are not a free-running
    generation: they are the token each side would have emitted at each position of the
    reference text, which is what makes the two strings comparable line by line.
    """

    def _decode(ids):
        return tokenizer.decode([int(t) for t in ids])

    logger.info("-" * 78)
    logger.info("Teacher-forced output (each token predicted from ground-truth context):")
    logger.info("  reference (HF) : {!r}", _decode(hf_top1))
    logger.info("  device    (TT) : {!r}", _decode(tt_top1))
    logger.info("  ground truth   : {!r}", _decode(truth))
    if _PRINT_STEPS <= 0:
        return
    n = min(_PRINT_STEPS, len(tt_top1))
    logger.info("  first {} predictions, token by token:", n)
    logger.info("      {:<11} {:<16} {:<16} {:<16} {}", "step", "truth", "HF", "TT", "match")
    for i in range(n):
        logger.info(
            "      {:<11} {!r:<16} {!r:<16} {!r:<16} {}",
            _where(i),
            _decode([truth[i]]),
            _decode([hf_top1[i]]),
            _decode([tt_top1[i]]),
            "ok" if int(tt_top1[i]) == int(hf_top1[i]) else "MISMATCH",
        )
    if len(tt_top1) > n:
        logger.info("      ... {} more predictions omitted", len(tt_top1) - n)


# ── shared run ────────────────────────────────────────────────────────────


def _prepare_teacher_forcing_run(prefill_len, max_new_tokens, mesh_device, request):
    """Shared setup for the token-accuracy and logit-PCC teacher-forced tests."""
    max_prefill = request.config.getoption("--max-prefill")
    if prefill_len > max_prefill:
        pytest.skip(f"prefill_len={prefill_len} > --max-prefill={max_prefill}")

    total_len = prefill_len + max_new_tokens + 1
    # Blocks must cover every position touched, and a multiple of 32 keeps the
    # chunked-SDPA page-table alignment the demo rounds to.
    num_blocks = max(32, -(-total_len // BLOCK_SIZE))
    num_blocks = -(-num_blocks // 32) * 32

    model, tokenizer, _ = build_full_depth_model(mesh_device, max_seq_len=num_blocks * BLOCK_SIZE)
    tokens = _build_tokens(tokenizer, total_len)
    logger.info(
        "Teacher forcing: prefill_len={} max_new_tokens={} ({} predictions, {} tokens, {} blocks x {})",
        prefill_len,
        max_new_tokens,
        max_new_tokens + 1,
        total_len,
        num_blocks,
        BLOCK_SIZE,
    )

    hf_rows = _hf_reference_rows(model.args.CKPT_DIR, tokens, prefill_len, max_new_tokens)

    page_table = allocate_paged_kv(model, num_blocks)
    tt_rows = _run_teacher_forced_tt(model, page_table, tokens, prefill_len, max_new_tokens)

    vocab = int(model.args.vocab_size)
    tt_rows, hf_rows = tt_rows[:, :vocab], hf_rows[:, :vocab]
    truth = tokens[0, prefill_len : prefill_len + tt_rows.shape[0]].long()
    return {
        "model_args": model.args,
        "tokenizer": tokenizer,
        "tt_rows": tt_rows,
        "hf_rows": hf_rows,
        "truth": truth,
    }


@torch.no_grad()
@parametrize_full_depth()
@pytest.mark.parametrize("prefill_len,max_new_tokens", _TF_LENGTHS)
def test_teacher_forcing_e2e(prefill_len, max_new_tokens, mesh_device, reset_seeds, ensure_gc, request):
    """Score prefill + every teacher-forced decode step, top-1 and top-5, TT vs HF."""
    run = _prepare_teacher_forcing_run(prefill_len, max_new_tokens, mesh_device, request)
    tt_rows, hf_rows, truth, tokenizer = run["tt_rows"], run["hf_rows"], run["truth"], run["tokenizer"]

    r = _rates(tt_rows, hf_rows)
    pccs = _logit_pccs(tt_rows, hf_rows)

    logger.info("=" * 78)
    logger.info(
        "Teacher-forced accuracy — {} layers, prefill_len={} max_new_tokens={} ({} predictions)",
        run["model_args"].n_layers,
        prefill_len,
        max_new_tokens,
        tt_rows.shape[0],
    )
    logger.info("=" * 78)
    _log_segment("prefill", r, tt_rows, hf_rows, truth, slice(0, 1))
    _log_segment("decode", r, tt_rows, hf_rows, truth, slice(1, None))
    _log_segment("e2e", r, tt_rows, hf_rows, truth, slice(None))
    logger.info("-" * 78)
    logger.info("Logit PCC (informational — asserted in test_teacher_forcing_logits_pcc):")
    _log_pcc_segment("prefill", pccs, slice(0, 1))
    _log_pcc_segment("decode", pccs, slice(1, None))
    _log_pcc_segment("e2e", pccs, slice(None))
    _log_per_step_pcc(pccs)
    _log_decision_metrics(tt_rows, hf_rows)
    _log_position_trend(r, pccs, hf_rows, truth)
    _log_confidence_split(r, hf_rows, truth, tokenizer)
    _log_generated_text(tokenizer, r["tt_top1"], r["hf_top1"], truth)

    mismatched = (~r["agree_top1"]).nonzero(as_tuple=True)[0].tolist()
    outside = (~r["agree_topk"]).nonzero(as_tuple=True)[0].tolist()
    if mismatched:
        logger.info("  first {} top-1 mismatches:", min(8, len(mismatched)))
        for i in mismatched[:8]:
            hf_choices = ", ".join(f"{tokenizer.decode([int(t)])!r}" for t in r["hf_topk"][i])
            logger.info(
                "      {:<11} truth={!r:<14} TT={!r:<14} in_hf_top{}={!s:<5} HF top{}=[{}]",
                _where(i),
                tokenizer.decode([int(truth[i])]),
                tokenizer.decode([int(r["tt_top1"][i])]),
                _TOP_K,
                bool(r["agree_topk"][i]),
                _TOP_K,
                hf_choices,
            )

    top1_rate = r["agree_top1"].float().mean().item()
    top5_rate = r["agree_topk"].float().mean().item()
    case_id = f"prefill_{prefill_len}-max_new_tokens_{max_new_tokens}"
    min_top1, min_top5, _, provenance, inherited = _resolve_floors(_model_key(run["model_args"]), case_id)
    _log_resolution(r, min_top1, min_top5, provenance, inherited)
    logger.info("=" * 78)

    n_pred = r["agree_top1"].numel()
    t1_lo, t1_hi = _wilson(int(r["agree_top1"].sum()), n_pred)
    unresolved = t1_lo < min_top1 <= t1_hi
    assert top1_rate >= min_top1 and top5_rate >= min_top5, (
        f"e2e teacher forcing below threshold: top1={top1_rate:.2%} "
        f"(CI [{t1_lo:.2%}, {t1_hi:.2%}], min {min_top1:.2%}), "
        f"top{_TOP_K}={top5_rate:.2%} (min {min_top5:.2%}); "
        f"{len(mismatched)}/{tt_rows.shape[0]} top-1 mismatches, {len(outside)} outside HF top-{_TOP_K}. "
        f"Floors from: {provenance}."
        + (
            " NOTE: the top-1 confidence interval straddles the floor, so this run does not resolve pass from "
            "fail at this sample size."
            if unresolved
            else ""
        )
    )


@torch.no_grad()
@parametrize_full_depth()
@pytest.mark.parametrize("prefill_len,max_new_tokens", _TF_LENGTHS)
def test_teacher_forcing_logits_pcc(prefill_len, max_new_tokens, mesh_device, reset_seeds, ensure_gc, request):
    """Assert full-vocab logit PCC (TT vs HF) at every teacher-forced step."""
    run = _prepare_teacher_forcing_run(prefill_len, max_new_tokens, mesh_device, request)
    tt_rows, hf_rows = run["tt_rows"], run["hf_rows"]

    case_id = f"prefill_{prefill_len}-max_new_tokens_{max_new_tokens}"
    _, _, threshold, provenance, inherited = _resolve_floors(_model_key(run["model_args"]), case_id)
    pccs = _logit_pccs(tt_rows, hf_rows)

    logger.info("=" * 78)
    logger.info(
        "Teacher-forced logit PCC — {} layers, prefill_len={} max_new_tokens={} ({} steps, threshold={:.4f})",
        run["model_args"].n_layers,
        prefill_len,
        max_new_tokens,
        len(pccs),
        threshold,
    )
    logger.info("=" * 78)
    _log_pcc_segment("prefill", pccs, slice(0, 1))
    _log_pcc_segment("decode", pccs, slice(1, None))
    _log_pcc_segment("e2e", pccs, slice(None))
    _log_per_step_pcc(pccs)
    _log_decision_metrics(tt_rows, hf_rows)
    _log_generated_text(run["tokenizer"], _greedy_tokens(tt_rows), _greedy_tokens(hf_rows), run["truth"])
    logger.info("=" * 78)

    logger.info("  floors from : {}", provenance)
    if inherited:
        logger.warning(
            "  This (model, length) has no measured floor of its own — it is being gated on numbers from a "
            "different run. Measure and add a _MEASURED_FLOORS row rather than editing an existing one."
        )

    failures = [f"{_where(i)} PCC={p:.6f}" for i, p in enumerate(pccs) if p < threshold]
    assert not failures, (
        f"{len(failures)}/{len(pccs)} teacher-forced steps below logit PCC threshold {threshold:.4f}: "
        f"{', '.join(failures[:8])}{' ...' if len(failures) > 8 else ''}. "
        f"Floors from: {provenance}"
    )
