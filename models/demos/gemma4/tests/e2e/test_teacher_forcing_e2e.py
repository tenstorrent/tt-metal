# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end teacher-forced evaluation: token accuracy and logit PCC.

Drives the real generation path — ``prefill_forward_text`` followed by the
``decode_forward`` loop, the same sequence ``text_demo_v2`` runs — but replaces
the sampled token at every decode step with the ground-truth token. That is the
teacher-forcing method ``TokenAccuracy`` implements for tt_transformers
(``models/tt_transformers/demo/simple_text_demo.py``): because the model is
re-anchored to truth each step, a single bad prediction cannot derail the rest
of the sequence, so every step is an independent measurement instead of one
diverged sample.

Ground-truth text defaults to *A Tale of Two Cities* (same corpus as
``models/tt_transformers/tests/generate_reference_hf.py``). Override with
``GEMMA4_TF_TEXT_FILE``.

Predictions scored, for ``prefill_len`` prompt tokens and ``max_new_tokens``
decode steps:

  * **prefill** — 1 prediction. Prefill over tokens ``0..prefill_len-1`` predicts
    token ``prefill_len``.
  * **decode**  — ``max_new_tokens`` predictions. Step ``j`` is fed ground-truth
    token ``prefill_len+j`` at position ``prefill_len+j`` and predicts token
    ``prefill_len+j+1``.
  * **e2e**     — the ``max_new_tokens+1`` combined.

Each is reported top-1 and top-5, against two references:

  * **vs HF** — TT's token equal to HF's token (top-1), and TT's token inside
    HF's top-5 (top-5). This is the device-correctness signal and the assertion.
    It matches ``TokenAccuracy.compute_accuracy``, which scores the prediction
    against the reference's ``top5_tokens`` rather than against the text.
  * **vs truth** — the next token actually in the text. Reported for TT and HF
    alike; informational, since it measures the checkpoint, not the device.

Both sides are picked the same way and the reference is loaded the same way as
the rest of the Gemma4 HF-reference tests:

  * tokens come from the demo's sampler, ``text_demo_v2._host_sample``, on the
    demo's defaults (``temperature=0`` → greedy argmax, ``top_p=0.08`` unused at
    temperature 0), applied to the TT and the HF logits alike. TT decodes with
    ``sampling_params=None`` so the full logits row comes back -- the demo's
    ``GEMMA4_HOST_SAMPLE=1`` branch -- which top-5 and PCC both need.
  * HF is loaded through ``test_factory.load_hf_reference_model`` -- the shared
    bf16 / ``trust_remote_code`` / ``GEMMA4_HF_DEVICE_MAP`` path, so every test
    that scores against HF loads it identically.

Reading the pair: top-1 agreement falling while top-5 holds at 100% means TT is
reordering near-ties, the ordinary consequence of BFP8 weights or LoFi fidelity.
Top-5 falling too means TT is emitting tokens the reference does not rank at
all — a real divergence, not rounding.

``test_teacher_forcing_e2e`` also reports three diagnostics that a raw match
rate cannot give you, because a sub-100%% rate has two unrelated causes and they
want opposite fixes:

  * **flip classification** — every top-1 flip is scored by the *reference's own*
    top-1 minus top-2 margin. Flips inside ``GEMMA4_TF_CONFIDENT_GAP`` (default
    5.0 logits) are positions the reference itself barely decided, which two
    numeric paths are expected to reorder; only *confident* flips indicate a
    defect. The rank of TT's pick under HF is printed separately, as severity —
    it used to be the thing gated on, which made "confident" a restatement of
    "outside HF top-5" rather than an independent signal.
  * **position trend** — accuracy and PCC per position bin, printed beside the
    reference's own accuracy, margin and entropy over the same bins. Both halves
    are needed: agreement falling while the reference's margin stays flat means
    state accumulating with position (KV cache, page table, positions); agreement
    and margin falling together means the text simply got harder and the trend
    says nothing about the device; flat means per-step numerics, which is what
    dtype and fidelity move.
  * **resolution** — each rate with its Wilson interval next to the floor it is
    gated on, and the provenance of that floor. A rate whose interval straddles
    the floor is an unresolved measurement, not a defect; floors are keyed per
    length case so a change to ``_TF_LENGTHS`` cannot inherit one silently.
  * **decision-relevant distance** — KL(HF||TT) and max |delta logit| over HF's
    top-``GEMMA4_TF_LOGIT_CMP_K``. Full-vocab PCC spreads its weight over 262k
    mostly-irrelevant entries and is compressed further by the tanh
    ``final_logit_softcapping`` (30.0), so it tracks sampling behaviour poorly.

Both tests also print the teacher-forced output itself — the token sequence TT
emitted and the one HF emitted, decoded to text and shown beside the ground
truth. Since every prediction is anchored to truth, the two strings are aligned
position by position and can be read against each other directly.
``GEMMA4_TF_PRINT_STEPS`` (default 16, 0 to disable) also prints the first steps
as a truth / HF / TT token table.

Tests in this module:

  * ``test_teacher_forcing_e2e`` — top-1 / top-5 token accuracy (TT vs HF and vs
    truth), printed per prefill / decode / e2e segment.
  * ``test_e2e_logits_pcc`` — full-vocab logit PCC (TT vs HF) at every teacher-
    forced step, mirroring ``models/tt_transformers/tests/test_model.py``.

Run. Test ids are ``[wormhole_b0-<case>-<mesh>]``, e.g.
``test_teacher_forcing_e2e[wormhole_b0-prefill_512-max_new_tokens_500-1x8]``
(default: 512-token prefill, 500 max-new-tokens). Use ``--timeout=0``:

  HF_MODEL=google/gemma-4-31B-it MESH_DEVICE=1x8 GEMMA4_HF_DEVICE_MAP=auto \\
    pytest models/demos/gemma4/tests/e2e/test_teacher_forcing_e2e.py -k 1x8 -sv --timeout=0

  # 12B (same lengths / token-accuracy floors; logit-PCC floor is under gemma-4-12B-it)
  HF_MODEL=google/gemma-4-12B-it MESH_DEVICE=1x8 GEMMA4_HF_DEVICE_MAP=auto \\
    pytest "models/demos/gemma4/tests/e2e/test_teacher_forcing_e2e.py::test_teacher_forcing_e2e[wormhole_b0-prefill_512-max_new_tokens_500-1x8]" -sv --timeout=0

  # token accuracy only -- select by node id, NOT by ``-k``. The module filename
  # is itself a keyword, so ``-k test_teacher_forcing_e2e`` also matches every
  # test_e2e_logits_pcc case in this file.
  pytest models/demos/gemma4/tests/e2e/test_teacher_forcing_e2e.py::test_teacher_forcing_e2e -k 1x8 -sv --timeout=0

  # logit PCC only (this name is unique, so -k is safe)
  pytest models/demos/gemma4/tests/e2e/test_teacher_forcing_e2e.py -k "test_e2e_logits_pcc and 1x8" -sv --timeout=0

  # one exact case
  pytest "models/demos/gemma4/tests/e2e/test_teacher_forcing_e2e.py::test_teacher_forcing_e2e[wormhole_b0-prefill_512-max_new_tokens_500-1x8]" -sv --timeout=0
"""

from __future__ import annotations

import bz2
import math
import os
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.gemma4.demo.text_demo_v2 import _host_sample, create_tt_page_table
from models.demos.gemma4.tt.generator import Gemma4Generator
from models.demos.gemma4.tt.precision import Gemma4Precision, dtype_to_str
from models.tt_transformers.tt.common import PagedAttentionConfig

from ..test_factory import (
    _get_model_path,
    get_pcc_threshold,
    hf_reference_model_device,
    load_hf_reference_model,
    parametrize_mesh_with_fabric,
    skip_if_config_only_checkpoint,
)

# Prefill length + decode steps. Total tokens consumed = prefill_len + max_new_tokens + 1
# (final decode step still needs a ground-truth target). Default matches
# tt-transformers TokenAccuracy. Add ``pytest.param(...)`` rows to collect more cases.
_TF_LENGTHS = [
    pytest.param(512, 500, id="prefill_512-max_new_tokens_500"),
]

_BLOCK_SIZE = 64

# Agreement floors for the accuracy assertion — not a correctness target, and
# only valid at the length they were measured at. Keyed by the ``_TF_LENGTHS``
# case id so that adding or changing a length case cannot silently inherit a
# floor calibrated somewhere else: an unlisted case falls back to
# ``_FALLBACK_FLOORS`` and the run logs a warning naming the length the numbers
# actually came from.
#
# History, because it is the reason this table exists: 0.75 / 0.92 were measured
# on gemma-4-31B-it, WH T3K 1x8, p128_n128 (baseline 79.07% / 95.35%). The
# default case was later changed to 512 / 500 without a remeasure, so the long
# case has been asserting against a short-case floor. Add a row here once
# someone remeasures on T3K — do not edit the p128 row to make a 512-case run
# pass.
#
# ``test_full_model`` 0.98 PCC is a ~6-token prompt; longer teacher-forced
# sequences accumulate bfp8 error, so these gates stay below that.
_MEASURED_FLOORS = {
    # case id: (min_top1, min_top5, provenance)
    "prefill_128-max_new_tokens_128": (
        0.75,
        0.92,
        "gemma-4-31B-it WH T3K 1x8 p128_n128 (measured 79.07% / 95.35%)",
    ),
}
_FALLBACK_FLOORS = (
    0.75,
    0.92,
    "gemma-4-31B-it WH T3K 1x8 p128_n128 — NOT measured at this case's length",
)

# Env overrides win over the table. ``None`` means "use the table".
_MIN_TOP1_ENV = os.getenv("GEMMA4_TF_MIN_TOP1")
_MIN_TOP5_ENV = os.getenv("GEMMA4_TF_MIN_TOP5")

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

_TALE_OF_TWO_CITIES = Path(__file__).resolve().parents[4] / "tt_transformers" / "tests" / "tale-of-two-cities.txt.bz2"

# Used only when the Tale corpus is unavailable (e.g. sparse checkout).
_FALLBACK_TEXT = """The history of computing hardware begins with mechanical devices built to
automate arithmetic. Charles Babbage designed the Analytical Engine in the nineteenth century,
and Ada Lovelace wrote what is now regarded as the first algorithm intended for a machine."""


def _load_text():
    """Teacher-forcing text.

    Priority: ``GEMMA4_TF_TEXT_FILE`` env → Tale of Two Cities (tt_transformers
    corpus) → small built-in fallback.
    """
    path = os.getenv("GEMMA4_TF_TEXT_FILE")
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

    Repetition is harmless under teacher forcing: every position is re-anchored
    to truth, so predictions stay independent, and TT and HF are measured on
    identical input either way.
    """
    text = _load_text()
    ids = tokenizer.encode(text, add_special_tokens=True)
    if len(ids) < total_len:
        body = tokenizer.encode(text, add_special_tokens=False)
        while len(ids) < total_len:
            ids.extend(body)
    return torch.tensor(ids[:total_len], dtype=torch.int32).unsqueeze(0)


def _log_parameters(*, model_path, mesh_device, model_args, prefill_len, max_new_tokens, precision):
    """Print every parameter that can move the accuracy numbers.

    Weight dtype, math fidelity and the CCL/matmul env knobs all change
    teacher-forced accuracy, so a number is only meaningful beside the
    configuration that produced it.
    """
    mesh_shape = tuple(mesh_device.shape)
    module_dtypes = {
        name: dtype_to_str(precision.get(name, ttnn.bfloat16))
        for name in ("shared_mlp", "attention", "experts", "router", "lm_head", "embedding")
    }
    # Every knob that can move these numbers, printed whether set or not. An A/B
    # run is only interpretable if its log names the setting it was testing:
    # earlier sweeps produced pairs of runs that were bit-identical because the
    # intended knob never appeared here and never took effect.
    env_knobs = {
        key: os.getenv(key, "<unset>")
        for key in (
            "GEMMA4_PRECISION_OVERRIDE",
            "GEMMA4_PREFILL_SDPA_FIDELITY",
            "GEMMA4_DECODE_SDPA_FIDELITY",
            "GEMMA4_LM_HEAD_FIDELITY",
            "GEMMA4_PREFILL_MATMUL_LOFI",
            "GEMMA4_ATTN_DRAM_SHARD",
            "GEMMA4_DECODE_QKV_L1",
            "GEMMA4_SHARDED_NORM",
            "GEMMA4_CCL_PACKET_BYTES",
            "GEMMA4_HOST_SAMPLE",
            "GEMMA4_TF_TEXT_FILE",
            "GEMMA4_TF_TEMPERATURE",
            "GEMMA4_TF_TOP_P",
            "GEMMA4_TF_CONFIDENT_GAP",
            "GEMMA4_TF_MIN_TOP1",
            "GEMMA4_TF_MIN_TOP5",
            "GEMMA4_HF_DEVICE_MAP",
        )
    }

    logger.info("=" * 78)
    logger.info("End-to-end teacher forcing — run parameters")
    logger.info("=" * 78)
    logger.info("  model_path          : {}", model_path)
    logger.info(
        "  mesh_shape          : {}x{}  ({} devices)", mesh_shape[0], mesh_shape[1], mesh_device.get_num_devices()
    )
    logger.info("  hidden_size         : {}", model_args.hidden_size)
    logger.info("  num_hidden_layers   : {}", model_args.num_hidden_layers)
    logger.info("  vocab_size          : {}", model_args.vocab_size)
    logger.info("  sliding_window      : {}", getattr(model_args, "sliding_window", None))
    logger.info("  prefill_len         : {}", prefill_len)
    logger.info("  max_new_tokens      : {}", max_new_tokens)
    logger.info("  predictions scored  : {}  (1 prefill + {} decode)", max_new_tokens + 1, max_new_tokens)
    logger.info("  text source         : {}", os.getenv("GEMMA4_TF_TEXT_FILE") or _TALE_OF_TWO_CITIES)
    logger.info("  top_k               : {}", _TOP_K)
    logger.info(
        "  sampling            : temperature={} top_p={} ({})",
        _TEMPERATURE,
        _TOP_P,
        "greedy argmax" if _TEMPERATURE <= 0 else "top-p sampling (noisy accuracy)",
    )
    logger.info("  weight dtypes       :")
    for name, dtype_str in module_dtypes.items():
        logger.info("      {:<12}: {}", name, dtype_str)
    logger.info("  env knobs           :")
    for key, value in env_knobs.items():
        logger.info("      {:<28}: {}", key, value)
    logger.info("=" * 78)


def _as_row(logits):
    """Normalize a prefill/decode logits return to a 1-D [vocab] host row.

    Batch is 1 throughout, so flattening the leading dims leaves a single row
    for decode and the next-token row last for prefill; either way the final
    row is the one being scored. Handles [vocab], [B,vocab] and [B,1,vocab].
    """
    out = logits if isinstance(logits, torch.Tensor) else torch.as_tensor(logits)
    out = out.float().cpu()
    return out.reshape(-1, out.shape[-1])[-1]


def _run_teacher_forced_e2e(generator, tt_kv_cache, page_table, tokens, prefill_len, max_new_tokens):
    """Prefill the prompt, then decode ``max_new_tokens`` steps feeding truth back.

    Returns the TT logits rows for each scored prediction, in order:
    ``[prefill] + [decode_0 .. decode_{max_new_tokens-1}]``.
    """
    prompt = tokens[:, :prefill_len]
    prompt_lens = torch.tensor([prefill_len], dtype=torch.long)

    rows = []

    # Prefill over 0..P-1 → predicts token P. sampling_params=None keeps host
    # logits (device sampling would return a token and hide the distribution).
    prefill_out = generator.prefill_forward_text(
        prompt,
        page_table=page_table,
        kv_cache=tt_kv_cache,
        prompt_lens=prompt_lens,
        enable_trace=False,
        warmup_prefill=False,
        sampling_params=None,
    )
    rows.append(_as_row(prefill_out))

    # Decode step j is fed ground-truth token prefill_len+j at that position and
    # predicts prefill_len+j+1 — this substitution is the teacher forcing.
    current_pos = torch.tensor([prefill_len], dtype=torch.long)
    for j in range(max_new_tokens):
        forced = tokens[:, prefill_len + j].reshape(1, 1).long()
        decode_out, _ = generator.decode_forward(
            forced,
            current_pos,
            enable_trace=False,
            page_table=page_table,
            kv_cache=tt_kv_cache,
            sampling_params=None,
        )
        rows.append(_as_row(decode_out))
        current_pos = current_pos + 1

    return torch.stack(rows, dim=0)


def _hf_reference_rows(model_path, tokens, prefill_len, max_new_tokens):
    """HF logits rows aligned to the TT predictions.

    TT prediction ``i`` predicts token ``prefill_len + i``, which HF produces at
    row ``prefill_len + i - 1``. One HF forward covers every row, because HF
    prefill is itself teacher-forced by causal masking.
    """
    end = prefill_len + max_new_tokens
    # Shared loader (bf16, trust_remote_code, GEMMA4_HF_DEVICE_MAP) so every test
    # scores against an identically loaded HF.
    hf_model = load_hf_reference_model(model_path)
    try:
        device = hf_reference_model_device(hf_model)
        with torch.no_grad():
            out = hf_model(tokens[:, :end].long().to(device))
        return out.logits[0, prefill_len - 1 : end, :].float().cpu()
    finally:
        del hf_model


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


def _resolve_floors(case_id):
    """Assertion floors for one length case, with the provenance of the numbers.

    Returns ``(min_top1, min_top5, provenance, inherited)``. ``inherited`` is True
    when this case has no row of its own in ``_MEASURED_FLOORS`` and is therefore
    being gated on numbers measured at a different length — the caller logs that
    as a warning rather than letting it pass silently, which is how the 512/500
    default came to assert against a p128 floor.

    Env overrides (``GEMMA4_TF_MIN_TOP1`` / ``GEMMA4_TF_MIN_TOP5``) win over the
    table and are reported as their own provenance.
    """
    row = _MEASURED_FLOORS.get(case_id)
    inherited = row is None
    top1, top5, provenance = row if row is not None else _FALLBACK_FLOORS
    overrides = []
    if _MIN_TOP1_ENV is not None:
        top1 = float(_MIN_TOP1_ENV)
        overrides.append("GEMMA4_TF_MIN_TOP1")
    if _MIN_TOP5_ENV is not None:
        top5 = float(_MIN_TOP5_ENV)
        overrides.append("GEMMA4_TF_MIN_TOP5")
    if overrides:
        provenance = f"env override ({', '.join(overrides)})"
        inherited = False
    return top1, top5, provenance, inherited


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


def _prepare_teacher_forcing_run(prefill_len, max_new_tokens, mesh_device, request):
    """Shared setup for token-accuracy and logit-PCC teacher-forced e2e tests."""
    skip_if_config_only_checkpoint()
    if math.prod(tuple(mesh_device.shape)) < 4:
        pytest.skip("teacher-forcing e2e is a multi-chip CI gate (1x4/1x8)")

    max_prefill = request.config.getoption("--max-prefill")
    if prefill_len > max_prefill:
        pytest.skip(f"prefill_len={prefill_len} > --max-prefill={max_prefill}")

    model_path = _get_model_path()
    total_len = prefill_len + max_new_tokens + 1
    max_seq_len = max(4096, ((total_len + _BLOCK_SIZE - 1) // _BLOCK_SIZE) * _BLOCK_SIZE)

    paged_cfg = PagedAttentionConfig(
        block_size=_BLOCK_SIZE,
        max_num_blocks=max(1, (max_seq_len + _BLOCK_SIZE - 1) // _BLOCK_SIZE),
    )

    generator, tt_kv_cache, tokenizer = Gemma4Generator.from_pretrained(
        mesh_device=mesh_device,
        model_path=model_path,
        max_batch_size=1,
        max_seq_len=max_seq_len,
        paged_attention_config=paged_cfg,
    )
    model_args = generator.model_args[0]

    precision = Gemma4Precision.load(model_path, tuple(mesh_device.shape), hf_config=model_args)
    _log_parameters(
        model_path=model_path,
        mesh_device=mesh_device,
        model_args=model_args,
        prefill_len=prefill_len,
        max_new_tokens=max_new_tokens,
        precision=precision,
    )

    page_table = create_tt_page_table(1, paged_cfg)
    tokens = _build_tokens(tokenizer, total_len)

    generator.warmup_model_prefill(
        kv_cache=tt_kv_cache,
        enable_trace=False,
        can_sample_on_device=False,
        greedy_only=True,
    )

    tt_rows = _run_teacher_forced_e2e(generator, tt_kv_cache, page_table, tokens, prefill_len, max_new_tokens)
    hf_rows = _hf_reference_rows(model_path, tokens, prefill_len, max_new_tokens)

    vocab = int(model_args.vocab_size)
    tt_rows = tt_rows[:, :vocab]
    hf_rows = hf_rows[:, :vocab]
    truth = tokens[0, prefill_len : prefill_len + tt_rows.shape[0]].long()

    return {
        "model_path": model_path,
        "tokenizer": tokenizer,
        "tt_rows": tt_rows,
        "hf_rows": hf_rows,
        "truth": truth,
        "prefill_len": prefill_len,
        "max_new_tokens": max_new_tokens,
    }


@pytest.mark.timeout(3600)
@parametrize_mesh_with_fabric()
@pytest.mark.parametrize("prefill_len,max_new_tokens", _TF_LENGTHS)
def test_teacher_forcing_e2e(prefill_len, max_new_tokens, mesh_device, reset_seeds, request):
    """Score prefill + every teacher-forced decode step, top-1 and top-5."""
    run = _prepare_teacher_forcing_run(prefill_len, max_new_tokens, mesh_device, request)
    tt_rows = run["tt_rows"]
    hf_rows = run["hf_rows"]
    truth = run["truth"]
    tokenizer = run["tokenizer"]

    r = _rates(tt_rows, hf_rows)
    pccs = _logit_pccs(tt_rows, hf_rows)

    logger.info("=" * 78)
    logger.info(
        "Teacher-forced accuracy — prefill_len={} max_new_tokens={} ({} predictions)",
        prefill_len,
        max_new_tokens,
        tt_rows.shape[0],
    )
    logger.info("=" * 78)
    _log_segment("prefill", r, tt_rows, hf_rows, truth, slice(0, 1))
    _log_segment("decode", r, tt_rows, hf_rows, truth, slice(1, None))
    _log_segment("e2e", r, tt_rows, hf_rows, truth, slice(None))
    logger.info("-" * 78)
    logger.info("Logit PCC (informational — asserted in test_e2e_logits_pcc):")
    _log_pcc_segment("prefill", pccs, slice(0, 1))
    _log_pcc_segment("decode", pccs, slice(1, None))
    _log_pcc_segment("e2e", pccs, slice(None))
    _log_per_step_pcc(pccs)
    _log_decision_metrics(tt_rows, hf_rows)
    _log_position_trend(r, pccs, hf_rows, truth)
    _log_confidence_split(r, hf_rows, truth, tokenizer)
    _log_generated_text(tokenizer, r["tt_top1"], r["hf_top1"], truth)
    logger.info("=" * 78)

    outside = (~r["agree_topk"]).nonzero(as_tuple=True)[0].tolist()
    mismatched = (~r["agree_top1"]).nonzero(as_tuple=True)[0].tolist()
    if mismatched:
        logger.info("  first {} top-1 mismatches:", min(8, len(mismatched)))
        for i in mismatched[:8]:
            where = "prefill" if i == 0 else f"decode[{i - 1}]"
            hf_choices = ", ".join(f"{tokenizer.decode([int(t)])!r}" for t in r["hf_topk"][i])
            logger.info(
                "      {:<11} truth={!r:<14} TT={!r:<14} in_hf_top{}={!s:<5} HF top{}=[{}]",
                where,
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
    min_top1, min_top5, provenance, inherited = _resolve_floors(case_id)
    _log_resolution(r, min_top1, min_top5, provenance, inherited)

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
            " NOTE: this case has no measured floor of its own, and its top-1 confidence interval straddles the "
            "floor — the measurement does not resolve pass from fail. Remeasure on T3K and add a "
            "_MEASURED_FLOORS row before treating this as a device defect."
            if inherited and unresolved
            else (
                " NOTE: the top-1 confidence interval straddles the floor, so this run does not resolve pass "
                "from fail at this sample size."
                if unresolved
                else ""
            )
        )
        + " See the parameter block above for the weight dtypes and fidelity knobs in effect."
    )


@pytest.mark.gemma4_hf_direct_parity
@pytest.mark.timeout(3600)
@parametrize_mesh_with_fabric()
@pytest.mark.parametrize("prefill_len,max_new_tokens", _TF_LENGTHS)
def test_e2e_logits_pcc(prefill_len, max_new_tokens, mesh_device, reset_seeds, request):
    """Assert full-vocab logit PCC (TT vs HF) at every teacher-forced step."""
    run = _prepare_teacher_forcing_run(prefill_len, max_new_tokens, mesh_device, request)
    tt_rows = run["tt_rows"]
    hf_rows = run["hf_rows"]

    pcc_threshold = float(os.getenv("GEMMA4_TF_MIN_LOGIT_PCC", get_pcc_threshold(request)))
    pccs = _logit_pccs(tt_rows, hf_rows)

    logger.info("=" * 78)
    logger.info(
        "Teacher-forced logit PCC — prefill_len={} max_new_tokens={} ({} steps, threshold={:.4f})",
        prefill_len,
        max_new_tokens,
        len(pccs),
        pcc_threshold,
    )
    logger.info("=" * 78)
    _log_pcc_segment("prefill", pccs, slice(0, 1))
    _log_pcc_segment("decode", pccs, slice(1, None))
    _log_pcc_segment("e2e", pccs, slice(None))
    _log_per_step_pcc(pccs)
    _log_generated_text(run["tokenizer"], _greedy_tokens(tt_rows), _greedy_tokens(hf_rows), run["truth"])
    logger.info("=" * 78)

    failures = []
    for i, pcc in enumerate(pccs):
        where = "prefill" if i == 0 else f"decode[{i - 1}]"
        if pcc < pcc_threshold:
            failures.append(f"{where} PCC={pcc:.6f}")

    assert not failures, (
        f"{len(failures)}/{len(pccs)} teacher-forced steps below logit PCC threshold "
        f"{pcc_threshold:.4f}: {', '.join(failures[:8])}"
        f"{' ...' if len(failures) > 8 else ''}"
    )
