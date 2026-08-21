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
    pytest models/demos/gemma4/tests/unit/test_teacher_forcing_e2e.py -k 1x8 -sv --timeout=0

  # 12B (same lengths / token-accuracy floors; logit-PCC floor is under gemma-4-12B-it)
  HF_MODEL=google/gemma-4-12B-it MESH_DEVICE=1x8 GEMMA4_HF_DEVICE_MAP=auto \\
    pytest "models/demos/gemma4/tests/unit/test_teacher_forcing_e2e.py::test_teacher_forcing_e2e[wormhole_b0-prefill_512-max_new_tokens_500-1x8]" -sv --timeout=0

  # token accuracy only -- select by node id, NOT by ``-k``. The module filename
  # is itself a keyword, so ``-k test_teacher_forcing_e2e`` also matches every
  # test_e2e_logits_pcc case in this file.
  pytest models/demos/gemma4/tests/unit/test_teacher_forcing_e2e.py::test_teacher_forcing_e2e -k 1x8 -sv --timeout=0

  # logit PCC only (this name is unique, so -k is safe)
  pytest models/demos/gemma4/tests/unit/test_teacher_forcing_e2e.py -k "test_e2e_logits_pcc and 1x8" -sv --timeout=0

  # one exact case
  pytest "models/demos/gemma4/tests/unit/test_teacher_forcing_e2e.py::test_teacher_forcing_e2e[wormhole_b0-prefill_512-max_new_tokens_500-1x8]" -sv --timeout=0
"""

from __future__ import annotations

import bz2
import os
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.gemma4.demo.text_demo_v2 import create_tt_page_table
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
from .teacher_forcing_report import (
    _TEMPERATURE,
    _TOP_K,
    _TOP_P,
    _greedy_tokens,
    _log_confidence_split,
    _log_decision_metrics,
    _log_generated_text,
    _log_pcc_segment,
    _log_per_step_pcc,
    _log_position_trend,
    _log_resolution,
    _log_segment,
    _logit_pccs,
    _rates,
    _wilson,
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


def _prepare_teacher_forcing_run(prefill_len, max_new_tokens, mesh_device, request):
    """Shared setup for token-accuracy and logit-PCC teacher-forced e2e tests."""
    skip_if_config_only_checkpoint()

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
