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

Predictions scored, for a prompt of ``P`` tokens and ``N`` decode steps:

  * **prefill** — 1 prediction. Prefill over tokens ``0..P-1`` predicts token
    ``P``.
  * **decode**  — ``N`` predictions. Step ``j`` is fed ground-truth token
    ``P+j`` at position ``P+j`` and predicts token ``P+j+1``.
  * **e2e**     — the ``N+1`` combined.

Each is reported top-1 and top-5, against two references:

  * **vs HF** — TT's argmax equal to HF's argmax (top-1), and TT's argmax inside
    HF's top-5 (top-5). This is the device-correctness signal and the assertion.
    It matches ``TokenAccuracy.compute_accuracy``, which scores the prediction
    against the reference's ``top5_tokens`` rather than against the text.
  * **vs truth** — the next token actually in the text. Reported for TT and HF
    alike; informational, since it measures the checkpoint, not the device.

Reading the pair: top-1 agreement falling while top-5 holds at 100% means TT is
reordering near-ties, the ordinary consequence of BFP8 weights or LoFi fidelity.
Top-5 falling too means TT is emitting tokens the reference does not rank at
all — a real divergence, not rounding.

Tests in this module:

  * ``test_teacher_forcing_e2e`` — top-1 / top-5 token accuracy (TT vs HF and vs
    truth), printed per prefill / decode / e2e segment.
  * ``test_e2e_logits_pcc`` — full-vocab logit PCC (TT vs HF) at every teacher-
    forced step, mirroring ``models/tt_transformers/tests/test_model.py``.

Run:

  HF_MODEL=google/gemma-4-31B-it MESH_DEVICE=1x8 \\
    pytest models/demos/gemma4/tests/unit/test_teacher_forcing_e2e.py -sv

  # token accuracy only
  ... -k "test_teacher_forcing_e2e" -sv

  # logit PCC only
  ... -k "test_e2e_logits_pcc" -sv

  # single case
  ... -k "p64_n64 and 1x8" -sv
"""

from __future__ import annotations

import bz2
import os
from pathlib import Path

import pytest
import torch
from loguru import logger
from transformers import AutoModelForCausalLM

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.gemma4.demo.text_demo_v2 import create_tt_page_table
from models.demos.gemma4.tt.generator import Gemma4Generator
from models.demos.gemma4.tt.precision import Gemma4Precision, dtype_to_str
from models.tt_transformers.tt.common import PagedAttentionConfig

from ..test_factory import (
    _get_model_path,
    get_pcc_threshold,
    parametrize_mesh_with_fabric,
    skip_if_config_only_checkpoint,
)

# (prompt_len, max_new_tokens) — prompt_len + max_new_tokens + 1 tokens are
# consumed, since the final decode step still needs a ground-truth target.
_CASES = [(64, 64), (128, 128)]
_CASE_IDS = [f"p{p}_n{n}" for p, n in _CASES]

_BLOCK_SIZE = 64

# Baselined to the current measured behaviour of the generator path, not to a
# device-correctness target. Measured on gemma-4-31B-it, WH T3K 1x8, p128_n128
# (all-bf16 weights): top1=77.52%, top5=93.80% over 129 predictions. The bar
# sits just under that, so these assertions detect a *regression* from today's
# numbers; they do not certify TT matches HF. Raise them as the gap closes —
# note test_full_model[wormhole_b0-1x8] holds 0.98 PCC on the same checkpoint,
# so the loss is in this end-to-end path rather than the layer math.
_MIN_TOP1_AGREEMENT = float(os.getenv("GEMMA4_TF_MIN_TOP1", "0.75"))
_MIN_TOP5_AGREEMENT = float(os.getenv("GEMMA4_TF_MIN_TOP5", "0.92"))

# Matches TokenAccuracy's top-5 in models/tt_transformers/demo/simple_text_demo.py.
_TOP_K = int(os.getenv("GEMMA4_TF_TOP_K", "5"))

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


def _log_parameters(*, model_path, mesh_device, model_args, prompt_len, max_new_tokens, precision):
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
    env_knobs = {
        key: os.getenv(key, "<unset>")
        for key in (
            "GEMMA4_PREFILL_MATMUL_LOFI",
            "GEMMA4_CCL_PACKET_BYTES",
            "GEMMA4_HOST_SAMPLE",
            "GEMMA4_TF_TEXT_FILE",
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
    logger.info("  prompt_len (P)      : {}", prompt_len)
    logger.info("  max_new_tokens (N)  : {}", max_new_tokens)
    logger.info("  predictions scored  : {}  (1 prefill + {} decode)", max_new_tokens + 1, max_new_tokens)
    logger.info("  text source         : {}", os.getenv("GEMMA4_TF_TEXT_FILE") or _TALE_OF_TWO_CITIES)
    logger.info("  top_k               : {}", _TOP_K)
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


def _run_teacher_forced_e2e(generator, tt_kv_cache, page_table, tokens, prompt_len, max_new_tokens):
    """Prefill the prompt, then decode ``max_new_tokens`` steps feeding truth back.

    Returns the TT logits rows for each scored prediction, in order:
    ``[prefill] + [decode_0 .. decode_{N-1}]``.
    """
    prompt = tokens[:, :prompt_len]
    prompt_lens = torch.tensor([prompt_len], dtype=torch.long)

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

    # Decode step j is fed ground-truth token P+j at position P+j, and predicts
    # P+j+1 — this substitution is the teacher forcing.
    current_pos = torch.tensor([prompt_len], dtype=torch.long)
    for j in range(max_new_tokens):
        forced = tokens[:, prompt_len + j].reshape(1, 1).long()
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


def _hf_reference_rows(model_path, tokens, prompt_len, max_new_tokens):
    """HF logits rows aligned to the TT predictions.

    TT prediction ``i`` predicts token ``prompt_len + i``, which HF produces at
    row ``prompt_len + i - 1``. One HF forward covers every row, because HF
    prefill is itself teacher-forced by causal masking.
    """
    end = prompt_len + max_new_tokens
    hf_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    hf_model.eval()
    try:
        with torch.no_grad():
            out = hf_model(tokens[:, :end].long())
        return out.logits[0, prompt_len - 1 : end, :].float()
    finally:
        del hf_model


def _vs_truth(rows, truth):
    """(top-1, top-k) hit rate of ``rows`` against the ground-truth next tokens."""
    if rows.shape[0] == 0:
        return (0.0, 0.0)
    top1 = rows.argmax(dim=-1)
    topk = rows.topk(_TOP_K, dim=-1).indices
    return (
        (top1 == truth).float().mean().item(),
        (topk == truth.unsqueeze(-1)).any(dim=-1).float().mean().item(),
    )


def _rates(tt_rows, hf_rows):
    """TT-vs-HF top-1 equality and top-k containment, per prediction."""
    tt_top1 = tt_rows.argmax(dim=-1)
    hf_top1 = hf_rows.argmax(dim=-1)
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


def _prepare_teacher_forcing_run(prompt_len, max_new_tokens, mesh_device, request):
    """Shared setup for token-accuracy and logit-PCC teacher-forced e2e tests."""
    skip_if_config_only_checkpoint()

    max_prefill = request.config.getoption("--max-prefill")
    if prompt_len > max_prefill:
        pytest.skip(f"prompt_len={prompt_len} > --max-prefill={max_prefill}")

    model_path = _get_model_path()
    total_len = prompt_len + max_new_tokens + 1
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
        prompt_len=prompt_len,
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

    tt_rows = _run_teacher_forced_e2e(generator, tt_kv_cache, page_table, tokens, prompt_len, max_new_tokens)
    hf_rows = _hf_reference_rows(model_path, tokens, prompt_len, max_new_tokens)

    vocab = int(model_args.vocab_size)
    tt_rows = tt_rows[:, :vocab]
    hf_rows = hf_rows[:, :vocab]
    truth = tokens[0, prompt_len : prompt_len + tt_rows.shape[0]].long()

    return {
        "model_path": model_path,
        "tokenizer": tokenizer,
        "tt_rows": tt_rows,
        "hf_rows": hf_rows,
        "truth": truth,
        "prompt_len": prompt_len,
        "max_new_tokens": max_new_tokens,
    }


@pytest.mark.timeout(3600)
@parametrize_mesh_with_fabric()
@pytest.mark.parametrize("prompt_len,max_new_tokens", _CASES, ids=_CASE_IDS)
def test_teacher_forcing_e2e(prompt_len, max_new_tokens, mesh_device, reset_seeds, request):
    """Score prefill + every teacher-forced decode step, top-1 and top-5."""
    run = _prepare_teacher_forcing_run(prompt_len, max_new_tokens, mesh_device, request)
    tt_rows = run["tt_rows"]
    hf_rows = run["hf_rows"]
    truth = run["truth"]
    tokenizer = run["tokenizer"]

    r = _rates(tt_rows, hf_rows)
    pccs = _logit_pccs(tt_rows, hf_rows)

    logger.info("=" * 78)
    logger.info("Teacher-forced accuracy — P={} N={} ({} predictions)", prompt_len, max_new_tokens, tt_rows.shape[0])
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

    assert top1_rate >= _MIN_TOP1_AGREEMENT and top5_rate >= _MIN_TOP5_AGREEMENT, (
        f"e2e teacher forcing below threshold: top1={top1_rate:.2%} (min {_MIN_TOP1_AGREEMENT:.2%}), "
        f"top{_TOP_K}={top5_rate:.2%} (min {_MIN_TOP5_AGREEMENT:.2%}); "
        f"{len(mismatched)}/{tt_rows.shape[0]} top-1 mismatches, {len(outside)} outside HF top-{_TOP_K}. "
        f"See the parameter block above for the weight dtypes and fidelity knobs in effect."
    )


@pytest.mark.gemma4_hf_direct_parity
@pytest.mark.timeout(3600)
@parametrize_mesh_with_fabric()
@pytest.mark.parametrize("prompt_len,max_new_tokens", _CASES, ids=_CASE_IDS)
def test_e2e_logits_pcc(prompt_len, max_new_tokens, mesh_device, reset_seeds, request):
    """Assert full-vocab logit PCC (TT vs HF) at every teacher-forced step."""
    run = _prepare_teacher_forcing_run(prompt_len, max_new_tokens, mesh_device, request)
    tt_rows = run["tt_rows"]
    hf_rows = run["hf_rows"]

    pcc_threshold = float(os.getenv("GEMMA4_TF_MIN_LOGIT_PCC", get_pcc_threshold(request)))
    pccs = _logit_pccs(tt_rows, hf_rows)

    logger.info("=" * 78)
    logger.info(
        "Teacher-forced logit PCC — P={} N={} ({} steps, threshold={:.4f})",
        prompt_len,
        max_new_tokens,
        len(pccs),
        pcc_threshold,
    )
    logger.info("=" * 78)
    _log_pcc_segment("prefill", pccs, slice(0, 1))
    _log_pcc_segment("decode", pccs, slice(1, None))
    _log_pcc_segment("e2e", pccs, slice(None))
    _log_per_step_pcc(pccs)
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
