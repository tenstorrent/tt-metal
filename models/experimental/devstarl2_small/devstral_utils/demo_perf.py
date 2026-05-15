# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Readable wall-clock / context logging for Devstral demo scripts."""

from __future__ import annotations

import os
import time
from typing import Any

from loguru import logger

_BLOCK_WIDTH = 62

# Short keys passed from demos → human labels in log output.
_PERF_LABELS: dict[str, str] = {
    "hf.load_processor_and_model": "Load processor + HF model",
    "hf.model.generate": "HF generate()",
    "hf.processor.batch_decode": "Decode output tokens",
    "tt.hf_processor_chat_template_and_pixel_batch": "Processor (chat template + pixels)",
    "tt.open_mesh_device": "Open TT mesh device",
    "tt.model_args_init_and_load_state_dict": "ModelArgs + checkpoint load",
    "tt.ModelArgs_init_and_load_state_dict": "ModelArgs + checkpoint load",
    "tt.TtDevstral2SmallModel_init": "TtDevstral2SmallModel init",
    "tt.TtMinistral3Model_init": "TtMinistral3Model init",
    "tt.vision_projector_and_rows_to_torch": "Vision projector → host rows",
    "tt.lm_head_setup": "LM head setup",
    "tt.sampling_generator_setup": "SamplingGenerator setup",
    "tt.tokenizer_for_decode": "Tokenizer (decode / EOS)",
    "tt.tokenizer_load_and_apply_chat_template": "Tokenizer + chat template",
    "tt.prompt_prefill_tt_prefill_hidden_states_from_ids": "Prompt TT prefill (full stack)",
    "tt.verify_hf_reference_vs_tt_prefill": "PCC verify (HF ref vs TT prefill)",
    "tt.hf_baseline_model.generate": "HF baseline generate()",
    "tt.host_input_ids_to_tt_replicated": "Upload prompt ids to device",
    "tt.autoregressive_loop_wall": "Autoregressive loop (wall clock)",
    "tt.tt_replicated_ids_to_torch_long": "Pull generated ids to host",
}


def _fmt_int(n: int | float) -> str:
    return f"{int(n):,}"


def fmt_duration(seconds: float) -> str:
    """Format a duration for logs (ms below 1s, else seconds with 2 decimals)."""
    ms = seconds * 1000.0
    if ms < 1000.0:
        return f"{ms:8.2f} ms"
    return f"{seconds:8.2f} s "


def perf_log_generation_steps() -> bool:
    """If true, log per-token timings during autoregressive loops (can be noisy)."""
    return os.environ.get("DEVSTRAL2_DEMO_PERF_LOG_STEPS", "").lower() in ("1", "true", "yes")


def _perf_label(phase_key: str) -> str:
    if phase_key in _PERF_LABELS:
        return _PERF_LABELS[phase_key]
    return phase_key.replace("tt.", "").replace("hf.", "").replace("_", " ").replace(".", " · ").title()


def log_block(title: str, rows: list[tuple[str, Any]], *, width: int = _BLOCK_WIDTH) -> None:
    """Log a titled key/value block (one ``logger.info`` call, multi-line)."""
    if not rows:
        return
    bar = "─" * width
    label_w = max(len(label) for label, _ in rows)
    lines = [bar, f"  {title}", bar]
    for label, value in rows:
        lines.append(f"  {label:<{label_w}}  {value}")
    lines.append(bar)
    logger.info("\n".join(lines))


def log_perf_ms(phase_key: str, t0: float) -> None:
    """Log elapsed time for one setup or inference phase."""
    elapsed = time.perf_counter() - t0
    label = _perf_label(phase_key)
    logger.info(f"  ▶ {label:<44} {fmt_duration(elapsed)}")


def log_hf_context_before(*, prompt_tokens: int, max_new_tokens: int) -> None:
    log_block(
        "HF · context",
        [
            ("Prompt tokens", _fmt_int(prompt_tokens)),
            ("Max new tokens", _fmt_int(max_new_tokens)),
        ],
    )


def log_hf_context_after(*, output_tokens: int, prompt_tokens: int) -> None:
    new_tok = output_tokens - prompt_tokens
    log_block(
        "HF · context (after generate)",
        [
            ("Output sequence", f"{_fmt_int(output_tokens)} tokens"),
            ("New tokens", _fmt_int(new_tok)),
        ],
    )


def log_tt_context_budget(
    *,
    prompt_tokens: int,
    budget_need_tokens: int,
    rope_max_seq_len: int,
    kv_cache_seq_dim: int,
) -> None:
    log_block(
        "TT · context budget",
        [
            ("Prompt tokens", _fmt_int(prompt_tokens)),
            ("Run budget (prompt + gen + margin)", f"≈ {_fmt_int(budget_need_tokens)}"),
            ("RoPE max seq len", _fmt_int(rope_max_seq_len)),
            ("KV cache seq dim", _fmt_int(kv_cache_seq_dim)),
        ],
    )


def log_tt_prefill_context(
    *,
    language_prompt_tokens: int,
    tt_prefill_padded_len: int,
    rope_max_seq_len: int,
) -> None:
    pad_note = ""
    if tt_prefill_padded_len != language_prompt_tokens:
        pad_note = f"  (+{_fmt_int(tt_prefill_padded_len - language_prompt_tokens)} pad for TT alignment)"
    log_block(
        "TT · prompt prefill",
        [
            ("Language prompt", f"{_fmt_int(language_prompt_tokens)} tokens"),
            ("TT prefill length", f"{_fmt_int(tt_prefill_padded_len)} tokens{pad_note}"),
            ("RoPE max seq len", _fmt_int(rope_max_seq_len)),
        ],
    )


def log_generation_start(
    *,
    backend: str,
    current_sequence_tokens: int,
    prompt_tokens: int,
    max_new_tokens: int,
    mode: str,
    lm_mode: str,
    sampling_mode: str,
    extra: str | None = None,
) -> None:
    rows: list[tuple[str, Any]] = [
        ("Backend", backend),
        ("Sequence now", f"{_fmt_int(current_sequence_tokens)} tokens  (prompt {_fmt_int(prompt_tokens)})"),
        ("Max new tokens", _fmt_int(max_new_tokens)),
        ("Sampling", mode),
        ("LM head", lm_mode),
        ("Token pick", sampling_mode),
    ]
    if extra:
        rows.append(("Note", extra))
    log_block(f"{backend} · generation start", rows)


def log_generation_done(*, backend: str, final_sequence_tokens: int, prompt_tokens: int) -> None:
    new_tok = final_sequence_tokens - prompt_tokens
    log_block(
        f"{backend} · generation done",
        [
            ("Final sequence", f"{_fmt_int(final_sequence_tokens)} tokens"),
            ("New tokens", _fmt_int(new_tok)),
        ],
    )


def log_model_output(*, backend: str, text: str, new_tokens: int) -> None:
    """Log generated text in a bordered block."""
    bar = "─" * _BLOCK_WIDTH
    logger.info(f"{bar}\n  {backend} · output  ({_fmt_int(new_tokens)} new tokens)\n{bar}\n{text}\n{bar}")


def log_gen_step_timings(
    *,
    step_index: int,
    sequence_tokens: int,
    phase_seconds: dict[str, float],
) -> None:
    """Per-step timing (only when ``DEVSTRAL2_DEMO_PERF_LOG_STEPS`` is set)."""
    parts = [f"{label} {fmt_duration(sec).strip()}" for label, sec in phase_seconds.items()]
    logger.info(f"  · step {step_index:>3}  │  seq {_fmt_int(sequence_tokens):>6}  │  " + "  │  ".join(parts))


def log_gen_step_summary(
    *,
    n_steps: int,
    wall_seconds: float,
    avg_phase_seconds: dict[str, float],
    throughput_label: str = "Throughput",
    throughput_value: float | None = None,
) -> None:
    """Table of average per-step phase times after the autoregressive loop."""
    if n_steps <= 0:
        return

    bar = "─" * _BLOCK_WIDTH
    total_avg = sum(avg_phase_seconds.values()) or 1.0
    label_w = max(len(k) for k in avg_phase_seconds) if avg_phase_seconds else 10

    lines = [
        bar,
        f"  Generation timing  ({_fmt_int(n_steps)} step(s), wall {fmt_duration(wall_seconds).strip()})",
        bar,
        f"  {'Phase':<{label_w}}   {'avg/step':>10}   {'%':>6}",
    ]
    for phase, avg_s in avg_phase_seconds.items():
        pct = 100.0 * avg_s / total_avg
        lines.append(f"  {phase:<{label_w}}   {fmt_duration(avg_s).strip():>10}   {pct:5.1f}%")

    if throughput_value is not None:
        lines.append(bar)
        lines.append(f"  {throughput_label:<{label_w}}   {throughput_value:>10.3f} tok/s")
    lines.append(bar)
    logger.info("\n".join(lines))
