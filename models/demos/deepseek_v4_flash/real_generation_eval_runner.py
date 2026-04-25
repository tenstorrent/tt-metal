# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Compact generation/eval-shaped wrapper for the real DSV4 Flash decode smoke.

This module intentionally stays on the verified real multi-token smoke path:
real input IDs, host embedding lookup, layers 2->3 by default, carried per-layer
cache state, final norm, and sliced-vocab LM head unless full-vocab smoke is
requested. It is not a vLLM or tt-inference-server integration.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from models.demos.deepseek_v4_flash.real_decode_logits_smoke import DEFAULT_DECODE_LOGITS_TOP_K
from models.demos.deepseek_v4_flash.real_decode_stack_logits_smoke import DEFAULT_DECODE_STACK_LAYERS
from models.demos.deepseek_v4_flash.real_input_stack_logits_smoke import DEFAULT_INPUT_STACK_MAX_EMBEDDING_ROWS
from models.demos.deepseek_v4_flash.real_multi_token_decode_smoke import (
    DEFAULT_MULTI_TOKEN_MAX_BYTES,
    DEFAULT_MULTI_TOKEN_MAX_TENSORS,
    run_real_multi_token_decode_smoke,
)
from models.demos.deepseek_v4_flash.real_prefill_cache_prep_smoke import DEFAULT_SEQUENCE_LENGTH

REAL_GENERATION_EVAL_RUNNER_SCHEMA_VERSION = 1
RUNNER_NAME = "deepseek_v4_flash_real_generation_eval_runner"
DEFAULT_REAL_SNAPSHOT_DIR = Path("/proj_sw/user_dev/moconnor/deepseek_v4_flash_hf")
DEFAULT_EVAL_DECODE_STEPS = 2


@dataclass(frozen=True)
class _ResolvedInput:
    input_ids: list[int] | None
    source: str
    source_path: str | None
    prompt_label: str | None


def run_real_generation_eval_runner(
    snapshot_dir: str | Path,
    *,
    layers: Sequence[int] = DEFAULT_DECODE_STACK_LAYERS,
    prefill_seq_len: int = DEFAULT_SEQUENCE_LENGTH,
    decode_steps: int = DEFAULT_EVAL_DECODE_STEPS,
    input_ids: Sequence[int] | None = None,
    input_ids_path: str | Path | None = None,
    prompt_path: str | Path | None = None,
    input_id_start: int = 0,
    prompt_label: str | None = None,
    embedding_mode: str = "slice",
    max_embedding_rows: int = DEFAULT_INPUT_STACK_MAX_EMBEDDING_ROWS,
    vocab_mode: str = "slice",
    full_vocab_smoke: bool = False,
    vocab_start: int = 0,
    vocab_size: int | None = None,
    top_k: int = DEFAULT_DECODE_LOGITS_TOP_K,
    max_tensors: int = DEFAULT_MULTI_TOKEN_MAX_TENSORS,
    max_bytes: int = DEFAULT_MULTI_TOKEN_MAX_BYTES,
    cpu_only: bool = False,
    device_id: int = 0,
    prefill_hidden_pcc: float = 0.99,
    layer_hidden_pcc: float = 0.99,
    final_norm_pcc: float = 0.999,
    logits_pcc: float = 0.99,
    rtol: float = 8e-2,
    atol: float = 8e-2,
    residual_atol: float = 3e-1,
    logits_rtol: float = 1e-1,
    logits_atol: float = 1.0,
    top_logit_atol: float = 1.0,
) -> dict[str, Any]:
    snapshot_dir = Path(snapshot_dir).expanduser().resolve()
    total_tokens = int(prefill_seq_len) + int(decode_steps)
    resolved_input = _resolve_input(
        snapshot_dir=snapshot_dir,
        total_tokens=total_tokens,
        input_ids=input_ids,
        input_ids_path=input_ids_path,
        prompt_path=prompt_path,
        prompt_label=prompt_label,
    )
    effective_vocab_mode = "full" if full_vocab_smoke else vocab_mode
    effective_vocab_start = 0 if full_vocab_smoke else vocab_start
    effective_vocab_size = None if full_vocab_smoke else vocab_size

    runner_start = time.perf_counter()
    smoke_result = run_real_multi_token_decode_smoke(
        snapshot_dir,
        layers=layers,
        prefill_seq_len=prefill_seq_len,
        decode_steps=decode_steps,
        input_ids=resolved_input.input_ids,
        input_id_start=input_id_start,
        prompt_label=resolved_input.prompt_label,
        embedding_mode=embedding_mode,  # type: ignore[arg-type]
        max_embedding_rows=max_embedding_rows,
        vocab_mode=effective_vocab_mode,  # type: ignore[arg-type]
        vocab_start=effective_vocab_start,
        vocab_size=effective_vocab_size,
        top_k=top_k,
        max_tensors=max_tensors,
        max_bytes=max_bytes,
        cpu_only=cpu_only,
        device_id=device_id,
        prefill_hidden_pcc=prefill_hidden_pcc,
        layer_hidden_pcc=layer_hidden_pcc,
        final_norm_pcc=final_norm_pcc,
        logits_pcc=logits_pcc,
        rtol=rtol,
        atol=atol,
        residual_atol=residual_atol,
        logits_rtol=logits_rtol,
        logits_atol=logits_atol,
        top_logit_atol=top_logit_atol,
    )
    runner_wall_seconds = _seconds_since(runner_start)
    return summarize_real_generation_eval_result(
        smoke_result,
        resolved_input=resolved_input,
        runner_wall_seconds=runner_wall_seconds,
        full_vocab_smoke=full_vocab_smoke,
    )


def summarize_real_generation_eval_result(
    smoke_result: Mapping[str, Any],
    *,
    resolved_input: _ResolvedInput,
    runner_wall_seconds: float,
    full_vocab_smoke: bool,
) -> dict[str, Any]:
    timing = _compact_timing(smoke_result.get("timing", {}), decode_steps=int(smoke_result["decode_steps"]))
    timing["runner_wall_seconds"] = runner_wall_seconds
    steps = [_compact_step(step) for step in smoke_result["steps"]]
    return {
        "schema_version": REAL_GENERATION_EVAL_RUNNER_SCHEMA_VERSION,
        "runner": RUNNER_NAME,
        "wrapped_smoke_schema_version": smoke_result["schema_version"],
        "mode": smoke_result["mode"],
        "passed": bool(smoke_result["passed"]),
        "snapshot_dir": smoke_result["snapshot_dir"],
        "layers": smoke_result["layers"],
        "prefill_sequence_length": smoke_result["prefill_sequence_length"],
        "decode_steps": smoke_result["decode_steps"],
        "positions": {
            "per_step": smoke_result["current_positions"],
            "next_position": smoke_result["next_position"],
        },
        "input": {
            "source": resolved_input.source,
            "source_path": resolved_input.source_path,
            "prompt_label": smoke_result["input"]["prompt_label"],
            "token_ids": smoke_result["input"]["token_ids"],
            "prefill_token_ids": smoke_result["input"]["prefill_token_ids"],
            "supplied_decode_token_ids": smoke_result["input"]["supplied_decode_token_ids"],
            "decode_feed_mode": smoke_result["decode_feed_mode"],
            "deterministic_notice": (
                "deterministic contiguous input IDs were used"
                if resolved_input.source == "deterministic_contiguous_input_ids"
                else None
            ),
        },
        "generated": smoke_result["generated"],
        "top_k": {
            "k": len(steps[0]["reference_top_k"]) if steps else 0,
            "per_step": [
                {
                    "step_index": step["step_index"],
                    "reference": step["reference_top_k"],
                    "ttnn": step["ttnn_top_k"],
                }
                for step in steps
            ],
        },
        "vocab": {
            **smoke_result["vocab"],
            "full_vocab_smoke_requested": bool(full_vocab_smoke),
        },
        "payload_bytes": smoke_result["payload_bytes"],
        "host_boundaries": smoke_result["host_boundaries"],
        "ttnn_ops": smoke_result["ttnn_ops"],
        "timing": timing,
        "correctness": {
            "passed": bool(smoke_result["passed"]),
            "ttnn_compared_to_torch": smoke_result["mode"] == "ttnn",
            "top1_ids_match": smoke_result["generated"]["top1_ids_match"],
            "top_level_accuracy": smoke_result["accuracy"],
            "per_step": [
                {
                    "step_index": step["step_index"],
                    "passed": step["passed"],
                    "accuracy": step["accuracy"],
                }
                for step in steps
            ],
        },
        "steps": steps,
        "limitations": {
            "serving_eval_boundaries": smoke_result["stack_scope"]["serving_eval_boundaries"],
            "generation": smoke_result["generated"]["limitation"],
            "host_visible_cache_boundaries_preserved": True,
        },
    }


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a compact DeepSeek V4 Flash real generation/eval smoke.")
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=Path(os.environ.get("DSV4_FLASH_REAL_SNAPSHOT_DIR", str(DEFAULT_REAL_SNAPSHOT_DIR))),
    )
    parser.add_argument("--layers", type=int, nargs="+", default=list(DEFAULT_DECODE_STACK_LAYERS))
    parser.add_argument("--prefill-seq-len", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument("--decode-steps", type=int, default=DEFAULT_EVAL_DECODE_STEPS)
    parser.add_argument("--input-ids", type=int, nargs="+")
    parser.add_argument("--input-ids-path", type=Path)
    parser.add_argument(
        "--prompt-path",
        type=Path,
        help="Optional text prompt file tokenized with the local HF snapshot tokenizer. Requires enough tokens.",
    )
    parser.add_argument("--input-id-start", type=int, default=0)
    parser.add_argument("--prompt-label")
    parser.add_argument("--embedding-mode", choices=("slice", "full"), default="slice")
    parser.add_argument("--max-embedding-rows", type=int, default=DEFAULT_INPUT_STACK_MAX_EMBEDDING_ROWS)
    parser.add_argument(
        "--vocab-mode",
        choices=("full", "slice"),
        default=os.environ.get("DSV4_FLASH_REAL_GENERATION_EVAL_VOCAB_MODE", "slice"),
    )
    parser.add_argument("--full-vocab-smoke", action="store_true")
    parser.add_argument(
        "--vocab-start",
        type=int,
        default=_optional_int_env("DSV4_FLASH_REAL_GENERATION_EVAL_VOCAB_START", 0),
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=_optional_int_env("DSV4_FLASH_REAL_GENERATION_EVAL_VOCAB_SIZE"),
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_DECODE_LOGITS_TOP_K)
    parser.add_argument("--max-tensors", type=int, default=DEFAULT_MULTI_TOKEN_MAX_TENSORS)
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MULTI_TOKEN_MAX_BYTES)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--prefill-hidden-pcc", type=float, default=0.99)
    parser.add_argument("--layer-hidden-pcc", type=float, default=0.99)
    parser.add_argument("--final-norm-pcc", type=float, default=0.999)
    parser.add_argument("--logits-pcc", type=float, default=0.99)
    parser.add_argument("--rtol", type=float, default=8e-2)
    parser.add_argument("--atol", type=float, default=8e-2)
    parser.add_argument("--residual-atol", type=float, default=3e-1)
    parser.add_argument("--logits-rtol", type=float, default=1e-1)
    parser.add_argument("--logits-atol", type=float, default=1.0)
    parser.add_argument("--top-logit-atol", type=float, default=1.0)
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--verbose-logs", action="store_true")
    return parser


def main() -> None:
    args = create_arg_parser().parse_args()
    if not args.verbose_logs:
        os.environ.setdefault("TT_LOGGER_LEVEL", "FATAL")
        os.environ.setdefault("LOGGER_LEVEL", "Warning")

    result = run_real_generation_eval_runner(
        args.snapshot_dir,
        layers=args.layers,
        prefill_seq_len=args.prefill_seq_len,
        decode_steps=args.decode_steps,
        input_ids=args.input_ids,
        input_ids_path=args.input_ids_path,
        prompt_path=args.prompt_path,
        input_id_start=args.input_id_start,
        prompt_label=args.prompt_label,
        embedding_mode=args.embedding_mode,
        max_embedding_rows=args.max_embedding_rows,
        vocab_mode=args.vocab_mode,
        full_vocab_smoke=args.full_vocab_smoke,
        vocab_start=args.vocab_start,
        vocab_size=args.vocab_size,
        top_k=args.top_k,
        max_tensors=args.max_tensors,
        max_bytes=args.max_bytes,
        cpu_only=args.cpu_only,
        device_id=args.device_id,
        prefill_hidden_pcc=args.prefill_hidden_pcc,
        layer_hidden_pcc=args.layer_hidden_pcc,
        final_norm_pcc=args.final_norm_pcc,
        logits_pcc=args.logits_pcc,
        rtol=args.rtol,
        atol=args.atol,
        residual_atol=args.residual_atol,
        logits_rtol=args.logits_rtol,
        logits_atol=args.logits_atol,
        top_logit_atol=args.top_logit_atol,
    )
    print(json.dumps(result, indent=2 if args.pretty else None, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


def _resolve_input(
    *,
    snapshot_dir: Path,
    total_tokens: int,
    input_ids: Sequence[int] | None,
    input_ids_path: str | Path | None,
    prompt_path: str | Path | None,
    prompt_label: str | None,
) -> _ResolvedInput:
    provided = sum(value is not None for value in (input_ids, input_ids_path, prompt_path))
    if provided > 1:
        raise ValueError("Pass only one of input_ids, input_ids_path, or prompt_path")
    if input_ids is not None:
        values = _normalize_token_ids(input_ids, total_tokens=total_tokens, label="input_ids")
        return _ResolvedInput(values, "explicit_input_ids", None, prompt_label)
    if input_ids_path is not None:
        path = Path(input_ids_path).expanduser().resolve()
        values = _normalize_token_ids(_read_token_ids(path), total_tokens=total_tokens, label="input_ids_path")
        return _ResolvedInput(values, "prompt_ids_path", str(path), prompt_label or path.name)
    if prompt_path is not None:
        path = Path(prompt_path).expanduser().resolve()
        values = _tokenize_prompt_path(path, snapshot_dir=snapshot_dir, total_tokens=total_tokens)
        return _ResolvedInput(values, "prompt_text_path_hf_tokenizer", str(path), prompt_label or path.name)
    return _ResolvedInput(None, "deterministic_contiguous_input_ids", None, prompt_label)


def _read_token_ids(path: Path) -> list[int]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"{path} is empty")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = [part for part in text.replace(",", " ").split() if part]
    if isinstance(parsed, Mapping):
        if "input_ids" not in parsed:
            raise ValueError(f"{path} JSON object must contain an input_ids field")
        parsed = parsed["input_ids"]
    if not isinstance(parsed, Sequence) or isinstance(parsed, (str, bytes)):
        raise ValueError(f"{path} must contain a JSON list of token IDs or whitespace/comma-separated integers")
    return [int(value) for value in parsed]


def _tokenize_prompt_path(path: Path, *, snapshot_dir: Path, total_tokens: int) -> list[int]:
    prompt = path.read_text(encoding="utf-8")
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is required for --prompt-path; use --input-ids or --input-ids-path") from exc

    tokenizer = AutoTokenizer.from_pretrained(str(snapshot_dir), trust_remote_code=True, local_files_only=True)
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    return _normalize_token_ids(token_ids[:total_tokens], total_tokens=total_tokens, label="prompt_path token IDs")


def _normalize_token_ids(values: Sequence[int], *, total_tokens: int, label: str) -> list[int]:
    normalized = [int(value) for value in values]
    if len(normalized) != int(total_tokens):
        raise ValueError(f"{label} must contain exactly {total_tokens} token IDs, got {len(normalized)}")
    return normalized


def _compact_step(step: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "step_index": step["step_index"],
        "current_position": step["current_position"],
        "next_position": step["next_position"],
        "feed_token_id": step["feed_token_id"],
        "reference_top_k": step["reference"]["top_k"],
        "ttnn_top_k": step.get("ttnn", {}).get("top_k", []),
        "output_shapes": step["output_shapes"],
        "passed": step["passed"],
        "accuracy": step["accuracy"],
    }


def _compact_timing(timing: Mapping[str, Any], *, decode_steps: int) -> dict[str, Any]:
    ttnn_timing = timing.get("ttnn", {})
    ttnn_decode_total = ttnn_timing.get("decode_total_seconds") if isinstance(ttnn_timing, Mapping) else None
    reference_decode_total = timing.get("decode_build_total_seconds")
    decode_denominator = ttnn_decode_total if ttnn_decode_total is not None else reference_decode_total
    decode_denominator_name = (
        "ttnn_decode_total_seconds" if ttnn_decode_total is not None else "torch_reference_decode_build_total_seconds"
    )
    return {
        "setup_load_seconds": timing.get("setup_load_seconds"),
        "prefill_build_seconds": timing.get("prefill_build_seconds"),
        "decode_build_total_seconds": reference_decode_total,
        "decode_build_step_seconds": timing.get("decode_build_step_seconds", []),
        "logits_build_total_seconds": timing.get("logits_build_total_seconds"),
        "logits_build_step_seconds": timing.get("logits_build_step_seconds", []),
        "ttnn_prefill_seconds": ttnn_timing.get("prefill_total_seconds") if isinstance(ttnn_timing, Mapping) else None,
        "ttnn_prefill_layer_seconds": ttnn_timing.get("prefill_layer_seconds", [])
        if isinstance(ttnn_timing, Mapping)
        else [],
        "ttnn_decode_total_seconds": ttnn_decode_total,
        "ttnn_decode_step_seconds": ttnn_timing.get("decode_step_seconds", [])
        if isinstance(ttnn_timing, Mapping)
        else [],
        "ttnn_logits_total_seconds": (
            ttnn_timing.get("logits_total_seconds") if isinstance(ttnn_timing, Mapping) else None
        ),
        "ttnn_logits_step_seconds": ttnn_timing.get("logits_step_seconds", [])
        if isinstance(ttnn_timing, Mapping)
        else [],
        "ttnn_total_seconds": timing.get("ttnn_total_seconds"),
        "end_to_end_wall_seconds": timing.get("end_to_end_wall_seconds"),
        "decode_tokens_per_sec_per_user": _tokens_per_second(decode_steps, decode_denominator),
        "decode_tokens_per_sec_denominator": decode_denominator_name,
    }


def _tokens_per_second(tokens: int, seconds: Any) -> float | None:
    if seconds is None:
        return None
    seconds = float(seconds)
    if seconds <= 0.0:
        return None
    return round(float(tokens) / seconds, 6)


def _seconds_since(start: float) -> float:
    return round(max(0.0, time.perf_counter() - start), 6)


def _optional_int_env(name: str, default: int | None = None) -> int | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


if __name__ == "__main__":
    main()
