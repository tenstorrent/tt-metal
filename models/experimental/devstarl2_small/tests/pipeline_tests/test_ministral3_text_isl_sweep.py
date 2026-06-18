# SPDX-FileCopyrightText: (c) 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import gc
import io
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

import ttnn
from models.experimental.devstarl2_small.demo import tt_image_demo


OUTPUT_SEQ_LEN = 200
_DEFAULT_ISLS = (4096, 8192, 16384, 32768, 65536, 131072, 262144)
_DEFAULT_MESSAGES_JSON = Path("models/experimental/devstarl2_small/demo/messages_256k_text.json")
_SYNTHETIC_PROMPT_TOKEN_COUNT = 300_000

_TTFT_RE = re.compile(r"TTFT \(prompt -> 1st new tok\)\s+([0-9.]+) ms")
_FIRST_TRACED_RE = re.compile(r"First traced decode step latency\s+([0-9.]+) ms")
_STEADY_LAT_RE = re.compile(r"Steady-state decode latency / tok\s+([0-9.]+) ms")
_STEADY_RE = re.compile(r"Steady-state decode throughput\s+([0-9.]+) tok/s")
_E2E_RE = re.compile(r"End-to-end throughput\s+([0-9.]+) tok/s")


@dataclass
class _SweepResult:
    sweep_id: str
    isl: int
    mesh_width: int
    prefill_tokens: int
    decode_t_s_u: float
    decode_t_s: float
    ttft_ms: float | None = None
    first_traced_ms: float | None = None
    steady_latency_ms: float | None = None
    e2e_tok_s: float | None = None
    skipped_reason: str | None = None
    model_output: str | None = None


def _parse_isl_sweep() -> tuple[int, ...]:
    raw = os.environ.get("DEVSTRAL_TEXT_ISL_SWEEP_VALUES")
    if raw is None:
        return _DEFAULT_ISLS
    out = []
    for part in raw.split(","):
        token = part.strip().lower()
        if not token:
            continue
        multiplier = 1024 if token.endswith("k") else 1
        if token.endswith("k"):
            token = token[:-1]
        out.append(int(token) * multiplier)
    return tuple(out)


def _mesh_width() -> int:
    raw = os.environ.get("DEVSTRAL_TEXT_ISL_SWEEP_MESH_WIDTH")
    if raw is not None:
        return int(raw)
    return 4


def _metric(pattern: re.Pattern[str], output: str, name: str) -> float:
    match = pattern.search(output)
    if match is None:
        raise AssertionError(f"Could not parse {name} from image demo output:\n{output}")
    return float(match.group(1))


def _format_isl(isl: int) -> str:
    if isl >= 1024 and isl % 1024 == 0:
        return f"{isl // 1024}k"
    return str(isl)


def _sweep_id(isl: int) -> str:
    return f"b1_isl{_format_isl(isl)}"


def _ensure_messages_json(path: Path) -> None:
    if path.exists():
        return
    if path != _DEFAULT_MESSAGES_JSON:
        raise FileNotFoundError(f"Missing benchmark messages JSON: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    prompt = " ".join(["token"] * _SYNTHETIC_PROMPT_TOKEN_COUNT)
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ]
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _messages_text(path: Path) -> str:
    raw = json.loads(path.read_text(encoding="utf-8"))
    block = raw["scenarios"][raw.get("default_scenario")] if "scenarios" in raw else raw
    pieces = []
    for message in block.get("messages", []):
        content = message.get("content", "")
        if isinstance(content, str):
            pieces.append(content)
        else:
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    pieces.append(str(item.get("text", "")))
    text = "\n".join(piece for piece in pieces if piece)
    if not text:
        raise ValueError(f"{path}: no text content found in messages")
    return text


_PROCESSOR_CACHE = None


def _processor():
    global _PROCESSOR_CACHE
    if _PROCESSOR_CACHE is None:
        _PROCESSOR_CACHE = tt_image_demo.AutoProcessor.from_pretrained(
            tt_image_demo._DEFAULT_MODEL_ID,
            trust_remote_code=True,
            fix_mistral_regex=True,
            cache_dir=os.getenv("HF_TOKENIZER_CACHE") or os.getenv("HF_HUB_CACHE") or None,
            local_files_only=False,
            token=os.getenv("HF_TOKEN") or None,
        )
    return _PROCESSOR_CACHE


def _text_from_token_count(source_text: str, token_count: int) -> str:
    tokenizer = getattr(_processor(), "tokenizer", _processor())
    source_tokens = tokenizer.encode(source_text)
    if not source_tokens:
        return source_text
    if token_count <= 0:
        return ""
    if len(source_tokens) < token_count:
        repeats = (token_count + len(source_tokens) - 1) // len(source_tokens)
        source_tokens = (source_tokens * repeats)[:token_count]
    else:
        source_tokens = source_tokens[:token_count]
    return tokenizer.decode(source_tokens)


def _multimodal_messages(text: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": text},
            ],
        }
    ]


def _prompt_token_count(
    messages: list[dict], image_path: Path, vision_max_edge: int, vision_square_pixels: int | None
) -> int:
    image = tt_image_demo.Image.open(image_path).convert("RGB")
    image = tt_image_demo._prepare_vision_image(image, vision_max_edge, vision_square_pixels)
    processor = _processor()
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    proc_out = processor(text=prompt, images=image, return_tensors="pt")
    return int(proc_out["input_ids"].shape[1])


def _fit_multimodal_messages(
    source_text: str,
    max_seq_len: int,
    image_path: Path,
    vision_max_edge: int,
    vision_square_pixels: int | None,
) -> tuple[list[dict], int]:
    min_messages = _multimodal_messages(_text_from_token_count(source_text, 0))
    min_tokens = _prompt_token_count(min_messages, image_path, vision_max_edge, vision_square_pixels)
    if min_tokens > max_seq_len:
        return min_messages, min_tokens

    low = 0
    high = max(1, max_seq_len - min_tokens)
    best_messages = min_messages
    best_tokens = min_tokens

    while True:
        messages = _multimodal_messages(_text_from_token_count(source_text, high))
        tokens = _prompt_token_count(messages, image_path, vision_max_edge, vision_square_pixels)
        if tokens > max_seq_len:
            break
        best_messages = messages
        best_tokens = tokens
        low = high
        high *= 2

    while low + 1 < high:
        mid = (low + high) // 2
        messages = _multimodal_messages(_text_from_token_count(source_text, mid))
        tokens = _prompt_token_count(messages, image_path, vision_max_edge, vision_square_pixels)
        if tokens <= max_seq_len:
            best_messages = messages
            best_tokens = tokens
            low = mid
        else:
            high = mid

    return best_messages, best_tokens


def _clear_memory_after_isl() -> None:
    for attr in vars(tt_image_demo.ModelArgs).values():
        cache_clear = getattr(attr, "cache_clear", None)
        if cache_clear is not None:
            with contextlib.suppress(Exception):
                cache_clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _run_image_demo_for_isl(
    monkeypatch,
    *,
    source_text: str,
    mesh_width: int,
    isl: int,
    prefill_chunk_size: int,
    image_path: Path,
    vision_max_edge: int,
    vision_square_pixels: int | None,
) -> _SweepResult:
    sweep_id = _sweep_id(isl)
    messages, prefill_tokens = _fit_multimodal_messages(
        source_text, isl, image_path, vision_max_edge, vision_square_pixels
    )
    if prefill_tokens > isl:
        return _SweepResult(
            sweep_id=sweep_id,
            isl=isl,
            mesh_width=mesh_width,
            prefill_tokens=prefill_tokens,
            decode_t_s_u=0.0,
            decode_t_s=0.0,
            skipped_reason=f"prompt is {prefill_tokens} tokens but context window max_seq_len={isl}",
        )

    stdout = io.StringIO()
    with monkeypatch.context() as scoped_monkeypatch, contextlib.redirect_stdout(stdout):
        scoped_monkeypatch.setattr(tt_image_demo, "MODEL_LOADING_MESSAGES", messages)
        tt_image_demo.run_tt(
            tt_image_demo._DEFAULT_MODEL_ID,
            image_path,
            mesh_width=mesh_width,
            text_layers=None,
            max_new_tokens=OUTPUT_SEQ_LEN,
            greedy=True,
            temperature=tt_image_demo._DEFAULT_SAMPLE_TEMPERATURE,
            seed=0,
            lm_head_max_device_cols=None,
            vision_max_edge=vision_max_edge,
            vision_square_pixels=vision_square_pixels,
            prefill_chunk_size=prefill_chunk_size,
            clear_weight_cache=False,
            perf=False,
        )
    output = stdout.getvalue()

    steady_tok_s = _metric(_STEADY_RE, output, "steady-state decode throughput")
    return _SweepResult(
        sweep_id=sweep_id,
        isl=isl,
        mesh_width=mesh_width,
        prefill_tokens=prefill_tokens,
        decode_t_s_u=steady_tok_s,
        decode_t_s=steady_tok_s,
        ttft_ms=_metric(_TTFT_RE, output, "TTFT"),
        first_traced_ms=_metric(_FIRST_TRACED_RE, output, "first traced decode step latency"),
        steady_latency_ms=_metric(_STEADY_LAT_RE, output, "steady-state decode latency"),
        e2e_tok_s=_metric(_E2E_RE, output, "end-to-end throughput"),
        model_output=output if isl == 262144 else None,
    )


def _print_result(result: _SweepResult) -> None:
    print()
    print("------------------------------------------------------------------------------")
    status = "skipped" if result.skipped_reason else "complete"
    print(f"  ISL {_format_isl(result.isl)} {status}  (mesh_width={result.mesh_width})")
    print("------------------------------------------------------------------------------")
    print(f"  Prefill tokens                     {result.prefill_tokens:>10}")
    if result.skipped_reason:
        print(f"  Skip reason                        {result.skipped_reason}")
    else:
        assert result.ttft_ms is not None
        assert result.first_traced_ms is not None
        assert result.steady_latency_ms is not None
        assert result.e2e_tok_s is not None
        print(f"  TTFT (prompt -> 1st new tok)       {result.ttft_ms:>10.2f} ms")
        print(f"  First traced decode step latency   {result.first_traced_ms:>10.2f} ms")
        print(f"  Steady-state decode latency / tok  {result.steady_latency_ms:>10.2f} ms")
        print(f"  Steady-state decode throughput     {result.decode_t_s_u:>10.3f} tok/s/user")
        print(f"  Aggregate decode throughput        {result.decode_t_s:>10.3f} tok/s")
        print(f"  End-to-end throughput              {result.e2e_tok_s:>10.3f} tok/s")
        if result.model_output is not None:
            print("------------------------------------------------------------------------------")
            print("  Model output for ISL 256k")
            print("------------------------------------------------------------------------------")
            print(result.model_output.rstrip())
    print("------------------------------------------------------------------------------")


def _print_report(results: list[_SweepResult], messages_json: Path) -> None:
    print()
    print("------------------------------------------------------------------------------")
    print("  Devstral image+text ISL sweep decode throughput summary")
    print("------------------------------------------------------------------------------")
    print(f"  messages_json: {messages_json}")
    print(f"  output_seq_len: {OUTPUT_SEQ_LEN}")
    print("------------------------------------------------------------------------------")
    print("| config       | max_seq_len | prefill_tok | decode_t/s/u | decode_t/s | status  |")
    print("| ------------ | ----------- | ----------- | ------------ | ---------- | ------- |")
    for result in results:
        status = "skipped" if result.skipped_reason else "ok"
        print(
            f"| {result.sweep_id:<12} | {result.isl:>11} | {result.prefill_tokens:>11} | "
            f"{result.decode_t_s_u:>12.2f} | {result.decode_t_s:>10.2f} | {status:<7} |"
        )
    print("------------------------------------------------------------------------------")


@pytest.mark.timeout(28800)
@pytest.mark.models_performance_bare_metal
def test_devstral_image_text_isl_sweep_perf(monkeypatch):
    messages_json = Path(os.environ.get("DEVSTRAL_TEXT_ISL_SWEEP_MESSAGES_JSON", str(_DEFAULT_MESSAGES_JSON)))
    _ensure_messages_json(messages_json)
    source_text = _messages_text(messages_json)

    mesh_width = _mesh_width()
    if mesh_width > ttnn.get_num_devices():
        pytest.skip(f"mesh_width={mesh_width} requested but only {ttnn.get_num_devices()} device(s) are visible.")

    prefill_chunk_size = int(os.environ.get("DEVSTRAL_TEXT_ISL_SWEEP_PREFILL_CHUNK_SIZE", "8192"))
    image_path = Path(os.environ.get("DEVSTRAL_TEXT_ISL_SWEEP_IMAGE", str(tt_image_demo._sample_image_path())))
    vision_max_edge = int(os.environ.get("DEVSTRAL_TEXT_ISL_SWEEP_VISION_MAX_EDGE", "0"))
    vision_square_pixels_raw = os.environ.get("DEVSTRAL_TEXT_ISL_SWEEP_VISION_SQUARE_PIXELS")
    vision_square_pixels = int(vision_square_pixels_raw) if vision_square_pixels_raw else None

    results = []
    for isl in _parse_isl_sweep():
        print(f"running ISL={_format_isl(isl)} mesh_width={mesh_width}")
        try:
            result = _run_image_demo_for_isl(
                monkeypatch,
                source_text=source_text,
                mesh_width=mesh_width,
                isl=isl,
                prefill_chunk_size=prefill_chunk_size,
                image_path=image_path,
                vision_max_edge=vision_max_edge,
                vision_square_pixels=vision_square_pixels,
            )
        finally:
            _clear_memory_after_isl()
        results.append(result)
        _print_result(result)

    _print_report(results, messages_json)
