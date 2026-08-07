# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Devstral multimodal ISL sweep: prefill/decode perf vs context length (tt_image_demo).

Builds image+text prompts from a corpus file, tiles text when ISL exceeds corpus length,
runs ``tt_image_demo.run_tt`` per ISL point, and reports decode throughput plus a short
generation preview.

Run order: 256k first (cold device), then 4k → 128k ascending. No env vars — defaults are in this file;
mesh width follows ``MESH_DEVICE`` like other Devstral tests (P150 → 1, BH-QB → 4).
"""

from __future__ import annotations

import bz2
import contextlib
import gc
import io
import os
import re
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

import ttnn
from models.experimental.devstral2_small.demo import tt_image_demo

OUTPUT_SEQ_LEN = 200
PREFILL_CHUNK_SIZE = 8192
DEFAULT_ISLS = (4096, 8192, 16384, 32768, 65536, 131072, 262144)
DEFAULT_PROMPT_FILE = Path("models/tt_transformers/tests/tale-of-two-cities.txt.bz2")
DEFAULT_IMAGE = tt_image_demo._sample_image_path()

_DEMO_PERF_SEP = "──────────────────────────────────────────────────────────────"
_RE_TTFT = re.compile(r"TTFT \(prompt -> 1st new tok\)\s+([0-9.]+) ms")
_RE_FIRST_TRACED = re.compile(r"First traced decode step latency\s+([0-9.]+) ms")
_RE_STEADY_LAT = re.compile(r"Steady-state decode latency / tok\s+([0-9.]+) ms")
_RE_STEADY_TPS = re.compile(r"Steady-state decode throughput\s+([0-9.]+) tok/s")
_RE_E2E_TPS = re.compile(r"End-to-end throughput\s+([0-9.]+) tok/s")


@dataclass(frozen=True)
class SweepConfig:
    prompt_file: Path
    mesh_width: int
    prefill_chunk_size: int
    image_path: Path
    vision_max_edge: int
    vision_square_pixels: int | None
    isls: tuple[int, ...]

    @staticmethod
    def default() -> SweepConfig:
        return SweepConfig(
            prompt_file=DEFAULT_PROMPT_FILE,
            mesh_width=mesh_width_from_platform(),
            prefill_chunk_size=PREFILL_CHUNK_SIZE,
            image_path=DEFAULT_IMAGE,
            vision_max_edge=0,
            vision_square_pixels=None,
            isls=DEFAULT_ISLS,
        )


def mesh_width_from_platform() -> int:
    """Match demo/tests: P150 → 1, BH-QB → 4, else min(4, visible devices)."""
    env = os.environ.get("MESH_DEVICE")
    if env in ("P150", "P150x1"):
        return 1
    if env in ("BH-QB", "P150x4"):
        return 4
    return min(4, ttnn.get_num_devices())


@dataclass
class SweepResult:
    sweep_id: str
    isl: int
    mesh_width: int
    prefill_tokens: int
    decode_tps: float
    ttft_ms: float | None = None
    first_traced_ms: float | None = None
    steady_latency_ms: float | None = None
    e2e_tps: float | None = None
    skipped_reason: str | None = None
    generated_text: str | None = None

    @property
    def ok(self) -> bool:
        return self.skipped_reason is None


class PromptBuilder:
    """Caches tokenizer, corpus tokens, and vision image for multimodal prompt fitting."""

    def __init__(self) -> None:
        self._processor = None
        self._source_tokens: list[int] | None = None
        self._vision_cache: dict[tuple[str, int, int | None], object] = {}

    def load_corpus(self, path: Path) -> list[int]:
        log(f"Loading prompt text from {path}...")
        with bz2.open(path, "rt", encoding="utf-8") as f:
            text = f.read()
        log(f"  Loaded {len(text)} chars from {path.name}")
        self._source_tokens = self._tokenize(text)
        log(f"  Ready: {len(self._source_tokens)} text tokens")
        return self._source_tokens

    @property
    def source_tokens(self) -> list[int]:
        if self._source_tokens is None:
            raise RuntimeError("call load_corpus() first")
        return self._source_tokens

    def fit_messages(
        self,
        target_isl: int,
        image_path: Path,
        vision_max_edge: int,
        vision_square_pixels: int | None,
    ) -> tuple[list[dict], int]:
        source_tokens = self.source_tokens
        source_len = len(source_tokens)
        log(
            f"  Fitting prompt to ISL {format_isl(target_isl)} "
            f"(image={image_path.name}, corpus={source_len} text tokens)..."
        )

        best_messages = multimodal_messages("")
        best_tokens = self.count_tokens(
            best_messages, image_path, vision_max_edge, vision_square_pixels, label="baseline"
        )
        log(f"  Baseline (image + template): {best_tokens} tokens")

        if best_tokens > target_isl:
            log(f"  Skip: baseline {best_tokens} tokens exceeds ISL {format_isl(target_isl)}")
            return best_messages, best_tokens

        text_budget = max(0, target_isl - best_tokens)
        if text_budget > source_len:
            reps = (text_budget + source_len - 1) // source_len
            log(
                f"  Corpus shorter than ISL; tiling ~{reps}x "
                f"(~{text_budget} text tokens needed, perf workload only)"
            )

        if text_budget <= 0:
            return best_messages, best_tokens

        lo, hi = 0, text_budget
        while lo <= hi:
            mid = (lo + hi) // 2
            messages = multimodal_messages(self.text_for_token_count(mid))
            tokens = self.count_tokens(
                messages, image_path, vision_max_edge, vision_square_pixels, label=f"text_tokens={mid}"
            )
            if tokens <= target_isl:
                best_messages, best_tokens = messages, tokens
                lo = mid + 1
                pct = tokens * 100 // max(target_isl, 1)
                log(f"  Fit: {tokens}/{target_isl} prompt tokens ({pct}% ISL, text_tokens={mid})")
            else:
                hi = mid - 1

        if best_tokens < target_isl:
            log(
                f"  Note: fitted {best_tokens} tokens (< ISL {format_isl(target_isl)}; "
                "multimodal tokenization is not linear in text length)"
            )
        log(f"  Fitted: {best_tokens}/{target_isl} prompt tokens")
        return best_messages, best_tokens

    def text_for_token_count(self, token_count: int) -> str:
        if token_count <= 0:
            return ""
        ids = tile_tokens(self.source_tokens, token_count)
        return self.tokenizer().decode(ids)

    def count_tokens(
        self,
        messages: list[dict],
        image_path: Path,
        vision_max_edge: int,
        vision_square_pixels: int | None,
        *,
        label: str,
    ) -> int:
        log(f"  Token count ({label})...")
        image = self.vision_image(image_path, vision_max_edge, vision_square_pixels)
        processor = self.processor()
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        n_tokens = int(processor(text=prompt, images=image, return_tensors="pt")["input_ids"].shape[1])
        log(f"  Token count ({label}): {n_tokens} prompt tokens")
        return n_tokens

    def preview_generation(self, text: str, max_tokens: int = OUTPUT_SEQ_LEN) -> str:
        if not text:
            return "(no generated text captured)"
        tokenizer = self.tokenizer()
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) <= max_tokens:
            return text
        return tokenizer.decode(ids[:max_tokens]) + " …"

    def processor(self):
        if self._processor is None:
            model_id = tt_image_demo._DEFAULT_MODEL_ID
            log(f"Loading HuggingFace processor ({model_id})...")
            self._processor = tt_image_demo.AutoProcessor.from_pretrained(
                model_id,
                trust_remote_code=True,
                fix_mistral_regex=True,
                cache_dir=os.getenv("HF_TOKENIZER_CACHE") or os.getenv("HF_HUB_CACHE") or None,
                local_files_only=False,
                token=os.getenv("HF_TOKEN") or None,
            )
            log("Processor loaded.")
        return self._processor

    def tokenizer(self):
        return getattr(self.processor(), "tokenizer", self.processor())

    def vision_image(self, image_path: Path, vision_max_edge: int, vision_square_pixels: int | None):
        key = (str(image_path.resolve()), vision_max_edge, vision_square_pixels)
        if key not in self._vision_cache:
            log(f"  Loading vision image {image_path.name}...")
            image = tt_image_demo.Image.open(image_path).convert("RGB")
            image = tt_image_demo._prepare_vision_image(image, vision_max_edge, vision_square_pixels)
            self._vision_cache[key] = image
            log("  Vision image ready.")
        return self._vision_cache[key]

    def _tokenize(self, text: str) -> list[int]:
        log(f"  Tokenizing source ({len(text)} chars)...")
        tokens = self.tokenizer().encode(text)
        log(f"  Tokenized: {len(tokens)} tokens")
        return tokens


def run_order(isls: tuple[int, ...]) -> tuple[int, ...]:
    """Peak ISL first (cold device), then ascending sweep for the rest."""
    unique = tuple(sorted(set(isls)))
    if len(unique) <= 1:
        return unique
    return (unique[-1],) + unique[:-1]


def format_isl(isl: int) -> str:
    if isl >= 1024 and isl % 1024 == 0:
        return f"{isl // 1024}k"
    return str(isl)


def sweep_id(isl: int) -> str:
    return f"b1_isl{format_isl(isl)}"


def multimodal_messages(text: str) -> list[dict]:
    return [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]


def tile_tokens(source_tokens: list[int], count: int) -> list[int]:
    if count <= 0 or not source_tokens:
        return []
    if count <= len(source_tokens):
        return source_tokens[:count]
    reps = (count + len(source_tokens) - 1) // len(source_tokens)
    return (source_tokens * reps)[:count]


def parse_demo_metrics(stdout: str) -> tuple[float, float, float, float, float]:
    def _one(pattern: re.Pattern[str], name: str) -> float:
        match = pattern.search(stdout)
        if match is None:
            raise AssertionError(f"Could not parse {name} from demo output:\n{stdout}")
        return float(match.group(1))

    return (
        _one(_RE_STEADY_TPS, "steady-state decode throughput"),
        _one(_RE_TTFT, "TTFT"),
        _one(_RE_FIRST_TRACED, "first traced decode step latency"),
        _one(_RE_STEADY_LAT, "steady-state decode latency"),
        _one(_RE_E2E_TPS, "end-to-end throughput"),
    )


def extract_generated_text(stdout: str) -> str:
    sep = stdout.rfind(_DEMO_PERF_SEP)
    return stdout[sep + len(_DEMO_PERF_SEP) :].strip() if sep >= 0 else ""


def clear_model_caches() -> None:
    for attr in vars(tt_image_demo.ModelArgs).values():
        cache_clear = getattr(attr, "cache_clear", None)
        if cache_clear is not None:
            with contextlib.suppress(Exception):
                cache_clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def run_isl_point(
    monkeypatch,
    *,
    prompts: PromptBuilder,
    config: SweepConfig,
    isl: int,
    messages: list[dict],
    prefill_tokens: int,
) -> SweepResult:
    sid = sweep_id(isl)
    base = dict(sweep_id=sid, isl=isl, mesh_width=config.mesh_width, prefill_tokens=prefill_tokens)

    if prefill_tokens > isl:
        return SweepResult(
            **base,
            decode_tps=0.0,
            skipped_reason=f"prompt is {prefill_tokens} tokens but max_seq_len={isl}",
        )

    log(
        f"  Running TT demo: prefill={prefill_tokens}, decode={OUTPUT_SEQ_LEN}, "
        f"chunk_size={config.prefill_chunk_size}"
    )
    buf = io.StringIO()
    with monkeypatch.context() as mp, contextlib.redirect_stdout(buf):
        mp.setattr(tt_image_demo, "MODEL_LOADING_MESSAGES", messages)
        tt_image_demo.run_tt(
            tt_image_demo._DEFAULT_MODEL_ID,
            config.image_path,
            mesh_width=config.mesh_width,
            text_layers=None,
            max_new_tokens=OUTPUT_SEQ_LEN,
            greedy=True,
            temperature=tt_image_demo._DEFAULT_SAMPLE_TEMPERATURE,
            seed=0,
            lm_head_max_device_cols=None,
            vision_max_edge=config.vision_max_edge,
            vision_square_pixels=config.vision_square_pixels,
            prefill_chunk_size=config.prefill_chunk_size,
            clear_weight_cache=False,
            perf=False,
        )

    stdout = buf.getvalue()
    steady_tps, ttft_ms, first_traced_ms, steady_lat_ms, e2e_tps = parse_demo_metrics(stdout)
    return SweepResult(
        **base,
        decode_tps=steady_tps,
        ttft_ms=ttft_ms,
        first_traced_ms=first_traced_ms,
        steady_latency_ms=steady_lat_ms,
        e2e_tps=e2e_tps,
        generated_text=extract_generated_text(stdout),
    )


def log(msg: str) -> None:
    print(msg, flush=True)


def banner(title: str) -> None:
    line = "-" * 78
    print(f"\n{line}\n  {title}\n{line}")


def print_result(result: SweepResult, prompts: PromptBuilder) -> None:
    banner(
        f"ISL {format_isl(result.isl)} {'skipped' if result.skipped_reason else 'complete'}  "
        f"(mesh_width={result.mesh_width})"
    )
    print(f"  Prefill tokens                     {result.prefill_tokens:>10}")

    if result.skipped_reason:
        print(f"  Skip reason                        {result.skipped_reason}")
    else:
        assert result.ttft_ms is not None
        assert result.first_traced_ms is not None
        assert result.steady_latency_ms is not None
        assert result.e2e_tps is not None
        print(f"  TTFT (prompt -> 1st new tok)       {result.ttft_ms:>10.2f} ms")
        print(f"  First traced decode step latency   {result.first_traced_ms:>10.2f} ms")
        print(f"  Steady-state decode latency / tok  {result.steady_latency_ms:>10.2f} ms")
        print(f"  Steady-state decode throughput     {result.decode_tps:>10.3f} tok/s/user")
        print(f"  End-to-end throughput              {result.e2e_tps:>10.3f} tok/s")
        if result.generated_text is not None:
            banner(f"Generated text (first {OUTPUT_SEQ_LEN} new tokens)")
            print(prompts.preview_generation(result.generated_text))
    print("-" * 78)


def print_summary(results: list[SweepResult], config: SweepConfig) -> None:
    banner("Devstral image+text ISL sweep decode throughput summary")
    print(f"  prompt_file: {config.prompt_file}")
    print(f"  output_seq_len: {OUTPUT_SEQ_LEN}")
    print("-" * 78)
    print("| config       | max_seq_len | prefill_tok | decode_t/s/u | decode_t/s | status  |")
    print("| ------------ | ----------- | ----------- | ------------ | ---------- | ------- |")
    for r in sorted(results, key=lambda x: x.isl):
        status = "skipped" if r.skipped_reason else "ok"
        print(
            f"| {r.sweep_id:<12} | {r.isl:>11} | {r.prefill_tokens:>11} | "
            f"{r.decode_tps:>12.2f} | {r.decode_tps:>10.2f} | {status:<7} |"
        )
    print("-" * 78)


@pytest.mark.timeout(2400)
@pytest.mark.models_performance_bare_metal
def test_devstral_image_text_isl_sweep_perf(monkeypatch):
    config = SweepConfig.default()
    prompts = PromptBuilder()
    prompts.load_corpus(config.prompt_file)

    log(f"Probing TT devices (mesh_width={config.mesh_width})...")
    num_devices = ttnn.get_num_devices()
    log(f"  Visible TT devices: {num_devices}")
    if config.mesh_width > num_devices:
        pytest.skip(f"mesh_width={config.mesh_width} requested but only {num_devices} device(s) visible.")

    run_isls = run_order(config.isls)
    peak_isl = max(config.isls) if config.isls else 0
    log(
        f"Sweep: isls={[format_isl(i) for i in config.isls]}, "
        f"run_order={[format_isl(i) for i in run_isls]} "
        f"(peak {format_isl(peak_isl)} first, then ascending), "
        f"image={config.image_path.name}, prefill_chunk_size={config.prefill_chunk_size}"
    )

    results: list[SweepResult] = []
    for isl in run_isls:
        log(f"running ISL={format_isl(isl)} mesh_width={config.mesh_width}")
        try:
            messages, prefill_tokens = prompts.fit_messages(
                isl, config.image_path, config.vision_max_edge, config.vision_square_pixels
            )
            result = run_isl_point(
                monkeypatch,
                prompts=prompts,
                config=config,
                isl=isl,
                messages=messages,
                prefill_tokens=prefill_tokens,
            )
        finally:
            clear_model_caches()

        results.append(result)
        print_result(result, prompts)

    print_summary(results, config)
