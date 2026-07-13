# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
FastAPI server for Gemma4 text generation on Tenstorrent hardware.

Uses the same stack as ``text_demo_v2.py`` (``Gemma4Generator`` + host sampling)
and reports TTFT / decode tok/s with the same conventions as the demo
(decode iteration 0 excluded from steady-state tok/s).

OpenAI-compatible surface for Hermes / OpenAI clients:
    GET  /health
    GET  /v1/models
    POST /v1/chat/completions   (tools, tool_choice, tool role, max_completion_tokens)
    POST /generate              (raw prompt; debugging)

Run from tt-metal root::

    export TT_METAL_HOME=~/tt-metal PYTHONPATH=~/tt-metal ARCH_NAME=blackhole
    export HF_HUB_OFFLINE=1 HF_HOME=~/.cache/huggingface
    export HF_MODEL=$HF_HOME/hub/models--google--gemma-4-31B-it/snapshots/main
    export TT_CACHE_PATH=$HF_HOME/tt_cache/google--gemma-4-31B-it
    export MESH_DEVICE=P150x4

    uv pip install -r models/demos/gemma4/demo/requirements-server.txt
    bash models/demos/gemma4/demo/run_server.sh

Or::

    python -m uvicorn models.demos.gemma4.demo.serve_fastapi:app --host 0.0.0.0 --port 8000

Environment:
    GEMMA4_MAX_SEQ_LEN       Max context (power of 2 recommended for long prompts)
    GEMMA4_MAX_NEW_TOKENS    Default max tokens to generate per request
    GEMMA4_BOUNDED_SLIDING   1/0 — auto-on when max_seq_len > 16384 if unset
    GEMMA4_PAGE_BLOCK_SIZE   Paged-attention block size (default 64 if max_seq_len > 4096)
    GEMMA4_ENABLE_TRACE      1/0 (default 1)
    GEMMA4_INSTRUCT          Use chat template (default 1)
    GEMMA4_WARMUP_TOKENS     Decode warmup tokens at startup (default 2)
    GEMMA4_SAMPLE_ON_DEVICE  1/0 — on-device sampling (default 0; host is faster for
                             batch-1 greedy on Gemma4 31B @ 1×4). Use 1 for vLLM-style
                             serving where avoiding full-vocab readback matters at scale.
    GEMMA4_SPECULATIVE       1/0 — greedy speculative decode with the it-assistant drafter
    GEMMA4_ASSISTANT_MODEL   Drafter checkpoint (default: <HF_MODEL>-assistant)
    GEMMA4_SPEC_DRAFT_LEN    High-band draft candidates/iter (default 4; use 16 on QB2)
    GEMMA4_SPEC_ADAPTIVE_K   1/0 — per-request K from prompt class (default 1 when speculative)
    GEMMA4_SPEC_DRAFT_LEN_MID Mid-band K for summarize-like prompts (default 8)
    GEMMA4_SPEC_DRAFT_LEN_LOW Low-band K for creative/open-ended prompts (default 6)
    GEMMA4_SPEC_TRACE        1/0 — fused metal trace (default 1)
    GEMMA4_SPEC_FUSED_RESEED 1/0 — fused trace seed mode (default 1=reseed; 0=shift hangs)
    GEMMA4_SPEC_HOST_RESEED  1/0 — exact reseed loop (higher acceptance, slower)
    GEMMA4_BOUNDED_SLIDING   Must be 0 for speculative decode
    GEMMA4_FUSED_GREEDY      1/0 — plain fused greedy decode (non-spec; default 0)
    GEMMA4_FUSED_GREEDY_TRACE 1/0 — metal trace for fused greedy (default 1)
    GEMMA4_REPETITION_PENALTY     Host repetition penalty (default 1.15; 1.0=off)
    GEMMA4_REPETITION_MAX_STREAK  Stop after N identical tokens in a row (default 12)
    GEMMA4_MODEL_ID          id advertised on /v1/models (also always lists ``main``)
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import queue
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Literal, Union

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

import ttnn
from models.common.sampling import SamplingParams
from models.demos.gemma4.demo.sampling_utils import RepetitionStreakGuard, host_sample
from models.demos.gemma4.demo.text_demo_v2 import _device_params, _model_path, create_tt_page_table
from models.demos.gemma4.tt.generator import Gemma4Generator
from models.perf.benchmarking_utils import BenchmarkProfiler
from models.tt_transformers.tt.common import PagedAttentionConfig, preprocess_inputs_prefill


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() in ("1", "true", "yes")


def _device_sampling_available(generator: Gemma4Generator) -> bool:
    if not _env_bool("GEMMA4_SAMPLE_ON_DEVICE", False):
        return False
    model = generator.model[0]
    return getattr(model, "sampling", None) is not None and getattr(model, "_supports_on_device_sampling", False)


def _build_device_sampling_params(temperature: float, top_p: float) -> SamplingParams:
    """Match Gemma3 demo: greedy uses top_k=1 so device argmax matches host."""
    if temperature <= 0:
        return SamplingParams(temperature=0.0, top_k=1, top_p=1.0)
    return SamplingParams(temperature=temperature, top_k=32, top_p=top_p)


def _normalize_sampled_tokens(tokens: torch.Tensor, batch_size: int) -> torch.Tensor:
    if tokens.dim() == 1:
        return tokens.reshape(batch_size, 1)
    if tokens.shape != (batch_size, 1):
        return tokens.reshape(batch_size, 1)
    return tokens


def _mesh_shape() -> ttnn.MeshShape:
    mesh_device = os.environ.get("MESH_DEVICE", "P150x4")
    shape_map = {
        "N150": (1, 1),
        "N300": (1, 2),
        "P150": (1, 1),
        "P300": (1, 2),
        "P150x4": (1, 4),
        "P150x8": (1, 8),
        "T3K": (1, 8),
    }
    rows, cols = shape_map.get(mesh_device, (1, 4))
    override = os.environ.get("GEMMA4_MESH_SHAPE")
    if override:
        parts = [int(x.strip()) for x in override.split(",")]
        if len(parts) != 2:
            raise ValueError("GEMMA4_MESH_SHAPE must be 'rows,cols'")
        rows, cols = parts
    return ttnn.MeshShape(rows, cols)


@dataclass
class ServerConfig:
    model_path: str
    max_seq_len: int = 1024
    default_max_tokens: int = 128
    batch_size: int = 1
    page_block_size: int = 32
    paged_attention: bool = True
    bounded_sliding: bool = False
    enable_trace: bool = True
    instruct: bool = True
    stop_at_eos: bool = True
    warmup_tokens: int = 2
    model_id: str = "google/gemma-4-31B-it"
    speculative: bool = False
    spec_draft_len: int = 4
    spec_adaptive_k: bool = True
    repetition_penalty: float = 1.15
    repetition_max_streak: int = 12

    @classmethod
    def from_env(cls) -> ServerConfig:
        max_seq_len = int(os.environ.get("GEMMA4_MAX_SEQ_LEN", "1024"))
        page_block_size = int(os.environ.get("GEMMA4_PAGE_BLOCK_SIZE", "0"))
        if page_block_size <= 0:
            page_block_size = 64 if max_seq_len > 4096 else 32

        bs_env = os.environ.get("GEMMA4_BOUNDED_SLIDING")
        if bs_env is None:
            bounded_sliding = max_seq_len > 16384
        else:
            bounded_sliding = bs_env.lower() in ("1", "true", "yes")

        speculative = _env_bool("GEMMA4_SPECULATIVE", False)
        # Adaptive K defaults ON when speculative is enabled; set
        # GEMMA4_SPEC_ADAPTIVE_K=0 to pin every request to SPEC_DRAFT_LEN.
        adaptive_raw = os.environ.get("GEMMA4_SPEC_ADAPTIVE_K")
        if adaptive_raw is None:
            spec_adaptive_k = speculative
        else:
            spec_adaptive_k = adaptive_raw.lower() in ("1", "true", "yes")

        model_path = _model_path()
        return cls(
            model_path=model_path,
            max_seq_len=max_seq_len,
            default_max_tokens=int(os.environ.get("GEMMA4_MAX_NEW_TOKENS", "128")),
            batch_size=int(os.environ.get("GEMMA4_BATCH", "1")),
            page_block_size=page_block_size,
            bounded_sliding=bounded_sliding,
            enable_trace=_env_bool("GEMMA4_ENABLE_TRACE", True),
            instruct=_env_bool("GEMMA4_INSTRUCT", True),
            stop_at_eos=_env_bool("GEMMA4_STOP_AT_EOS", True),
            warmup_tokens=int(os.environ.get("GEMMA4_WARMUP_TOKENS", "2")),
            model_id=os.environ.get("GEMMA4_MODEL_ID", os.path.basename(model_path.rstrip("/")) or "gemma4"),
            speculative=speculative,
            spec_draft_len=int(os.environ.get("GEMMA4_SPEC_DRAFT_LEN", "4")),
            spec_adaptive_k=spec_adaptive_k,
            repetition_penalty=float(os.environ.get("GEMMA4_REPETITION_PENALTY", "1.15")),
            repetition_max_streak=int(os.environ.get("GEMMA4_REPETITION_MAX_STREAK", "12")),
        )


@dataclass
class InferenceMetrics:
    ttft_ms: float
    decode_tok_s_per_user: float
    decode_tok_s: float
    prompt_tokens: int
    generated_tokens: int
    total_ms: float
    speculative: bool = False
    mean_accept: float | None = None
    tokens_per_iter: float | None = None
    draft_len: int | None = None
    adaptive_workload: str | None = None

    def as_dict(self) -> dict[str, Any]:
        out = {
            "ttft_ms": round(self.ttft_ms, 2),
            "decode_tok_s_per_user": round(self.decode_tok_s_per_user, 3),
            "decode_tok_s": round(self.decode_tok_s, 3),
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "total_ms": round(self.total_ms, 2),
        }
        if self.speculative:
            out["speculative"] = True
            if self.mean_accept is not None:
                out["mean_accept"] = round(self.mean_accept, 3)
            if self.tokens_per_iter is not None:
                out["tokens_per_iter"] = round(self.tokens_per_iter, 3)
            if self.draft_len is not None:
                out["draft_len"] = self.draft_len
            if self.adaptive_workload is not None:
                out["adaptive_workload"] = self.adaptive_workload
        return out


def _metrics_with_sampling(result: GenerationResult, sample_on_device: bool) -> dict[str, Any]:
    metrics = result.metrics.as_dict()
    metrics["sample_on_device"] = sample_on_device
    return metrics


@dataclass
class GenerationResult:
    text: str
    token_ids: list[int]
    metrics: InferenceMetrics


@dataclass
class Gemma4ServerState:
    config: ServerConfig
    mesh_device: ttnn.MeshDevice
    generator: Gemma4Generator
    tt_kv_cache: Any
    page_table: torch.Tensor
    paged_attention_config: PagedAttentionConfig | None
    use_device_sampling: bool = False
    use_speculative: bool = False
    spec_decoder: Any = None
    lock: threading.Lock = field(default_factory=threading.Lock)
    ready: bool = False

    @classmethod
    def load(cls, config: ServerConfig) -> Gemma4ServerState:
        device_params = _device_params()
        fabric_config = device_params.pop("fabric_config")
        ttnn.set_fabric_config(fabric_config)
        mesh_device = ttnn.open_mesh_device(mesh_shape=_mesh_shape(), **device_params)

        page_max_num_blocks = config.batch_size * math.ceil(config.max_seq_len / config.page_block_size)
        paged_attention_config = (
            PagedAttentionConfig(
                block_size=config.page_block_size,
                max_num_blocks=page_max_num_blocks,
            )
            if config.paged_attention
            else None
        )

        bounded_sliding = config.bounded_sliding and config.paged_attention
        if config.speculative and bounded_sliding:
            logger.warning("GEMMA4_SPECULATIVE=1 requires unbounded sliding KV; forcing GEMMA4_BOUNDED_SLIDING=0")
            bounded_sliding = False
        logger.info(
            f"Loading Gemma4 from {config.model_path} "
            f"(max_seq_len={config.max_seq_len}, bounded_sliding={bounded_sliding}, "
            f"speculative={config.speculative})"
        )
        generator, tt_kv_cache, _tokenizer = Gemma4Generator.from_pretrained(
            mesh_device=mesh_device,
            model_path=config.model_path,
            max_batch_size=config.batch_size,
            max_seq_len=config.max_seq_len,
            paged_attention_config=paged_attention_config,
            bounded_sliding_kv_cache=bounded_sliding,
        )
        model_args = generator.model_args[0]
        page_table = create_tt_page_table(config.batch_size, paged_attention_config)

        if bounded_sliding:
            from models.demos.gemma4.tt.attention.kv_cache_hybrid import build_hybrid_page_tables

            n_layers = model_args.num_hidden_layers
            sliding_mask = [model_args.layer_types[i] == "sliding_attention" for i in range(n_layers)]
            per_layer_pts = build_hybrid_page_tables(
                n_layers,
                sliding_mask,
                num_users=config.batch_size,
                block_size=config.page_block_size,
                max_seq_len=config.max_seq_len,
                sliding_window=model_args.sliding_window,
            )
            generator.model[0]._active_page_tables_per_layer = per_layer_pts
            logger.info(f"Bounded sliding: installed {len(per_layer_pts)} per-layer page tables")

        use_device_sampling = _device_sampling_available(generator)
        logger.info(f"Sampling path: {'on-device' if use_device_sampling else 'host (full logits readback)'}")

        logger.info("Warming up prefill traces...")
        generator.warmup_model_prefill(
            kv_cache=tt_kv_cache,
            enable_trace=config.enable_trace,
            can_sample_on_device=use_device_sampling,
            greedy_only=True,
        )

        state = cls(
            config=config,
            mesh_device=mesh_device,
            generator=generator,
            tt_kv_cache=tt_kv_cache,
            page_table=page_table,
            paged_attention_config=paged_attention_config,
            use_device_sampling=use_device_sampling,
            use_speculative=config.speculative,
        )
        if config.speculative:
            state._init_spec_decoder()
            state._warmup_spec_decode()
        else:
            state._warmup_decode()
        state.ready = True
        return state

    def _init_spec_decoder(self) -> None:
        from models.demos.gemma4.tt.common import create_assistant_model, default_assistant_model_path
        from models.demos.gemma4.tt.spec_decode import SpeculativeDecoder

        assistant_path = default_assistant_model_path(self.config.model_path)
        if not os.getenv("GEMMA4_ASSISTANT_MODEL"):
            logger.info(f"GEMMA4_ASSISTANT_MODEL unset; defaulting drafter to {assistant_path}")
        target = self.generator.model[0]
        _, assistant = create_assistant_model(
            mesh_device=self.mesh_device,
            target_model=target,
            mesh_config=target.mesh_config,
            ccl_manager=target.ccl_manager,
            assistant_path=assistant_path,
        )
        self.spec_decoder = SpeculativeDecoder(
            target_model=target,
            assistant_model=assistant,
            mesh_device=self.mesh_device,
            tt_kv_cache=self.tt_kv_cache,
            page_table_torch=self.page_table,
            stop_tokens=self.generator.model_args[0].tokenizer.stop_tokens,
            draft_len=self.config.spec_draft_len,
        )
        _trace_env = os.environ.get("GEMMA4_SPEC_TRACE")
        self.spec_decoder._use_trace = True if _trace_env is None else (_trace_env == "1")
        if self.spec_decoder._use_trace and not self.spec_decoder._fused_reseed:
            logger.warning("GEMMA4_SPEC_FUSED_RESEED=0 with fused trace hangs on replay #2; forcing reseed mode")
            self.spec_decoder._fused_reseed = True
        logger.info(
            f"Speculative decoder ready (draft_len={self.config.spec_draft_len}, "
            f"trace={self.spec_decoder._use_trace}, "
            f"reseed={self.spec_decoder._fused_reseed}, "
            f"host_reseed={os.environ.get('GEMMA4_SPEC_HOST_RESEED', '0')})"
        )

    def _per_layer_kv_cache(self):
        """Return ``[[k, v], ...]`` peeling the ``from_pretrained`` model wrapper."""
        kv = self.tt_kv_cache
        if kv is None:
            return None
        if (
            isinstance(kv, (list, tuple))
            and kv
            and isinstance(kv[0], (list, tuple))
            and kv[0]
            and isinstance(kv[0][0], (list, tuple))
        ):
            kv = kv[0]
        return kv

    def _reset_paged_kv_cache(self) -> None:
        """Zero the paged KV cache so a new prompt does not inherit stale decode state."""
        kv = self._per_layer_kv_cache()
        if kv is None:
            return
        for layer_cache in kv:
            for cache_tensor in layer_cache:
                ttnn.fill(
                    cache_tensor,
                    0.0,
                    memory_config=cache_tensor.memory_config(),
                    output_tensor=cache_tensor,
                )
        ttnn.synchronize_device(self.mesh_device)

    def _release_traces_for_new_request(self) -> None:
        """Drop live metal traces before each served request.

        Hermes multi-turn chats re-prefill the full tools catalog (~16k→32k pad)
        eagerly. Leaving decode traces from the previous turn registered makes
        Metal treat that allocation as unsafe and hang mid-prefill on turn 2+.
        Trace capture re-happens on the first short/decode path that needs it.
        """
        release = getattr(self.generator, "release_captured_traces", None)
        if release is not None:
            release(decode=True, prefill=True)
        fused = getattr(self, "_plain_fused_decoder", None)
        if fused is not None and hasattr(fused, "release_fused_trace"):
            fused.release_fused_trace()
        if self.spec_decoder is not None and hasattr(self.spec_decoder, "release_fused_trace"):
            self.spec_decoder.release_fused_trace()

    def _warmup_decode(self) -> None:
        if self.config.warmup_tokens <= 0:
            return
        logger.info(f"Running decode warmup ({self.config.warmup_tokens} tokens)...")
        self._reset_paged_kv_cache()
        self.generate(
            prompt="Hello",
            max_tokens=self.config.warmup_tokens,
            temperature=0.0,
            top_p=1.0,
            instruct=self.config.instruct,
            stop_at_eos=False,
        )

    def _warmup_spec_decode(self) -> None:
        """Match text_demo_v2: spec warmup after the drafter is loaded, with a clean KV."""
        if self.config.warmup_tokens <= 0:
            return
        logger.info(
            f"Running spec-decode warmup ({self.config.warmup_tokens} tokens, "
            f"draft_len={self.config.spec_draft_len})..."
        )
        self._reset_paged_kv_cache()
        self.generate(
            prompt="The capital of France is",
            max_tokens=self.config.warmup_tokens,
            temperature=0.0,
            top_p=1.0,
            instruct=False,
            stop_at_eos=False,
        )
        # Leave KV clean for the first user request; fused trace stays captured.
        self._reset_paged_kv_cache()

    def close(self) -> None:
        for submesh in self.mesh_device.get_submeshes():
            ttnn.close_mesh_device(submesh)
        ttnn.close_mesh_device(self.mesh_device)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

    def generate_stream(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        instruct: bool,
        stop_at_eos: bool,
        repetition_penalty: float | None = None,
        repetition_max_streak: int | None = None,
        draft_len: int | None = None,
    ) -> Iterator[tuple[str, int | None, InferenceMetrics | None]]:
        """Yield ``(text_piece, token_id, metrics)`` per generated token."""
        yield from self._run_generation_loop(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            instruct=instruct,
            stop_at_eos=stop_at_eos,
            repetition_penalty=repetition_penalty,
            repetition_max_streak=repetition_max_streak,
            draft_len=draft_len,
        )

    def generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        instruct: bool,
        stop_at_eos: bool,
        repetition_penalty: float | None = None,
        repetition_max_streak: int | None = None,
        draft_len: int | None = None,
    ) -> GenerationResult:
        pieces: list[str] = []
        token_ids: list[int] = []
        metrics: InferenceMetrics | None = None
        for piece, tok_id, final_metrics in self._run_generation_loop(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            instruct=instruct,
            stop_at_eos=stop_at_eos,
            repetition_penalty=repetition_penalty,
            repetition_max_streak=repetition_max_streak,
            draft_len=draft_len,
        ):
            if final_metrics is not None:
                metrics = final_metrics
            elif tok_id is not None:
                token_ids.append(tok_id)
                pieces.append(piece)
        assert metrics is not None
        return GenerationResult(text="".join(pieces), token_ids=token_ids, metrics=metrics)

    def _select_request_draft_len(
        self,
        prompt: str,
        *,
        prompt_tokens: int,
        max_tokens: int,
        draft_len_override: int | None,
    ) -> tuple[int, str | None]:
        """Resolve per-request K (adaptive or pinned) and apply it to the decoder."""
        from models.demos.gemma4.tt.adaptive_draft_len import AdaptiveDraftLenConfig, select_adaptive_draft_len

        cfg = AdaptiveDraftLenConfig.from_env(default_draft_len=self.config.spec_draft_len)
        if not self.config.spec_adaptive_k:
            cfg = AdaptiveDraftLenConfig(
                enabled=False,
                k_high=self.config.spec_draft_len,
                k_mid=cfg.k_mid,
                k_low=cfg.k_low,
                long_prompt_tokens=cfg.long_prompt_tokens,
                short_prompt_tokens=cfg.short_prompt_tokens,
            )
        k, workload = select_adaptive_draft_len(
            prompt,
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_tokens,
            config=cfg,
            override=draft_len_override,
        )
        if self.spec_decoder is not None:
            changed = self.spec_decoder.set_draft_len(k)
            if changed:
                logger.info(
                    f"Adaptive draft_len -> {k}"
                    + (f" (workload={workload})" if workload else " (override/pin)")
                    + " — fused trace will recapture"
                )
            elif workload is not None:
                logger.info(f"Adaptive draft_len={k} (workload={workload})")
        return k, workload

    def _run_generation_loop(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        instruct: bool,
        stop_at_eos: bool,
        repetition_penalty: float | None = None,
        repetition_max_streak: int | None = None,
        draft_len: int | None = None,
    ) -> Iterator[tuple[str, int | None, InferenceMetrics | None]]:
        config = self.config
        batch_size = config.batch_size
        repetition_penalty = config.repetition_penalty if repetition_penalty is None else repetition_penalty
        repetition_max_streak = config.repetition_max_streak if repetition_max_streak is None else repetition_max_streak
        if repetition_penalty < 1.0:
            repetition_penalty = 1.0
        if repetition_max_streak < 0:
            repetition_max_streak = 0
        profiler = BenchmarkProfiler()
        run_start = time.perf_counter()
        profiler.start("run")

        model_args_list = self.generator.model_args
        model_args = model_args_list[0]
        tokenizer = model_args.tokenizer

        # Clamp max_tokens to the space left after the prompt. max_seq_len is the
        # TOTAL window (prompt + completion); clients (e.g. Hermes with
        # context_length=max_seq_len) often send max_tokens == the full window,
        # which leaves no room for the prompt and trips preprocess_inputs_prefill's
        # `max_prefill_len > 0` assert mid-stream. Reserve the real encoded prompt
        # length (+1 slack) and cap generation to what actually fits.
        prompt_len = len(model_args.encode_prompt(prompt, instruct=instruct))
        room_for_gen = config.max_seq_len - prompt_len - 1
        if room_for_gen <= 0:
            raise ValueError(
                f"prompt ({prompt_len}) leaves no room for generation within "
                f"max_seq_len ({config.max_seq_len}); shorten the prompt or raise GEMMA4_MAX_SEQ_LEN"
            )
        if max_tokens > room_for_gen:
            logger.warning(
                f"Clamping max_tokens {max_tokens} -> {room_for_gen} "
                f"(prompt={prompt_len}, max_seq_len={config.max_seq_len})"
            )
            max_tokens = room_for_gen

        input_tokens_prefill_pt, encoded_prompts, decoding_pos, prefill_lens = preprocess_inputs_prefill(
            [prompt] * batch_size,
            tokenizer,
            model_args_list,
            instruct,
            max_tokens,
            max_prefill_len=config.max_seq_len,
        )
        input_tokens_prefill_pt = torch.stack(input_tokens_prefill_pt).view(batch_size, -1)

        # Release traces BEFORE KV zero-fill: allocation during a long eager
        # prefill is unsafe while any metal trace from the prior request is live.
        self._release_traces_for_new_request()
        self._reset_paged_kv_cache()

        device_sampling_params = _build_device_sampling_params(temperature, top_p) if self.use_device_sampling else None
        use_spec = (
            self.use_speculative
            and self.spec_decoder is not None
            and batch_size == 1
            and temperature <= 0
            and device_sampling_params is None
        )
        if use_spec:
            selected_k, adaptive_workload = self._select_request_draft_len(
                prompt,
                prompt_tokens=prompt_len,
                max_tokens=max_tokens,
                draft_len_override=draft_len,
            )
            yield from self._run_speculative_generation_loop(
                input_tokens_prefill_pt=input_tokens_prefill_pt,
                encoded_prompts=encoded_prompts,
                decoding_pos=decoding_pos,
                prefill_lens=prefill_lens,
                max_tokens=max_tokens,
                stop_at_eos=stop_at_eos,
                selected_draft_len=selected_k,
                adaptive_workload=adaptive_workload,
                tokenizer=tokenizer,
                run_start=run_start,
                profiler=profiler,
                repetition_max_streak=repetition_max_streak,
            )
            return

        from models.demos.gemma4.tt.plain_fused_decode import PlainFusedGreedyDecoder, fused_greedy_enabled

        # Fused greedy: batch=1 greedy only, no PLI / bounded sliding. Spec path
        # above wins when speculative is enabled. Host repetition_penalty cannot
        # be applied inside the device argmax loop — when fused is on we skip the
        # penalty (streak guard still stops runaway repeats).
        use_fused_greedy = (
            fused_greedy_enabled()
            and batch_size == 1
            and temperature <= 0
            and device_sampling_params is None
            and not config.bounded_sliding
            and self.page_table is not None
            and not getattr(self.generator.model[0], "hidden_size_per_layer_input", 0)
        )
        if fused_greedy_enabled() and not use_fused_greedy and temperature <= 0:
            logger.info(
                "GEMMA4_FUSED_GREEDY set but skipped "
                f"(batch={batch_size}, temp={temperature}, "
                f"bounded_sliding={config.bounded_sliding})"
            )
        elif use_fused_greedy and repetition_penalty > 1.0:
            logger.info(
                f"GEMMA4_FUSED_GREEDY: ignoring host repetition_penalty={repetition_penalty} "
                "(device argmax path; streak guard still active)"
            )

        profiler.start("inference_prefill")
        prefill_out = self.generator.prefill_forward_text(
            input_tokens_prefill_pt,
            page_table=self.page_table,
            kv_cache=self.tt_kv_cache,
            prompt_lens=decoding_pos,
            warmup_prefill=False,
            sampling_params=device_sampling_params,
            # Eager prefill when fused greedy owns the CCL decode trace.
            enable_trace=False if use_fused_greedy else config.enable_trace,
        )
        if device_sampling_params is not None:
            prefill_tokens, _prefill_lp = prefill_out
            prefilled_token = prefill_tokens.long()
        else:
            prefilled_token = host_sample(
                prefill_out,
                temperature,
                top_p,
                input_ids=torch.tensor(encoded_prompts[0], dtype=torch.long),
                repetition_penalty=repetition_penalty,
            )
        profiler.end("inference_prefill")

        prefilled_flat = prefilled_token.view(batch_size, -1).squeeze(-1)
        all_outputs = [encoded_prompts[b][: prefill_lens[b]] for b in range(batch_size)]
        for user in range(batch_size):
            all_outputs[user].append(int(prefilled_flat[user].item()))

        streak_guards = [RepetitionStreakGuard(repetition_max_streak) for _ in range(batch_size)]

        # Stream the first (prefill) token for user 0 immediately, then each
        # decode token as it is produced — instead of buffering the whole
        # response and flushing at the end.
        first_tok = int(prefilled_flat[0].item())
        user_done = [False] * batch_size
        for user in range(batch_size):
            tok = int(prefilled_flat[user].item())
            if streak_guards[user].observe(tok):
                user_done[user] = True
            elif stop_at_eos and tok in tokenizer.stop_tokens:
                user_done[user] = True

        if not user_done[0] and first_tok not in tokenizer.stop_tokens:
            yield tokenizer.decode([first_tok]), first_tok, None

        current_pos = torch.tensor([decoding_pos[b] for b in range(batch_size)])
        out_tok = prefilled_flat.reshape(batch_size, 1)
        iteration = 0
        users_decoding = not all(user_done)

        profiler.start("inference_decode")
        if use_fused_greedy and users_decoding:
            fused = getattr(self, "_plain_fused_decoder", None)
            if fused is None:
                fused = PlainFusedGreedyDecoder(
                    target_model=self.generator.model[0],
                    mesh_device=self.mesh_device,
                    tt_kv_cache=self.tt_kv_cache,
                    page_table_torch=self.page_table,
                    stop_tokens=tokenizer.stop_tokens if stop_at_eos else None,
                )
                self._plain_fused_decoder = fused
            # Recapture is safe across requests; release first if a stale trace
            # exists from a prior hang (release is a no-op when None).
            decode_t0 = time.perf_counter()
            remaining = max_tokens
            token_q: queue.Queue = queue.Queue()
            sentinel = object()
            result: dict[str, Any] = {}

            def _fused_worker() -> None:
                try:
                    gen = fused.generate(
                        anchor_token=int(prefilled_flat[0].item()),
                        anchor_pos=int(current_pos[0].item()),
                        max_new_tokens=remaining,
                        token_callback=token_q.put,
                        repetition_max_streak=repetition_max_streak,
                    )
                    result["generated"] = gen
                except Exception as exc:
                    result["error"] = exc
                finally:
                    token_q.put(sentinel)

            worker = threading.Thread(target=_fused_worker, name="plain-fused-generate", daemon=True)
            worker.start()
            stop_seen = False
            while True:
                tok_id = token_q.get()
                if tok_id is sentinel:
                    break
                if stop_seen:
                    continue
                if stop_at_eos and tok_id in tokenizer.stop_tokens:
                    stop_seen = True
                    continue
                iteration += 1
                yield tokenizer.decode([tok_id]), tok_id, None
            worker.join()
            decode_elapsed = time.perf_counter() - decode_t0
            if "error" in result:
                raise result["error"]
            generated = result.get("generated", [])
            iteration = len(generated)
            steady_s = getattr(fused, "_last_replay_s", decode_elapsed)
            decode_tps_u = iteration / steady_s if steady_s > 0 else 0.0
            profiler.end("inference_decode")
            profiler.end("run")
            metrics = InferenceMetrics(
                ttft_ms=profiler.get_duration("inference_prefill") * 1000,
                decode_tok_s_per_user=decode_tps_u,
                decode_tok_s=decode_tps_u,
                prompt_tokens=prefill_lens[0],
                generated_tokens=iteration,
                total_ms=(time.perf_counter() - run_start) * 1000,
            )
            logger.info(
                f"TTFT={metrics.ttft_ms:.1f}ms plain-fused={metrics.decode_tok_s_per_user:.2f} tok/s/u "
                f"prompt={metrics.prompt_tokens} gen={metrics.generated_tokens}"
            )
            yield "", None, metrics
            return

        while users_decoding:
            profiler.start(f"inference_decode_time_{iteration}")
            decode_out, _ = self.generator.decode_forward(
                out_tok,
                current_pos,
                enable_trace=config.enable_trace,
                page_table=self.page_table,
                kv_cache=self.tt_kv_cache,
                sampling_params=device_sampling_params,
            )
            if device_sampling_params is not None:
                out_tok = _normalize_sampled_tokens(decode_out, batch_size)
            else:
                penalty_ids = torch.tensor(all_outputs[0], dtype=torch.long)
                out_tok = host_sample(
                    decode_out,
                    temperature,
                    top_p,
                    input_ids=penalty_ids,
                    repetition_penalty=repetition_penalty,
                )
            profiler.end(f"inference_decode_time_{iteration}")

            current_pos += 1
            tok0 = int(out_tok[0, 0].item())
            for user in range(batch_size):
                tok = int(out_tok[user, 0].item())
                if user_done[user]:
                    continue
                if streak_guards[user].observe(tok):
                    user_done[user] = True
                elif tok not in tokenizer.stop_tokens:
                    all_outputs[user].append(tok)
                elif stop_at_eos:
                    user_done[user] = True
                if all(user_done):
                    users_decoding = False

            # Emit user 0's token live (stop tokens are never streamed, matching
            # the buffered behavior).
            if not user_done[0] and tok0 not in tokenizer.stop_tokens:
                yield tokenizer.decode([tok0]), tok0, None

            iteration += 1
            if iteration >= max_tokens:
                users_decoding = False
        profiler.end("inference_decode")
        profiler.end("run")

        prefill_s = profiler.get_duration("inference_prefill")
        steady_iters = max(iteration - 1, 0)
        total_decode = sum(profiler.get_duration(f"inference_decode_time_{i}") for i in range(1, iteration))
        decode_tps_u = steady_iters / total_decode if total_decode > 0 and steady_iters > 0 else 0.0
        decode_tps = decode_tps_u * batch_size

        metrics = InferenceMetrics(
            ttft_ms=prefill_s * 1000,
            decode_tok_s_per_user=decode_tps_u,
            decode_tok_s=decode_tps,
            prompt_tokens=prefill_lens[0],
            generated_tokens=iteration,
            total_ms=(time.perf_counter() - run_start) * 1000,
        )
        logger.info(
            f"TTFT={metrics.ttft_ms:.1f}ms decode={metrics.decode_tok_s_per_user:.2f} tok/s/u "
            f"prompt={metrics.prompt_tokens} gen={metrics.generated_tokens}"
        )

        yield "", None, metrics

    def _run_speculative_generation_loop(
        self,
        input_tokens_prefill_pt,
        encoded_prompts,
        decoding_pos,
        prefill_lens,
        max_tokens: int,
        stop_at_eos: bool,
        tokenizer,
        run_start: float,
        profiler: BenchmarkProfiler,
        repetition_max_streak: int = 0,
        selected_draft_len: int | None = None,
        adaptive_workload: str | None = None,
    ) -> Iterator[tuple[str, int | None, InferenceMetrics | None]]:
        config = self.config
        spec = self.spec_decoder
        assert spec is not None
        draft_len_used = int(selected_draft_len if selected_draft_len is not None else spec.draft_len)

        profiler.start("inference_prefill")
        # Run prefill EAGERLY (no prefill CCL trace). The speculative decode loop
        # replays a single long-lived fused CCL trace; if a prefill trace replay
        # were interleaved before a fused replay, the two distinct CCL traces
        # deadlock the mesh (hang at fused execute / recapture). Eager prefill
        # keeps the fused trace the only CCL trace on the queue.
        prefill_out = self.generator.prefill_forward_text(
            input_tokens_prefill_pt,
            page_table=self.page_table,
            kv_cache=self.tt_kv_cache,
            prompt_lens=decoding_pos,
            warmup_prefill=False,
            enable_trace=False,
        )
        if hasattr(prefill_out, "deallocate"):
            prefill_out.deallocate(True)
        profiler.end("inference_prefill")

        prompt_len = int(decoding_pos[0])
        anchor_pos = prompt_len - 1
        anchor_token = int(encoded_prompts[0][anchor_pos])

        profiler.start("inference_decode")
        decode_t0 = time.perf_counter()

        # spec.generate() runs the whole fused loop to completion, so streaming
        # requires a producer/consumer bridge: run generation in a worker thread
        # that pushes each committed token (via token_callback) into a queue, and
        # yield tokens here as they arrive. Only the worker touches the device;
        # this thread just drains the queue, so there is no concurrent device use.
        token_q: queue.Queue = queue.Queue()
        sentinel = object()
        result: dict[str, Any] = {}

        def _spec_worker() -> None:
            try:
                gen, acc = spec.generate(
                    anchor_token=anchor_token,
                    anchor_pos=anchor_pos,
                    max_new_tokens=max_tokens,
                    token_callback=token_q.put,
                    repetition_max_streak=repetition_max_streak,
                )
                result["generated"] = gen
                result["accepts"] = acc
            except Exception as exc:  # surface worker failure to the request thread
                result["error"] = exc
            finally:
                token_q.put(sentinel)

        worker = threading.Thread(target=_spec_worker, name="spec-generate", daemon=True)
        worker.start()

        stop_seen = False
        while True:
            tok_id = token_q.get()
            if tok_id is sentinel:
                break
            if stop_seen:
                continue  # keep draining callback tokens until the worker signals done
            if stop_at_eos and tok_id in tokenizer.stop_tokens:
                stop_seen = True
                continue
            yield tokenizer.decode([tok_id]), tok_id, None

        worker.join()
        decode_elapsed = time.perf_counter() - decode_t0
        profiler.end("inference_decode")
        profiler.end("run")

        if "error" in result:
            raise result["error"]
        generated = result.get("generated", [])
        accepts = result.get("accepts", [])

        n_tokens = len(generated)
        n_iters = len(accepts)
        mean_accept = (sum(accepts) / n_iters) if n_iters else 0.0
        tok_s_u = n_tokens / decode_elapsed if decode_elapsed > 0 else 0.0
        metrics = InferenceMetrics(
            ttft_ms=profiler.get_duration("inference_prefill") * 1000,
            decode_tok_s_per_user=tok_s_u,
            decode_tok_s=tok_s_u,
            prompt_tokens=prompt_len,
            generated_tokens=n_tokens,
            total_ms=(time.perf_counter() - run_start) * 1000,
            speculative=True,
            mean_accept=mean_accept,
            tokens_per_iter=mean_accept + 1.0,
            draft_len=draft_len_used,
            adaptive_workload=adaptive_workload,
        )
        workload_note = f" workload={adaptive_workload}" if adaptive_workload else ""
        logger.info(
            f"TTFT={metrics.ttft_ms:.1f}ms spec-decode={metrics.decode_tok_s_per_user:.2f} tok/s/u "
            f"accept={mean_accept:.2f}/{draft_len_used}{workload_note} "
            f"prompt={metrics.prompt_tokens} gen={n_tokens}"
        )

        # Tokens were already streamed live above via the worker callback; only
        # the final metrics record remains to be emitted.
        yield "", None, metrics


# Gemma4 emits: <|tool_call>call:NAME{ARGS}<tool_call|>
# ARGS may be JSON (`{"k":"v"}`), Gemma native (`k:<|"|>v<|"|>`), or double-wrapped
# JSON (`{{"k":"v"}}`) when OpenAI-style argument strings are round-tripped through
# the HF chat template (which always wraps arguments in `{...}`).
_GEMMA_TOOL_CALL_RE = re.compile(
    r"<\|tool_call>call:([A-Za-z0-9_\.\-:]+)\{(.*?)}<tool_call\|>",
    re.DOTALL,
)
_GEMMA_THOUGHT_RE = re.compile(
    r"<\|channel>thought\n?(.*?)\n?<channel\|>",
    re.DOTALL,
)


class FunctionCall(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    arguments: str | dict[str, Any] = "{}"


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str | None = None
    type: Literal["function"] = "function"
    function: FunctionCall


class ChatMessage(BaseModel):
    """OpenAI-style chat message, including Hermes tool turns."""

    model_config = ConfigDict(extra="ignore")

    role: Literal["system", "user", "assistant", "tool", "function"]
    content: Union[str, list[Any], None] = None
    name: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None


class ChatCompletionRequest(BaseModel):
    """OpenAI chat.completions body (Hermes sends tools + max_completion_tokens)."""

    model_config = ConfigDict(extra="ignore")

    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int | None = Field(default=None, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    temperature: float = 0.0
    top_p: float = 1.0
    stream: bool = False
    stop_at_eos: bool | None = None
    repetition_penalty: float | None = Field(default=None, ge=1.0)
    repetition_max_streak: int | None = Field(default=None, ge=0)
    draft_len: int | None = Field(
        default=None,
        ge=1,
        description="Pin speculative draft length K for this request (skips adaptive K).",
    )
    tools: list[dict[str, Any]] | None = None
    tool_choice: Union[str, dict[str, Any], None] = None


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    prompt: str
    max_tokens: int | None = Field(default=None, ge=1)
    temperature: float = 0.0
    top_p: float = 1.0
    instruct: bool | None = None
    stop_at_eos: bool | None = None
    stream: bool = False
    repetition_penalty: float | None = Field(default=None, ge=1.0)
    repetition_max_streak: int | None = Field(default=None, ge=0)
    draft_len: int | None = Field(
        default=None,
        ge=1,
        description="Pin speculative draft length K for this request (skips adaptive K).",
    )


def _flatten_content(content: Union[str, list[Any], None]) -> str:
    """Normalize OpenAI multimodal content parts to a single text string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                if part.get("type") == "text" and "text" in part:
                    parts.append(str(part["text"]))
                elif "text" in part:
                    parts.append(str(part["text"]))
        return "".join(parts)
    return str(content)


def _normalize_tool_arguments(arguments: str | dict[str, Any] | None) -> str:
    if arguments is None:
        return "{}"
    if isinstance(arguments, dict):
        return json.dumps(arguments, ensure_ascii=False)
    text = str(arguments).strip()
    # Chat template wraps args in `{...}`; OpenAI JSON strings then become `{{...}}`.
    if text.startswith("{{") and text.endswith("}}"):
        inner = text[1:-1]
        try:
            json.loads(inner)
            return inner
        except json.JSONDecodeError:
            pass
    return text


def _messages_to_hf_chat(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    """Convert OpenAI/Hermes messages into HF chat-template message dicts."""
    chat: list[dict[str, Any]] = []
    for msg in messages:
        role = "tool" if msg.role == "function" else msg.role
        entry: dict[str, Any] = {"role": role}
        text = _flatten_content(msg.content)
        if text:
            entry["content"] = text
        elif msg.tool_calls:
            # Assistant tool-call turns often have null content.
            entry["content"] = None
        else:
            entry["content"] = text

        if msg.name:
            entry["name"] = msg.name
        if msg.tool_call_id:
            entry["tool_call_id"] = msg.tool_call_id
        if msg.tool_calls:
            entry["tool_calls"] = [
                {
                    "id": tc.id or f"call_{i}",
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": _normalize_tool_arguments(tc.function.arguments),
                    },
                }
                for i, tc in enumerate(msg.tool_calls)
            ]
        chat.append(entry)
    return chat


def _tools_for_template(
    tools: list[dict[str, Any]] | None,
    tool_choice: Union[str, dict[str, Any], None],
) -> list[dict[str, Any]] | None:
    if tool_choice == "none":
        return None
    return tools or None


def _coerce_tool_arguments_json(raw: str) -> str:
    """Best-effort normalize Gemma arg payloads to an OpenAI JSON object string."""
    text = raw.strip()
    if text.startswith("{{") and text.endswith("}}"):
        text = text[1:-1]
    if not text:
        return "{}"
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return json.dumps(parsed, ensure_ascii=False)
        return json.dumps({"value": parsed}, ensure_ascii=False)
    except json.JSONDecodeError:
        pass
    # Gemma native style: key:<|"|>value<|"|>,key2:123
    if ":" in text and not text.startswith("{"):
        try:
            # Wrap so json can help? Just return a synthetic {"raw": ...}.
            return json.dumps({"raw": text}, ensure_ascii=False)
        except Exception:
            return "{}"
    if text.startswith("{") and text.endswith("}"):
        return text
    return json.dumps({"value": text}, ensure_ascii=False)


def _parse_gemma_tool_calls(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Split assistant text into visible content + OpenAI ``tool_calls`` list."""
    tool_calls: list[dict[str, Any]] = []
    for idx, match in enumerate(_GEMMA_TOOL_CALL_RE.finditer(text)):
        name = match.group(1)
        args_json = _coerce_tool_arguments_json(match.group(2))
        tool_calls.append(
            {
                "id": f"call_{uuid.uuid4().hex[:8]}_{idx}",
                "type": "function",
                "function": {"name": name, "arguments": args_json},
            }
        )

    visible = _GEMMA_TOOL_CALL_RE.sub("", text)
    visible = _GEMMA_THOUGHT_RE.sub("", visible).strip()
    # Drop orphan channel / turn markers that sometimes leak into decode text.
    for junk in ("<|channel>thought", "<channel|>", "<|turn>model", "<turn|>"):
        visible = visible.replace(junk, "")
    visible = visible.strip()
    return visible, tool_calls


def _extract_user_prompt(messages: list[ChatMessage]) -> str:
    user_parts = [_flatten_content(m.content) for m in messages if m.role == "user"]
    user_parts = [p for p in user_parts if p]
    if not user_parts:
        raise HTTPException(status_code=400, detail="At least one user message is required")
    return user_parts[-1]


def _render_chat_prompt(
    messages: list[ChatMessage],
    tokenizer,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: Union[str, dict[str, Any], None] = None,
) -> str:
    """Render the full conversation via the model's chat template.

    Applying the model's own chat template over ALL messages (system + every
    turn) — rather than forwarding only the last user message — is what makes
    the model see real conversational structure and emit its end-of-turn token
    (<turn|>, id 106) at the right place. Feeding a bare, context-less turn
    pushes greedy decode off-distribution and it degenerates into repeat-until-
    max_tokens.

    Returns a fully-templated string (add_generation_prompt=True). The caller
    passes it downstream with instruct=False so it is tokenized verbatim: the
    template already carries <bos> and the turn markers, so re-tokenizing with
    add_special_tokens=False round-trips to a single BOS (verified) and stays
    compatible with the long-prompt left-clip path.

    Hermes / OpenAI tool schemas are passed through ``tools=`` so Gemma emits
    ``<|tool>declaration:…`` + ``<|tool_call>call:…`` tokens.
    """
    if not any(m.role == "user" for m in messages):
        raise HTTPException(status_code=400, detail="At least one user message is required")
    chat = _messages_to_hf_chat(messages)
    template_tools = _tools_for_template(tools, tool_choice)
    try:
        kwargs: dict[str, Any] = {"add_generation_prompt": True, "tokenize": False}
        if template_tools:
            kwargs["tools"] = template_tools
        return tokenizer.apply_chat_template(chat, **kwargs)
    except Exception as exc:
        # Roles the template doesn't understand or a template error: fall back
        # to the last user message rather than 500.
        logger.warning(f"apply_chat_template failed ({exc}); falling back to last user message")
        raise _FallbackToLastUser() from exc


class _FallbackToLastUser(Exception):
    """Signal that chat-template rendering failed and the caller should fall back."""


server_state: Gemma4ServerState | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global server_state
    config = ServerConfig.from_env()
    server_state = await asyncio.to_thread(Gemma4ServerState.load, config)
    logger.info("Gemma4 server ready")
    yield
    if server_state is not None:
        await asyncio.to_thread(server_state.close)
        server_state = None


app = FastAPI(title="Gemma4 TT Inference Server", lifespan=lifespan)


@app.get("/health")
async def health():
    if server_state is None or not server_state.ready:
        raise HTTPException(status_code=503, detail="Model not loaded")
    from models.demos.gemma4.tt.plain_fused_decode import fused_greedy_enabled

    return {
        "status": "ok",
        "model": server_state.config.model_id,
        "sample_on_device": server_state.use_device_sampling,
        "speculative": server_state.use_speculative,
        "spec_draft_len": server_state.config.spec_draft_len,
        "spec_adaptive_k": server_state.config.spec_adaptive_k,
        "fused_greedy": fused_greedy_enabled() and not server_state.use_speculative,
    }


@app.get("/v1/models")
async def list_models():
    assert server_state is not None
    # Hermes custom_providers often select model id "main"; advertise both.
    ids = []
    for mid in (server_state.config.model_id, "main"):
        if mid and mid not in ids:
            ids.append(mid)
    return {
        "object": "list",
        "data": [{"id": mid, "object": "model"} for mid in ids],
    }


async def _generate_locked(**kwargs) -> GenerationResult:
    assert server_state is not None

    def _call():
        with server_state.lock:
            return server_state.generate(**kwargs)

    return await asyncio.to_thread(_call)


def _stream_locked(*, parse_tools: bool = False, **kwargs) -> Iterator[str]:
    """SSE chat.completion.chunk stream.

    When ``parse_tools`` is True (request included a tools schema), buffer the
    full generation so Gemma ``<|tool_call>…`` spans can be parsed into OpenAI
    ``tool_calls`` before any content is emitted. Hermes' streaming path expects
    structured tool_calls rather than raw marker text.
    """
    assert server_state is not None
    with server_state.lock:
        model_id = server_state.config.model_id
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        role_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_id,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(role_chunk)}\n\n"

        if parse_tools:
            pieces: list[str] = []
            final_metrics: InferenceMetrics | None = None
            for piece, _tok_id, metrics in server_state.generate_stream(**kwargs):
                if metrics is not None:
                    final_metrics = metrics
                elif piece:
                    pieces.append(piece)
            text = "".join(pieces)
            content, tool_calls = _parse_gemma_tool_calls(text)
            if content:
                chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_id,
                    "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            if tool_calls:
                for i, tc in enumerate(tool_calls):
                    delta = {
                        "tool_calls": [
                            {
                                "index": i,
                                "id": tc["id"],
                                "type": "function",
                                "function": {
                                    "name": tc["function"]["name"],
                                    "arguments": tc["function"]["arguments"],
                                },
                            }
                        ]
                    }
                    chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_id,
                        "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
            finish_reason = "tool_calls" if tool_calls else "stop"
            final = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_id,
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
            }
            if final_metrics is not None:
                final["metrics"] = final_metrics.as_dict() | {"sample_on_device": server_state.use_device_sampling}
            yield f"data: {json.dumps(final)}\n\n"
            yield "data: [DONE]\n\n"
            return

        for piece, _tok_id, metrics in server_state.generate_stream(**kwargs):
            if metrics is not None:
                final = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_id,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    "metrics": metrics.as_dict() | {"sample_on_device": server_state.use_device_sampling},
                }
                yield f"data: {json.dumps(final)}\n\n"
                yield "data: [DONE]\n\n"
                return
            if not piece:
                continue
            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_id,
                "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"


def _chat_completion_json(result: GenerationResult, model_id: str) -> dict[str, Any]:
    content, tool_calls = _parse_gemma_tool_calls(result.text)
    message: dict[str, Any] = {"role": "assistant", "content": content or None}
    if tool_calls:
        message["tool_calls"] = tool_calls
        # OpenAI clients expect content null (not "") when only tool_calls fire.
        if not content:
            message["content"] = None
    finish_reason = "tool_calls" if tool_calls else "stop"
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": result.metrics.prompt_tokens,
            "completion_tokens": result.metrics.generated_tokens,
            "total_tokens": result.metrics.prompt_tokens + result.metrics.generated_tokens,
        },
        "metrics": _metrics_with_sampling(result, server_state.use_device_sampling),
    }


def _stream_generate_locked(**kwargs) -> Iterator[str]:
    assert server_state is not None
    with server_state.lock:
        for piece, _tok_id, metrics in server_state.generate_stream(**kwargs):
            if metrics is not None:
                yield f"data: {json.dumps({'metrics': metrics.as_dict() | {'sample_on_device': server_state.use_device_sampling}, 'done': True})}\n\n"
                return
            if piece:
                yield f"data: {json.dumps({'text': piece})}\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if server_state is None or not server_state.ready:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Render the full conversation with the model's chat template. The rendered
    # string is already templated, so it is generated with instruct=False. If
    # rendering fails, fall back to the last user message + config.instruct.
    tokenizer = server_state.generator.model_args[0].tokenizer
    try:
        prompt = _render_chat_prompt(
            request.messages,
            tokenizer,
            tools=request.tools,
            tool_choice=request.tool_choice,
        )
        instruct = False
    except _FallbackToLastUser:
        prompt = _extract_user_prompt(request.messages)
        instruct = server_state.config.instruct
    max_tokens = request.max_tokens or request.max_completion_tokens or server_state.config.default_max_tokens
    stop_at_eos = request.stop_at_eos if request.stop_at_eos is not None else server_state.config.stop_at_eos
    # Echo requested model id (Hermes often uses "main") but fall back to config.
    model_id = request.model or server_state.config.model_id
    gen_kwargs = dict(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        instruct=instruct,
        stop_at_eos=stop_at_eos,
        repetition_penalty=request.repetition_penalty,
        repetition_max_streak=request.repetition_max_streak,
        draft_len=request.draft_len,
    )
    parse_tools = bool(request.tools) and request.tool_choice != "none"

    if request.stream:
        return StreamingResponse(
            _stream_locked(parse_tools=parse_tools, **gen_kwargs),
            media_type="text/event-stream",
        )

    try:
        result = await _generate_locked(**gen_kwargs)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _chat_completion_json(result, model_id)


@app.post("/generate")
async def generate(request: GenerateRequest):
    if server_state is None or not server_state.ready:
        raise HTTPException(status_code=503, detail="Model not loaded")

    max_tokens = request.max_tokens or server_state.config.default_max_tokens
    instruct = request.instruct if request.instruct is not None else server_state.config.instruct
    stop_at_eos = request.stop_at_eos if request.stop_at_eos is not None else server_state.config.stop_at_eos
    gen_kwargs = dict(
        prompt=request.prompt,
        max_tokens=max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        instruct=instruct,
        stop_at_eos=stop_at_eos,
        repetition_penalty=request.repetition_penalty,
        repetition_max_streak=request.repetition_max_streak,
        draft_len=request.draft_len,
    )

    if request.stream:
        return StreamingResponse(_stream_generate_locked(**gen_kwargs), media_type="text/event-stream")

    try:
        result = await _generate_locked(**gen_kwargs)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "text": result.text,
        "token_ids": result.token_ids,
        "metrics": _metrics_with_sampling(result, server_state.use_device_sampling),
    }
