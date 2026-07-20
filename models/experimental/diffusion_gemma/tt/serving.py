# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Block-granular serving session for DiffusionGemma (#47466 / #47488).

This module is the **vLLM-free** core of the serving path. The one-shot
``tt.generate.generate_text_from_checkpoint_state`` orchestration prefills the
prompt and runs every block in a single call; serving needs the same work split
into per-request steps:

- ``prefill``  — tokenize (caller-owned) → write prompt K/V → build the stateful
  denoise logits fn (``DenoiseLogitsAdapter``) that reads the frozen prompt K/V
  and applies self-conditioning.
- ``decode_block`` — run one canvas denoise trajectory (the on-device Gumbel-max
  / entropy-budget / renoise loop), commit the clean argmax, append its K/V, and
  emit **one 256-token block**. Advance the absolute position by ``canvas_length``.

The vLLM adapter (``tt/generator_vllm.py``) is a thin wrapper over this session:
its ``prefill_forward`` maps to :meth:`BlockDiffusionServingSession.prefill` plus
the first :meth:`decode_block`, and each ``decode_forward`` maps to one
:meth:`decode_block`. Keeping the block-emission state machine here (with no vLLM
import) means the reduced-surface serving driver
(``tests/test_serving_block_contract.py`` / ``demo/serving_smoke.py``) can exercise
the exact contract on device wherever ``ttnn`` runs, without the container-gated
vLLM stack.

Cache ownership: this session uses the **model-owned contiguous** K/V cache
(``tt_model.tt_kv_cache``) — the "generator/standalone" ownership mode. The
diffusion denoise-read path (``read_prompt_kv_cache_by_layer`` →
``ttnn.slice`` over ``[B, heads, max_seq, head_dim]``) reads the frozen prompt
prefix from that contiguous cache, so a single session tracks one active
sequence. Routing the frozen-prefix read through a vLLM paged cache + per-request
block tables (for concurrent batched serving) is part of the upstream #47488
runner/scheduler + cache-ownership change and the batched-canvas-decode work
(#47557); see ``doc/vllm_integration/README.md``.
"""

from __future__ import annotations

import time
from numbers import Integral
from typing import NamedTuple

import torch
from loguru import logger

from models.experimental.diffusion_gemma.config import DiffusionConfig
from models.experimental.diffusion_gemma.tt import sampling as TS
from models.experimental.diffusion_gemma.tt.denoise_forward import (
    make_generation_logits_fn_builder_from_checkpoint_state,
)
from models.experimental.diffusion_gemma.tt.generate import (
    _contains_stop_token,
    _infer_generation_vocab_size,
    _infer_context_limit,
    _normalize_eos_token_ids,
    _pad_prompt_tokens_for_prefill,
    _validate_prompt_tokens,
    denoise_and_commit_block,
    make_seeded_chunked_gumbel_noise_fn,
    make_seeded_gumbel_noise_fn,
    make_seeded_host_canvas_init_fn,
    make_seeded_host_gumbel_noise_fn,
    make_seeded_host_noise_tokens_fn,
    prefill_prompt_tokens,
)
from models.experimental.diffusion_gemma.tt.prefix_cache import PrefixKVCache, prefix_cache_enabled

# Sampling modes exposed to the serving layer. "chunked" is the DEFAULT: the
# no-materialize on-device Gumbel-max sampler (argmax(logits/T + Gumbel)) that is
# distribution-faithful to the model's reference EntropyBoundSampler / HF
# multinomial(softmax(logits/T)) AND fits full-depth 256K. "argmax" is the greedy
# RUN-first path (no Gumbel materialization, also fits 256K) — kept as an opt-in
# speed/determinism mode. "host"/"device" are seeded-Gumbel debug/reference paths
# that materialize the full-vocab Gumbel (OOM at 256K), for token-exact validation.
GUMBEL_MODES = ("chunked", "argmax", "host", "device")


def _validate_next_block_capacity(tt_model, *, start_pos: int, canvas_length: int) -> None:
    """Reject a whole-canvas commit before any denoise/device execution."""
    context_limit = _infer_context_limit(tt_model)
    if context_limit is None:
        return
    end_pos = start_pos + canvas_length
    if end_pos > context_limit:
        raise ValueError(
            "next committed canvas exceeds the model context window: "
            f"{start_pos} + {canvas_length} = {end_pos} > {context_limit}"
        )


def _argmax_gumbel_noise_fn(block_idx: int):
    """Block-level Gumbel hook whose per-step noise is ``None`` → clean argmax.

    Mirrors ``demo/text_demo.py --argmax-sampling``. ``gumbel_max(logits, T, None)``
    returns the temperature-scaled argmax, so the committed clean argmax equals the
    sampled canvas and no full-vocab Gumbel tensor is allocated.
    """
    if isinstance(block_idx, bool) or not isinstance(block_idx, Integral):
        raise ValueError("Gumbel block index must be an integer")

    def _per_step(step: int):
        del step
        return None

    return _per_step


class BlockEmission(NamedTuple):
    """One committed 256-token block plus the per-block serving metrics."""

    tokens: torch.Tensor  # [batch, canvas_length] committed clean-argmax ids (host)
    block_idx: int
    start_pos: int
    next_pos: int
    num_denoise_steps: int
    halted: bool  # denoise loop hit the stable+confident early halt
    stop: bool  # an EOS / stop token was committed in this block
    latency_s: float
    denoise_latency_s: float
    commit_latency_s: float


class BlockDiffusionServingSession:
    """Per-request block-diffusion decode state for the serving path.

    Reuses the existing ``tt.generate`` block engine unchanged; the only net-new
    behaviour is persisting the absolute position and the stateful logits fn
    across per-block decode calls (which ``generate_blocks`` keeps internal to a
    single call).
    """

    def __init__(
        self,
        tt_model,
        dg_state_dict,
        *,
        config: DiffusionConfig | None = None,
        tokenizer=None,
        vocab_size: int | None = None,
        seed: int = 0,
        gumbel_mode: str = "chunked",
        gumbel_vocab_chunk_size: int = 1024,
        eos_token_id=None,
        stop_token_ids=None,
        page_table=None,
        page_tables_per_layer=None,
        prefix_cache: PrefixKVCache | None = None,
        adapter_kwargs: dict | None = None,
        denoise_block_fn=None,
        logits_fn_builder_factory=make_generation_logits_fn_builder_from_checkpoint_state,
    ):
        if gumbel_mode not in GUMBEL_MODES:
            raise ValueError(f"gumbel_mode must be one of {GUMBEL_MODES}, got {gumbel_mode!r}")
        self.tt_model = tt_model
        self.dg_state_dict = dg_state_dict
        self.config = DiffusionConfig() if config is None else config
        self.canvas_length = self.config.canvas_length
        self.page_table = page_table
        self.page_tables_per_layer = page_tables_per_layer
        self.gumbel_mode = gumbel_mode
        self.gumbel_vocab_chunk_size = gumbel_vocab_chunk_size
        # Explicit denoise-loop selection for the traced serving decode path. None ⇒ the
        # env-gated dispatcher (``generate._resolve_default_denoise_block_fn`` — the current
        # default, eager unless a DG_DENOISE_* flag is set). The vLLM adapter passes the traced
        # fn here when it honors ``enable_trace``, so the persistent session's logits fn caches
        # ONE traced controller (captured on block 0, ``execute_trace``-replayed every block).
        self._denoise_block_fn = denoise_block_fn
        # Frozen prompt-prefix KV reuse (APC prototype, #47466). Off unless a
        # PrefixKVCache is attached AND DG_PREFIX_CACHE is set; the paged frozen
        # prefix read is #47488. Reuse only kicks in for a model-owned contiguous
        # cache (page_table=None) — a paged/vLLM cache owns its own prefix caching.
        self.prefix_cache = prefix_cache
        self.prefill_reused = False
        self.prefill_time_s = 0.0

        if vocab_size is None:
            vocab_size = _infer_generation_vocab_size(tokenizer, tt_model)
        if vocab_size is None:
            raise ValueError("vocab_size is required (pass it, or a tokenizer/model with vocab metadata)")
        self.vocab_size = int(vocab_size)

        # Stop policy mirrors generate_text_from_checkpoint_state defaults.
        if stop_token_ids is None and eos_token_id is None:
            eos_token_id = getattr(tokenizer, "eos_token_id", None)
        self.stop_token_ids = stop_token_ids if stop_token_ids is not None else eos_token_id
        if self.stop_token_ids is not None:
            _normalize_eos_token_ids(self.stop_token_ids)

        mesh_device = tt_model.mesh_device
        self._init_canvas_fn = make_seeded_host_canvas_init_fn(
            mesh_device, batch=1, canvas_len=self.canvas_length, vocab_size=self.vocab_size, seed=seed
        )
        self._noise_tokens_fn = make_seeded_host_noise_tokens_fn(
            mesh_device, batch=1, canvas_len=self.canvas_length, vocab_size=self.vocab_size, seed=seed + 1
        )
        self._gumbel_noise_fn = self._build_gumbel_noise_fn(mesh_device, seed + 2)

        adapter_kwargs = dict(adapter_kwargs or {})
        if "config" not in adapter_kwargs:
            adapter_config = getattr(tt_model, "hf_config", None)
            if adapter_config is not None:
                adapter_kwargs["config"] = adapter_config
        self._logits_fn_builder = logits_fn_builder_factory(dg_state_dict, **adapter_kwargs)

        self._logits_fn = None
        self.prompt_len = None
        self.cache_len = None
        self.next_pos = None
        self.block_idx = 0
        self.finished = False

    def _build_gumbel_noise_fn(self, mesh_device, gumbel_seed: int):
        if self.gumbel_mode == "argmax":
            return _argmax_gumbel_noise_fn
        if self.gumbel_mode == "chunked":
            return make_seeded_chunked_gumbel_noise_fn(
                seed=TS._validate_ttnn_rand_seed(gumbel_seed),
                vocab_chunk_size=self.gumbel_vocab_chunk_size,
            )
        if self.gumbel_mode == "host":
            return make_seeded_host_gumbel_noise_fn(
                mesh_device, batch=1, canvas_len=self.canvas_length, vocab_size=self.vocab_size, seed=gumbel_seed
            )
        return make_seeded_gumbel_noise_fn(
            mesh_device,
            batch=1,
            canvas_len=self.canvas_length,
            vocab_size=self.vocab_size,
            seed=TS._validate_ttnn_rand_seed(gumbel_seed),
        )

    def _prefix_reuse_active(self) -> bool:
        # Reuse only for the model-owned contiguous cache (a paged/vLLM cache owns
        # its own APC — #47488) and only when opted in via DG_PREFIX_CACHE.
        return (
            self.prefix_cache is not None
            and self.page_table is None
            and self.page_tables_per_layer is None
            and prefix_cache_enabled()
        )

    def prefill(self, prompt_tokens: torch.Tensor) -> int:
        """Write prompt K/V into the frozen cache and build the denoise logits fn.

        ``prompt_tokens`` is ``[1, prompt_len]`` host token ids (already
        chat-templated/tokenized by the caller — vLLM owns tokenization). Any
        valid prompt length is accepted; prefill pads to a 32-tile multiple
        internally and reports both the logical ``prompt_len`` and the aligned
        ``cache_len`` used for the frozen-prefix read. Returns ``cache_len``.

        When a :class:`~models.experimental.diffusion_gemma.tt.prefix_cache.PrefixKVCache`
        is attached and ``DG_PREFIX_CACHE`` is on, a prompt whose aligned token span
        is a byte-identical leading span of the resident contiguous cache **skips the
        prefill forward entirely** (frozen prompt-prefix reuse — see the design note).
        The committed output is bit-identical to a fresh prefill because causal
        prefill makes position ``i``'s K/V a pure function of ``tokens[0:i]`` + the
        absolute (RoPE) position ``i``.
        """
        _validate_prompt_tokens(prompt_tokens)
        if prompt_tokens.shape[0] != 1:
            raise NotImplementedError(
                "BlockDiffusionServingSession is single-sequence; the vLLM adapter "
                "manages one session per active request (batched canvas decode is #47557)"
            )
        prompt_len = int(prompt_tokens.shape[1])
        aligned = _pad_prompt_tokens_for_prefill(prompt_tokens)
        cache_len = int(aligned.shape[1])
        aligned_ids = aligned.reshape(-1).tolist()

        self.prefill_reused = False
        self.prefill_time_s = 0.0
        reuse_active = self._prefix_reuse_active()
        plan = self.prefix_cache.plan(aligned_ids, prompt_len, cache_len) if reuse_active else None

        if plan is not None and plan.reuse:
            # [0:cache_len] is already correct in the contiguous cache — skip prefill.
            self.prefix_cache.note_reuse(plan, prefill_time_saved_s=self.prefix_cache.avg_prefill_time_s)
            logger.info(
                f"[prefix_cache] REUSE prompt_len={prompt_len} cache_len={cache_len} "
                f"(resident_cache_len={self.prefix_cache.resident_cache_len}) — prefill skipped"
            )
        else:
            if plan is not None:
                self.prefix_cache.note_miss(plan)
                if plan.shorter_prefix:
                    logger.info(
                        f"[prefix_cache] shorter-prefix miss: incoming aligned prompt is a byte-identical "
                        f"proper prefix ({plan.cache_len} of resident {self.prefix_cache.resident_cache_len}), "
                        f"but bf16 SDPA reduction-length makes it non-bit-exact → full prefill "
                        f"(reuse only with allow_shorter_prefix, approximate tier)"
                    )
                elif plan.partial_prefix:
                    logger.info(
                        f"[prefix_cache] partial-prefix miss: matched {plan.matched_len} aligned "
                        f"tokens, suffix differs/extends → full prefill (needs chunked prefill / #47488)"
                    )
            t0 = time.perf_counter()
            prefill = prefill_prompt_tokens(
                self.tt_model,
                prompt_tokens,
                page_table=self.page_table,
                page_tables_per_layer=self.page_tables_per_layer,
            )
            self.prefill_time_s = time.perf_counter() - t0
            prompt_len = prefill.prompt_len
            cache_len = prefill.cache_len
            if reuse_active:
                self.prefix_cache.observe_prefill_time(self.prefill_time_s)

        # After either path, the contiguous cache holds THIS prompt's [0:cache_len].
        if reuse_active:
            self.prefix_cache.record(aligned_ids, prompt_len, cache_len)

        self.prompt_len = prompt_len
        self.cache_len = cache_len
        self.next_pos = cache_len
        self.block_idx = 0
        self.finished = False
        self.prefill_reused = bool(plan is not None and plan.reuse)
        self._logits_fn = self._logits_fn_builder(
            self.tt_model,
            prompt_tokens=prompt_tokens,
            prompt_len=cache_len,
            page_table=self.page_table,
            page_tables_per_layer=self.page_tables_per_layer,
        )
        return cache_len

    def decode_block(self) -> BlockEmission:
        """Denoise + commit one canvas → emit one 256-token block.

        This is the block-granular unit the serving runner drives once per decode
        step (mapping cleanly to the #47488 per-block runner/scheduler contract).
        """
        if self._logits_fn is None or self.next_pos is None:
            raise RuntimeError("decode_block called before prefill")
        if self.finished:
            raise RuntimeError("decode_block called after the sequence already emitted a stop token")

        start_pos = self.next_pos
        _validate_next_block_capacity(self.tt_model, start_pos=start_pos, canvas_length=self.canvas_length)
        block_idx = self.block_idx
        gumbel_for_block = self._gumbel_noise_fn(block_idx) if self._gumbel_noise_fn else None
        noise_for_block = self._noise_tokens_fn(block_idx) if self._noise_tokens_fn else None
        init_canvas = self._init_canvas_fn(block_idx, start_pos)

        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        block = denoise_and_commit_block(
            self.tt_model,
            self._logits_fn,
            init_canvas,
            self.config,
            start_pos=start_pos,
            gumbel_noise_fn=gumbel_for_block,
            noise_tokens_fn=noise_for_block,
            page_table=self.page_table,
            page_tables_per_layer=self.page_tables_per_layer,
            denoise_block_fn=self._denoise_block_fn,
            timings=timings,
        )
        latency_s = time.perf_counter() - t0

        self.next_pos = block.next_pos
        self.block_idx += 1
        stop = _contains_stop_token(block.committed, self.stop_token_ids)
        if stop:
            self.finished = True
        trajectory = block.trajectory
        return BlockEmission(
            tokens=block.committed,
            block_idx=block_idx,
            start_pos=start_pos,
            next_pos=block.next_pos,
            num_denoise_steps=trajectory.num_steps,
            halted=bool(trajectory.halted),
            stop=stop,
            latency_s=latency_s,
            denoise_latency_s=timings["denoise_s"],
            commit_latency_s=timings["commit_s"],
        )

    def trace_stats(self) -> list[dict]:
        """Snapshot any per-request traced-controller counters before reset."""
        if self._logits_fn is None:
            return []
        stats = []
        for attr in (
            "_traced_denoise_controller",
            "_traced_denoise_multistep_controller",
            "_traced_early_halt_controller",
        ):
            controller = getattr(self._logits_fn, attr, None)
            if controller is not None and hasattr(controller, "stats"):
                stats.append(controller.stats())
        return stats

    def reset(self) -> None:
        """Release per-request Metal traces, buffers, and logits state."""
        logits_fn = self._logits_fn
        try:
            if logits_fn is not None:
                for attr in (
                    "_traced_denoise_controller",
                    "_traced_denoise_multistep_controller",
                    "_traced_early_halt_controller",
                ):
                    controller = getattr(logits_fn, attr, None)
                    if controller is not None:
                        try:
                            controller.release()
                        except BaseException as cleanup_error:
                            logger.error(f"failed to release serving controller {attr}: {cleanup_error}")
                        finally:
                            delattr(logits_fn, attr)
                if hasattr(logits_fn, "reset"):
                    try:
                        logits_fn.reset()
                    except BaseException as cleanup_error:
                        logger.error(f"failed to reset serving logits state: {cleanup_error}")
        finally:
            self._logits_fn = None
            self.next_pos = None
            self.finished = False
            self.block_idx = 0
