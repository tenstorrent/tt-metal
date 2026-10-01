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
(``tests/test_serving.py`` / ``tests/serving_smoke.py``) can exercise
the exact contract on device wherever ``ttnn`` runs, without the container-gated
vLLM stack.

Cache ownership: this session uses the **model-owned hybrid paged** K/V cache
(``tt_model.tt_kv_cache``) — the "generator/standalone" ownership mode. Sliding
layers keep a circular 1024-token physical window and full-attention layers keep
the full served context; DG-local readers expose the contiguous prefix/window
shape denoise attention consumes. A single deterministic identity page-table set
still backs one active sequence. Arbitrary vLLM block-pool ownership and
concurrent per-request tables remain #47488/#47557.
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
from models.experimental.diffusion_gemma.tt.degeneracy import DegenerateBlockError
from models.experimental.diffusion_gemma.tt.generate import (
    _contains_stop_token,
    _infer_generation_vocab_size,
    _infer_context_limit,
    _normalize_eos_token_ids,
    _pad_prompt_tokens_for_prefill,
    _validate_prompt_tokens,
    denoise_and_commit_block,
    make_seeded_gumbel_noise_fn,
    make_seeded_host_canvas_init_fn,
    make_seeded_host_noise_tokens_fn,
    prefill_prompt_tokens,
)

# Sampling modes exposed to the serving layer. "device" (on-device seeded SFPU RNG,
# lane-salted per tt-metal#52024) is the production mode and the only one usable under
# up-front capture (DG_UPFRONT_CAPTURE=1); "argmax" is the greedy speed/determinism control.
# Injecting a torch run's exact noise for HF<->TT parity is a different concern:
# tt/generate.py:make_host_gumbel_noise_fn. If full-vocab noise ever OOMs at 256K contexts,
# pursue the TP-sharded denoise terminal (unimplemented design note at the bottom of
# tt/sampling.py).
GUMBEL_MODES = ("argmax", "device")


def _resolve_degeneracy_stop_ids(explicit, *, stop_token_ids, tokenizer) -> set | None:
    """Which ids the degeneracy guard may treat as terminal padding.

    Preference order: what the caller declared, then the session's own stop policy, then every
    special id the tokenizer knows. The tokenizer fallback is what makes the vLLM path correct
    without hand-transcribing this checkpoint's ``eos_token_id`` list into the source: a canvas
    tail made of <eos>/<end_of_turn>/<pad> is padding under any tokenizer, and no special id is
    content the model is answering with. Returns ``None`` when nothing is knowable, which leaves
    the guard on its whole-canvas rule.
    """
    for candidate in (explicit, stop_token_ids):
        ids = _normalize_eos_token_ids(candidate, kind="stop_token_ids") if candidate is not None else None
        if ids:
            return set(ids)
    special = set()
    for attr in ("all_special_ids", "eos_token_id"):
        value = getattr(tokenizer, attr, None)
        if value is None:
            continue
        try:
            special |= _normalize_eos_token_ids(value, kind="stop_token_ids") or set()
        except ValueError:
            # A mock/partial tokenizer can expose non-int specials; the guard must degrade, not
            # take down generation.
            continue
    if special:
        return special
    logger.warning(
        "[serving] degeneracy guard has no stop-token ids (no explicit set, no session stop policy, "
        "no tokenizer specials): a terminal <eos> canvas cannot be told from a collapsed one, so "
        "the whole-canvas rule applies and normal completions may be rejected"
    )
    return None


def _validate_next_block_capacity(
    tt_model, *, start_pos: int, canvas_length: int, served_context_limit: int | None = None
) -> None:
    """Reject a whole-canvas commit before any denoise/device execution."""
    context_limit = _infer_context_limit(tt_model)
    if served_context_limit is not None:
        context_limit = (
            int(served_context_limit) if context_limit is None else min(int(context_limit), int(served_context_limit))
        )
    if context_limit is None:
        return
    end_pos = start_pos + canvas_length
    if end_pos > context_limit:
        raise ValueError(
            "next committed canvas exceeds the model context window: "
            f"{start_pos} + {canvas_length} = {end_pos} > {context_limit}"
        )


def _bind_retry_noise(gumbel_noise_fn, block_idx: int):
    """Adapt a ``(block_idx, attempt)`` noise factory to the ``(attempt)`` retry contract.

    ``denoise_and_commit_block`` passes only the attempt number, and it refuses any factory that is
    not marked ``supports_retry`` -- so the marker has to ride along, otherwise binding the block
    would turn a working retry into a hard error.
    """
    if gumbel_noise_fn is None:
        return None

    def retry_noise_for_attempt(attempt: int):
        return gumbel_noise_fn(block_idx, attempt)

    retry_noise_for_attempt.supports_retry = getattr(gumbel_noise_fn, "supports_retry", False)
    return retry_noise_for_attempt


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
        gumbel_mode: str = "device",
        eos_token_id=None,
        stop_token_ids=None,
        degeneracy_stop_token_ids=None,
        page_table=None,
        page_tables_per_layer=None,
        prefill_execution_len: int | None = None,
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
        # The model-owned hybrid cache attaches its identity page tables to the model;
        # a caller without its own tables (demo, tests) serves against those directly.
        self.page_tables_per_layer = (
            page_tables_per_layer
            if page_tables_per_layer is not None
            else getattr(tt_model, "_dg_hybrid_page_tables_per_layer", None)
        )
        self.prefill_execution_len = None if prefill_execution_len is None else int(prefill_execution_len)
        self.gumbel_mode = gumbel_mode
        # ``None`` selects ordinary eager denoise. The vLLM wrapper passes only the
        # up-front model-lifetime denoise function when DG_UPFRONT_CAPTURE is enabled.
        self._denoise_block_fn = denoise_block_fn
        # No prompt-prefix KV reuse; both fields feed the prefill_block0 metric.
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
        # The degeneracy guard needs to know which ids are stop tokens for a DIFFERENT reason than
        # the stop policy does: to tell an answer's terminal <eos> padding from a collapsed canvas.
        # Those two must not share one field: the vLLM path deliberately sets stop_token_ids=[]
        # ("vLLM owns the stop decision"), and a guard fed from that field would reject every
        # terminal block as a wall of one token.
        self.degeneracy_stop_token_ids = _resolve_degeneracy_stop_ids(
            degeneracy_stop_token_ids,
            stop_token_ids=self.stop_token_ids,
            tokenizer=tokenizer,
        )

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
        # Optional model-lifetime adapter injected by the vLLM wrapper after startup capture.
        # A session only borrows this object; reset must detach it without releasing its traces.
        self._persistent_adapter = None
        self.prompt_len = None
        self.cache_len = None
        self.next_pos = None
        self.block_idx = 0
        self.finished = False

    def _build_gumbel_noise_fn(self, mesh_device, gumbel_seed: int):
        if self.gumbel_mode == "argmax":
            return _argmax_gumbel_noise_fn
        return make_seeded_gumbel_noise_fn(
            mesh_device,
            batch=1,
            canvas_len=self.canvas_length,
            vocab_size=self.vocab_size,
            seed=TS._validate_ttnn_rand_seed(gumbel_seed),
        )

    def attach_persistent_adapter(self, adapter) -> None:
        """Borrow a startup-captured adapter for this request's prefill/decode lifetime."""
        if adapter is None:
            raise ValueError("persistent adapter must not be None")
        if self._logits_fn is not None or self.next_pos is not None:
            raise RuntimeError("persistent adapter must be attached before session prefill")
        self._persistent_adapter = adapter

    def prefill(self, prompt_tokens: torch.Tensor) -> int:
        """Write prompt K/V into the frozen cache and build the denoise logits fn.

        ``prompt_tokens`` is ``[1, prompt_len]`` host token ids (already
        chat-templated/tokenized by the caller — vLLM owns tokenization). Any
        valid prompt length is accepted; prefill pads to a 32-tile multiple
        internally and reports both the logical ``prompt_len`` and the aligned
        ``cache_len`` used for the frozen-prefix read. ``prefill_execution_len``
        may make the compute tensor larger without moving that logical cache
        boundary. Returns ``cache_len``.

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

        # prefill_reused stays False (no prefix-cache reuse; vLLM advertises
        # supports_prefix_caching: False). The field and prefill_time_s are KEPT —
        # generator_vllm emits both in the prefill_block0 metric.
        self.prefill_reused = False
        self.prefill_time_s = 0.0
        execution_len = getattr(self, "prefill_execution_len", None)
        if self._persistent_adapter is not None:
            logger.info(
                f"DG_UPFRONT_MARK prefill_device_begin prompt_len={prompt_len} cache_len={cache_len} "
                f"execution_len={execution_len or cache_len}"
            )
        t0 = time.perf_counter()
        execution_kwargs = {"execution_len": execution_len} if execution_len is not None else {}
        prefill = prefill_prompt_tokens(
            self.tt_model,
            prompt_tokens,
            page_table=self.page_table,
            page_tables_per_layer=self.page_tables_per_layer,
            **execution_kwargs,
        )
        self.prefill_time_s = time.perf_counter() - t0
        prompt_len = prefill.prompt_len
        cache_len = prefill.cache_len
        if self._persistent_adapter is not None:
            logger.info(f"DG_UPFRONT_MARK prefill_device_end prompt_len={prompt_len} cache_len={cache_len}")

        self.prompt_len = prompt_len
        self.cache_len = cache_len
        self.next_pos = cache_len
        self.block_idx = 0
        self.finished = False
        if self._persistent_adapter is not None:
            logger.info(f"DG_UPFRONT_MARK rebind_begin cache_len={cache_len}")
            self._persistent_adapter.rebind_prompt(cache_len, true_prompt_len=prompt_len)
            logger.info(f"DG_UPFRONT_MARK rebind_end cache_len={cache_len}")
            self._logits_fn = self._persistent_adapter
        else:
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
        served_context_limit = (
            getattr(self._logits_fn, "_reveal_p_max", None)
            if self._logits_fn is getattr(self, "_persistent_adapter", None)
            and getattr(self._logits_fn, "use_reveal_mask", False)
            else None
        )
        _validate_next_block_capacity(
            self.tt_model,
            start_pos=start_pos,
            canvas_length=self.canvas_length,
            served_context_limit=served_context_limit,
        )
        block_idx = self.block_idx
        gumbel_for_block = self._gumbel_noise_fn(block_idx) if self._gumbel_noise_fn else None
        noise_for_block = self._noise_tokens_fn(block_idx) if self._noise_tokens_fn else None
        init_canvas = self._init_canvas_fn(block_idx, start_pos)

        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        try:
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
                # The guard's stop set, NOT the session's stop policy (see __init__): this argument
                # only tells the degeneracy check which ids are terminal padding. Never pass
                # self.stop_token_ids here — the vLLM path leaves it empty by design.
                stop_token_ids=self.degeneracy_stop_token_ids,
                # Both are needed by DG_DEGENERACY_POLICY=retry: the noise factory so a retry can
                # draw different noise, and the canvas factory because the denoise path consumes
                # the canvas it is handed. The noise factory must be bound to THIS block:
                # denoise_and_commit_block calls retry_noise_fn(attempt) with one positional
                # argument.
                retry_noise_fn=_bind_retry_noise(self._gumbel_noise_fn, block_idx),
                retry_init_canvas_fn=lambda: self._init_canvas_fn(block_idx, start_pos),
            )
        except DegenerateBlockError as degenerate:
            # The canvas was NOT committed. End the request here and return a zero-token terminal
            # emission, so the caller keeps every healthy block it already received instead of
            # losing the whole response to an exception.
            logger.warning(f"[serving] ending request at block {block_idx}: {degenerate}")
            self.finished = True
            self.block_idx += 1
            return BlockEmission(
                tokens=torch.zeros((1, 0), dtype=torch.long),
                block_idx=block_idx,
                start_pos=start_pos,
                next_pos=start_pos,
                num_denoise_steps=0,
                halted=False,
                stop=True,
                latency_s=time.perf_counter() - t0,
                denoise_latency_s=timings.get("denoise_s", 0.0),
                commit_latency_s=timings.get("commit_s", 0.0),
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
        """Snapshot model-lifetime up-front controller counters before detach."""
        if self._logits_fn is None:
            return []
        controller = getattr(self._logits_fn, "_upfront_traced_denoise_controller", None)
        return [controller.stats()] if controller is not None and hasattr(controller, "stats") else []

    def reset(self) -> None:
        """Release eager logits state or detach a borrowed persistent adapter."""
        logits_fn = self._logits_fn
        persistent = logits_fn is not None and logits_fn is getattr(self, "_persistent_adapter", None)
        try:
            if logits_fn is not None and not persistent:
                if hasattr(logits_fn, "reset"):
                    try:
                        logits_fn.reset()
                    except BaseException as cleanup_error:
                        logger.error(f"failed to reset serving logits state: {cleanup_error}")
        finally:
            self._logits_fn = None
            self._persistent_adapter = None
            self.next_pos = None
            self.finished = False
            self.block_idx = 0
