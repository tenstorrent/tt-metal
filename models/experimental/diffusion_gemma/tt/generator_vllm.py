# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DiffusionGemma vLLM adapter for the tenstorrent/vllm TT plugin (#47466 / #47488).

DiffusionGemma is a **block-diffusion** model: a single decode step denoises a
256-token canvas and commits it, so the model emits a **256-token BLOCK per decode
step**, not one token. This adapter is written to that *block-granular* contract.
The whole denoise loop (bidirectional canvas attention, three-phase KV, on-device
Gumbel-max / entropy-budget / renoise sampling, self-conditioning) lives inside
``prefill_forward`` / ``decode_forward`` via the existing ``tt.generate`` engine —
the runner passes only tokens / page_table / kv_cache / start_pos / prompt_lens /
sampling; the tt-metal model owns forward + attention + KV.

Structure
---------
The block-emission state machine is the vLLM-free
:class:`~models.experimental.diffusion_gemma.tt.serving.BlockDiffusionServingSession`;
this file is the thin vLLM interface wrapper over it:

- ``prefill_forward`` → :meth:`BlockDiffusionServingSession.prefill` (write prompt
  K/V, build the stateful denoise logits fn) followed by the first
  :meth:`decode_block` (block 0), mirroring the autoregressive contract where
  prefill returns the first token — here it returns the first 256-token block.
- ``decode_forward`` → one :meth:`decode_block` per active request (block N).

Because the block-emission core has no vLLM import, the reduced-surface serving
driver drives the identical contract on device without the (container-gated) vLLM
stack.

Contract gaps handled here vs deferred to #47488 (upstream tenstorrent/vllm)
--------------------------------------------------------------------------
The current TT runner assumes **one committed token per decode step**. Emitting a
256-token block therefore needs the runner/scheduler to (a) accept a
``[num_reqs, 256]`` block output, (b) advance ``num_computed_tokens`` /
``num_tokens`` by ``canvas_length`` per decode step, and (c) bound-check
``start_idx + 256 <= max_model_len``. That runner+scheduler change is **#47488**;
this adapter is written to that block contract so it works once #47488 lands.

Cache ownership
---------------
The diffusion denoise-read path reads the frozen prompt prefix from the
**model-owned paged hybrid** ``tt_model.tt_kv_cache``. Sliding layers retain a
1024-token circular window while full-attention layers retain ``max_model_len``;
deterministic identity page tables keep the existing one-cache/one-active-sequence
ownership contract. :meth:`allocate_kv_cache` returns those existing handles (no
double allocation). This is not vLLM block-pool ownership: routing arbitrary
per-request block tables for concurrent batched serving remains #47488/#47557.
The model-owned hybrid KV layout is the only served cache layout.

**Do not edit ``models/demos/gemma4/``.** The backbone is imported and reused
unchanged; the ``get_kv_cache_spec`` hybrid layer-type logic is copied (not
imported) so this adapter stays self-contained.
"""

from __future__ import annotations

import json
import os
import time

import torch
from loguru import logger

import ttnn
from models.experimental.diffusion_gemma.checkpoint import build_tt_model_from_checkpoint_dir
from models.experimental.diffusion_gemma.config import DiffusionConfig
from models.experimental.diffusion_gemma.tt.generate import fixed_prefill_chunks_enabled, prefill_prompt_tokens
from models.experimental.diffusion_gemma.tt.hybrid_kv import (
    attach_model_owned_hybrid_kv,
    model_owned_hybrid_kv_model_kwargs,
)
from models.experimental.diffusion_gemma.tt.serving import BlockDiffusionServingSession
from models.experimental.diffusion_gemma.tt.traced_denoise import (
    UPFRONT_DENOISE_STEPS,
    set_active_reveal_pmax,
    set_default_reveal_pmax,
    upfront_capture_enabled,
    upfront_traced_denoise_block,
)
from models.tt_transformers.tt.generator_vllm import HybridAttentionForCausalLM


# Served default Gumbel source: the on-device seeded SFPU RNG (see the __init__ note).
# Requires the lane-salted Blackhole rand kernel (tt-metal#52024); without it this default
# corrupts generated text.
DEFAULT_VLLM_GUMBEL_MODE = "device"
PREFILL_BUCKETS = tuple(1 << exponent for exponent in range(7, 19))  # 128 ... 262144


def _resolve_checkpoint_dir(hf_config):
    """Locate the DiffusionGemma checkpoint from the vLLM hf_config / env."""
    for attr in ("_name_or_path", "name_or_path"):
        path = getattr(hf_config, attr, None)
        if path:
            return path
    env_path = os.environ.get("DG_CKPT")
    if env_path:
        return env_path
    raise ValueError("DiffusionGemma checkpoint path not found on hf_config (_name_or_path) or DG_CKPT env var")


def _round_down_to_tile(value: int) -> int:
    """Round ``value`` down to a ``ttnn.TILE_SIZE`` multiple."""
    return (value // ttnn.TILE_SIZE) * ttnn.TILE_SIZE


def _validate_upfront_capture_configuration(
    *,
    canvas_length: int,
    max_denoise_steps: int,
    gumbel_mode: str,
    max_model_len: int | None = None,
) -> int:
    """Validate the fail-loud startup contract and return the fixed prefix span.

    ``DG_DENOISE_REVEAL_PMAX`` stays an accepted explicit override, but when it is
    unset the span is derived from ``max_model_len`` (vLLM's ``--max-model-len``),
    which is exactly what the served bound already is. The remaining contract items
    stay fail-loud: ``DG_TRACE_REGION_SIZE`` mirrors a reservation this process
    cannot query back from the device, and the prefill warmup shapes cannot be
    guessed from anything the wrapper knows.
    """
    if max_denoise_steps != UPFRONT_DENOISE_STEPS:
        raise RuntimeError(
            f"DG_UPFRONT_CAPTURE requires max_denoise_steps={UPFRONT_DENOISE_STEPS}, " f"got {max_denoise_steps}"
        )
    if gumbel_mode != "device":
        raise RuntimeError(
            "DG_UPFRONT_CAPTURE requires DG_VLLM_GUMBEL_MODE='device'; "
            f"got {gumbel_mode!r}. 'device' is the only materialized source: the W4-validated "
            "on-device seeded SFPU RNG (no per-step host RNG or PCIe DMA). "
            "'argmax' is not a materialized full-tensor source and is unsupported "
            "by the up-front controller."
        )

    # This process cannot read the reserved trace region back from the device (Metal takes
    # it as an open-time constructor argument and exposes no getter), so the operator must
    # mirror the reservation here. Defaulting it would silence the guard without reserving
    # anything, and a trace-region overflow poisons the device (needs `tt-smi -r`).
    raw_trace_region = os.environ.get("DG_TRACE_REGION_SIZE", "").strip()
    _trace_region_remedy = (
        " Reserve it with the vLLM --additional-config tt.trace_region_size and mirror the "
        "same value in DG_TRACE_REGION_SIZE, or set DG_UPFRONT_CAPTURE=0 to run the eager loop."
    )
    if not raw_trace_region:
        raise RuntimeError(
            "DG_UPFRONT_CAPTURE requires an explicit integer DG_TRACE_REGION_SIZE > 0." + _trace_region_remedy
        )
    try:
        trace_region_size = int(raw_trace_region)
    except ValueError as exc:
        raise RuntimeError(
            "DG_UPFRONT_CAPTURE requires an integer DG_TRACE_REGION_SIZE > 0." + _trace_region_remedy
        ) from exc
    if trace_region_size <= 0:
        raise RuntimeError("DG_UPFRONT_CAPTURE requires DG_TRACE_REGION_SIZE > 0." + _trace_region_remedy)

    raw_pmax = os.environ.get("DG_DENOISE_REVEAL_PMAX", "").strip()
    if not raw_pmax:
        if max_model_len is None:
            raise RuntimeError(
                "DG_UPFRONT_CAPTURE requires an explicit bounded DG_DENOISE_REVEAL_PMAX "
                "when no max_model_len is available to derive it from; "
                "the full allocated KV span is not an acceptable fallback"
            )
        # Round DOWN. The model-owned KV cache is allocated with seq dim == max_model_len
        # verbatim and ttnn keeps that logical (unpadded) shape, so rounding UP would make
        # p_max exceed the allocated span for every non-tile-multiple served bound and abort
        # startup. The rounded-off tokens were never addressable anyway: the reachable span
        # is capped by the cache, not by p_max.
        p_max = _round_down_to_tile(int(max_model_len))
        logger.info(
            f"[DiffusionGemma vLLM] DG_DENOISE_REVEAL_PMAX unset; derived fixed reveal span "
            f"p_max={p_max} from max_model_len={max_model_len}"
        )
    else:
        try:
            p_max = int(raw_pmax)
        except ValueError as exc:
            raise RuntimeError("DG_DENOISE_REVEAL_PMAX must be an integer") from exc
    if p_max <= 0 or p_max % ttnn.TILE_SIZE != 0:
        raise RuntimeError(f"DG_DENOISE_REVEAL_PMAX must be a positive {ttnn.TILE_SIZE}-token multiple, got {p_max}")
    minimum = ttnn.TILE_SIZE + int(canvas_length)
    if p_max < minimum:
        raise RuntimeError(
            "DG_DENOISE_REVEAL_PMAX cannot fit the startup prompt and one canvas: "
            f"{p_max} < {ttnn.TILE_SIZE} + {canvas_length} = {minimum}"
        )
    return p_max


def _metric(event: str, **fields) -> None:
    """Emit a stable JSON marker for live OpenAI-server evidence."""
    logger.info("DG_VLLM_METRIC " + json.dumps({"event": event, **fields}, sort_keys=True, default=str))


def _strict_prefill_lens() -> bool:
    """Whether an unwarmed prefill length should kill the engine instead of the request.

    Default OFF. A bit-exactness gate legitimately wants the run to stop, because an unwarmed shape
    means the comparison is no longer the thing it claims to measure -- but a serving deployment
    wants the server to survive one bad request. Since the raise is engine-fatal, that has to be a
    choice rather than the default.
    """
    return os.environ.get("DG_UPFRONT_STRICT_PREFILL_LENS", "0").strip().lower() in ("1", "true", "yes", "on")


def _lazy_prefill_recapture_enabled() -> bool:
    """Whether a cold prefill shape may rebuild the resident denoise trace.

    The rebuild is deliberately release-before-compile: no TT program-cache miss
    is allowed while a Metal trace is resident. Set
    ``DG_UPFRONT_LAZY_PREFILL_RECAPTURE=0`` to restore the legacy per-request
    rejection path.
    """
    return os.environ.get("DG_UPFRONT_LAZY_PREFILL_RECAPTURE", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _coarse_prefill_buckets_enabled() -> bool:
    """Whether prefill compute shapes use 128..256K power-of-two buckets."""
    return os.environ.get("DG_UPFRONT_COARSE_PREFILL_BUCKETS", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _reveal_buckets_enabled() -> bool:
    """Whether the up-front denoise trace binds a per-request reveal-span bucket.

    Default ON. With DG_REVEAL_OUTPUT_BUDGET unset, admission provisions the
    full ceiling, so captures bind exactly the span the fixed-span deployment
    binds and mid-request growth can never fire. With DG_REVEAL_OUTPUT_BUDGET
    set, admission provisions prompt + canvas + budget, every recapture runs
    between requests, and mid-request growth remains only a hardened fallback
    for requests exceeding the budget (a failure costs one request, not the
    engine). DG_DENOISE_REVEAL_BUCKETS=0 restores the single deployment-wide
    span outright.
    """
    return os.environ.get("DG_DENOISE_REVEAL_BUCKETS", "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _reveal_bucket_floor() -> int:
    """Smallest reveal bucket (default 4096): below this the span no longer dominates."""
    raw = os.environ.get("DG_REVEAL_BUCKET_FLOOR", "").strip()
    floor = int(raw) if raw else 4096
    if floor <= 0 or floor % ttnn.TILE_SIZE:
        raise RuntimeError(f"DG_REVEAL_BUCKET_FLOOR must be a positive {ttnn.TILE_SIZE}-token multiple, got {floor}")
    return floor


def _reveal_bucket_downshift_enabled() -> bool:
    """Whether admission may recapture DOWN to a smaller bucket (default on).

    Downshift is what returns the speed after a long request parks the resident
    bucket high; the 4x hysteresis in ``_resolve_reveal_bucket_change`` keeps
    alternating short/long traffic from recapturing on every request.
    """
    return os.environ.get("DG_REVEAL_BUCKET_DOWNSHIFT", "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _reveal_output_budget() -> int | None:
    """Output tokens provisioned into the admission bucket (DG_REVEAL_OUTPUT_BUDGET).

    Unset (default): admission provisions the full ceiling — no mid-request
    recapture can ever fire, at the cost of capturing the worst-case span. An
    explicit value provisions ``prompt + canvas + budget`` — the deployment
    states how much output its workload emits (an eval pins max_gen_toks; a
    serving profile knows its cap) and every recapture happens at ADMISSION,
    between requests, where memory is quiet. Mid-request upshift remains only
    the fallback for requests that exceed the budget.
    """
    raw = os.environ.get("DG_REVEAL_OUTPUT_BUDGET", "").strip()
    if not raw:
        return None
    budget = int(raw)
    if budget < 0:
        raise RuntimeError(f"DG_REVEAL_OUTPUT_BUDGET must be >= 0, got {budget}")
    return budget


def _reveal_provisioned_span(prefix_len: int, *, canvas_length: int, ceiling: int) -> int:
    """Span the admission bucket must cover for a request starting at ``prefix_len``."""
    budget = _reveal_output_budget()
    if budget is None:
        return int(ceiling)
    return min(int(prefix_len) + int(canvas_length) + budget, int(ceiling))


def _resolve_reveal_bucket(needed_span: int, *, ceiling: int) -> int:
    """Smallest power-of-two bucket >= max(needed, floor), clipped to the ceiling.

    The ceiling (DG_DENOISE_REVEAL_PMAX / derived max_model_len span) is returned
    for spans between the last power of two below it and the ceiling itself, so a
    non-power-of-two ceiling like 261888 still yields a servable bucket. A needed
    span beyond the ceiling raises — same contract as the fixed-span validation.
    """
    needed_span = int(needed_span)
    ceiling = int(ceiling)
    if needed_span > ceiling:
        raise ValueError(f"reveal span {needed_span} exceeds the deployment ceiling {ceiling}")
    bucket = _reveal_bucket_floor()
    while bucket < needed_span:
        bucket *= 2
    return min(bucket, ceiling)


def _resolve_reveal_bucket_change(desired: int, resident: int) -> bool:
    """Whether a capture at ``resident`` should be rebuilt to serve ``desired``.

    Upshift is mandatory (the mask cannot reveal beyond the captured span);
    downshift is worth one recapture only when at least 4x of the span comes
    back, and only when enabled.
    """
    if desired > resident:
        return True
    return _reveal_bucket_downshift_enabled() and desired * 4 <= resident


def _aligned_prefill_len(prompt_len: int) -> int:
    return ((int(prompt_len) + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE


def _resolve_prefill_execution_len(prompt_len: int, *, max_model_len: int | None) -> int:
    """Map a logical prompt to its exact or coarse prefill compute shape."""
    aligned = _aligned_prefill_len(prompt_len)
    capacity = PREFILL_BUCKETS[-1] if max_model_len is None else int(max_model_len)
    if aligned > capacity:
        raise ValueError(f"aligned prefill length {aligned} exceeds max_model_len={capacity}")
    if not _coarse_prefill_buckets_enabled():
        return aligned
    for bucket in PREFILL_BUCKETS:
        if aligned <= bucket <= capacity:
            return bucket
    raise ValueError(
        f"no power-of-two prefill bucket can cover aligned length {aligned} within max_model_len={capacity}"
    )


def _prefill_execution_len_is_warmed(execution_len: int, warmed) -> bool:
    """Whether the runtime shape has an already-compiled prefill program.

    With ``DG_PREFILL_FIXED_CHUNKS=1`` every prompt executes as fixed
    ``DG_PREFILL_CHUNK_SIZE`` chunks, so any startup warmup compiles the one
    program used by every request length. On the legacy mixed path, only prompts
    above 32K share that chunked program; short prompts require their bucket.
    """
    execution_len = int(execution_len)
    warmed = frozenset(int(value) for value in warmed)
    if fixed_prefill_chunks_enabled():
        # Every prompt is padded to, and iterated as, the same fixed-size
        # DG_PREFILL_CHUNK_SIZE model call. Once startup has compiled one such
        # chunk, a different total prompt length cannot introduce a model
        # program-cache miss and therefore needs no trace rebuild.
        return bool(warmed)
    return execution_len in warmed or (execution_len > 32768 and any(warmed_len > 32768 for warmed_len in warmed))


def _committed_ids(tokens) -> list:
    """Flat python ids for one committed block, for the DG_VLLM_METRIC block_ids audit line.

    Token IDS rather than text on purpose: the generator does not own a detokenizer, and a text
    field would couple this log line to whichever tokenizer the server happens to hold. The scorer
    detokenizes from the checkpoint instead, so the log stays valid if the serving stack changes.
    """
    try:
        return [int(v) for v in tokens.reshape(-1).tolist()]
    except Exception:  # never let an audit line break a served request
        return []


def _dram_snapshot(mesh_device, *, synchronize: bool = True) -> dict:
    if synchronize:
        ttnn.synchronize_device(mesh_device)
    view = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
    gib = 2**30
    return {
        "used_gib": round(view.num_banks * view.total_bytes_allocated_per_bank / gib, 6),
        "free_gib": round(view.num_banks * view.total_bytes_free_per_bank / gib, 6),
        "total_gib": round(view.num_banks * view.total_bytes_per_bank / gib, 6),
    }


class DiffusionGemmaForCausalLM(HybridAttentionForCausalLM):
    """Block-diffusion TT bridge for the tenstorrent/vllm TT plugin.

    Registered as ``TTDiffusionGemmaForBlockDiffusion`` (HF arch
    ``DiffusionGemmaForBlockDiffusion`` → plugin ``TT`` prefix). Inherits the
    hybrid KV-cache scaffolding and per-layer page-table plumbing from
    :class:`HybridAttentionForCausalLM`; overrides the forward path to run the
    diffusion block engine instead of the autoregressive one.
    """

    # Serving-feature reality on the TT path:
    #  * prefix caching: the vLLM APC contract needs paged-cache ownership + a
    #    block pool (#47488), which is NOT wired here; the TT plugin also
    #    force-disables it for sliding-window models, and block-diffusion
    #    recomputes canvas K/V every step → declare False.
    #  * async decode: the per-BLOCK async contract is unproven without the #47488
    #    runner; never advertise async without proof → declare False (safe default).
    #  * on-device sampling: the canvas Gumbel-max / entropy-budget / renoise path
    #    runs on device (no host argmax, no full-logits readback) → True.
    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_async_decode": False,
        "supports_sample_on_device": True,
        "output_tokens_per_step": 256,
    }

    @classmethod
    def get_max_tokens_all_users(cls, *, max_model_len=None, max_num_seqs=1, **kwargs) -> int:
        """Advertise the capacity of DG's model-owned hybrid cache to vLLM.

        The generic TT fallback is 131K tokens, which leaves a 256K DG model
        permanently WAITING in the scheduler even though its physical full-layer
        cache was constructed for ``max_model_len``. This path owns one identity-
        mapped cache and supports one active sequence, so scheduler bookkeeping
        must cover that full served length.
        """
        if max_model_len is not None:
            if int(max_num_seqs) != 1:
                raise ValueError("DG model-owned hybrid KV supports max_num_seqs=1; " f"got {max_num_seqs}")
            return int(max_model_len)
        return super().get_max_tokens_all_users(
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            **kwargs,
        )

    def __init__(
        self,
        *args,
        dg_state_dict=None,
        tokenizer=None,
        config=None,
        gumbel_mode=DEFAULT_VLLM_GUMBEL_MODE,
        max_model_len=None,
        page_tables_per_layer=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._dg_state_dict = dg_state_dict
        self._tokenizer = tokenizer
        self._config = DiffusionConfig() if config is None else config
        self.canvas_length = self._config.canvas_length
        # The served bound, used to derive the fixed reveal span when the operator does not
        # pin DG_DENOISE_REVEAL_PMAX explicitly.
        self._max_model_len = None if max_model_len is None else int(max_model_len)
        self._model_owned_page_tables_per_layer = page_tables_per_layer
        # DEFAULT "device" is the on-device seeded SFPU Gumbel source: no per-step host
        # RNG and no per-step replicated PCIe DMA. It DEPENDS on the lane-salted Blackhole
        # rand kernel (tt-metal#52024); without that fix this default corrupts generated
        # text.
        #
        # MEMORY ENVELOPE: "device" materializes a full-vocabulary (262144) tensor per
        # step. If that envelope stops fitting at 256K, pursue the TP-sharded denoise
        # terminal (unimplemented design note at the bottom of tt/sampling.py). "argmax"
        # is the fast deterministic RUN control (requires DG_UPFRONT_CAPTURE=0; the
        # captured path raises on any other mode than "device"). The torch-noise
        # INJECTION helpers in tt/generate.py replay a torch run's exact noise for
        # HF<->TT determinism; they are not a serving mode.
        self._gumbel_mode = os.environ.get("DG_VLLM_GUMBEL_MODE", gumbel_mode)
        # One active session per batch row. A single model-owned hybrid cache backs
        # one active sequence today (see module docstring); the dict is keyed by
        # row so output formatting never assumes batch size 1.
        self._sessions: dict[int, BlockDiffusionServingSession] = {}
        # Model-level denoise has exactly two paths. DG_UPFRONT_CAPTURE (default ON) owns
        # one startup-captured adapter/controller for the model lifetime; DG_UPFRONT_CAPTURE=0
        # opts out and each request uses the ordinary eager denoise loop. The TT vLLM
        # ``trace_mode=all`` setting remains only the runner's compile/capture warmup signal.
        self._upfront = upfront_capture_enabled()
        self._persistent_adapter = None
        self._upfront_compile_phase_seen = False
        self._upfront_prefill_warmup_lens = frozenset()
        self._upfront_rebuild_in_progress = False
        self._upfront_rebuilds = 0
        self._upfront_pmax = (
            _validate_upfront_capture_configuration(
                canvas_length=self.canvas_length,
                max_denoise_steps=self._config.max_denoise_steps,
                gumbel_mode=self._gumbel_mode,
                max_model_len=self._max_model_len,
            )
            if self._upfront
            else None
        )
        # The controller is built inside the denoise-block entry point, which cannot take the
        # span as an argument without changing the denoise_block_fn protocol; register the
        # resolved value so it does not have to re-read (and re-require) the env var.
        set_default_reveal_pmax(self._upfront_pmax)
        # Per-request reveal-span buckets: _upfront_pmax stays the CEILING; captures bind
        # the active bucket registered immediately before each (re)capture. The startup BOS
        # capture only reveals one tile plus a canvas, so it binds the floor bucket — which
        # also makes startup capture itself cheaper than a worst-case-span capture.
        self._upfront_reveal_bucket = None
        if self._upfront and _reveal_buckets_enabled():
            self._upfront_reveal_bucket = _resolve_reveal_bucket(
                _reveal_provisioned_span(ttnn.TILE_SIZE, canvas_length=self.canvas_length, ceiling=self._upfront_pmax),
                ceiling=self._upfront_pmax,
            )
        # Register unconditionally: the active span is module-global in traced_denoise,
        # so a wrapper built after a bucketed one (tests, restarts in-process) must not
        # inherit a stale bucket — None restores env/default resolution.
        set_active_reveal_pmax(self._upfront_reveal_bucket)

    # ── construction ────────────────────────────────────────────────────
    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        max_seq_len=262144,
        n_layers=None,
        tt_data_parallel=1,
        optimizations: str = None,
    ):
        if optimizations not in (None, "performance"):
            raise ValueError("DiffusionGemma TT serving uses the full-model bf16 policy; no custom profiles")
        if tt_data_parallel != 1:
            # The 26B-A4B backbone is tensor-parallel (TP=4) on the (1,4) QB2 mesh;
            # attention data-parallel replicas are not part of the block-diffusion
            # serving path today.
            raise ValueError("DiffusionGemma TT serving is TP=4 single-replica (tt_data_parallel must be 1)")

        checkpoint_dir = _resolve_checkpoint_dir(hf_config)
        diffusion_config = DiffusionConfig()
        model_kwargs = dict(
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            dtype=ttnn.bfloat16,  # full-model policy: bf16 weights + bf16 KV cache
            create_kv_cache=True,  # overridden below with bounded paged KV by default
        )
        model_kwargs.update(
            model_owned_hybrid_kv_model_kwargs(
                max_seq_len=max_seq_len,
                max_batch_size=max_batch_size,
            )
        )
        if n_layers is not None:
            model_kwargs["num_layers"] = n_layers

        build_t0 = time.perf_counter()
        bundle = build_tt_model_from_checkpoint_dir(mesh_device, checkpoint_dir, **model_kwargs)
        page_tables_per_layer = attach_model_owned_hybrid_kv(
            bundle.tt_model,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
        )
        ttnn.synchronize_device(mesh_device)
        model_build_s = time.perf_counter() - build_t0
        dram = _dram_snapshot(mesh_device, synchronize=False)
        logger.info(
            f"[DiffusionGemma vLLM] built model: max_seq_len={max_seq_len} "
            f"n_layers={n_layers or 'full'} "
            f"model_owned_hybrid_kv=True "
            f"gumbel_mode={os.environ.get('DG_VLLM_GUMBEL_MODE', DEFAULT_VLLM_GUMBEL_MODE)}"
        )
        _metric(
            "model_build",
            max_seq_len=max_seq_len,
            num_layers=n_layers or 30,
            model_build_s=round(model_build_s, 6),
            gumbel_mode=os.environ.get("DG_VLLM_GUMBEL_MODE", DEFAULT_VLLM_GUMBEL_MODE),
            max_denoise_steps=diffusion_config.max_denoise_steps,
            model_owned_hybrid_kv=True,
            trace_region_size_env=int(os.environ.get("DG_TRACE_REGION_SIZE", "0")),
            selfcond_prechunk_embed=os.environ.get("DG_SELFCOND_PRECHUNK_EMBED", "1"),
            selfcond_logits_l1=os.environ.get("DG_SELFCOND_LOGITS_L1", "chain"),
            dram=dram,
        )
        return cls(
            [bundle.tt_model],
            [bundle.model_args],
            mesh_device,
            dg_state_dict=bundle.state_dict,
            tokenizer=bundle.tokenizer,
            config=diffusion_config,
            max_model_len=max_seq_len,
            page_tables_per_layer=page_tables_per_layer,
        )

    @property
    def cache_path(self):
        return self.model_args[0].weight_cache_path(ttnn.bfloat16)

    # ── vLLM VllmModelForTextGeneration protocol shims ──────────────────
    # vLLM's is_text_generation_model predicate inspects the resolved class for
    # embed_input_ids / forward / compute_logits. DiffusionGemma has no upstream
    # vLLM impl, so inspection lands here. Execution goes through prefill_forward /
    # decode_forward; these are never invoked.
    def embed_input_ids(self, input_ids):  # pragma: no cover - protocol shim
        raise NotImplementedError("DiffusionGemma is a TT bridge; embeddings happen on TT in decode_forward.")

    def forward(self, input_ids, positions, **kwargs):  # pragma: no cover - protocol shim
        raise NotImplementedError("DiffusionGemma is a TT bridge; the TT runner calls prefill_forward/decode_forward.")

    def compute_logits(self, hidden_states, **kwargs):  # pragma: no cover - protocol shim
        raise NotImplementedError("DiffusionGemma keeps logits on device; canvas sampling runs in decode_forward.")

    # ── KV cache ────────────────────────────────────────────────────────
    @classmethod
    def get_kv_cache_spec(cls, vllm_config):
        """Per-layer KV spec (copied from the gemma4 bridge geometry).

        DiffusionGemma's text backbone == Gemma-4 26B-A4B: sliding layers use
        ``head_dim`` (256) / ``num_key_value_heads``; full-attention layers use
        ``global_head_dim`` (512) / ``num_global_key_value_heads``. Every layer emits
        a ``FullAttentionSpec`` (uniform type) so vLLM merges them into ONE KV-cache
        group backed by the whole block pool — hybrid groups are disabled
        (``_HYBRID_KV_CACHE_GROUPS_ENABLED = False``) and the diffusion forward uses
        the non-hybrid single-page-table bookkeeping path, so a per-type spec would instead split
        into 6 groups sharing the pool and cap prefill admission at ~21824 tokens (see
        the sliding branch). The diffusion forward reads the model-owned hybrid
        cache, so this spec is the manager's bookkeeping, not the physical cache
        (#47488).
        """
        from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        parallel_config = vllm_config.parallel_config

        hf_config = model_config.hf_config
        text_config = getattr(hf_config, "text_config", hf_config)
        layer_types = getattr(text_config, "layer_types", None)
        if layer_types is None:
            raise ValueError(f"{cls.__name__}.get_kv_cache_spec requires text_config.layer_types")

        sliding_kv_heads = text_config.num_key_value_heads
        sliding_head_dim = text_config.head_dim
        sliding_window = getattr(text_config, "sliding_window", None)
        full_kv_heads = getattr(text_config, "num_global_key_value_heads", None) or sliding_kv_heads
        full_head_dim = getattr(text_config, "global_head_dim", None) or sliding_head_dim

        tp = parallel_config.tensor_parallel_size
        # Match Gemma4 split_qkv/cache allocation: when KV heads < TP, each
        # device receives one replicated/assigned KV head rather than zero.
        sliding_kv_heads_per_dev = 1 if sliding_kv_heads < tp else sliding_kv_heads // tp
        full_kv_heads_per_dev = 1 if full_kv_heads < tp else full_kv_heads // tp

        dtype = (
            model_config.dtype
            if cache_config.cache_dtype == "auto"
            else STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]
        )
        block_size = cache_config.block_size

        spec_per_layer = {}
        for i, lt in enumerate(layer_types):
            name = f"model.layers.{i}.self_attn"
            if lt == "sliding_attention":
                if sliding_window is None:
                    raise ValueError(f"layer_types[{i}] is sliding but sliding_window is None")
                # Hybrid kv-cache groups are disabled (``_HYBRID_KV_CACHE_GROUPS_ENABLED = False``,
                # inherited; the diffusion forward uses the non-hybrid single-page-table path): emit
                # ``FullAttentionSpec`` for sliding layers too, keeping their own sliding
                # num_kv_heads/head_size. vLLM then merges all same-type specs into ONE
                # ``UniformTypeKVCacheSpecs`` group so the full block pool backs every request,
                # instead of splitting into 6 groups (1 full + 5 sliding) that share the pool and
                # cap single-shot prefill admission at (num_gpu_blocks // 6) * block_size ~= 21824
                # tokens — the cause of the >21824-token (e.g. 32768) prefill WAITING-forever stall
                # (allocate_slots needs 6 * cdiv(L/64) blocks and returns None). The spec is vLLM
                # bookkeeping only; the model owns the physical contiguous cache (#47488). Mirrors
                # models/demos/gemma4/tt/generator_vllm.py.
                spec_per_layer[name] = FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=sliding_kv_heads_per_dev,
                    head_size=sliding_head_dim,
                    dtype=dtype,
                )
            elif lt == "full_attention":
                spec_per_layer[name] = FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=full_kv_heads_per_dev,
                    head_size=full_head_dim,
                    dtype=dtype,
                )
            else:
                raise ValueError(f"Unsupported layer_type {lt!r} at layer {i}")
        return spec_per_layer

    def _model_owned_kv_handles(self):
        """``[submesh][layer][k_or_v]`` handles into the model's own hybrid cache.

        Serving runs on the model-owned cache the model allocated at
        build time (`create_kv_cache=True`); both allocator entry points return
        those existing handles so vLLM's `kv_cache` arg points at the physical
        cache the diffusion forward actually reads/writes — no fresh DRAM, no
        double allocation (see the module docstring on cache ownership / #47488).
        """
        return [[[k, v] for (k, v) in model.tt_kv_cache] for model in self.model]

    def allocate_kv_cache(self, *args, **kwargs):
        # Legacy uniform entry point; the hybrid model uses allocate_kv_cache_per_layer.
        # Both return the model's own handles (no new DRAM) for the model-owned-cache
        # serving mode.
        del args, kwargs
        return self._model_owned_kv_handles()

    def allocate_kv_cache_per_layer(self, per_layer_specs):
        del per_layer_specs  # sizing bookkeeping only; physical cache is model-owned
        return self._model_owned_kv_handles()

    # ── warmup ──────────────────────────────────────────────────────────
    def warmup_model_prefill(self, kv_cache, enable_trace, can_sample_on_device, greedy_only: bool = False):
        del kv_cache, can_sample_on_device, greedy_only
        if not self._upfront:
            logger.info("[DiffusionGemma vLLM] warmup is a no-op for eager block diffusion")
            return
        if not enable_trace:
            # The TT vLLM runner performs a compile-only phase before its trace-capture phase.
            # The denoise controller already warms its exact programs immediately before capture;
            # defer here so later runner warmups cannot allocate buffers after an active trace.
            self._upfront_compile_phase_seen = True
            raw_warmup_lens = os.environ.get("DG_UPFRONT_PREFILL_WARMUP_LENS", "").strip()
            # Keep the exact one-tile shape for the BOS capture session. Coarse
            # buckets apply to real requests, whose minimum execution shape is 128.
            warmup_lens = {ttnn.TILE_SIZE}
            if raw_warmup_lens:
                for value in raw_warmup_lens.split(","):
                    prompt_len = int(value.strip())
                    if prompt_len <= 0 or prompt_len % ttnn.TILE_SIZE != 0:
                        raise RuntimeError(
                            "DG_UPFRONT_PREFILL_WARMUP_LENS values must be positive "
                            f"{ttnn.TILE_SIZE}-token multiples, got {prompt_len}"
                        )
                    if prompt_len > self._upfront_pmax:
                        raise RuntimeError(f"prefill warmup length {prompt_len} exceeds p_max={self._upfront_pmax}")
                    warmup_lens.add(
                        _resolve_prefill_execution_len(
                            prompt_len,
                            max_model_len=getattr(self, "_max_model_len", None),
                        )
                    )
            # One tile is always warmed, whether or not the caller listed it. It is the
            # only startup shape needed for the BOS capture session; every other cold
            # shape can now use the release-before-compile rebuild below.
            self._upfront_prefill_warmup_lens = frozenset(warmup_lens)
            for prompt_len in sorted(self._upfront_prefill_warmup_lens):
                logger.info(f"[DiffusionGemma vLLM] warming prefill shape {prompt_len} before trace capture")
                mock_tokens = torch.zeros((1, prompt_len), dtype=torch.long)
                page_tables = getattr(self, "_model_owned_page_tables_per_layer", None)
                page_table_kwargs = {"page_tables_per_layer": page_tables} if page_tables is not None else {}
                prefill_prompt_tokens(self.model[0], mock_tokens, **page_table_kwargs)
            ttnn.synchronize_device(self.model[0].mesh_device)
            logger.info("[DiffusionGemma vLLM] deferring up-front denoise capture to trace warmup phase")
            return
        if not self._upfront_compile_phase_seen:
            raise RuntimeError(
                "DG_UPFRONT_CAPTURE requires the startup compile warmup phase before capture; "
                "enable vLLM model warmup with TT trace_mode=all, "
                "or set DG_UPFRONT_CAPTURE=0 to run the eager loop"
            )
        if not getattr(self, "_upfront_prefill_warmup_lens", ()):
            raise RuntimeError(
                "DG_UPFRONT_CAPTURE requires its compile-only prefill warmup before trace capture; "
                "executing a program-cache miss while a trace is resident can corrupt active traces. "
                "Keep the default one-tile startup warmup, "
                "or set DG_UPFRONT_CAPTURE=0 to run the eager loop"
            )
        if self._persistent_adapter is not None:
            logger.info("[DiffusionGemma vLLM] up-front denoise capture already initialized")
            return

        p_max = _validate_upfront_capture_configuration(
            canvas_length=self.canvas_length,
            max_denoise_steps=self._config.max_denoise_steps,
            gumbel_mode=self._gumbel_mode,
            max_model_len=self._max_model_len,
        )
        cache_span = (
            # Direct-construction callers (tests, benches) have no vLLM max_model_len;
            # the hybrid attachment records the allocated span on the model itself.
            int(self._max_model_len or self.model[0]._dg_hybrid_max_seq_len)
            if bool(getattr(self.model[0], "_dg_model_owned_hybrid_kv", False))
            else min(int(k_cache.shape[-2]) for k_cache, _v_cache in self.model[0].tt_kv_cache)
        )
        if p_max > cache_span:
            raise RuntimeError(
                f"DG_DENOISE_REVEAL_PMAX={p_max} exceeds the smallest allocated model KV span {cache_span}"
            )
        if p_max == cache_span:
            # A span equal to the whole cache is legal but is the most expensive one: the
            # per-step prefix read takes the full-span branch (a whole-cache clone per layer
            # per step) and the persistent reveal mask is sized for the full span. Pinning
            # DG_DENOISE_REVEAL_PMAX to the context actually served is the cheap path.
            logger.warning(
                f"[DiffusionGemma vLLM] fixed reveal span p_max={p_max} equals the whole allocated "
                f"KV span: every denoise step reads the full prefix and the reveal mask is sized "
                f"for it. Set DG_DENOISE_REVEAL_PMAX to the context you actually serve to cut "
                f"per-step prefix cost."
            )
        self._upfront_pmax = p_max

        mock_token_id = getattr(self._tokenizer, "bos_token_id", None)
        if mock_token_id is None:
            mock_token_id = getattr(self._tokenizer, "eos_token_id", None)
        if mock_token_id is None:
            mock_token_id = 0
        mock_tokens = torch.tensor([[int(mock_token_id)]], dtype=torch.long)

        session = self._make_session()
        adapter = None
        try:
            cache_len = session.prefill(mock_tokens)
            emission, adapter, trace_stats = self._capture_prefilled_session(session)
            # Detach before resetting the throwaway shell: the wrapper now owns the adapter.
            session._logits_fn = None
            session.reset()
            self._persistent_adapter = adapter
        except BaseException:
            if adapter is None:
                adapter = session._logits_fn
            controller_attr = "_upfront_traced_denoise_controller"
            controller = getattr(adapter, controller_attr, None) if adapter is not None else None
            if controller is not None:
                try:
                    controller.release()
                except BaseException as cleanup_error:
                    logger.error(f"failed to release aborted up-front controller: {cleanup_error}")
                finally:
                    delattr(adapter, controller_attr)
            session.reset()
            logger.error(
                "[DiffusionGemma vLLM] up-front denoise capture failed; startup is aborted. "
                "A trace-region overflow may require `tt-smi -r` before retrying."
            )
            raise

        _metric(
            "upfront_capture",
            cache_len=cache_len,
            committed_tokens=int(emission.tokens.numel()),
            next_pos=emission.next_pos,
            reveal_pmax=p_max,
            trace_stats=trace_stats,
            dram=_dram_snapshot(self.model[0].mesh_device),
        )
        logger.info(
            f"[DiffusionGemma vLLM] captured persistent denoise trace at startup "
            f"(mock_cache_len={cache_len}, p_max={p_max})"
        )

    def _capture_prefilled_session(self, session):
        """Capture denoise on an already-prefilled session without publishing it."""
        adapter = session._logits_fn
        if adapter is None:
            raise RuntimeError("cannot capture an up-front controller before session prefill")
        adapter._upfront_capture_phase = True
        try:
            emission = session.decode_block()
        finally:
            if hasattr(adapter, "_upfront_capture_phase"):
                delattr(adapter, "_upfront_capture_phase")
        controller = getattr(adapter, "_upfront_traced_denoise_controller", None)
        if controller is None or not getattr(controller, "captured", False):
            raise RuntimeError("denoise did not leave a fully captured up-front controller")
        if not getattr(adapter, "use_reveal_mask", False):
            raise RuntimeError("denoise trace was not captured with a persistent reveal mask")
        return emission, adapter, session.trace_stats()

    @staticmethod
    def _release_unpublished_adapter(adapter, *, label: str) -> None:
        """Best-effort cleanup for a capture that was never made model-resident."""
        if adapter is None:
            return
        attr = "_upfront_traced_denoise_controller"
        controller = getattr(adapter, attr, None)
        if controller is not None:
            try:
                controller.release()
            except BaseException as cleanup_error:
                logger.error(f"failed to release {label} controller: {cleanup_error}")
            finally:
                if hasattr(adapter, attr):
                    delattr(adapter, attr)
        if hasattr(adapter, "reset"):
            try:
                adapter.reset()
            except BaseException as cleanup_error:
                logger.error(f"failed to release {label} adapter: {cleanup_error}")

    def _reserve_cold_recapture_holes(self, *, span: int | None = None) -> list:
        """Protect the four large holes needed to restart long-context capture.

        Releasing the resident trace exposes the large DRAM blocks used by the
        full-attention prefix+canvas K/V concats. A cold prefill can split those
        blocks into smaller allocations before recapture, leaving enough total
        free memory but no contiguous block large enough. Capture first needs two
        full-vocabulary Gumbel buffers, then simultaneous prefix+canvas K and V
        concats. Hold four concat-sized buffers across the prefill, then release
        them immediately before recapture so those allocations can reuse the
        preserved holes. ``span`` sizes the holes to the span the capture will
        actually bind (a reveal bucket); default is the deployment ceiling.
        """
        p_max = int(span) if span is not None else int(getattr(self, "_upfront_pmax", 0) or 0)
        if p_max < 65536:
            return []
        tt_model = self.model[0]
        text_config = getattr(tt_model.hf_config, "text_config", tt_model.hf_config)
        head_dim = int(getattr(text_config, "global_head_dim", None) or getattr(text_config, "head_dim"))
        shape = [1, 1, p_max + int(self.canvas_length), head_dim]
        reservations = []
        try:
            for _ in range(4):
                reservations.append(
                    ttnn.empty(
                        shape,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=tt_model.mesh_device,
                    )
                )
        except BaseException:
            for reservation in reservations:
                reservation.deallocate(True)
            raise
        _metric(
            "cold_prefill_recapture_holes_reserved",
            buffers=len(reservations),
            shape=shape,
            bytes_per_buffer=2 * (p_max + int(self.canvas_length)) * head_dim,
        )
        return reservations

    @staticmethod
    def _release_cold_recapture_holes(reservations) -> None:
        for reservation in reservations:
            reservation.deallocate(True)

    def _rebuild_for_cold_prefill(
        self,
        session,
        prompt_tokens,
        *,
        expected_cache_len: int,
        execution_len: int | None = None,
        capture_span: int | None = None,
    ):
        """Release the active trace, compile one cold prefill shape, and recapture.

        Returning only after the new 48-trace set is complete makes publication
        atomic from the serving wrapper's point of view. The old trace cannot be
        retained while compiling: that ordering can hang the multi-device CCL.
        """
        if getattr(self, "_upfront_rebuild_in_progress", False):
            raise RuntimeError("concurrent up-front cold-shape rebuild is unsupported")
        if getattr(self, "_sessions", {}):
            raise RuntimeError("cold-shape rebuild requires no active serving sessions")

        self._upfront_rebuild_in_progress = True
        execution_len = expected_cache_len if execution_len is None else int(execution_len)
        _metric(
            "cold_prefill_rebuild_begin",
            cache_len=expected_cache_len,
            execution_len=execution_len,
            warmed=sorted(getattr(self, "_upfront_prefill_warmup_lens", ())),
            rebuild_index=int(getattr(self, "_upfront_rebuilds", 0)) + 1,
        )
        try:
            # HARD SAFETY ORDER: release every resident trace/buffer, drain CQ0,
            # and only then execute the prefill program that may miss the cache.
            self.release_persistent_capture()
            ttnn.synchronize_device(self.model[0].mesh_device)
            _metric(
                "cold_prefill_trace_released",
                cache_len=expected_cache_len,
                execution_len=execution_len,
            )

            # Concat holes are sized to the span this capture binds: a ceiling-sized
            # 4x ~268 MB reservation stops fitting once heterogeneous bucket rebuilds
            # have fragmented DRAM. The [canvas, vocab] Gumbel/noise buffers need no
            # holes — they live in the model-lifetime pool and are never reallocated.
            recapture_holes = self._reserve_cold_recapture_holes(span=capture_span)
            try:
                cache_len = session.prefill(prompt_tokens)
                if cache_len != expected_cache_len:
                    raise RuntimeError(
                        f"cold prefill aligned to {cache_len}, expected {expected_cache_len}; "
                        "refusing to publish a mismatched trace"
                    )
                ttnn.synchronize_device(self.model[0].mesh_device)
            finally:
                self._release_cold_recapture_holes(recapture_holes)
            _metric("cold_prefill_compiled", cache_len=cache_len, execution_len=execution_len)

            emission, adapter, trace_stats = self._capture_prefilled_session(session)
            # The live request and wrapper now co-own the model-lifetime adapter.
            # Marking it persistent keeps session.reset from releasing it.
            session._persistent_adapter = adapter
            self._persistent_adapter = adapter
            warmed = set(getattr(self, "_upfront_prefill_warmup_lens", ()))
            warmed.add(execution_len)
            self._upfront_prefill_warmup_lens = frozenset(warmed)
            self._upfront_rebuilds = int(getattr(self, "_upfront_rebuilds", 0)) + 1
            _metric(
                "cold_prefill_rebuild_complete",
                cache_len=cache_len,
                execution_len=execution_len,
                rebuilds=self._upfront_rebuilds,
                warmed=sorted(warmed),
                trace_stats=trace_stats,
                dram=_dram_snapshot(self.model[0].mesh_device),
            )
            return cache_len, emission
        except BaseException as rebuild_error:
            adapter = getattr(session, "_logits_fn", None)
            self._persistent_adapter = None
            self._release_unpublished_adapter(adapter, label="failed cold-prefill")
            # Prevent BlockDiffusionServingSession.reset from resetting the same
            # adapter twice; the outer prefill failure path still clears the shell.
            session._logits_fn = None
            session._persistent_adapter = None
            try:
                ttnn.synchronize_device(self.model[0].mesh_device)
            except BaseException as cleanup_error:
                logger.error(f"failed to synchronize after cold-prefill rebuild error: {cleanup_error}")
            _metric(
                "cold_prefill_rebuild_failed",
                cache_len=expected_cache_len,
                execution_len=execution_len,
                error=repr(rebuild_error),
            )
            raise
        finally:
            self._upfront_rebuild_in_progress = False

    def _rebuild_for_reveal_bucket(self, session, prompt_tokens, *, expected_cache_len: int):
        """Bucket-only rebuild: prefill FIRST, then release-and-recapture.

        When the prefill shape is already warm the prefill cannot miss the
        program cache, so it may run while the old traces are still resident —
        with the full free pool available for its transients. Only afterwards is
        the old capture released and the new span captured back-to-back, with no
        big allocations in between beyond what the release itself just freed
        (the [canvas, vocab] noise buffers live in the model-lifetime pool).
        """
        if getattr(self, "_upfront_rebuild_in_progress", False):
            raise RuntimeError("concurrent up-front rebuild is unsupported")
        if getattr(self, "_sessions", {}):
            raise RuntimeError("reveal-bucket rebuild requires no active serving sessions")
        self._upfront_rebuild_in_progress = True
        _metric(
            "reveal_bucket_rebuild_begin",
            cache_len=expected_cache_len,
            rebuild_index=int(getattr(self, "_upfront_rebuilds", 0)) + 1,
        )
        try:
            cache_len = session.prefill(prompt_tokens)
            if cache_len != expected_cache_len:
                raise RuntimeError(
                    f"bucket-rebuild prefill aligned to {cache_len}, expected {expected_cache_len}; "
                    "refusing to publish a mismatched trace"
                )
            ttnn.synchronize_device(self.model[0].mesh_device)
            self.release_persistent_capture()
            ttnn.synchronize_device(self.model[0].mesh_device)
            emission, adapter, trace_stats = self._capture_prefilled_session(session)
            session._persistent_adapter = adapter
            self._persistent_adapter = adapter
            self._upfront_rebuilds = int(getattr(self, "_upfront_rebuilds", 0)) + 1
            _metric(
                "reveal_bucket_rebuild_complete",
                cache_len=cache_len,
                rebuilds=self._upfront_rebuilds,
                trace_stats=trace_stats,
                dram=_dram_snapshot(self.model[0].mesh_device),
            )
            return cache_len, emission
        except BaseException as rebuild_error:
            adapter = getattr(session, "_logits_fn", None)
            self._persistent_adapter = None
            self._release_unpublished_adapter(adapter, label="failed reveal-bucket rebuild")
            session._logits_fn = None
            session._persistent_adapter = None
            try:
                ttnn.synchronize_device(self.model[0].mesh_device)
            except BaseException as cleanup_error:
                logger.error(f"failed to synchronize after reveal-bucket rebuild error: {cleanup_error}")
            _metric("reveal_bucket_rebuild_failed", cache_len=expected_cache_len, error=repr(rebuild_error))
            raise
        finally:
            self._upfront_rebuild_in_progress = False

    def _reveal_upshift_needed_span(self, session) -> int | None:
        """Span the next block needs when it no longer fits the resident bucket.

        ``adapter.prompt_len`` is the request's CURRENT revealed base (it advances
        with every committed block), so a long generation crossing its capture
        bucket is detected here, at a block boundary, before the replay's own
        rebind validation would reject it.
        """
        if not (getattr(self, "_upfront", False) and _reveal_buckets_enabled()):
            return None
        adapter = getattr(self, "_persistent_adapter", None)
        controller = getattr(adapter, "_upfront_traced_denoise_controller", None) if adapter is not None else None
        if controller is None or not getattr(controller, "captured", False):
            return None
        resident = int(getattr(controller, "reveal_pmax", 0) or 0)
        needed = int(getattr(adapter, "prompt_len", 0) or 0) + self.canvas_length
        return needed if resident and needed > resident else None

    def _rebuild_for_reveal_upshift(self, session, *, needed_span: int, row: int):
        """Release the resident traces and recapture at a bigger bucket mid-request.

        Same primitive as the cold-prefill rebuild — the capture decodes the live
        session's next block while tracing, so this call RETURNS that block's
        emission — minus the prefill compile (the session keeps its cache; only
        the denoise span grows). Buckets double, so a generation reaching span S
        pays O(log2(S/floor)) recaptures over its whole lifetime.
        """
        if getattr(self, "_upfront_rebuild_in_progress", False):
            raise RuntimeError("concurrent up-front reveal upshift is unsupported")
        if any(existing is not session for existing in self._sessions.values()):
            # Recapture invalidates every session's traces; with more than one
            # active sequence this needs the batched-capture ownership story.
            raise RuntimeError("reveal-span upshift requires the growing request to be the only active session")
        new_bucket = _resolve_reveal_bucket(needed_span, ceiling=int(self._upfront_pmax))
        resident = int(self._upfront_reveal_bucket or 0)
        self._upfront_rebuild_in_progress = True
        _metric(
            "reveal_upshift_begin",
            row=row,
            needed_span=needed_span,
            resident=resident,
            desired=new_bucket,
            rebuild_index=int(getattr(self, "_upfront_rebuilds", 0)) + 1,
        )
        try:
            # NOT release_persistent_capture(): that detaches every active request and
            # resets the adapter, and this request must survive the recapture. Release
            # only the controller (traces + its device buffers); the adapter keeps its
            # live request state and the capture rebuilds its mask/RoPE buffers at the
            # new span (prepare_reveal_mask_buffers releases-then-resizes).
            adapter = self._persistent_adapter
            attr = "_upfront_traced_denoise_controller"
            controller = getattr(adapter, attr, None)
            if controller is not None:
                controller.release()
                delattr(adapter, attr)
            # The old-span mask bookkeeping (adapter._reveal_p_max) feeds the serving
            # capacity check BEFORE the capture re-prepares the buffers; releasing the
            # mask here lets that check fall back to the model context limit and the
            # capture rebuild everything at the new span.
            release_masks = getattr(adapter, "release_reveal_mask_buffers", None)
            if callable(release_masks):
                release_masks()
            self._persistent_adapter = None  # republished after a successful capture
            # The [canvas, vocab] Gumbel/noise buffers live in the model-lifetime
            # pool (traced_denoise._vocab_noise_pool), so the recapture copies into
            # them instead of reallocating 134 MB into a fragmented heap.
            ttnn.synchronize_device(self.model[0].mesh_device)
            set_active_reveal_pmax(new_bucket)
            recapture_holes = self._reserve_cold_recapture_holes(span=new_bucket)
            try:
                emission, adapter, trace_stats = self._capture_prefilled_session(session)
            finally:
                self._release_cold_recapture_holes(recapture_holes)
            session._persistent_adapter = adapter
            self._persistent_adapter = adapter
            self._upfront_reveal_bucket = new_bucket
            self._upfront_rebuilds = int(getattr(self, "_upfront_rebuilds", 0)) + 1
            _metric(
                "reveal_upshift_complete",
                row=row,
                bucket=new_bucket,
                rebuilds=self._upfront_rebuilds,
                trace_stats=trace_stats,
                dram=_dram_snapshot(self.model[0].mesh_device),
            )
            return emission
        except BaseException as upshift_error:
            set_active_reveal_pmax(resident or None)
            adapter = getattr(session, "_logits_fn", None)
            self._persistent_adapter = None
            self._release_unpublished_adapter(adapter, label="failed reveal-upshift")
            session._logits_fn = None
            session._persistent_adapter = None
            try:
                ttnn.synchronize_device(self.model[0].mesh_device)
            except BaseException as cleanup_error:
                logger.error(f"failed to synchronize after reveal-upshift error: {cleanup_error}")
            _metric(
                "reveal_upshift_failed",
                row=row,
                needed_span=needed_span,
                desired=new_bucket,
                error=repr(upshift_error),
            )
            raise
        finally:
            self._upfront_rebuild_in_progress = False

    def _restore_resident_capture(self) -> None:
        """Re-establish the model-lifetime capture after a failed mid-request rebuild.

        Same BOS mock-prompt capture as startup warmup, at whatever span is
        currently registered (the failure path restored the resident bucket).
        The failed request remains registered as finished until vLLM consumes
        its synthetic EOS block, but its live adapter state is detached, so
        memory is as quiet as any between-requests recapture. A failure HERE
        leaves the wrapper unable to serve anything and is engine-fatal,
        matching the startup contract.
        """
        if self._persistent_adapter is not None:
            return
        mock_token_id = getattr(self._tokenizer, "bos_token_id", None)
        if mock_token_id is None:
            mock_token_id = getattr(self._tokenizer, "eos_token_id", None)
        if mock_token_id is None:
            mock_token_id = 0
        mock_tokens = torch.tensor([[int(mock_token_id)]], dtype=torch.long)
        session = self._make_session()
        adapter = None
        try:
            session.prefill(mock_tokens)
            _emission, adapter, trace_stats = self._capture_prefilled_session(session)
            session._logits_fn = None
            session.reset()
            self._persistent_adapter = adapter
            _metric("resident_capture_restored", trace_stats=trace_stats)
        except BaseException:
            self._release_unpublished_adapter(
                adapter if adapter is not None else session._logits_fn,
                label="failed resident-capture restore",
            )
            session._logits_fn = None
            session.reset()
            logger.error(
                "[DiffusionGemma vLLM] failed to restore the resident capture after a "
                "reveal-upshift failure; the engine cannot continue"
            )
            raise

    def warmup_model_decode(self, *args, **kwargs):
        """No-op: model-level denoise needs no separate decode warmup."""
        del args, kwargs
        if self._upfront:
            logger.info("[DiffusionGemma vLLM] decode warmup is covered by up-front block-denoise capture")
        else:
            logger.info("[DiffusionGemma vLLM] decode warmup is a no-op for eager block diffusion")

    # ── block-granular forward ──────────────────────────────────────────
    def _prompt_tokens_for_row(self, tokens, prompt_lens, row):
        length = int(prompt_lens[row]) if prompt_lens is not None else tokens.shape[1]
        ids = tokens[row, :length].reshape(1, length).to(torch.long)
        return ids

    def _make_session(self, seed: int = 0, *, prefill_execution_len: int | None = None) -> BlockDiffusionServingSession:
        # Serving contract: vLLM owns the stop decision (EOS / stop strings /
        # max_tokens / ignore_eos), not the model. Disable the session's internal
        # EOS stop (``stop_token_ids=[]``) so a committed block that happens to
        # contain an EOS does NOT force the session to finish and emit synthetic
        # stop-padding on the next decode step — that would defeat ``ignore_eos``
        # and short-circuit real multi-block generation. The runner still returns
        # the whole 256-token committed canvas to vLLM, which trims at its own
        # stop point (block-diffusion #47488 scheduler-half contract). The
        # standalone ``serving_smoke`` driver keeps its own session-level stop.
        #
        # This does NOT disarm the degeneracy guard's stop set: the session resolves that
        # separately from the tokenizer's special ids (``_resolve_degeneracy_stop_ids``).
        denoise_block_fn = upfront_traced_denoise_block if self._upfront else None
        _metric(
            "session_create",
            upfront_capture=self._upfront,
            denoise_path=getattr(denoise_block_fn, "__name__", "denoise_block"),
            gumbel_mode=self._gumbel_mode,
            canvas_length=self.canvas_length,
            max_denoise_steps=self._config.max_denoise_steps,
            seed=seed,
            prefill_execution_len=prefill_execution_len,
        )
        return BlockDiffusionServingSession(
            self.model[0],
            self._dg_state_dict,
            config=self._config,
            tokenizer=self._tokenizer,
            gumbel_mode=self._gumbel_mode,
            seed=seed,
            stop_token_ids=[],
            page_tables_per_layer=getattr(self, "_model_owned_page_tables_per_layer", None),
            prefill_execution_len=prefill_execution_len,
            denoise_block_fn=denoise_block_fn,
        )

    def prefill_forward(
        self,
        tokens,
        page_table=None,
        kv_cache=None,
        prompt_lens=None,
        start_pos=None,
        page_tables_per_layer=None,
        sampling_params=None,
        empty_slots=None,
        **kwargs,
    ):
        """Write prompt K/V and emit block 0 for each request.

        Prompt length may be any value up to ``max_model_len`` — there is no
        divisibility requirement (the intrinsic 256-token *output* block
        granularity is not an input constraint). ``prefill_prompt_tokens`` pads to
        a 32-tile multiple internally. Returns ``[num_reqs, canvas_length]``
        committed block-0 token ids (block-granular output; see #47488).
        """
        del kv_cache, start_pos, page_tables_per_layer, sampling_params  # model-owned cache path
        num_reqs = tokens.shape[0]
        if num_reqs > 1:
            # One contiguous model cache backs one active sequence: a second
            # request's prefill would overwrite the first's frozen prompt K/V.
            # Concurrent batched serving needs the vLLM paged-cache ownership
            # change (#47488) + batched canvas decode (#47557). Fail loud rather
            # than silently corrupt — this is the recorded hard limit, not a
            # hardcoded batch-1 assumption in shapes/formatting.
            raise NotImplementedError(
                f"DiffusionGemma serving is single active sequence (got {num_reqs}); "
                "concurrent batched serving is #47488 (paged-cache ownership) + #47557 "
                "(batched canvas decode). Set --max-num-seqs 1."
            )
        blocks = []
        for row in range(num_reqs):
            if row in self._sessions:
                # Defensive cleanup if a runner does not deliver its finished-request
                # callback before reusing the single active row.
                self.release_request(row)
            prompt_tokens = self._prompt_tokens_for_row(tokens, prompt_lens, row)
            upfront = bool(getattr(self, "_upfront", False))
            if upfront:
                cache_len = _aligned_prefill_len(prompt_tokens.shape[1])
                coarse_buckets = _coarse_prefill_buckets_enabled()
                execution_len = (
                    _resolve_prefill_execution_len(
                        prompt_tokens.shape[1],
                        max_model_len=getattr(self, "_max_model_len", None),
                    )
                    if coarse_buckets
                    else cache_len
                )
                session = (
                    self._make_session(prefill_execution_len=execution_len) if coarse_buckets else self._make_session()
                )
            else:
                cache_len = None
                execution_len = None
                session = self._make_session()
            cold_rebuild = False
            desired_bucket = None
            if upfront:
                if self._persistent_adapter is None:
                    raise RuntimeError(
                        "DG_UPFRONT_CAPTURE is enabled but warmup_model_prefill has not completed successfully"
                    )
            if upfront and getattr(self, "_upfront_compile_phase_seen", False):
                p_max = getattr(self, "_upfront_pmax", None)
                if p_max is not None and cache_len + self.canvas_length > int(p_max):
                    # vLLM admits against the logical prompt length, while the
                    # model must reserve a tile-aligned prefill plus a complete
                    # canvas. Reject this one unservable request instead of
                    # raising out of execute_model and terminating EngineCore.
                    logger.error(
                        f"[DiffusionGemma vLLM] REJECTING request on row {row}: aligned prefill plus "
                        f"canvas exceeds fixed reveal span ({cache_len} + {self.canvas_length} > {p_max}). "
                        "Ending this request with an empty answer; the server stays up."
                    )
                    _metric(
                        "prefill_rejected",
                        row=row,
                        cache_len=cache_len,
                        execution_len=execution_len,
                        p_max=int(p_max),
                        reason="aligned_prefill_plus_canvas_exceeds_reveal_span",
                    )
                    session.finished = True
                    self._sessions[row] = session
                    blocks.append(self._stop_block(session))
                    continue
                if _reveal_buckets_enabled() and getattr(self, "_upfront_reveal_bucket", None) is not None:
                    candidate = _resolve_reveal_bucket(
                        max(
                            cache_len + self.canvas_length,
                            _reveal_provisioned_span(cache_len, canvas_length=self.canvas_length, ceiling=int(p_max)),
                        ),
                        ceiling=int(p_max),
                    )
                    if _resolve_reveal_bucket_change(candidate, int(self._upfront_reveal_bucket)):
                        # Committed (active span registered, tracker updated) only around
                        # the rebuild call below, so the rejection path and a failed
                        # rebuild leave the resident capture's bookkeeping intact.
                        desired_bucket = candidate
                        logger.info(
                            f"[DiffusionGemma vLLM] reveal bucket change "
                            f"{self._upfront_reveal_bucket} -> {desired_bucket} for cache_len={cache_len}; "
                            "releasing the resident trace before recapture"
                        )
                        _metric(
                            "reveal_bucket_change",
                            row=row,
                            cache_len=cache_len,
                            resident=int(self._upfront_reveal_bucket),
                            desired=desired_bucket,
                        )
                warmed = getattr(self, "_upfront_prefill_warmup_lens", frozenset())
                if not _prefill_execution_len_is_warmed(execution_len, warmed):
                    if _lazy_prefill_recapture_enabled():
                        cold_rebuild = True
                        logger.info(
                            f"[DiffusionGemma vLLM] cold prefill shape execution_len={execution_len} "
                            f"(cache_len={cache_len}); "
                            "releasing the resident trace before compile and recapture"
                        )
                    else:
                        # FAIL THIS REQUEST, NOT THE SERVER. In vLLM V1 an exception out of
                        # ``execute_model`` is unconditionally fatal: EngineCore exits, and every
                        # request already queued behind it is answered with an empty completion and
                        # HTTP 200. One unservable request must cost one request.
                        #
                        # Compiling the missing shape here is NOT the fallback: a program-cache miss
                        # while a trace is resident can hang the multi-device CCL (needs `tt-smi -r`).
                        # Padding up to the nearest warmed length is an unmeasured alternative (the
                        # reveal mask hides pad keys from the CANVAS, but prefill still writes their
                        # K/V, and the commit path is not the denoise path). So the request ends,
                        # loudly, with the same stop-id block the degeneracy guard's terminal path
                        # already uses.
                        logger.error(
                            f"[DiffusionGemma vLLM] REJECTING request on row {row}: aligned prefill "
                            f"execution length {execution_len} (cache_len={cache_len}) was not warmed before trace capture "
                            f"(warmed={sorted(warmed)}). Ending this request with an empty answer; the "
                            f"server stays up. Enable DG_UPFRONT_LAZY_PREFILL_RECAPTURE or add {cache_len} "
                            "to DG_UPFRONT_PREFILL_WARMUP_LENS."
                        )
                        _metric(
                            "prefill_rejected",
                            row=row,
                            cache_len=cache_len,
                            execution_len=execution_len,
                            warmed=sorted(warmed),
                        )
                        if _strict_prefill_lens():
                            # Bit-exactness gates want the run to stop rather than silently lose a
                            # sample, since an unwarmed shape invalidates the comparison.
                            session.reset()
                            raise RuntimeError(
                                f"up-front capture cannot serve unseen aligned prefill length {cache_len}; "
                                f"warm it before capture via DG_UPFRONT_PREFILL_WARMUP_LENS "
                                f"(configured={sorted(warmed)}). This raise is FATAL to the vLLM engine "
                                f"and is enabled by DG_UPFRONT_STRICT_PREFILL_LENS=1; unset it to reject "
                                f"the request instead."
                            )
                        # Register the row as an ALREADY-FINISHED session rather than dropping it.
                        # ``decode_forward`` raises when ``_sessions`` is empty, and that raise is
                        # engine-fatal too -- a dropped row would move the crash one step later
                        # instead of removing it. A finished session takes
                        # decode_forward's existing stop-id branch, and release_request cleans it up and
                        # emits the usual request_release line.
                        session.finished = True
                        self._sessions[row] = session
                        blocks.append(self._stop_block(session))
                        continue
            bucket_only_rebuild = desired_bucket is not None and not cold_rebuild
            if desired_bucket is not None:
                cold_rebuild = True
            if upfront and not cold_rebuild:
                session.attach_persistent_adapter(self._persistent_adapter)
            ttft_t0 = time.perf_counter()
            try:
                if cold_rebuild:
                    if desired_bucket is not None:
                        set_active_reveal_pmax(desired_bucket)
                    try:
                        if bucket_only_rebuild:
                            # Shape is warm: prefill-first ordering keeps the whole
                            # free pool available for the prefill's transients.
                            cache_len, emission = self._rebuild_for_reveal_bucket(
                                session,
                                prompt_tokens,
                                expected_cache_len=cache_len,
                            )
                        else:
                            cache_len, emission = self._rebuild_for_cold_prefill(
                                session,
                                prompt_tokens,
                                expected_cache_len=cache_len,
                                execution_len=execution_len,
                                capture_span=(
                                    desired_bucket
                                    if desired_bucket is not None
                                    else getattr(self, "_upfront_reveal_bucket", None)
                                ),
                            )
                    except BaseException:
                        if desired_bucket is not None:
                            # The failed capture never published; the next capture must
                            # bind whatever bucket the tracker still records.
                            set_active_reveal_pmax(self._upfront_reveal_bucket)
                        raise
                    if desired_bucket is not None:
                        self._upfront_reveal_bucket = desired_bucket
                else:
                    cache_len = session.prefill(prompt_tokens)
                    emission = session.decode_block()
            except BaseException:
                # The row is not registered in ``_sessions`` until block 0 succeeds, so
                # request-finished callbacks cannot clean this partially built session.
                session.reset()
                raise
            ttft_s = time.perf_counter() - ttft_t0
            dram = _dram_snapshot(self.model[0].mesh_device)
            logger.info(
                f"[DiffusionGemma vLLM] prefill row={row} prompt_len={session.prompt_len} "
                f"cache_len={cache_len} block0 next_pos={emission.next_pos} "
                f"steps={emission.num_denoise_steps} latency={emission.latency_s:.3f}s"
            )
            _metric(
                "prefill_block0",
                row=row,
                prompt_len=session.prompt_len,
                cache_len=cache_len,
                execution_len=getattr(session, "prefill_execution_len", None) or cache_len,
                prefill_s=round(session.prefill_time_s, 6),
                ttft_s=round(ttft_s, 6),
                block_idx=emission.block_idx,
                block_latency_s=round(emission.latency_s, 6),
                denoise_latency_s=round(emission.denoise_latency_s, 6),
                commit_latency_s=round(emission.commit_latency_s, 6),
                denoise_steps=emission.num_denoise_steps,
                committed_tokens=int(emission.tokens.numel()),
                start_pos=emission.start_pos,
                next_pos=emission.next_pos,
                halted=emission.halted,
                dram=dram,
            )
            # Committed ids for every block, so accuracy is observable DURING a multi-hour eval:
            # lm_eval writes its samples only at the end, and nothing else in the serving path
            # carries the generated text.
            _metric(
                "block_ids",
                row=row,
                block_idx=emission.block_idx,
                prompt_len=session.prompt_len,
                start_pos=emission.start_pos,
                ids=_committed_ids(emission.tokens),
            )
            self._sessions[row] = session
            blocks.append(self._emission_block(emission, session, row))
        return torch.cat(blocks, dim=0)

    def decode_forward(
        self,
        tokens=None,
        start_pos=None,
        page_table=None,
        kv_cache=None,
        enable_trace: bool = True,
        read_from_device: bool = True,
        sampling_params=None,
        page_tables_per_layer=None,
        reset_batch: bool = False,
        slot_remap=None,
        **kwargs,
    ):
        """Emit the next 256-token block for each active request.

        One call == one denoise+commit block per active session. ``start_pos``,
        page-table refresh, and any stale-input handling are per-BLOCK (not
        per-token): the model already holds each request's absolute position and
        committed K/V in its cache, so the runner-supplied per-token ``tokens`` /
        ``start_pos`` are advisory only on this path. ``read_from_device`` /
        async semantics are per-block; the committed block returns on host (only
        per-step [B,L] decision tensors are read back — the [B,L,vocab] logits stay
        on device).

        ``enable_trace`` is a TT-runner transport argument only. Model-level tracing
        is selected once by ``DG_UPFRONT_CAPTURE`` and cannot change mid-sequence.
        """
        del tokens, start_pos, page_table, kv_cache, enable_trace, read_from_device
        del sampling_params, page_tables_per_layer, reset_batch, slot_remap
        if not self._sessions:
            raise RuntimeError("decode_forward called with no active sessions (prefill_forward first)")
        rows = sorted(self._sessions)
        blocks = []
        for row in rows:
            session = self._sessions[row]
            if session.finished:
                # Request already emitted a stop token; pad with the stop id.
                # (With the serving contract above this is dead for max_num_seqs=1,
                # but a batched session may still self-finish; guard for empty.)
                blocks.append(self._stop_block(session))
                continue
            try:
                upshift_span = self._reveal_upshift_needed_span(session)
                if upshift_span is not None:
                    try:
                        emission = self._rebuild_for_reveal_upshift(session, needed_span=upshift_span, row=row)
                    except BaseException as upshift_error:
                        # ONE REQUEST costs ONE REQUEST (same contract as the unwarmed
                        # prefill rejection above): a failed growth recapture ends the
                        # growing request with its stop block, then restores a resident
                        # capture so the server keeps serving. Only a failed restore is
                        # engine-fatal.
                        logger.error(
                            f"[DiffusionGemma vLLM] reveal upshift failed on row {row}; ending this "
                            f"request and restoring the resident capture: {upshift_error!r}"
                        )
                        _metric("reveal_upshift_request_failed", row=row, error=repr(upshift_error))
                        # Keep an already-finished row until vLLM observes its
                        # EOS block and invokes the normal finished callback.
                        # Removing the sole row makes the next decode raise
                        # ``no active sessions`` and kills EngineCore.
                        session.finished = True
                        self._sessions[row] = session
                        self._restore_resident_capture()
                        blocks.append(self._stop_block(session))
                        continue
                else:
                    emission = session.decode_block()
            except BaseException:
                # Detach the failed request. A model-lifetime up-front capture remains
                # owned by the wrapper and is released only at terminal shutdown.
                self.release_request(row)
                raise
            logger.info(
                f"[DiffusionGemma vLLM] decode row={row} block={emission.block_idx} "
                f"start_pos={emission.start_pos} next_pos={emission.next_pos} "
                f"steps={emission.num_denoise_steps} halted={emission.halted} "
                f"stop={emission.stop} latency={emission.latency_s:.3f}s"
            )
            _metric(
                "decode_block",
                row=row,
                block_idx=emission.block_idx,
                block_latency_s=round(emission.latency_s, 6),
                denoise_latency_s=round(emission.denoise_latency_s, 6),
                commit_latency_s=round(emission.commit_latency_s, 6),
                denoise_steps=emission.num_denoise_steps,
                committed_tokens=int(emission.tokens.numel()),
                start_pos=emission.start_pos,
                next_pos=emission.next_pos,
                halted=emission.halted,
                stop=emission.stop,
            )
            _metric(
                "block_ids",
                row=row,
                block_idx=emission.block_idx,
                prompt_len=session.prompt_len,
                start_pos=emission.start_pos,
                ids=_committed_ids(emission.tokens),
            )
            blocks.append(self._emission_block(emission, session, row))
        return torch.cat(blocks, dim=0)

    def _stop_block(self, session) -> torch.Tensor:
        """A full ``[1, canvas_length]`` block of the session's stop id.

        Used wherever a row has no real tokens to contribute but must still fill its slot: a session
        that already finished, and a terminal emission from the degeneracy guard.
        """
        token_ids = session.stop_token_ids
        if not token_ids:
            # The vLLM serving session deliberately disables its own stop
            # policy, so its ``stop_token_ids`` is normally empty. Synthetic
            # terminal blocks still need a token vLLM recognizes as EOS.
            token_ids = getattr(getattr(self, "_tokenizer", None), "eos_token_id", None)
        if token_ids:
            ids = token_ids if isinstance(token_ids, (list, tuple)) else [token_ids]
            stop_id = int(ids[0])
        else:
            # A production wrapper always has a tokenizer. Keep a shape-safe
            # fallback for partial test/tokenizer objects rather than raising
            # from execute_model.
            logger.error("[DiffusionGemma vLLM] no EOS token is available for a synthetic terminal block")
            stop_id = 0
        return torch.full((1, self.canvas_length), stop_id, dtype=torch.long)

    def _emission_block(self, emission, session, row: int) -> torch.Tensor:
        """One row's ``[1, canvas_length]`` contribution for a block emission.

        A ZERO-token emission is the degeneracy guard's terminal signal: the canvas was refused and
        NOT committed, so the request ends here and keeps the healthy blocks it already produced
        (``serving.decode_block``). Every row must still fill its slot, so it pads with the stop id
        exactly as the already-finished path does.
        """
        count = int(emission.tokens.numel())
        if count == 0:
            logger.info(
                f"[DiffusionGemma vLLM] row={row} block={emission.block_idx} terminal (canvas refused); "
                f"emitting a stop-id block and ending the request"
            )
            return self._stop_block(session)
        if count != self.canvas_length:
            raise RuntimeError(
                f"decode_block returned {count} tokens for row {row} block {emission.block_idx}; "
                f"expected 0 (terminal) or {self.canvas_length} (a full canvas)"
            )
        return emission.tokens.reshape(1, self.canvas_length)

    def release_request(self, row: int) -> None:
        """Drop a finished request, preserving any model-lifetime up-front capture."""
        session = self._sessions.pop(row, None)
        if session is not None:
            trace_stats = session.trace_stats()
            prompt_len = session.prompt_len
            cache_len = session.cache_len
            blocks_emitted = session.block_idx
            session.reset()
            _metric(
                "request_release",
                row=row,
                prompt_len=prompt_len,
                cache_len=cache_len,
                blocks_emitted=blocks_emitted,
                trace_stats=trace_stats,
                dram=_dram_snapshot(self.model[0].mesh_device),
            )

    def release_persistent_capture(self) -> None:
        """Best-effort terminal shutdown of the model-lifetime adapter and trace buffers.

        Call immediately before mesh close; continuing inference after this terminal release is
        unsupported because the startup capture is intentionally not rebuilt mid-process.
        """
        for row in list(getattr(self, "_sessions", {})):
            try:
                self.release_request(row)
            except BaseException as cleanup_error:
                logger.error(f"failed to detach active request {row} during persistent release: {cleanup_error}")

        adapter = getattr(self, "_persistent_adapter", None)
        self._persistent_adapter = None
        if adapter is None:
            return

        attr = "_upfront_traced_denoise_controller"
        controller = getattr(adapter, attr, None)
        if controller is not None:
            try:
                controller.release()
            except BaseException as cleanup_error:
                logger.error(f"failed to release persistent serving controller {attr}: {cleanup_error}")
            finally:
                delattr(adapter, attr)
        if hasattr(adapter, "reset"):
            try:
                adapter.reset()
            except BaseException as cleanup_error:
                logger.error(f"failed to release persistent serving adapter: {cleanup_error}")

    def __del__(self):
        """Release DiffusionGemma-owned traces before inherited model/mesh teardown."""
        try:
            self.release_persistent_capture()
        except BaseException:
            # Interpreter shutdown may already have torn down logging/TTNN modules.
            pass
        try:
            super().__del__()
        except BaseException:
            pass
