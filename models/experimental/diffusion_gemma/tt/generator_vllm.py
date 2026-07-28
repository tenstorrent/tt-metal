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
stack. See ``doc/vllm_integration/README.md``.

Contract gaps handled here vs deferred to #47488 (upstream tenstorrent/vllm)
--------------------------------------------------------------------------
The current TT runner assumes **one committed token per decode step** — hard
``assert num_out_tokens == 1`` at ``model_runner.py:2471``, ``[sz, 1]`` sampled-id
shape (``:2378``, ``:1878``), single-token ``_build_runner_output`` (``:2437``),
and a ``+1`` host position advance (``_apply_sampled_tokens_to_state`` ``:2479`` /
``:2508``). Emitting a 256-token block therefore needs the runner/scheduler to (a)
accept a ``[num_reqs, 256]`` block output, (b) advance ``num_computed_tokens`` /
``num_tokens`` by ``canvas_length`` per decode step, and (c) bound-check
``start_idx + 256 <= max_model_len``. That runner+scheduler change is **#47488**;
this adapter is written to that block contract so it works once #47488 lands.

Cache ownership
---------------
The diffusion denoise-read path reads the frozen prompt prefix from the
**model-owned contiguous** ``tt_model.tt_kv_cache`` via ``ttnn.slice`` (not from a
vLLM paged block pool). Serving therefore runs in the generator/standalone
cache-ownership mode: the model owns its ``max_model_len`` cache and is driven with
``page_table=None``; :meth:`allocate_kv_cache` returns those existing handles (no
double allocation). Routing the frozen-prefix read through a vLLM paged cache +
per-request block tables (for concurrent batched serving) is part of #47488 and
the batched-canvas-decode work (#47557). Until then one contiguous cache backs one
active sequence.

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
from models.experimental.diffusion_gemma.tt.generate import prefill_prompt_tokens
from models.experimental.diffusion_gemma.tt.serving import BlockDiffusionServingSession
from models.experimental.diffusion_gemma.tt.traced_denoise import (
    UPFRONT_DENOISE_STEPS,
    set_default_reveal_pmax,
    upfront_capture_enabled,
    upfront_traced_denoise_block,
)
from models.tt_transformers.tt.generator_vllm import HybridAttentionForCausalLM


# Served default Gumbel source: the on-device permuted-vocab RNG (see the __init__ note).
# Requires the Blackhole ttnn.rand kernel fix; without it this default corrupts generated text.
DEFAULT_VLLM_GUMBEL_MODE = "device"


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
            "on-device permuted-vocab RNG (no per-step host RNG or PCIe DMA). "
            "'chunked'/'argmax' are not materialized full-tensor sources and are unsupported "
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

    # Serving-feature reality on the TT path (documented in the stage evidence):
    #  * prefix caching: force-disabled for sliding-window models (platform.py:512),
    #    and block-diffusion recomputes canvas K/V every step → declare False.
    #  * async decode: the per-BLOCK async contract is unproven without the #47488
    #    runner; never advertise async without proof → declare False (safe default).
    #  * on-device sampling: the canvas Gumbel-max / entropy-budget / renoise path
    #    runs on device (no host argmax, no full-logits readback) → True.
    #  * prefix caching: the vLLM APC contract needs paged-cache ownership + a
    #    block pool (#47488), which is NOT wired here → advertise False. A serving-layer
    #    frozen-prefix KV reuse prototype used to sit behind DG_PREFIX_CACHE; it was deleted
    #    2026-07-28 (only its exact-full-match tier could fire from serving, the proper-prefix
    #    tier measured 57/256 flipped tokens, and under the shipped up-front capture every
    #    admitted prompt length is pre-enumerated anyway). See doc/vllm_integration/prefix_cache/
    #    for the design note; the real path is #47488.
    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_async_decode": False,
        "supports_sample_on_device": True,
    }

    def __init__(
        self,
        *args,
        dg_state_dict=None,
        tokenizer=None,
        config=None,
        gumbel_mode=DEFAULT_VLLM_GUMBEL_MODE,
        max_model_len=None,
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
        # DEFAULT "device" is the on-device permuted-vocab Gumbel: it removes the ~313 ms/step
        # host RNG and the ~256 MiB/step replicated PCIe DMA that the deleted per-step host-torch
        # Gumbel mode paid, measured here at ~53.6 vs ~36.3 tokens/block/s steady (~1.48x).
        #
        # HISTORY, because this default has moved twice. It was flipped to "device" on 2026-07-24,
        # reverted to the host IID mode on 2026-07-25 after "device" corrupted generated text on
        # 2 of 4 matched seeds, and restored here once the CAUSE was fixed. The cause was never in
        # this module: for the production noise shape (1, 1, 256, vocab) the permuted-vocab draw
        # puts the 256 canvas positions on the ttnn.rand width axis, and the Blackhole SFPU PRNG
        # is a sliding window over one stream -- element (read t, lane i) carried stream[t + i],
        # so 64 of 256 positions held a byte-identical COPY of another position's noise and picked
        # the same token together. tt_metal/hw/ckernels/blackhole/.../ckernel_sfpu_rand.h now
        # advances the window per element; duplicate rows are gone and the same 4-seed A/B answers
        # correctly 4/4 on both arms with the degeneracy guard never firing.
        #
        # This default therefore DEPENDS on that kernel fix. The residual correlation it does not
        # remove (cross-position max |r| 0.618 against 0.035 for a host IID control) is pinned by
        # tests/ttnn/nightly/unit_tests/operations/rand/test_rand_independence.py; if that
        # regresses, revisit this default.
        #
        # WHY THERE IS NO LONGER A "host" MODE: the per-step full-vocab torch Gumbel source was
        # the suspected cause of TT language drift and was measured NOT to be -- it drifts on
        # exactly the same prompts as "device", repairs 0 of them, and costs 1.40x per request.
        # The real cause was the canvas attending the prefill pad keys, fixed in d0936d4da4f, so
        # the mode was deleted. (The torch-noise INJECTION helpers in tt/generate.py stay: they
        # replay a torch run's exact noise for HF<->TT determinism, they are not a serving mode.)
        #
        # MEMORY ENVELOPE: "device" materializes a full-vocabulary (262144) tensor per step.
        # context_contract.json records that materialization measured as an OOM in the DRAM left
        # after a 256K-KV allocation, where only the no-materialize "chunked" descriptor fits.
        # So at very large served contexts this default must be overridden with
        # DG_VLLM_GUMBEL_MODE=chunked, which in turn requires DG_UPFRONT_CAPTURE=0 (the up-front
        # controller needs a materialized source and rejects "chunked"/"argmax"). "chunked" also
        # carries a known QB2 1024-wide RNG distribution bias; "argmax" is the fast deterministic
        # RUN control.
        self._gumbel_mode = os.environ.get("DG_VLLM_GUMBEL_MODE", gumbel_mode)
        # One active session per batch row. A single contiguous model cache backs
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
        # Frozen prompt-prefix KV reuse (APC prototype, #47466): a single registry
        # shared across sessions so a request whose aligned prompt is a prefix of the
        # resident contiguous-cache prompt can skip its prefill. Inert unless
        # (the DG_PREFIX_CACHE reuse tier was deleted 2026-07-28); safe for
        # max_num_seqs=1 (one contiguous cache = one resident prompt).

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
            create_kv_cache=True,  # model owns its contiguous KV cache (see docstring)
        )
        if n_layers is not None:
            model_kwargs["num_layers"] = n_layers

        build_t0 = time.perf_counter()
        bundle = build_tt_model_from_checkpoint_dir(mesh_device, checkpoint_dir, **model_kwargs)
        ttnn.synchronize_device(mesh_device)
        model_build_s = time.perf_counter() - build_t0
        dram = _dram_snapshot(mesh_device, synchronize=False)
        logger.info(
            f"[DiffusionGemma vLLM] built model: max_seq_len={max_seq_len} "
            f"n_layers={n_layers or 'full'} "
            f"gumbel_mode={os.environ.get('DG_VLLM_GUMBEL_MODE', DEFAULT_VLLM_GUMBEL_MODE)}"
        )
        _metric(
            "model_build",
            max_seq_len=max_seq_len,
            num_layers=n_layers or 30,
            model_build_s=round(model_build_s, 6),
            gumbel_mode=os.environ.get("DG_VLLM_GUMBEL_MODE", DEFAULT_VLLM_GUMBEL_MODE),
            max_denoise_steps=diffusion_config.max_denoise_steps,
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
        the non-hybrid single-page-table path, so a per-type spec would instead split
        into 6 groups sharing the pool and cap prefill admission at ~21824 tokens (see
        the sliding branch). The diffusion forward reads the model-owned contiguous
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
        """``[submesh][layer][k_or_v]`` handles into the model's own contiguous cache.

        Serving runs on the model-owned contiguous cache the model allocated at
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
            if raw_warmup_lens:
                warmup_lens = set()
                for value in raw_warmup_lens.split(","):
                    prompt_len = int(value.strip())
                    if prompt_len <= 0 or prompt_len % ttnn.TILE_SIZE != 0:
                        raise RuntimeError(
                            "DG_UPFRONT_PREFILL_WARMUP_LENS values must be positive "
                            f"{ttnn.TILE_SIZE}-token multiples, got {prompt_len}"
                        )
                    if prompt_len + self.canvas_length > self._upfront_pmax:
                        raise RuntimeError(
                            f"prefill warmup length {prompt_len} leaves no canvas within p_max={self._upfront_pmax}"
                        )
                    warmup_lens.add(prompt_len)
                # One tile is always warmed, whether or not the caller listed it. The capture phase
                # below prefills a single BOS token to build its adapter, so the 32-aligned prefill
                # program is compiled on every startup regardless -- listing it costs nothing and no
                # extra bytes. Leaving it out is what let a 21-token out-of-band request (a `curl`
                # smoke test against a live server) reach the rejection below and kill the engine
                # 56 minutes into a 198-question eval.
                warmup_lens.add(ttnn.TILE_SIZE)
                self._upfront_prefill_warmup_lens = frozenset(warmup_lens)
                for prompt_len in sorted(self._upfront_prefill_warmup_lens):
                    logger.info(f"[DiffusionGemma vLLM] warming prefill shape {prompt_len} before trace capture")
                    mock_tokens = torch.zeros((1, prompt_len), dtype=torch.long)
                    prefill_prompt_tokens(self.model[0], mock_tokens)
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
                "DG_UPFRONT_CAPTURE requires a compile-only warmup with DG_UPFRONT_PREFILL_WARMUP_LENS; "
                "executing an unseen prefill shape after trace capture can corrupt active traces. "
                "List every aligned prompt length the server will admit, "
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
        cache_span = min(int(k_cache.shape[-2]) for k_cache, _v_cache in self.model[0].tt_kv_cache)
        if p_max > cache_span:
            raise RuntimeError(
                f"DG_DENOISE_REVEAL_PMAX={p_max} exceeds the smallest allocated model KV span {cache_span}"
            )
        if p_max == cache_span:
            # A span equal to the whole cache is legal but is the most expensive one: the
            # per-step prefix read takes the full-span branch (a whole-cache clone per layer
            # per step) and the persistent reveal mask is sized for the full span. Pinning
            # DG_DENOISE_REVEAL_PMAX to the context actually served is the cheap path — this
            # is exactly why the span used to be mandatory rather than derived.
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
            adapter = session._logits_fn
            adapter._upfront_capture_phase = True
            try:
                emission = session.decode_block()
            finally:
                delattr(adapter, "_upfront_capture_phase")
            controller = getattr(adapter, "_upfront_traced_denoise_controller", None)
            if controller is None or not getattr(controller, "captured", False):
                raise RuntimeError("startup denoise did not leave a fully captured up-front controller")
            if not getattr(adapter, "use_reveal_mask", False):
                raise RuntimeError("startup denoise trace was not captured with a persistent reveal mask")

            trace_stats = session.trace_stats()
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

    def _make_session(self, seed: int = 0) -> BlockDiffusionServingSession:
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
        # separately from the tokenizer's special ids (``_resolve_degeneracy_stop_ids``). The two
        # were one field until 2026-07-28, and emptying the stop policy here also blinded the guard,
        # so it rejected the terminal block of 110 of 198 requests on the 07-27 eval.
        denoise_block_fn = upfront_traced_denoise_block if self._upfront else None
        _metric(
            "session_create",
            upfront_capture=self._upfront,
            denoise_path=getattr(denoise_block_fn, "__name__", "denoise_block"),
            gumbel_mode=self._gumbel_mode,
            canvas_length=self.canvas_length,
            max_denoise_steps=self._config.max_denoise_steps,
            seed=seed,
        )
        return BlockDiffusionServingSession(
            self.model[0],
            self._dg_state_dict,
            config=self._config,
            tokenizer=self._tokenizer,
            gumbel_mode=self._gumbel_mode,
            seed=seed,
            stop_token_ids=[],
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
            session = self._make_session()
            if getattr(self, "_upfront", False):
                if self._persistent_adapter is None:
                    raise RuntimeError(
                        "DG_UPFRONT_CAPTURE is enabled but warmup_model_prefill has not completed successfully"
                    )
                session.attach_persistent_adapter(self._persistent_adapter)
            prompt_tokens = self._prompt_tokens_for_row(tokens, prompt_lens, row)
            if getattr(self, "_upfront_compile_phase_seen", False):
                cache_len = ((int(prompt_tokens.shape[1]) + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
                warmed = getattr(self, "_upfront_prefill_warmup_lens", frozenset())
                if cache_len not in warmed:
                    # FAIL THIS REQUEST, NOT THE SERVER.
                    #
                    # This used to `raise`, and in vLLM V1 an exception out of ``execute_model`` is
                    # unconditionally fatal: EngineCore exits, and every request already queued
                    # behind it is answered with an empty completion and HTTP 200. On 2026-07-28 a
                    # single out-of-band 21-token `curl` smoke test against a live server therefore
                    # destroyed 135 of 198 answers, 56 minutes into the run, and the eval still
                    # wrote a normal-looking 23.74% results file. One unservable request must cost
                    # one request.
                    #
                    # Compiling the missing shape here instead is NOT the fallback: it is the
                    # documented cause of a reproduced four-device AllBroadcast hang needing
                    # `tt-smi -r` (doc/optimize_perf/upfront_earlyhalt_gpqa_20260722.md -- warming
                    # the 160-token prefill before capture was the controlled fix). Padding up to
                    # the nearest warmed length is not free either while
                    # DG_DENOISE_HIDE_PREFILL_PADS is default OFF, since the extra pad keys are
                    # decision-changing. So the request ends, loudly, with the same stop-id block
                    # the degeneracy guard's terminal path already uses.
                    logger.error(
                        f"[DiffusionGemma vLLM] REJECTING request on row {row}: aligned prefill "
                        f"length {cache_len} was not warmed before trace capture "
                        f"(warmed={sorted(warmed)}). Ending this request with an empty answer; the "
                        f"server stays up. Add {cache_len} to DG_UPFRONT_PREFILL_WARMUP_LENS to "
                        f"serve prompts of this length."
                    )
                    _metric("prefill_rejected", row=row, cache_len=cache_len, warmed=sorted(warmed))
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
                    # ``decode_forward`` raises when ``_sessions`` is empty, and that raise is just
                    # as engine-fatal as the one being replaced here -- so a dropped row would move
                    # the crash one step later instead of removing it. A finished session takes
                    # decode_forward's existing stop-id branch, and release_request cleans it up and
                    # emits the usual request_release line.
                    session.finished = True
                    self._sessions[row] = session
                    blocks.append(self._stop_block(session))
                    continue
            ttft_t0 = time.perf_counter()
            try:
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
            # Committed ids for every block, so accuracy is observable DURING a multi-hour eval.
            # lm_eval writes its samples only at the end, and nothing else in the serving path
            # carries the generated text, so a 4-hour run was previously unobservable until it
            # finished. Scored by doc/decision_fidelity/gate/live_score.py.
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
        stop_id = 0
        if session.stop_token_ids:
            ids = (
                session.stop_token_ids
                if isinstance(session.stop_token_ids, (list, tuple))
                else [session.stop_token_ids]
            )
            stop_id = int(ids[0])
        return torch.full((1, self.canvas_length), stop_id, dtype=torch.long)

    def _emission_block(self, emission, session, row: int) -> torch.Tensor:
        """One row's ``[1, canvas_length]`` contribution for a block emission.

        A ZERO-token emission is the degeneracy guard's terminal signal: the canvas was refused and
        NOT committed, so the request ends here and keeps the healthy blocks it already produced
        (``serving.decode_block``). Every row must still fill its slot, so it pads with the stop id
        exactly as the already-finished path does. Reshaping the empty tensor instead is what killed
        EngineCore on the first degenerate block of a served run.
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
