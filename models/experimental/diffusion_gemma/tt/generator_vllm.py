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
from dataclasses import replace

import torch
from loguru import logger

import ttnn
from models.experimental.diffusion_gemma.checkpoint import build_tt_model_from_checkpoint_dir
from models.experimental.diffusion_gemma.config import DiffusionConfig
from models.experimental.diffusion_gemma.tt.generate import (
    denoise_flags_select_traced,
    select_traced_denoise_block_fn,
)
from models.experimental.diffusion_gemma.tt.prefix_cache import PrefixKVCache
from models.experimental.diffusion_gemma.tt.serving import BlockDiffusionServingSession
from models.experimental.diffusion_gemma.tt.traced_denoise import (
    reveal_mask_enabled,
    traced_denoise_block,
    upfront_capture_enabled,
)
from models.tt_transformers.tt.generator_vllm import HybridAttentionForCausalLM

MAX_DENOISE_STEPS = 48


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


def _with_vllm_max_denoise_steps(config: DiffusionConfig) -> DiffusionConfig:
    """Apply the DG-local serving step cap, rejecting non-model budgets."""
    raw = os.environ.get("DG_VLLM_MAX_DENOISE_STEPS")
    if raw is None:
        return config
    try:
        steps = int(raw)
    except ValueError as exc:
        raise ValueError("DG_VLLM_MAX_DENOISE_STEPS must be an integer in [1, 48]") from exc
    if not 1 <= steps <= MAX_DENOISE_STEPS:
        raise ValueError("DG_VLLM_MAX_DENOISE_STEPS must be in [1, 48]")
    return replace(config, max_denoise_steps=steps)


def _validate_upfront_capture_configuration(*, trace_enabled: bool, canvas_length: int) -> int:
    """Validate the fail-loud startup contract and return the explicit fixed prefix span."""
    if not reveal_mask_enabled():
        raise RuntimeError("DG_UPFRONT_CAPTURE requires DG_DENOISE_REVEAL_MASK=1")
    if not trace_enabled:
        raise RuntimeError("DG_UPFRONT_CAPTURE requires DG_VLLM_TRACE=1 or an explicit DG_DENOISE_* trace")
    if os.environ.get("DG_DENOISE_LAZY_CAPTURE", "0").lower() in ("1", "true", "yes", "on"):
        raise RuntimeError(
            "DG_UPFRONT_CAPTURE requires DG_DENOISE_LAZY_CAPTURE=0; all trace windows must be captured at startup"
        )

    raw_trace_region = os.environ.get("DG_TRACE_REGION_SIZE", "").strip()
    try:
        trace_region_size = int(raw_trace_region)
    except ValueError as exc:
        raise RuntimeError("DG_UPFRONT_CAPTURE requires an integer DG_TRACE_REGION_SIZE > 0") from exc
    if trace_region_size <= 0:
        raise RuntimeError("DG_UPFRONT_CAPTURE requires DG_TRACE_REGION_SIZE > 0")

    raw_pmax = os.environ.get("DG_DENOISE_REVEAL_PMAX", "").strip()
    if not raw_pmax:
        raise RuntimeError(
            "DG_UPFRONT_CAPTURE requires an explicit bounded DG_DENOISE_REVEAL_PMAX; "
            "the full allocated KV span is not an acceptable fallback"
        )
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
    #    block pool (#47488), which is NOT wired here → advertise False. A DG
    #    serving-layer frozen prompt-prefix KV reuse prototype exists behind the
    #    DG_PREFIX_CACHE env flag (see tt/prefix_cache.py + the prefix_cache design
    #    note); it reuses the model-owned contiguous cache across sessions but does
    #    NOT change the vLLM-advertised capability, so vLLM's own block pool stays
    #    disabled. Flipping this flag requires the #47488 paged path + its tests.
    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_async_decode": False,
        "supports_sample_on_device": True,
    }

    def __init__(self, *args, dg_state_dict=None, tokenizer=None, config=None, gumbel_mode="chunked", **kwargs):
        super().__init__(*args, **kwargs)
        self._dg_state_dict = dg_state_dict
        self._tokenizer = tokenizer
        self._config = _with_vllm_max_denoise_steps(DiffusionConfig() if config is None else config)
        self.canvas_length = self._config.canvas_length
        # Sampler at the served context. DEFAULT "chunked": the no-materialize on-device
        # Gumbel-max sampler that is distribution-faithful to the model's reference
        # EntropyBoundSampler (HF multinomial(softmax(logits/T))) AND fits full-depth 256K.
        # "argmax" (greedy RUN-first, also fits 256K) is opt-in via DG_VLLM_GUMBEL_MODE; the
        # full-vocab Gumbel ("host"/"device") OOMs at 256K (see doc/context_contract.json).
        self._gumbel_mode = os.environ.get("DG_VLLM_GUMBEL_MODE", gumbel_mode)
        # One active session per batch row. A single contiguous model cache backs
        # one active sequence today (see module docstring); the dict is keyed by
        # row so output formatting never assumes batch size 1.
        self._sessions: dict[int, BlockDiffusionServingSession] = {}
        # Traced serving decode (Metal TRACE capture/replay in the decode path). The serving
        # session reuses the generator's env-gated dispatcher; when trace is requested here we
        # pass the traced loop explicitly so the PERSISTENT session's logits fn caches ONE
        # controller (captured on block 0 in prefill_forward) and ``execute_trace``-replays it
        # every decode_forward block — NOT re-captured per block. Argmax can use any selected
        # traced variant; dynamic Gumbel modes force single-step replay (chunked uses a persistent
        # device seed + bounded uniform chunk buffer, host/device use a persistent full-noise
        # input). The path needs a sized ``DG_TRACE_REGION_SIZE`` at mesh open. Default follows the
        # env dispatcher (eager unless a ``DG_DENOISE_*`` flag is set) so a plain launch is
        # unchanged; ``DG_VLLM_TRACE=1`` opts this path into trace without the internal flag and
        # ``DG_VLLM_TRACE=0`` forces eager. See doc/vllm_integration/README.md (traced serving).
        self._trace_enabled = self._resolve_trace_pref()
        self._upfront = upfront_capture_enabled()
        self._persistent_adapter = None
        self._upfront_pmax = (
            _validate_upfront_capture_configuration(
                trace_enabled=self._trace_enabled,
                canvas_length=self.canvas_length,
            )
            if self._upfront
            else None
        )
        # Frozen prompt-prefix KV reuse (APC prototype, #47466): a single registry
        # shared across sessions so a request whose aligned prompt is a prefix of the
        # resident contiguous-cache prompt can skip its prefill. Inert unless
        # DG_PREFIX_CACHE is set (checked per-prefill inside the session); safe for
        # max_num_seqs=1 (one contiguous cache = one resident prompt).
        self._prefix_cache = PrefixKVCache()

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
        diffusion_config = _with_vllm_max_denoise_steps(DiffusionConfig())
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
            f"n_layers={n_layers or 'full'} gumbel_mode={os.environ.get('DG_VLLM_GUMBEL_MODE', 'chunked')}"
        )
        _metric(
            "model_build",
            max_seq_len=max_seq_len,
            num_layers=n_layers or 30,
            model_build_s=round(model_build_s, 6),
            gumbel_mode=os.environ.get("DG_VLLM_GUMBEL_MODE", "chunked"),
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
        del kv_cache, enable_trace, can_sample_on_device, greedy_only
        if not self._upfront:
            # The default path remains lazy and per-request exactly as before.
            logger.info("[DiffusionGemma vLLM] warmup is a no-op; block-diffusion warms on first prefill/decode")
            return
        if self._persistent_adapter is not None:
            logger.info("[DiffusionGemma vLLM] up-front denoise capture already initialized")
            return

        p_max = _validate_upfront_capture_configuration(
            trace_enabled=self._trace_enabled,
            canvas_length=self.canvas_length,
        )
        cache_span = min(int(k_cache.shape[-2]) for k_cache, _v_cache in self.model[0].tt_kv_cache)
        if p_max > cache_span:
            raise RuntimeError(
                f"DG_DENOISE_REVEAL_PMAX={p_max} exceeds the smallest allocated model KV span {cache_span}"
            )
        self._upfront_pmax = p_max

        mock_token_id = getattr(self._tokenizer, "bos_token_id", None)
        if mock_token_id is None:
            mock_token_id = getattr(self._tokenizer, "eos_token_id", None)
        if mock_token_id is None:
            mock_token_id = 0
        mock_tokens = torch.tensor([[int(mock_token_id)]], dtype=torch.long)

        session = self._make_session()
        try:
            cache_len = session.prefill(mock_tokens)
            emission = session.decode_block()
            adapter = session._logits_fn
            controllers = [
                getattr(adapter, attr, None)
                for attr in (
                    "_traced_denoise_controller",
                    "_traced_denoise_multistep_controller",
                    "_traced_early_halt_controller",
                )
            ]
            controllers = [controller for controller in controllers if controller is not None]
            if not controllers or not all(getattr(controller, "captured", False) for controller in controllers):
                raise RuntimeError("startup denoise did not leave a fully captured traced controller")
            if not getattr(adapter, "use_reveal_mask", False):
                raise RuntimeError("startup denoise trace was not captured with a persistent reveal mask")

            trace_stats = session.trace_stats()
            # Detach before resetting the throwaway shell: the wrapper now owns the adapter.
            session._logits_fn = None
            session.reset()
            self._persistent_adapter = adapter
        except BaseException:
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

    # ── block-granular forward ──────────────────────────────────────────
    def _prompt_tokens_for_row(self, tokens, prompt_lens, row):
        length = int(prompt_lens[row]) if prompt_lens is not None else tokens.shape[1]
        ids = tokens[row, :length].reshape(1, length).to(torch.long)
        return ids

    def _resolve_trace_pref(self) -> bool:
        """Decide whether the serving decode path traces (Metal capture/replay).

        Resolved ONCE at construction because block 0 (captured inside ``prefill_forward``) fixes
        the trace, so vLLM's per-decode ``enable_trace`` maps here. Explicit ``DG_VLLM_TRACE``
        wins; otherwise follow the env dispatcher (``DG_DENOISE_*`` flags). Argmax,
        injected/materialized host/device Gumbel, and bounded-memory chunked Gumbel can trace.
        """
        raw = os.environ.get("DG_VLLM_TRACE")
        want = raw.strip().lower() in ("1", "true", "yes", "on") if raw is not None else denoise_flags_select_traced()
        if want:
            logger.info(
                "[DiffusionGemma vLLM] traced serving decode ENABLED (Metal capture/replay); "
                "ensure DG_TRACE_REGION_SIZE is set at mesh open"
            )
        return want

    def _select_session_denoise_block_fn(self):
        if not self._trace_enabled:
            return None
        if self._gumbel_mode in ("host", "device", "chunked"):
            # Dynamic Gumbel refreshes a full-noise input (host/device) or a device seed
            # (chunked) between single-step replays. A grouped trace cannot refresh either
            # input inside its window, so force the single-step traced controller here.
            return traced_denoise_block
        return select_traced_denoise_block_fn()

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
        denoise_block_fn = self._select_session_denoise_block_fn()
        _metric(
            "session_create",
            trace_enabled=self._trace_enabled,
            denoise_path=getattr(denoise_block_fn, "__name__", "env_dispatch"),
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
            prefix_cache=self._prefix_cache,
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
            self._sessions[row] = session
            blocks.append(emission.tokens.reshape(1, self.canvas_length))
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

        ``enable_trace`` is honored at CONSTRUCTION (``self._trace_enabled``), not per call:
        the denoise trace is captured on block 0 inside ``prefill_forward`` and
        ``execute_trace``-replayed here, so the traced-vs-eager decision cannot change
        mid-sequence. The session created in ``prefill_forward`` already carries the traced
        (or eager) ``denoise_block_fn``; this call just drives one more block through it.
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
                stop_id = 0
                if session.stop_token_ids:
                    ids = (
                        session.stop_token_ids
                        if isinstance(session.stop_token_ids, (list, tuple))
                        else [session.stop_token_ids]
                    )
                    stop_id = int(ids[0])
                blocks.append(torch.full((1, self.canvas_length), stop_id, dtype=torch.long))
                continue
            try:
                emission = session.decode_block()
            except BaseException:
                # A replay/device failure must release every trace and remove the row
                # before the runner can attempt another request.
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
            blocks.append(emission.tokens.reshape(1, self.canvas_length))
        return torch.cat(blocks, dim=0)

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

        for attr in (
            "_traced_denoise_controller",
            "_traced_denoise_multistep_controller",
            "_traced_early_halt_controller",
        ):
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
