# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""vLLM adapter for Gemma4-31B via tt_symbiote.

Bridges the vLLM serving interface (initialize_vllm_model / prefill_forward /
decode_forward / allocate_kv_cache) to the HuggingFace model whose decoder
layers, norms, embedding, and text-model wrapper have been replaced with TTNN
equivalents through tt_symbiote's module replacement machinery.

The module replacement pattern and warmup sequence follow test_gemma4.py.
"""

import gc
import logging
import os
import resource
import time
from typing import Optional

import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Diagnostic probe (gated behind TT_SYMBIOTE_DIAG=1)
#
# Goal: attribute the "first ~9 requests fast, then 370x slowdown" regression
# observed on the 2026-04-17 benchmark sweep to one of:
#   (1) TTNN program-cache pollution     -> progcache grows unbounded
#   (2) NormalRun Python heap accumulation -> gc_objs / rss_mb grow
#   (3) HF cache reference retention     -> wall_ms grows with no obvious heap
# Output is one CSV-shaped line per prefill and one per N decode steps.
# Default off so production runs are unaffected.
# ---------------------------------------------------------------------------
_DIAG_ENABLED = os.environ.get("TT_SYMBIOTE_DIAG", "0") == "1"
_DIAG_DECODE_EVERY = max(1, int(os.environ.get("TT_SYMBIOTE_DIAG_DECODE_EVERY", "32")))
_DIAG_STATE = {"prefill": 0, "decode": 0}

# Watchdog: if a single decode_forward call exceeds this many wall-seconds we
# log a WATCHDOG line. Default 0 = disabled to avoid false positives during
# legitimate first-call compilation. Set to 30 to catch the post-9th-request
# slowdown described in the latency-regression plan.
_WATCHDOG_DECODE_SEC = float(os.environ.get("TT_SYMBIOTE_DECODE_WATCHDOG_SEC", "0"))

# ---------------------------------------------------------------------------
# Phase 2B knobs (latency-regression fix; see
# .cursor/plans/gemma4_latency_regression_fix_9682ee7e.plan.md):
#
#   TT_SYMBIOTE_USE_INFERENCE_MODE=1   prefer torch.inference_mode() over
#                                       torch.no_grad() to avoid one autograd
#                                       version-counter object per dispatched
#                                       op.
#   TT_SYMBIOTE_GC_EVERY_N_PREFILLS=8  force gc.collect() every N prefills.
#                                       0 disables. ~50-100 ms per collection.
#   TT_SYMBIOTE_SEVER_PKV=1            null outputs.past_key_values + del
#                                       outputs after each prefill/decode to
#                                       drop HF's hold on KV pages.
#
# All three default OFF after the 2026-04-20 hang investigation. The 16:38
# server log shows the unmodified code (no_grad / no severance / no GC) ran
# 33 sequential requests cleanly; the 18:27 run with inference_mode + sever
# enabled hung deterministically on the 6th prefill inside a TTNN op called
# from NormalRun.module_run (py-spy: ttnn/decorators.py:473, 99% CPU on the
# main thread). Each knob can be turned on individually to bisect.
# ---------------------------------------------------------------------------
_USE_INFERENCE_MODE = os.environ.get("TT_SYMBIOTE_USE_INFERENCE_MODE", "0") == "1"
_GC_EVERY_N_PREFILLS = max(0, int(os.environ.get("TT_SYMBIOTE_GC_EVERY_N_PREFILLS", "0")))
_SEVER_PKV = os.environ.get("TT_SYMBIOTE_SEVER_PKV", "0") == "1"

# ---------------------------------------------------------------------------
# Referrer probe (next-step diagnosis after the 2026-04-20 smoke run).
#
# The 33-request smoke confirmed the heap grows by ~5,686 objects per decode
# step but TPOT stayed flat at ~430ms -- so Phase 2B's outputs.past_key_values
# severance was a no-op for whatever pins the TorchTTNNTensor wrappers. We
# need to know *which* objects refer to a live wrapper to design the real
# fix (likely TTNNGemma4PagedAttentionKVCache, the NormalRun dispatcher, or
# HF's attention cache).
#
# This probe is OFF by default. When enabled it samples one wrapper every
# TT_SYMBIOTE_DIAG_REFS_EVERY prefills, walks gc.get_referrers(), and logs
# the top referrer types. gc.get_referrers() is O(heap) so we keep N small.
#   TT_SYMBIOTE_DIAG_REFS=1
#   TT_SYMBIOTE_DIAG_REFS_EVERY=8        (default; one snapshot every 8 prefills)
#   TT_SYMBIOTE_DIAG_REFS_TARGET=TorchTTNNTensor   (substring match on type name)
# ---------------------------------------------------------------------------
_DIAG_REFS_ENABLED = os.environ.get("TT_SYMBIOTE_DIAG_REFS", "0") == "1"
_DIAG_REFS_EVERY = max(1, int(os.environ.get("TT_SYMBIOTE_DIAG_REFS_EVERY", "8")))
_DIAG_REFS_TARGET = os.environ.get("TT_SYMBIOTE_DIAG_REFS_TARGET", "TorchTTNNTensor")


class _NoGrad:
    """Tiny re-entrant wrapper that prefers inference_mode but falls back to
    no_grad if a downstream op in tt_symbiote / HF rejects inference tensors.
    Stateful: once we've fallen back we stay fallen back for the rest of the
    process to avoid per-call try/except overhead.
    """

    _disabled = False

    def __enter__(self):
        if _USE_INFERENCE_MODE and not _NoGrad._disabled:
            try:
                self._cm = torch.inference_mode()
                self._cm.__enter__()
                self._mode = "inference_mode"
                return self
            except Exception as exc:
                logger.warning(
                    "torch.inference_mode() unavailable (%s); "
                    "falling back to torch.no_grad() for the rest of the run.",
                    exc,
                )
                _NoGrad._disabled = True
        self._cm = torch.no_grad()
        self._cm.__enter__()
        self._mode = "no_grad"
        return self

    def __exit__(self, exc_type, exc, tb):
        # Auto-fallback if a downstream op rejected inference-mode tensors.
        # Marking _disabled here flips the global so no further calls use it.
        if (
            exc is not None
            and self._mode == "inference_mode"
            and isinstance(exc, RuntimeError)
            and "inference" in str(exc).lower()
        ):
            logger.warning(
                "RuntimeError under torch.inference_mode(): %s. "
                "Auto-disabling inference_mode for the rest of the run; "
                "next call will retry under torch.no_grad().",
                exc,
            )
            _NoGrad._disabled = True
        return self._cm.__exit__(exc_type, exc, tb)


def _diag_progcache_entries(mesh_device) -> int:
    """Best-effort count of TTNN program-cache entries.

    Uses the public MeshDevice.num_program_cache_entries() binding from
    ttnn/core/distributed/distributed_nanobind.cpp. Falls back to -1 if the
    accessor is unavailable on the current commit.
    """
    try:
        return int(mesh_device.num_program_cache_entries())
    except Exception:
        return -1


def _diag_log(kind: str, mesh_device, wall_ms: float, extra: str = "") -> None:
    """Emit one CSV-shaped DIAG line. Caller is responsible for the gating."""
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
    gc_objs = len(gc.get_objects())
    progcache = _diag_progcache_entries(mesh_device)
    counter = _DIAG_STATE[kind]
    suffix = f" {extra}" if extra else ""
    logger.info(
        f"[DIAG] kind={kind} req#={_DIAG_STATE['prefill']} {kind}#={counter} "
        f"wall_ms={wall_ms:.1f} rss_mb={rss_mb} gc_objs={gc_objs} "
        f"progcache={progcache}{suffix}"
    )


def _diag_log_gc_stats(label: str) -> None:
    """Emit per-generation gc stats so we can see whether collections
    actually move the gen0/gen1/gen2 counts (i.e. whether Phase 2B's
    gc.collect() reclaims anything or just pushes garbage into older
    generations).
    """
    try:
        stats = gc.get_stats()
        counts = gc.get_count()
        threshold = gc.get_threshold()
    except Exception as exc:
        logger.info("[DIAG] gc_stats label=%s unavailable: %s", label, exc)
        return
    parts = []
    for gen, st in enumerate(stats):
        parts.append(
            f"g{gen}=col{st.get('collections', 0)}/" f"un{st.get('uncollectable', 0)}/" f"co{st.get('collected', 0)}"
        )
    logger.info(
        "[DIAG] gc_stats label=%s count=%s threshold=%s %s",
        label,
        counts,
        threshold,
        " ".join(parts),
    )


def _diag_log_referrers(target_substr: str, max_objects: int = 1) -> None:
    """Walk gc.get_referrers() for one live target object and log the top
    referrer types. Expensive (O(heap)); caller MUST gate on rate.

    We pick the first matching object (typically the oldest, deepest into
    accumulated state) so the referrer chain is the one that actually
    pinned it across requests.
    """
    try:
        sample = None
        sample_id = None
        for obj in gc.get_objects():
            tname = type(obj).__name__
            if target_substr in tname:
                sample = obj
                sample_id = id(obj)
                break
        if sample is None:
            logger.info(
                "[DIAG] refs target=%r not found in heap (no live instances)",
                target_substr,
            )
            return

        # gc.get_referrers itself appears in its own output; filter it out
        # via id-comparison before tallying.
        referrers = gc.get_referrers(sample)
        type_counts = {}
        sample_reprs = []
        for r in referrers:
            if id(r) == sample_id:
                continue
            tname = type(r).__name__
            type_counts[tname] = type_counts.get(tname, 0) + 1
            if len(sample_reprs) < 3:
                try:
                    sample_reprs.append(f"{tname}@{id(r):x}")
                except Exception:
                    pass

        ranked = sorted(type_counts.items(), key=lambda kv: -kv[1])[:8]
        ranked_str = " ".join(f"{t}={c}" for t, c in ranked)
        logger.info(
            "[DIAG] refs target=%s sample_id=%x n_referrers=%d " "top_types=[%s] examples=[%s]",
            type(sample).__name__,
            sample_id,
            len(referrers) - 1,  # exclude the get_referrers internal frame
            ranked_str,
            ", ".join(sample_reprs),
        )
        # Drop our local references so the probe doesn't itself pin the
        # sample for the next collection cycle.
        del sample, referrers
    except Exception as exc:
        logger.warning("[DIAG] refs probe failed: %s", exc)


class SymbioteGemma4ForCausalLM:
    """vLLM-compatible adapter for Gemma4-31B running on TT hardware via tt_symbiote.

    Implements the four-method contract expected by TTModelLoader / TTModelRunner:
        - initialize_vllm_model (classmethod, called once at startup)
        - prefill_forward (variable-length prompt encoding)
        - decode_forward (single-token autoregressive step)
        - allocate_kv_cache (returns the opaque KV cache object)
    """

    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_async_decode": False,
        "supports_multimodal": False,
    }

    # Sequence lengths primed during warmup. The benchmark sweep uses ISL=128
    # today; 1024 is a forward-looking entry that mirrors the Gemma-3 T3K
    # default returned by get_warmup_prefill_supported_seq_lens().
    WARMUP_PREFILL_SEQ_LENS = (128, 1024)

    # Mirrors Generator.already_warmed_up_prefill so TTModelRunner can
    # reset the flag between Phase 1 (compile) and Phase 2 (trace capture).
    already_warmed_up_prefill = False

    def __init__(self, model, mesh_device, kv_cache, hf_config):
        self.model = model
        self.mesh_device = mesh_device
        self.kv_cache = kv_cache
        self.hf_config = hf_config

    # ------------------------------------------------------------------
    # Tensor conversion: symbiote-wrapped -> host torch.Tensor
    # ------------------------------------------------------------------

    def _to_host_tensor(self, tensor):
        """Convert any tensor variant back to a plain torch.Tensor on host.

        The model forward pass may return:
        - ttnn.Tensor (native device tensor)
        - TorchTTNNTensor (symbiote wrapper, a torch.Tensor subclass with a
          .to_torch *property* that yields the unwrapped torch.Tensor)
        - torch.Tensor (if the dispatcher already converted)
        """
        import ttnn

        if isinstance(tensor, ttnn.Tensor):
            if self.mesh_device.get_num_devices() > 1:
                return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0]).float()
            return ttnn.to_torch(tensor).float()

        if isinstance(tensor, torch.Tensor):
            unwrapped = getattr(tensor, "to_torch", None)
            if unwrapped is not None and isinstance(unwrapped, torch.Tensor):
                return unwrapped.float()
            return tensor.float()

        raise TypeError(
            f"Unexpected logits type {type(tensor).__name__}; " "expected ttnn.Tensor, TorchTTNNTensor, or torch.Tensor"
        )

    # ------------------------------------------------------------------
    # Class method: model loading & weight conversion
    # ------------------------------------------------------------------

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        max_seq_len=4096,
        tt_data_parallel=1,
        optimizations: Optional[str] = None,
    ):
        """Load HF Gemma4 and replace modules with TTNN equivalents.

        Follows the loading and replacement logic from test_gemma4.py: two-pass
        module replacement, weight preprocessing, device transfer, and dual
        paged KV cache allocation. Warmup itself is driven by vLLM through
        warmup_model_prefill / warmup_model_decode below.
        """
        import transformers

        major = int(transformers.__version__.split(".")[0])
        assert major >= 5, f"Gemma4 requires transformers>=5.0.0, found {transformers.__version__}"

        from transformers import AutoModelForCausalLM

        from models.experimental.tt_symbiote.modules.gemma4_attention import (
            TTNNGemma4PagedAttentionKVCache,
        )
        from models.experimental.tt_symbiote.modules.gemma4_modules import (
            TTNNGemma4DecoderLayer,
            TTNNGemma4LMHead,
            TTNNGemma4ScaledEmbedding,
        )
        from models.experimental.tt_symbiote.models.gemma4_text import TTNNGemma4TextModel
        from models.experimental.tt_symbiote.modules.normalization import TTNNDistributedRMSNorm
        from models.experimental.tt_symbiote.utils.device_management import set_device
        from models.experimental.tt_symbiote.utils.module_replacement import register_module_replacement_dict

        model_name = hf_config._name_or_path
        logger.info(f"Loading HF model: {model_name}")

        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16)

        # Capture model_device BEFORE module replacement -- after replacement,
        # TTNN modules lack _parameters so model.parameters() would fail.
        model_device = next(model.parameters()).device

        # Gemma4 is multimodal: model.model is Gemma4Model (wrapper),
        # the actual text model lives at model.model.language_model
        text_model = model.model.language_model
        decoder_class = text_model.layers[0].__class__
        norm_class = text_model.layers[0].input_layernorm.__class__
        embed_class = text_model.embed_tokens.__class__
        text_model_class = text_model.__class__

        # Exclude vision_tower and embed_vision modules from replacement
        # (their norms have incompatible dims for multi-device sharding)
        exclude_vision = {name for name, _ in model.named_modules() if "vision_tower" in name or "embed_vision" in name}

        # Pass 1: decoder layers, norms, embedding, and lm_head
        nn_to_ttnn = {
            decoder_class: TTNNGemma4DecoderLayer,
            norm_class: TTNNDistributedRMSNorm,
            embed_class: TTNNGemma4ScaledEmbedding,
            torch.nn.Linear: TTNNGemma4LMHead,
        }
        modules = register_module_replacement_dict(
            model, nn_to_ttnn, model_config=None, exclude_replacement=exclude_vision
        )

        # Pass 2: text model wrapper (handles input_ids -> embedding on device,
        # iterates layers without ModuleList slicing which breaks TTNNModule)
        nn_to_ttnn_model = {text_model_class: TTNNGemma4TextModel}
        modules.update(register_module_replacement_dict(model, nn_to_ttnn_model, model_config=None))

        set_device(model, mesh_device)

        logger.info(f"Preprocessing {len(modules)} TTNN modules weights...")
        for name, mod in tqdm(modules.items(), desc="Preprocessing & moving weights"):
            mod.preprocess_weights()
            mod.move_weights_to_device()

        # Allocate dual paged-attention KV cache (sliding + global)
        text_config = model.config.text_config
        global_indices = {i for i, lt in enumerate(text_config.layer_types) if lt == "full_attention"}
        kv_cache = TTNNGemma4PagedAttentionKVCache(
            text_config=text_config,
            global_layer_indices=global_indices,
            device=mesh_device,
        )
        kv_cache.to_device(mesh_device)

        model.eval()
        torch.set_grad_enabled(False)

        # Patch model.device so HF resolves the device after TTNN replacement
        # removed all standard torch parameters. Kept even though we no longer
        # call model.generate() here -- prefill_forward / decode_forward still
        # invoke model.forward and downstream code may probe model.device.
        try:
            _ = model.device
        except (AttributeError, StopIteration):
            pass
        type(model).device = property(lambda self: model_device)

        # Warmup is handled by vLLM's TTModelRunner.warmup_model() -> our
        # warmup_model_prefill / warmup_model_decode methods, gated by
        # override_tt_config['enable_model_warmup']. We deliberately do NOT
        # warm up here so the same code paths used at serving time are also
        # the ones primed during startup.
        return cls(model, mesh_device, kv_cache, hf_config)

    # ------------------------------------------------------------------
    # Prefill: variable-length prompt encoding
    # ------------------------------------------------------------------

    def prefill_forward(self, tokens, page_table, kv_cache, prompt_lens, **kwargs):
        """Encode a full prompt sequence and populate the KV cache.

        Args:
            tokens: input token IDs, shape [batch, seq_len]
            page_table: vLLM page table (unused; adapter manages its own KV cache)
            kv_cache: opaque KV cache object (our self.kv_cache, passed back by runner)
            prompt_lens: actual prompt lengths per batch element
            **kwargs: additional TTModelInput fields (absorbed)

        Returns:
            torch.Tensor of shape [batch, seq_len, vocab_size] (float32).
        """
        batch_size = tokens.shape[0]
        seq_len = tokens.shape[1]

        input_ids = tokens.view(batch_size, seq_len)
        attention_mask = torch.ones_like(input_ids)
        position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)

        self.kv_cache.reset()

        diag_t0 = time.perf_counter() if _DIAG_ENABLED else 0.0

        with _NoGrad():
            outputs = self.model.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=self.kv_cache,
                use_cache=True,
            )

        logits = self._to_host_tensor(outputs.logits)

        # Phase 2B: opt-in severance of HF's past_key_values reference. Off by
        # default after the 2026-04-20 hang on prefill #6 inside a TTNN op;
        # we suspect the early drop interacts with the symbiote dispatcher's
        # accumulated tensor state. Re-enable with TT_SYMBIOTE_SEVER_PKV=1
        # to bisect once we have a hardened harness.
        if _SEVER_PKV:
            outputs.past_key_values = None
            del outputs

        if logits.dim() == 2:
            logits = logits.unsqueeze(0)

        if _DIAG_ENABLED:
            _DIAG_STATE["prefill"] += 1
            _DIAG_STATE["decode"] = 0  # reset per-request decode counter
            _diag_log(
                "prefill",
                self.mesh_device,
                (time.perf_counter() - diag_t0) * 1000.0,
                extra=f"isl={seq_len} bs={batch_size}",
            )

        # Phase 2B: bound heap growth across long sweeps. Cyclic GC alone
        # eventually reclaims the wrappers but only after multi-second pauses
        # at high object counts; periodic eager collection keeps each pass
        # cheap (~50-100 ms).
        if _GC_EVERY_N_PREFILLS > 0:
            self._gc_prefill_counter = getattr(self, "_gc_prefill_counter", 0) + 1
            if self._gc_prefill_counter % _GC_EVERY_N_PREFILLS == 0:
                gc_t0 = time.perf_counter() if _DIAG_ENABLED else 0.0
                gc_objs_before = len(gc.get_objects()) if _DIAG_ENABLED else 0
                collected = gc.collect()
                if _DIAG_ENABLED:
                    gc_objs_after = len(gc.get_objects())
                    logger.info(
                        "[DIAG] gc.collect() #%d freed=%d gc_objs_before=%d " "gc_objs_after=%d delta=%d wall_ms=%.1f",
                        self._gc_prefill_counter // _GC_EVERY_N_PREFILLS,
                        collected,
                        gc_objs_before,
                        gc_objs_after,
                        gc_objs_before - gc_objs_after,
                        (time.perf_counter() - gc_t0) * 1000.0,
                    )
                    _diag_log_gc_stats(label=f"post_collect_{self._gc_prefill_counter}")

        # Sampled, gated referrer probe: every N prefills walk the referrer
        # chain of one live wrapper to expose what pins it across requests.
        # O(heap) so default OFF; opt in with TT_SYMBIOTE_DIAG_REFS=1.
        if _DIAG_REFS_ENABLED:
            self._diag_refs_counter = getattr(self, "_diag_refs_counter", 0) + 1
            if self._diag_refs_counter % _DIAG_REFS_EVERY == 0:
                _diag_log_referrers(_DIAG_REFS_TARGET)

        return logits

    # ------------------------------------------------------------------
    # Decode: single-token autoregressive step
    # ------------------------------------------------------------------

    def decode_forward(
        self,
        tokens,
        start_pos,
        page_table,
        kv_cache,
        enable_trace=False,
        read_from_device=True,
        **kwargs,
    ):
        """Generate logits for one decode step.

        Args:
            tokens: current token IDs, shape [batch, 1]
            start_pos: cache position for each sequence in the batch
            page_table: vLLM page table (unused; adapter manages its own KV cache)
            kv_cache: opaque KV cache object
            enable_trace: whether TTNN trace capture is active (accepted but
                currently unused -- warmup already traced in initialize_vllm_model)
            read_from_device: if True, convert output to host torch.Tensor;
                if False, return the device tensor for async pipelines
            **kwargs: additional TTModelInput fields

        Returns:
            torch.Tensor of shape [batch, 1, vocab_size] (float32) when
            read_from_device is True.
        """
        batch_size = tokens.shape[0]
        input_ids = tokens.view(batch_size, 1)

        if isinstance(start_pos, int):
            cache_position = torch.tensor([start_pos], dtype=torch.long)
        else:
            cache_position = start_pos

        # Time every decode_forward when DIAG or watchdog is on; both share t0.
        time_decode = _DIAG_ENABLED or (_WATCHDOG_DECODE_SEC > 0)
        decode_t0 = time.perf_counter() if time_decode else 0.0

        with _NoGrad():
            outputs = self.model.forward(
                input_ids=input_ids,
                past_key_values=self.kv_cache,
                use_cache=True,
                cache_position=cache_position,
            )

        if not read_from_device:
            # Async path: caller (TTAsyncDecodeController) needs the device
            # tensor still alive, so we only sever past_key_values here. The
            # outputs container is released as `outputs` falls out of scope at
            # return; we keep a private reference to logits so it survives.
            logits_dev = outputs.logits
            if _SEVER_PKV:
                outputs.past_key_values = None
                del outputs
            if time_decode:
                self._maybe_emit_decode_diag(decode_t0, batch_size, async_path=True)
            return logits_dev

        logits = self._to_host_tensor(outputs.logits)
        if _SEVER_PKV:
            outputs.past_key_values = None
            del outputs

        if logits.dim() == 2:
            logits = logits.unsqueeze(0)

        if time_decode:
            self._maybe_emit_decode_diag(decode_t0, batch_size, async_path=False)

        return logits

    def _maybe_emit_decode_diag(self, decode_t0, batch_size, async_path):
        """Shared post-decode bookkeeping: DIAG sampling + watchdog tripwire."""
        elapsed = time.perf_counter() - decode_t0
        if _DIAG_ENABLED:
            _DIAG_STATE["decode"] += 1
            if _DIAG_STATE["decode"] % _DIAG_DECODE_EVERY == 0:
                suffix = f"bs={batch_size}" + (" async=1" if async_path else "")
                _diag_log("decode", self.mesh_device, elapsed * 1000.0, extra=suffix)
        if _WATCHDOG_DECODE_SEC > 0 and elapsed > _WATCHDOG_DECODE_SEC:
            logger.warning(
                "[WATCHDOG] decode_forward took %.2fs (limit=%.2fs) "
                "req#=%d decode#=%d bs=%d. Likely candidates: TTNN program-cache "
                "growth, NormalRun heap accumulation, or HF cache retention.",
                elapsed,
                _WATCHDOG_DECODE_SEC,
                _DIAG_STATE["prefill"],
                _DIAG_STATE["decode"],
                batch_size,
            )

    def process_decode_output_host(self, tt_out, is_tokens=False):
        """Post-process decode output on the host side.

        Called by TTAsyncDecodeController.finalize_decode when
        read_from_device was False during decode_forward.
        """
        if isinstance(tt_out, torch.Tensor):
            return tt_out.float()
        return self._to_host_tensor(tt_out)

    # ------------------------------------------------------------------
    # Warmup: driven by TTModelRunner.warmup_model() when
    # override_tt_config['enable_model_warmup'] is True.
    # ------------------------------------------------------------------

    def warmup_model_prefill(
        self,
        enable_trace: bool = False,
        kv_cache=None,
        can_sample_on_device: bool = False,
        non_greedy_decoding_on_device: bool = False,
        **kwargs,
    ):
        """Sweep prefill_forward across WARMUP_PREFILL_SEQ_LENS to populate
        the TTNN program cache for every ISL the benchmark may exercise.

        Args follow the vLLM TTModelRunner.warmup_model contract; kv_cache is
        ignored because this adapter owns its own paged KV cache (allocate_kv_cache
        returns self.kv_cache and runner-allocated buffers are unused).
        """
        if self.already_warmed_up_prefill:
            return
        self.already_warmed_up_prefill = True

        if enable_trace:
            # tt_symbiote runs in NormalRun (eager) mode for this model;
            # trace_mode is pinned to "none" in the model spec so Phase 2
            # should not be invoked. Log defensively if it ever is.
            logger.warning(
                "warmup_model_prefill called with enable_trace=True but adapter "
                "runs in NormalRun mode; ignoring trace capture request."
            )

        max_seq_len = getattr(self.hf_config, "max_position_embeddings", 4096)
        for isl in self.WARMUP_PREFILL_SEQ_LENS:
            if isl > max_seq_len:
                continue
            logger.info(f"Symbiote warmup: prefill at ISL={isl}")
            dummy_tokens = torch.zeros((1, isl), dtype=torch.long)
            self.prefill_forward(
                tokens=dummy_tokens,
                page_table=None,
                kv_cache=self.kv_cache,
                prompt_lens=[isl],
            )
            self.kv_cache.reset()
        logger.info("Symbiote warmup: prefill done.")

    def warmup_model_decode(
        self,
        enable_trace: bool = False,
        read_from_device: bool = True,
        kv_cache=None,
        max_batch_size: int = 1,
        num_blocks: int = 1,
        can_sample_on_device: bool = False,
        non_greedy_decoding_on_device: bool = False,
        **kwargs,
    ):
        """Drive decode_forward at the longest warmed ISL so the program cache
        also covers the decode-time shapes (different from prefill shapes).

        kv_cache and num_blocks are accepted for contract compatibility but
        unused -- this adapter manages its own KV cache pages internally.
        """
        if enable_trace:
            logger.warning(
                "warmup_model_decode called with enable_trace=True but adapter "
                "runs in NormalRun mode; ignoring trace capture request."
            )

        max_seq_len = getattr(self.hf_config, "max_position_embeddings", 4096)
        eligible = [s for s in self.WARMUP_PREFILL_SEQ_LENS if s <= max_seq_len]
        if not eligible:
            logger.warning(
                "warmup_model_decode: no WARMUP_PREFILL_SEQ_LENS entry fits "
                f"max_position_embeddings={max_seq_len}; skipping decode warmup."
            )
            return
        isl = max(eligible)

        # decode_forward expects an existing KV cache state. If Phase 1 prefill
        # already ran in this session it left the cache populated for ISL=1024,
        # but it also called reset() at the end -- so we always re-prime here.
        dummy_tokens = torch.zeros((1, isl), dtype=torch.long)
        self.prefill_forward(
            tokens=dummy_tokens,
            page_table=None,
            kv_cache=self.kv_cache,
            prompt_lens=[isl],
        )

        logger.info(f"Symbiote warmup: decode for {max_batch_size} batch(es), " f"4 steps starting at pos={isl}")
        for step in range(4):
            pos = isl + step
            self.decode_forward(
                tokens=torch.zeros((max_batch_size, 1), dtype=torch.long),
                start_pos=torch.tensor([pos], dtype=torch.long),
                page_table=None,
                kv_cache=self.kv_cache,
                enable_trace=False,
                read_from_device=read_from_device,
            )
        self.kv_cache.reset()
        logger.info("Symbiote warmup: decode done.")

    # ------------------------------------------------------------------
    # KV cache allocation (no-op: adapter owns the cache)
    # ------------------------------------------------------------------

    def allocate_kv_cache(self, kv_cache_shape=None, dtype=None, num_layers=None):
        """Return the pre-allocated dual paged-attention KV cache.

        TTModelRunner calls this once and passes the result into every
        prefill_forward / decode_forward invocation unchanged.
        """
        return self.kv_cache
