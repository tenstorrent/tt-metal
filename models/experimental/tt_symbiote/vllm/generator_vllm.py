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
# One CSV-shaped log line per prefill and one per N decode steps, recording
# wall_ms / rss_mb / gc_objs / progcache. Used to attribute slowdowns to TTNN
# program-cache pollution vs Python heap accumulation vs HF cache retention.
# Default off so production runs are unaffected.
# ---------------------------------------------------------------------------
_DIAG_ENABLED = os.environ.get("TT_SYMBIOTE_DIAG", "0") == "1"
_DIAG_DECODE_EVERY = max(1, int(os.environ.get("TT_SYMBIOTE_DIAG_DECODE_EVERY", "32")))
_DIAG_STATE = {"prefill": 0, "decode": 0}

# Watchdogs: log a WATCHDOG line when a single prefill / decode exceeds the
# configured wall-second budget. Cannot preempt a stuck C++ TTNN op but
# surface the hang clearly in the server log so the operator can correlate
# with the request index. Default 0 = disabled (avoids false positives during
# legitimate first-call compilation).
_WATCHDOG_PREFILL_SEC = float(os.environ.get("TT_SYMBIOTE_PREFILL_WATCHDOG_SEC", "0"))
_WATCHDOG_DECODE_SEC = float(os.environ.get("TT_SYMBIOTE_DECODE_WATCHDOG_SEC", "0"))

# Periodic intra-request decode sync. With async decode (read_from_device=False)
# the TTNN command queue accumulates work across all decode steps of one
# request before the final read drains it. Forcing synchronize_device every
# N decode steps caps the live queue depth and complements the model-internal
# cross-request prefill barrier in tt_symbiote/models/gemma4_text.py. Costs
# ~1 ms per sync; with N=32 that's ~4 syncs per 128-step decode (negligible
# vs ~430 ms/step). Set to 0 to disable.
_SYNC_EVERY_N_DECODES = max(0, int(os.environ.get("TT_SYMBIOTE_SYNC_EVERY_N_DECODES", "0")))


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

        # The cross-request prefill barrier lives in the model wrapper
        # (tt_symbiote/models/gemma4_text.py, gated by
        # TT_SYMBIOTE_GEMMA4_PREFILL_SYNC, ON by default). Adding a second
        # barrier here would just double-sync; trust the model.
        time_prefill = _DIAG_ENABLED or (_WATCHDOG_PREFILL_SEC > 0)
        diag_t0 = time.perf_counter() if time_prefill else 0.0

        with torch.no_grad():
            outputs = self.model.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=self.kv_cache,
                use_cache=True,
            )

        logits = self._to_host_tensor(outputs.logits)

        if logits.dim() == 2:
            logits = logits.unsqueeze(0)

        if time_prefill:
            elapsed = time.perf_counter() - diag_t0
            if _DIAG_ENABLED:
                _DIAG_STATE["prefill"] += 1
                _DIAG_STATE["decode"] = 0  # reset per-request decode counter
                _diag_log(
                    "prefill",
                    self.mesh_device,
                    elapsed * 1000.0,
                    extra=f"isl={seq_len} bs={batch_size}",
                )
            if _WATCHDOG_PREFILL_SEC > 0 and elapsed > _WATCHDOG_PREFILL_SEC:
                logger.warning(
                    "[WATCHDOG] prefill_forward took %.2fs (limit=%.2fs) "
                    "req#=%d isl=%d bs=%d. Most likely candidates: stale TTNN "
                    "command-queue work from a prior async decode (verify "
                    "TT_SYMBIOTE_GEMMA4_PREFILL_SYNC=1) or a hang inside "
                    "tt_symbiote gemma4_attention prefill kernels.",
                    elapsed,
                    _WATCHDOG_PREFILL_SEC,
                    _DIAG_STATE["prefill"],
                    seq_len,
                    batch_size,
                )

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

        # Time every decode_forward when DIAG, watchdog, or periodic sync is on.
        time_decode = _DIAG_ENABLED or (_WATCHDOG_DECODE_SEC > 0) or (_SYNC_EVERY_N_DECODES > 0)
        decode_t0 = time.perf_counter() if time_decode else 0.0

        with torch.no_grad():
            outputs = self.model.forward(
                input_ids=input_ids,
                past_key_values=self.kv_cache,
                use_cache=True,
                cache_position=cache_position,
            )

        if not read_from_device:
            # Async path: caller (TTAsyncDecodeController) needs the device
            # tensor still alive; keep a private reference to logits so it
            # survives once `outputs` falls out of scope at return.
            logits_dev = outputs.logits
            if time_decode:
                self._maybe_emit_decode_diag(decode_t0, batch_size, async_path=True)
            return logits_dev

        logits = self._to_host_tensor(outputs.logits)

        if logits.dim() == 2:
            logits = logits.unsqueeze(0)

        if time_decode:
            self._maybe_emit_decode_diag(decode_t0, batch_size, async_path=False)

        return logits

    def _maybe_emit_decode_diag(self, decode_t0, batch_size, async_path):
        """Shared post-decode bookkeeping: DIAG sampling + watchdog tripwire +
        periodic device sync."""
        elapsed = time.perf_counter() - decode_t0
        if _DIAG_ENABLED:
            _DIAG_STATE["decode"] += 1
            if _DIAG_STATE["decode"] % _DIAG_DECODE_EVERY == 0:
                suffix = f"bs={batch_size}" + (" async=1" if async_path else "")
                _diag_log("decode", self.mesh_device, elapsed * 1000.0, extra=suffix)
        else:
            _DIAG_STATE["decode"] += 1

        # Periodic intra-request sync: cap TTNN command-queue depth during
        # async decode bursts so the next prefill never collides with hundreds
        # of in-flight ops. Off when _SYNC_EVERY_N_DECODES==0.
        if _SYNC_EVERY_N_DECODES > 0 and _DIAG_STATE["decode"] % _SYNC_EVERY_N_DECODES == 0:
            sync_t0 = time.perf_counter() if _DIAG_ENABLED else 0.0
            try:
                import ttnn

                ttnn.synchronize_device(self.mesh_device)
            except Exception as exc:
                logger.warning("ttnn.synchronize_device (decode) failed: %s", exc)
            if _DIAG_ENABLED:
                logger.info(
                    "[DIAG] sync_during_decode decode#=%d wall_ms=%.1f",
                    _DIAG_STATE["decode"],
                    (time.perf_counter() - sync_t0) * 1000.0,
                )

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
