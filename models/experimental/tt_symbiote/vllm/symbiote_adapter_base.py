# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared base class for tt_symbiote vLLM adapters.

Factors out the boilerplate every per-model adapter needs (DIAG instrumentation,
watchdog, periodic decode-stride sync, _to_host_tensor, prefill_forward and
decode_forward bodies, the two warmup loops, and the KV-cache hand-off). A
concrete adapter subclass typically becomes ~50-100 lines: the only required
overrides are MODEL_KEY (used in watchdog log strings), WARMUP_PREFILL_SEQ_LENS
(class attribute), and _build_model_and_kv_cache (classmethod that loads the
HF model, runs the model-specific TTNN module replacement, and allocates the
paged KV cache).

See models/experimental/tt_symbiote/vllm/generator_vllm.py (Gemma-4) and
generator_vllm_ling.py (Ling-mini-2.0) for canonical subclass examples.

The base intentionally does NOT centralise transformers compatibility shims
(e.g. is_torch_fx_available / ROPE_INIT_FUNCTIONS['default']). Those need to
be applied at module-import time of the adapter file, before HF dynamic-import
chains fire, so they live in the per-model adapter file.
"""

from __future__ import annotations

import gc
import logging
import os
import resource
import time
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level DIAG state and helpers.
#
# State is module-level (not class-level) so it is shared across subclasses;
# in practice a single vLLM process serves a single model so the lack of
# per-model isolation is intentional. A single counter pair makes the DIAG
# request index match the operator-visible request index.
# ---------------------------------------------------------------------------
_DIAG_ENABLED = os.environ.get("TT_SYMBIOTE_DIAG", "0") == "1"
_DIAG_DECODE_EVERY = max(1, int(os.environ.get("TT_SYMBIOTE_DIAG_DECODE_EVERY", "32")))
_DIAG_STATE = {"prefill": 0, "decode": 0}

# Watchdogs: log a [WATCHDOG] line when a single prefill / decode exceeds the
# configured wall-second budget. Cannot preempt a stuck C++ TTNN op but
# surfaces the hang clearly in the server log so the operator can correlate
# with the request index. Default 0 = disabled (avoids false positives during
# legitimate first-call compilation).
_WATCHDOG_PREFILL_SEC = float(os.environ.get("TT_SYMBIOTE_PREFILL_WATCHDOG_SEC", "0"))
_WATCHDOG_DECODE_SEC = float(os.environ.get("TT_SYMBIOTE_DECODE_WATCHDOG_SEC", "0"))

# Periodic intra-request decode sync. With async decode (read_from_device=False)
# the TTNN command queue accumulates work across all decode steps of one
# request before the final read drains it. Forcing synchronize_device every
# N decode steps caps the live queue depth and complements the model-internal
# cross-request prefill barrier (see e.g. tt_symbiote/models/gemma4_text.py
# or tt_symbiote/models/bailing_moe_v2.py). Costs ~1 ms per sync; with N=32
# that's ~4 syncs per 128-step decode (negligible vs ~430 ms/step). Set to 0
# to disable.
_SYNC_EVERY_N_DECODES = max(0, int(os.environ.get("TT_SYMBIOTE_SYNC_EVERY_N_DECODES", "0")))


def _diag_progcache_entries(mesh_device) -> int:
    """Best-effort count of TTNN program-cache entries.

    Uses the public MeshDevice.num_program_cache_entries() binding from
    ttnn/core/distributed/distributed_nanobind.cpp. Returns -1 if the
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


class SymbioteAdapterBase:
    """vLLM-compatible adapter base for tt_symbiote models on TT hardware.

    Implements the four-method contract expected by TTModelLoader / TTModelRunner:
        - initialize_vllm_model (classmethod, called once at startup)
        - prefill_forward (variable-length prompt encoding)
        - decode_forward (single-token autoregressive step)
        - allocate_kv_cache (returns the opaque KV cache object)

    Plus the two warmup hooks the runner invokes when override_tt_config
    enables warmup:
        - warmup_model_prefill
        - warmup_model_decode
    """

    # Default capabilities. Subclasses can override.
    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_async_decode": False,
        "supports_multimodal": False,
    }

    # ---- Subclass overrides ----

    # Used to format watchdog log strings as TT_SYMBIOTE_<MODEL_KEY>_PREFILL_SYNC.
    # Subclasses MUST override this so the log message guides the reader to the
    # right env-var name.
    MODEL_KEY: str = "MODEL"

    # Human-readable name of the model's TTNN prefill-attention kernel module,
    # cited in the [WATCHDOG] prefill log line so the operator can immediately
    # `grep` for the right tt_symbiote source file when triaging a hang.
    # Examples: "gemma4_attention" (gemma4-31B), "bailing attention" (Ling).
    WATCHDOG_PREFILL_KERNEL_HINT: str = "model attention"

    # Sequence lengths primed during warmup. Subclasses MUST cover every ISL
    # the benchmark sweep exercises against the spec's max_context cap. Values
    # exceeding hf_config.max_position_embeddings are filtered at runtime in
    # warmup_model_prefill.
    WARMUP_PREFILL_SEQ_LENS: Tuple[int, ...] = (128,)

    # Minimum required transformers major version. The base asserts this in
    # initialize_vllm_model so a stale environment fails fast rather than
    # surfacing as a cryptic missing-attribute error inside HF custom code.
    REQUIRED_TRANSFORMERS_MAJOR: int = 5

    # Mirrors Generator.already_warmed_up_prefill so TTModelRunner can
    # reset the flag between Phase 1 (compile) and Phase 2 (trace capture).
    already_warmed_up_prefill = False

    # ---- Construction ----

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
        max_seq_len: int = 4096,
        tt_data_parallel: int = 1,
        optimizations: Optional[str] = None,
        **kwargs,
    ):
        """Load HF model, replace modules with TTNN equivalents, allocate KV cache.

        The model-specific load + replacement + KV-cache build happens inside
        the subclass's _build_model_and_kv_cache classmethod. The base runs:

        1. transformers major-version assert.
        2. Subclass _build_model_and_kv_cache call.
        3. model.eval() + torch.set_grad_enabled(False).
        4. type(model).device = property(lambda self: model_device) so HF code
           that probes model.device still works after replacement removed all
           standard torch parameters.

        Warmup itself is driven by vLLM's TTModelRunner.warmup_model() ->
        warmup_model_prefill / warmup_model_decode below, gated by
        override_tt_config['enable_model_warmup']. We deliberately do NOT
        warm up here so the same code paths used at serving time are also
        the ones primed during startup.
        """
        import transformers

        major = int(transformers.__version__.split(".")[0])
        assert major >= cls.REQUIRED_TRANSFORMERS_MAJOR, (
            f"{cls.__name__} requires transformers>={cls.REQUIRED_TRANSFORMERS_MAJOR}.0.0, "
            f"found {transformers.__version__}"
        )

        model_name = getattr(hf_config, "_name_or_path", "<unknown>")
        logger.info(f"Loading HF model: {model_name}")

        model, kv_cache, model_device = cls._build_model_and_kv_cache(
            hf_config=hf_config,
            mesh_device=mesh_device,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            tt_data_parallel=tt_data_parallel,
            optimizations=optimizations,
            **kwargs,
        )

        model.eval()
        torch.set_grad_enabled(False)

        # Patch model.device so HF code that resolves the device after TTNN
        # replacement removed all standard torch parameters still works. The
        # subclass is responsible for any temporary device-property patches
        # needed during set_device (e.g. Ling's brief "cpu" patch); this
        # patch is the final, authoritative one.
        try:
            _ = model.device
        except (AttributeError, StopIteration):
            pass
        type(model).device = property(lambda self: model_device)

        return cls(model, mesh_device, kv_cache, hf_config)

    @classmethod
    def _build_model_and_kv_cache(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        max_seq_len,
        tt_data_parallel,
        optimizations,
        **kwargs,
    ):
        """Subclass hook: load HF model, replace modules, allocate KV cache.

        Returns:
            (model, kv_cache, model_device): model_device is the host torch
            device captured BEFORE module replacement (used by the base to
            patch model.device after replacement).
        """
        raise NotImplementedError(f"{cls.__name__} must implement _build_model_and_kv_cache")

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
        # (e.g. tt_symbiote/models/gemma4_text.py or bailing_moe_v2.py),
        # gated by TT_SYMBIOTE_<MODEL_KEY>_PREFILL_SYNC (ON by default).
        # Adding a second barrier here would just double-sync.
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
                    "req#=%d isl=%d bs=%d. Likely candidates: stale TTNN "
                    "command-queue work from a prior async decode (verify "
                    "TT_SYMBIOTE_%s_PREFILL_SYNC=1) or a hang inside "
                    "tt_symbiote %s prefill kernels.",
                    elapsed,
                    _WATCHDOG_PREFILL_SEC,
                    _DIAG_STATE["prefill"],
                    seq_len,
                    batch_size,
                    self.MODEL_KEY,
                    self.WATCHDOG_PREFILL_KERNEL_HINT,
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
        _DIAG_STATE["decode"] += 1
        if _DIAG_ENABLED and _DIAG_STATE["decode"] % _DIAG_DECODE_EVERY == 0:
            suffix = f"bs={batch_size}" + (" async=1" if async_path else "")
            _diag_log("decode", self.mesh_device, elapsed * 1000.0, extra=suffix)

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
        """Post-process decode output on host. Called by the async controller
        when read_from_device was False during decode_forward."""
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

        kv_cache is ignored because this adapter owns its own paged KV cache
        (allocate_kv_cache returns self.kv_cache).
        """
        if self.already_warmed_up_prefill:
            return
        self.already_warmed_up_prefill = True

        if enable_trace:
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
        also covers decode-time shapes (different from prefill shapes).

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

        # decode_forward expects an existing KV cache state. Re-prime here so
        # warmup_model_decode can be called independently of warmup_model_prefill.
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
        """Return the pre-allocated paged-attention KV cache.

        TTModelRunner calls this once and passes the result into every
        prefill_forward / decode_forward invocation unchanged.
        """
        return self.kv_cache


__all__ = ["SymbioteAdapterBase"]
