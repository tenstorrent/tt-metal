# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""vLLM adapter for Ling-mini-2.0 (BailingMoeV2) via tt_symbiote.

Bridges vLLM's serving interface (initialize_vllm_model, prefill_forward,
decode_forward, allocate_kv_cache) to the HF BailingMoeV2 model whose
decoder layers, norm, embedding, rotary embedding, linear projections,
SiLU, and text-model wrapper have been replaced with TTNN equivalents.

The module replacement pattern follows tests/test_ling_mini_2_0.py.
The adapter shape mirrors generator_vllm.py (Gemma-4 sibling); differences
(single paged KV cache, no vision tower, three-pass replacement) are noted
inline.
"""

import gc
import logging
import os
import resource
import time
from math import ceil
from typing import Optional

import torch
from torch import nn
from tqdm import tqdm

# Transformers 5.x compatibility shims for Ling-mini-2.0's HF custom code.
# These must be applied before from_pretrained triggers the dynamic import.

# (1) is_torch_fx_available was removed in transformers 5.x. Returning False
#     skips the torch.fx wrapping which is a tracing-only optimisation.
import transformers.utils.import_utils as _tui

if not hasattr(_tui, "is_torch_fx_available"):
    _tui.is_torch_fx_available = lambda: False

# (2) ROPE_INIT_FUNCTIONS['default'] was dropped in transformers 5.x. The HF
#     custom rotary embedding sets rope_type='default' when rope_scaling is
#     absent. We add the key back with the original plain inv-freq formula
#     (base ** (-2i/dim)). The HF rotary emb is replaced by
#     TTNNBailingRotaryEmbedding anyway, so this only needs to survive __init__.
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS as _ROPE_INIT

if "default" not in _ROPE_INIT:

    def _default_rope_init(config, device=None, seq_len=None, **kwargs):
        base = getattr(config, "rope_theta", 10000.0)
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
        head_dim = getattr(config, "head_dim", None) or (config.hidden_size // config.num_attention_heads)
        dim = int(head_dim * partial_rotary_factor)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64, device=device).float() / dim))
        return inv_freq, 1.0  # attention_factor unused for default RoPE

    _ROPE_INIT["default"] = _default_rope_init

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
# with the request index. Default 0 = disabled.
_WATCHDOG_PREFILL_SEC = float(os.environ.get("TT_SYMBIOTE_PREFILL_WATCHDOG_SEC", "0"))
_WATCHDOG_DECODE_SEC = float(os.environ.get("TT_SYMBIOTE_DECODE_WATCHDOG_SEC", "0"))

# Periodic intra-request decode sync. With async decode (read_from_device=False)
# the TTNN command queue accumulates work across all decode steps of one
# request before the final read drains it. Forcing synchronize_device every
# N decode steps caps the live queue depth and complements the model-internal
# cross-request prefill barrier in tt_symbiote/models/bailing_moe_v2.py.
# Set to 0 to disable.
_SYNC_EVERY_N_DECODES = max(0, int(os.environ.get("TT_SYMBIOTE_SYNC_EVERY_N_DECODES", "0")))


def _diag_progcache_entries(mesh_device) -> int:
    """Best-effort count of TTNN program-cache entries."""
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


class SymbioteBailingMoeV2ForCausalLM:
    """vLLM-compatible adapter for Ling-mini-2.0 (BailingMoeV2) on TT hardware.

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

    # Sequence lengths primed during warmup. Covers every ISL the
    # benchmark sweep exercises against the spec's max_context=2048 cap:
    # rows (128,128), (128,1024), (1024,128), (2048,128) all hit a warmed
    # program-cache bucket so first-request TTFT does not pay JIT compile
    # cost. Values <= max_position_embeddings are filtered at runtime in
    # warmup_model_prefill. If max_context is raised in the future, extend
    # this tuple to match (e.g. add 3072 for max_context=3072).
    WARMUP_PREFILL_SEQ_LENS = (128, 1024, 2048)

    # Mirrors Generator.already_warmed_up_prefill so TTModelRunner can reset
    # the flag between Phase 1 (compile) and Phase 2 (trace capture).
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
        """Convert any tensor variant back to a plain torch.Tensor on host."""
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
        max_seq_len: int = 8192,
        tt_data_parallel: int = 1,
        optimizations: Optional[str] = None,
    ):
        """Load HF BailingMoeV2 and replace modules with TTNN equivalents.

        Mirrors the loading and replacement logic from test_ling_mini_2_0.py:
        three-pass module replacement, weight preprocessing, device transfer,
        and a single paged KV cache allocation. Warmup itself is driven by
        vLLM through warmup_model_prefill / warmup_model_decode below.
        """
        from transformers import AutoModelForCausalLM

        from models.experimental.tt_symbiote.models.bailing_moe_v2 import (
            TTNNBailingMoeV2Model,
        )
        from models.experimental.tt_symbiote.modules.activation import TTNNSilu
        from models.experimental.tt_symbiote.modules.attention import (
            PagedAttentionConfig,
            TTNNPagedAttentionKVCache,
        )
        from models.experimental.tt_symbiote.modules.decoder_layer import (
            TTNNBailingMoEDecoderLayerPadded,
        )
        from models.experimental.tt_symbiote.modules.embedding import (
            TTNNBailingPaddedEmbedding,
            TTNNBailingRotaryEmbedding,
        )
        from models.experimental.tt_symbiote.modules.linear import (
            TTNNLinearIColShardedWRowSharded,
        )
        from models.experimental.tt_symbiote.modules.normalization import (
            TTNNDistributedRMSNorm,
        )
        from models.experimental.tt_symbiote.utils.device_management import set_device
        from models.experimental.tt_symbiote.utils.module_replacement import (
            register_module_replacement_dict,
        )

        model_name = hf_config._name_or_path
        logger.info(f"Loading HF model: {model_name}")

        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16)

        # Capture model_device BEFORE module replacement -- after replacement,
        # TTNN modules lack _parameters so model.parameters() would fail.
        model_device = next(model.parameters()).device

        # BailingMoeV2 is text-only: the BailingMoeV2Model lives at model.model.
        text_model = model.model
        decoder_class = text_model.layers[0].__class__
        norm_class = text_model.norm.__class__
        rotary_class = text_model.rotary_emb.__class__
        text_model_class = text_model.__class__

        # Three-pass replacement (mirrors test_ling_mini_2_0.py:87-99). The
        # passes are sequenced so per-instance class replacements happen
        # before the blanket nn.Linear / nn.SiLU sweep, and the text-model
        # wrapper replacement runs last so its from_torch sees its children
        # already replaced.
        nn_to_ttnn = {
            decoder_class: TTNNBailingMoEDecoderLayerPadded,
            norm_class: TTNNDistributedRMSNorm,
            nn.Embedding: TTNNBailingPaddedEmbedding,
            rotary_class: TTNNBailingRotaryEmbedding,
        }
        nn_to_ttnn2 = {
            nn.Linear: TTNNLinearIColShardedWRowSharded,
            nn.SiLU: TTNNSilu,
        }
        nn_to_ttnn_3 = {text_model_class: TTNNBailingMoeV2Model}

        modules1 = register_module_replacement_dict(model, nn_to_ttnn, model_config=None)
        modules2 = register_module_replacement_dict(model, nn_to_ttnn2, model_config=None)
        modules3 = register_module_replacement_dict(model, nn_to_ttnn_3, model_config=None)
        all_modules = {**modules1, **modules2, **modules3}

        # After replacing all nn.Modules with TTNNModules, HF's model.device
        # (which calls next(self.parameters())) fails. Patch it to return cpu
        # while set_device runs; restored to the captured host device below.
        type(model).device = property(lambda self: torch.device("cpu"))

        set_device(model, mesh_device)

        logger.info(f"Preprocessing {len(all_modules)} TTNN modules weights...")
        for name, mod in tqdm(all_modules.items(), desc="Preprocessing & moving weights"):
            mod.preprocess_weights()
            mod.move_weights_to_device()

        # Allocate a single paged-attention KV cache sized to the configured
        # max_seq_len. block_size matches vllm_args["block_size"]=64 in the
        # ModelSpecTemplate; max_num_blocks covers max_seq_len * max_batch_size.
        block_size = 64
        max_num_blocks = max(1, ceil(max_seq_len / block_size)) * max(1, max_batch_size)
        kv_cache = TTNNPagedAttentionKVCache(
            num_layers=hf_config.num_hidden_layers,
            num_kv_heads=hf_config.num_key_value_heads,
            head_dim=hf_config.head_dim,
            config=PagedAttentionConfig(
                block_size=block_size,
                max_num_blocks=max_num_blocks,
                batch_size=max_batch_size,
            ),
            device=None,
        ).to_device(mesh_device)

        model.eval()
        torch.set_grad_enabled(False)

        # Restore the HF-side device probe to the original (host) device
        # captured before replacement.
        try:
            _ = model.device
        except (AttributeError, StopIteration):
            pass
        type(model).device = property(lambda self: model_device)

        return cls(model, mesh_device, kv_cache, hf_config)

    # ------------------------------------------------------------------
    # Prefill: variable-length prompt encoding
    # ------------------------------------------------------------------

    def prefill_forward(self, tokens, page_table, kv_cache, prompt_lens, **kwargs):
        """Encode a full prompt sequence and populate the KV cache.

        Args:
            tokens: input token IDs, shape [batch, seq_len]
            page_table: vLLM page table (unused; adapter manages its own KV cache)
            kv_cache: opaque KV cache object (passed back unchanged from allocate_kv_cache)
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
        # (tt_symbiote/models/bailing_moe_v2.py, gated by
        # TT_SYMBIOTE_LING_PREFILL_SYNC, ON by default). Adding a second
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
                _DIAG_STATE["decode"] = 0
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
                    "TT_SYMBIOTE_LING_PREFILL_SYNC=1) or a hang inside "
                    "tt_symbiote bailing attention prefill kernels.",
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
                currently unused)
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
                "[WATCHDOG] decode_forward took %.2fs (limit=%.2fs) " "req#=%d decode#=%d bs=%d.",
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

        max_seq_len = getattr(self.hf_config, "max_position_embeddings", 8192)
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

        max_seq_len = getattr(self.hf_config, "max_position_embeddings", 8192)
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
