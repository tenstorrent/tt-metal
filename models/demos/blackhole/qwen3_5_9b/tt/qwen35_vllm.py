# tt/qwen35_vllm.py
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Local vLLM wrapper for Qwen3.5-9B: a thin tt_transformers Generator subclass.

Qwen3.5-9B is a hybrid model: 8 full-attention layers (paged KV, stateless across prefill) plus
24 Gated DeltaNet (GDN) layers carrying a recurrent + conv state that accumulates across the whole
sequence. Standard tt_transformers models are stateless beyond paged KV, so the standard contract
assumes token-padding is numerically free and the decode trace is position-general. Neither holds
for GDN — which is the root of every place this model must diverge.

What conforms to the standard Generator (Llama/DeepSeek/Qwen-VL):
  - Decode, end to end. The model implements the standard decode contract (prepare_inputs_decode /
    ttnn_decode_forward / process_output_decode); current_pos and page_table are device input
    tensors the standard replay updates per step. The inherited WarmupForwardMixin captures the
    decode trace at position 0 during warmup; Generator.decode_forward replays it at serving.

What must diverge, and why:
  - Prefill bucketing. GDN forbids token-padding inside its recurrent scan, so prefill pads to a
    fixed bucket and passes an EXACT valid_len (the standard contract only plumbs get_last_token,
    floored to a 32-multiple — too lossy for the GDN mask). See model.prefill_masked_bucket.
  - Chunk-outer trace. At 128K a whole-length prefill trace is infeasible; we capture ONE
    2048-token chunk trace and replay it N times, carrying GDN/KV state in place. See
    model.prefill_traced_chunked / capture_prefill_trace_chunked.
  - State-reset guard. The stock trace capture runs the forward twice (compile + capture), which
    advances GDN state non-idempotently. Harmless only because every new sequence re-zeros the
    bound GDN buffers before consuming a token (model._reset_gdn_state_for_new_sequence).

Generator drives decode; all prefill is model-owned (prefill_masked_bucket / prefill_traced_chunked)
via prefill_dispatch. GDN recurrent state and the attention KV caches are model-bound, so the
kv_cache contract param is accepted but unused.
"""
import math
import os

import torch

import ttnn
from models.demos.blackhole.qwen3_5_9b.tt.common import create_tt_model
from models.demos.blackhole.qwen3_5_9b.tt.generator_interface import prefill_dispatch
from models.tt_transformers.tt.generator import Generator

_PREFILL_WARMUP_CHUNK = 2048
_PREFILL_WARMUP_BUCKET = 4096
_BLOCK_SIZE = 64


class Qwen35ForCausalLM(Generator):
    """vLLM-compatible wrapper for Qwen3.5-9B on Blackhole P150."""

    model_capabilities = {"supports_prefix_caching": False, "supports_async_decode": False}

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        max_seq_len,
        tt_data_parallel=1,
        optimizations=None,
        **kwargs,
    ):
        # HF_MODEL is the single source of truth; vLLM passes the hub name / local path
        # via hf_config._name_or_path. Resolve to a local dir (download if needed).
        name_or_path = hf_config._name_or_path
        if name_or_path and not os.path.isdir(name_or_path):
            from huggingface_hub import snapshot_download

            name_or_path = snapshot_download(name_or_path)
        args, model, _ = create_tt_model(
            mesh_device, max_batch_size=max_batch_size, max_seq_len=max_seq_len, hf_model=name_or_path
        )
        return cls([model], [args], mesh_device)

    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        """Allocate paged KV (8 attn layers) + external GDN state; returns the 8 KV pairs."""
        return self.model[0].allocate_kv_caches(kv_cache_shape, ttnn.bfloat16, batch_size=1)

    def prefill_forward(self, tokens, page_table, kv_cache, prompt_lens, **kwargs):
        """All prefill is model-owned (Generator drives decode only)."""
        logits = prefill_dispatch(
            self.model[0], tokens, page_table, prompt_lens, use_trace=kwargs.get("enable_trace", False)
        )
        logits = ttnn.to_torch(logits)
        # The vLLM runner unpacks (logits, rope_deltas) because the HF config has mrope_section;
        # zero deltas are correct for our text-only port.
        rope_deltas = torch.zeros(logits.shape[0], dtype=torch.long)
        return logits, rope_deltas

    def decode_forward(self, *args, **kwargs):
        # Standard path (default): the decode trace is captured at position 0 during warmup by the
        # inherited WarmupForwardMixin, then replayed here by Generator.decode_forward — identical
        # to Llama/DeepSeek/Qwen-VL. current_pos and page_table are device input tensors the
        # standard replay updates per step, so a pos-0 capture is position-general.
        #
        # Dormant fallback (QWEN35_DECODE_PRIME=1, paired with the warmup_model_decode no-op): if
        # the pos-0 warmup capture ever proves insufficient for GDN, lazily capture the decode
        # trace on the FIRST decode, at the REAL post-prefill position and recurrent state, via
        # prime_decode_trace (GDN-state snapshot/restore around the stock two-pass capture). This
        # should not be needed — every new sequence re-zeros the GDN state at prefill
        # (model._reset_gdn_state_for_new_sequence), so the warmup capture's residue can't leak in.
        if os.environ.get("QWEN35_DECODE_PRIME") == "1" and not getattr(self, "_decode_trace_primed", False):
            from models.demos.blackhole.qwen3_5_9b.tt.generator_interface import prime_decode_trace

            # Set the guard BEFORE calling so the re-entrant decode_forward it triggers skips
            # priming and just performs the capture.
            self._decode_trace_primed = True
            tokens = args[0] if args else kwargs.get("tokens")
            start_pos = args[1] if len(args) > 1 else kwargs.get("start_pos")
            prime_decode_trace(self, self.model[0], tokens, start_pos, kwargs.get("page_table"))

        return super().decode_forward(*args, **kwargs)

    def warmup_model_prefill(self, kv_cache, enable_trace, *args, **kwargs):
        # The plugin's warmup_model() is two-phase: it calls this first with
        # enable_trace=False (compile), then resets ``already_warmed_up_prefill``
        # and calls again with enable_trace=True (capture). Only the traced phase
        # captures the chunk-prefill trace; capture_prefill_trace_chunked compiles
        # its own programs before capturing, so the non-traced phase is a no-op.
        # The guard attribute MUST be named ``already_warmed_up_prefill`` so the
        # plugin's between-phase reset (model_runner.warmup_model) clears it.
        if not enable_trace:
            return
        if getattr(self, "already_warmed_up_prefill", False):
            return
        self.already_warmed_up_prefill = True
        # Size the captured chunk-trace page table to the FULL allocated KV cache
        # (max_model_len worth of blocks), so served ISL matches the tt-metal demo's
        # 128K — not a hardcoded 4096. The chunk-outer trace still captures only one
        # _PREFILL_WARMUP_CHUNK-token chunk, so this is just a larger page-table
        # tensor, not more compute/trace memory. kv_cache[0][0] is the first attention
        # layer's K cache, shape [max_num_blocks, n_kv_heads, block_size, head_dim].
        if kv_cache:
            # Round up to a multiple of 32: the paged/chunked SDPA requires the page-table
            # width (stick size) to be % 32 == 0 (the allocated block count carries a slack
            # block, e.g. 257, which is not 32-aligned). prefill_traced_chunked pads each
            # request's page table up to this width before replay.
            num_blocks = math.ceil(int(kv_cache[0][0].shape[0]) / 32) * 32
        else:
            num_blocks = math.ceil(_PREFILL_WARMUP_BUCKET / _BLOCK_SIZE)
        page_table = torch.arange(num_blocks, dtype=torch.int32).reshape(1, num_blocks)
        self.model[0].capture_prefill_trace_chunked(self.mesh_device, page_table, chunk_size=_PREFILL_WARMUP_CHUNK)

    def warmup_model_decode(self, *args, **kwargs):
        # Standard path (default): defer to the inherited WarmupForwardMixin, which captures the
        # paged-SDPA + GDN decode trace at position 0 during warmup. Qwen sets
        # _supports_on_device_sampling=False, so the orchestrator passes can_sample_on_device=False
        # and exactly one greedy trace is captured; serving replays it with per-step input updates.
        #
        # Dormant fallback (QWEN35_DECODE_PRIME=1): skip the warmup capture entirely; decode_forward
        # primes the trace lazily at the real post-prefill position instead. See decode_forward.
        if os.environ.get("QWEN35_DECODE_PRIME") == "1":
            return
        return super().warmup_model_decode(*args, **kwargs)
