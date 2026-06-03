# tt/qwen35_vllm.py
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Local vLLM wrapper for Qwen3.5-9B: a thin tt_transformers Generator subclass.

Generator drives decode; all prefill is model-owned (prefill_paged / prefill_traced_chunked)
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
        num_blocks = math.ceil(_PREFILL_WARMUP_BUCKET / _BLOCK_SIZE)
        page_table = torch.arange(num_blocks, dtype=torch.int32).reshape(1, num_blocks)
        self.model[0].capture_prefill_trace_chunked(self.mesh_device, page_table, chunk_size=_PREFILL_WARMUP_CHUNK)

    def warmup_model_decode(self, kv_cache, enable_trace, max_batch_size, num_blocks, *args, **kwargs):
        if not enable_trace:
            return
        dummy = torch.zeros(1, 1, dtype=torch.long)
        start = torch.zeros(1, dtype=torch.int64)
        pt = torch.arange(num_blocks, dtype=torch.int32).reshape(1, num_blocks)
        self.decode_forward(dummy, start, page_table=pt, kv_cache=kv_cache, enable_trace=True, read_from_device=False)
