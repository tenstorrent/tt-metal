# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""vLLM wrapper for Qwen3.5-9B on Blackhole P150.

Implements the methods required by the vLLM TT plugin:
- initialize_vllm_model (classmethod)
- allocate_kv_cache
- prefill_forward
- decode_forward
- warmup_model_prefill / warmup_model_decode (used by model_runner.warmup_model)
"""
import math
import os

import torch
import torch.nn as nn

import ttnn
from models.demos.blackhole.qwen3_5_9b.tt.model import Qwen35Model

_PREFILL_WARMUP_BUCKET = 4096
_PREFILL_WARMUP_CHUNK = 2048
_BLOCK_SIZE = 64


class TTQwen35ForCausalLM(nn.Module):
    """vLLM-compatible wrapper for Qwen3.5-9B on Blackhole P150."""

    model_capabilities = {
        "supports_async_decode": False,
        "supports_prefix_caching": False,
    }

    def __init__(self, model: Qwen35Model, device, **kwargs):
        # Accept **kwargs so vllm's duck-type protocol check `supports_kw(__init__, "vllm_config")`
        # passes; the TT execution path constructs us via initialize_vllm_model, not vllm_config.
        super().__init__()
        self.model = model
        self.device = device
        self.attention_layer_indices = [i for i in range(model.args.n_layers) if model.args.is_full_attention_layer(i)]
        self.num_attention_layers = len(self.attention_layer_indices)
        self.already_warmed_up_prefill = False

    # ---- vLLM protocol stubs ------------------------------------------------
    # vLLM's `is_text_generation_model` (interfaces_base.py) probes for these
    # via runtime_checkable Protocol. The TT plugin never calls them — the
    # runner dispatches through prefill_forward/decode_forward — but they must
    # exist with the right signatures for the protocol check to pass.

    def embed_input_ids(self, input_ids):
        raise NotImplementedError("TT path uses prefill_forward/decode_forward")

    def forward(self, input_ids, positions, **kwargs):
        raise NotImplementedError("TT path uses prefill_forward/decode_forward")

    def compute_logits(self, hidden_states):
        raise NotImplementedError("TT path returns logits directly from prefill/decode")

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
        """Factory method called by TTModelLoader."""
        # HF_MODEL is the single source of truth. vLLM passes the hub name / local
        # path via hf_config._name_or_path; resolve a local dir (download if needed)
        # and feed it through from_pretrained's hf_model kwarg, which sets HF_MODEL.
        name_or_path = hf_config._name_or_path
        if name_or_path and os.path.isdir(name_or_path):
            checkpoint_dir = name_or_path
        else:
            from huggingface_hub import snapshot_download

            checkpoint_dir = snapshot_download(name_or_path)
        model = Qwen35Model.from_pretrained(
            device=mesh_device,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            hf_model=checkpoint_dir,
        )
        return cls(model, mesh_device)

    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        """Allocate paged KV caches for the 8 attention layers + DeltaNet external state.

        Delegates to Qwen35Model.allocate_kv_caches so DeltaNet recurrent/conv state
        buffers are allocated alongside the paged KV caches (required by the trace
        capture paths in model.py).

        Args:
            kv_cache_shape: (num_blocks, num_kv_heads, block_size, head_dim)
            dtype: torch dtype from vLLM (mapped to ttnn.bfloat16)
            num_layers: layer count from vLLM (ignored; 8 are allocated internally)
        Returns:
            List of 8 [K_cache, V_cache] pairs.
        """
        return self.model.allocate_kv_caches(kv_cache_shape, ttnn.bfloat16, batch_size=1)

    def prefill_forward(self, tokens, page_table, kv_cache, prompt_lens, **kwargs):
        """Prefill: process prompt, fill paged cache.

        Args:
            tokens: torch.Tensor [batch, seq_len]
            page_table: torch.Tensor [batch, max_blocks_per_seq]
            kv_cache: list from allocate_kv_cache (already attached to model)
            prompt_lens: list[int] with one entry
        Returns:
            (logits, rope_deltas): tuple where logits is [batch, 1, vocab_size]
            and rope_deltas is [batch] zeros. The runner (`model_runner.py:1670`)
            unpacks this tuple unconditionally because the HF config has
            `mrope_section` (M-RoPE), even though our text-only port doesn't
            use M-RoPE — zero deltas are correct for pure text input.
        """
        page_table_tt = ttnn.from_torch(
            page_table,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
        )

        logits_tt = self.model.prefill_paged(tokens, page_table_tt)
        logits = ttnn.to_torch(logits_tt)  # [B, 1, vocab_size]
        batch_size = logits.shape[0]
        rope_deltas = torch.zeros(batch_size, dtype=torch.long)
        return logits, rope_deltas

    def decode_forward(
        self, tokens, start_pos, page_table, kv_cache, enable_trace=False, read_from_device=True, **kwargs
    ):
        """Decode: single-token generation step.

        Args:
            tokens: torch.Tensor [batch, 1]
            start_pos: torch.Tensor [batch] or int -- current position
            page_table: torch.Tensor [batch, max_blocks_per_seq]
            kv_cache: list from allocate_kv_cache (already attached to model)
        Returns:
            logits: torch.Tensor [batch, 1, vocab_size] (3D)
        """
        page_table_tt = ttnn.from_torch(
            page_table,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
        )

        if isinstance(start_pos, torch.Tensor):
            current_pos = start_pos[0].item()
        else:
            current_pos = int(start_pos)

        logits_tt = self.model.decode_paged(tokens, current_pos, page_table_tt)
        logits = ttnn.to_torch(logits_tt)  # [B, 1, vocab_size]
        return logits

    def warmup_model_prefill(
        self,
        kv_cache,
        enable_trace,
        can_sample_on_device,
        non_greedy_decoding_on_device,
    ):
        """Compile prefill programs (and optionally capture trace) for the bucket size."""
        if self.already_warmed_up_prefill:
            return
        self.already_warmed_up_prefill = True

        bucket = _PREFILL_WARMUP_BUCKET
        num_blocks = math.ceil(bucket / _BLOCK_SIZE)
        dummy_tokens = torch.zeros(1, bucket, dtype=torch.long)
        dummy_page_table = torch.arange(num_blocks, dtype=torch.int32).reshape(1, num_blocks)

        if enable_trace:
            self.model.capture_prefill_trace_paged(
                self.device,
                dummy_page_table,
                bucket_size=bucket,
                chunk_size=_PREFILL_WARMUP_CHUNK,
            )
        else:
            _ = self.model.prefill_paged(dummy_tokens, dummy_page_table)

    def warmup_model_decode(
        self,
        kv_cache,
        enable_trace,
        max_batch_size,
        num_blocks,
        can_sample_on_device,
        non_greedy_decoding_on_device,
    ):
        """Compile decode programs (and optionally capture trace) for batch=1 paged decode."""
        dummy_tokens = torch.zeros(1, 1, dtype=torch.long)
        dummy_page_table = torch.arange(num_blocks, dtype=torch.int32).reshape(1, num_blocks)

        if enable_trace:
            self.model.capture_decode_trace_paged(self.device, dummy_page_table)
        else:
            _ = self.model.decode_paged(dummy_tokens, 0, dummy_page_table)
