# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""vLLM wrapper for Qwen3.5-9B on Blackhole P150.

Implements the 4 methods required by tt-inference-server:
- initialize_vllm_model (classmethod)
- allocate_kv_cache
- prefill_forward
- decode_forward
"""
import torch
import torch.nn as nn

import ttnn
from models.demos.blackhole.qwen3_5_9b.tt.qwen35_model import Qwen35Model


class TTQwen35ForCausalLM(nn.Module):
    """vLLM-compatible wrapper for Qwen3.5-9B on Blackhole P150."""

    def __init__(self, model: Qwen35Model, device):
        super().__init__()
        self.model = model
        self.device = device
        self.attention_layer_indices = [i for i in range(model.args.n_layers) if model.args.is_full_attention_layer(i)]
        self.num_attention_layers = len(self.attention_layer_indices)

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
        """Factory method called by tt_loader.py."""
        from huggingface_hub import snapshot_download

        checkpoint_dir = snapshot_download(hf_config._name_or_path)
        model = Qwen35Model.from_pretrained(
            device=mesh_device,
            checkpoint_dir=checkpoint_dir,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
        )
        return cls(model, mesh_device)

    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        """Allocate paged KV caches for the 8 attention layers only.

        Args:
            kv_cache_shape: (num_blocks, num_kv_heads, block_size, head_dim)
            dtype: torch dtype (ignored -- we use ttnn.bfloat16)
            num_layers: 32 from vLLM (ignored -- we allocate 8)
        Returns:
            List of 8 [K_cache, V_cache] pairs.
        """
        kv_caches = []
        for _ in range(self.num_attention_layers):
            cache_k = ttnn.from_torch(
                torch.zeros(kv_cache_shape, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            cache_v = ttnn.from_torch(
                torch.zeros(kv_cache_shape, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            kv_caches.append([cache_k, cache_v])

        self.model.set_paged_kv_caches(kv_caches)
        return kv_caches

    def prefill_forward(self, tokens, page_table, kv_cache, prompt_lens, **kwargs):
        """Prefill: process prompt, fill paged cache.

        Args:
            tokens: torch.Tensor [batch, seq_len]
            page_table: torch.Tensor [batch, max_blocks_per_seq]
            kv_cache: list from allocate_kv_cache (already attached to model)
            prompt_lens: list[int] with one entry
        Returns:
            logits: torch.Tensor [batch, 1, vocab_size] (3D)
        """
        page_table_tt = ttnn.from_torch(
            page_table,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
        )

        logits_tt = self.model.prefill_paged(tokens, page_table_tt)
        logits = ttnn.to_torch(logits_tt)  # [B, 1, vocab_size]
        return logits

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
