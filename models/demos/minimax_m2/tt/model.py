# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
MiniMax-M2.5 TTNN model: DecoderLayer + MiniMaxM2Model — Galaxy mesh (8,4).

Trace-safe decode: forward_decode_trace uses tensor positions for
paged_update_cache, SDPA decode, and RoPE embedding lookup.
All shapes are fixed → safe for Metal trace capture/replay.
"""

import torch

import ttnn
from models.demos.gpt_oss.tt.ccl import CCLManager
from models.tt_transformers.tt.common import PagedAttentionConfig

from .attention import TtMiniMaxAttention
from .model_config import MiniMaxM2TTConfig, make_mesh_config
from .moe import TtMiniMaxMoE
from .rms_norm import TtRMSNorm
from .rope import PartialRoPESetup


def _mesh_mapper(device):
    if isinstance(device, ttnn.MeshDevice):
        return ttnn.ReplicateTensorToMesh(device)
    return None


class TtDecoderLayer:
    """Pre-norm decoder layer: attention + MoE with residual connections."""

    def __init__(
        self,
        device,
        state_dict: dict,
        config: MiniMaxM2TTConfig,
        layer_idx: int,
        mesh_config=None,
        ccl_manager=None,
        max_seq_len: int = 4096,
        max_batch_size: int = 1,
        paged_attention_config: PagedAttentionConfig = None,
    ):
        prefix = f"model.layers.{layer_idx}."
        eps = config.rms_norm_eps
        rep = _mesh_mapper(device)

        self.input_layernorm = TtRMSNorm(device, state_dict[prefix + "input_layernorm.weight"], eps, mesh_mapper=rep)
        self.post_attention_layernorm = TtRMSNorm(
            device, state_dict[prefix + "post_attention_layernorm.weight"], eps, mesh_mapper=rep
        )
        self.self_attn = TtMiniMaxAttention(
            device,
            state_dict,
            config,
            layer_idx,
            mesh_config=mesh_config,
            ccl_manager=ccl_manager,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            paged_attention_config=paged_attention_config,
        )
        self.moe = TtMiniMaxMoE(
            device,
            state_dict,
            config,
            layer_idx,
            mesh_config=mesh_config,
            ccl_manager=ccl_manager,
        )

    def forward(self, x, cos, sin, attention_mask=None, is_causal=True):
        """Full-sequence forward (no KV-cache)."""
        residual = x
        normed = self.input_layernorm(x)
        attn_out = self.self_attn(normed, cos, sin, attention_mask, is_causal=is_causal)
        normed.deallocate(True)
        x = ttnn.add(residual, attn_out)
        residual.deallocate(True)
        attn_out.deallocate(True)

        residual = x
        normed = self.post_attention_layernorm(x)
        moe_out = self.moe(normed)
        normed.deallocate(True)
        x = ttnn.add(residual, moe_out)
        residual.deallocate(True)
        moe_out.deallocate(True)

        return x

    def forward_prefill(self, x, cos, sin, user_id: int = 0, page_table: ttnn.Tensor = None):
        """Prefill: fills device-resident KV cache."""
        residual = x
        normed = self.input_layernorm(x)
        attn_out = self.self_attn.forward_prefill(normed, cos, sin, user_id=user_id, page_table=page_table)
        normed.deallocate(True)
        x = ttnn.add(residual, attn_out)
        residual.deallocate(True)
        attn_out.deallocate(True)

        residual = x
        normed = self.post_attention_layernorm(x)
        moe_out = self.moe(normed)
        normed.deallocate(True)
        x = ttnn.add(residual, moe_out)
        residual.deallocate(True)
        moe_out.deallocate(True)
        return x

    def forward_decode(self, x, cos, sin, cur_pos: int):
        """Decode: single token using device-resident KV cache (not trace-safe)."""
        residual = x
        normed = self.input_layernorm(x)
        attn_out = self.self_attn.forward_decode(normed, cos, sin, cur_pos)
        normed.deallocate(True)
        x = ttnn.add(residual, attn_out)
        residual.deallocate(True)
        attn_out.deallocate(True)

        residual = x
        normed = self.post_attention_layernorm(x)
        moe_out = self.moe(normed)
        normed.deallocate(True)
        x = ttnn.add(residual, moe_out)
        residual.deallocate(True)
        moe_out.deallocate(True)
        return x

    def forward_decode_trace(self, x, cos, sin, position_idx: ttnn.Tensor, page_table: ttnn.Tensor = None):
        """Trace-safe decode: tensor position for attention, device routing for MoE."""
        residual = x
        normed = self.input_layernorm(x)
        attn_out = self.self_attn.forward_decode_trace(normed, cos, sin, position_idx, page_table=page_table)
        normed.deallocate(True)
        x = ttnn.add(residual, attn_out)
        residual.deallocate(True)
        attn_out.deallocate(True)

        residual = x
        normed = self.post_attention_layernorm(x)
        moe_out = self.moe(normed)
        normed.deallocate(True)
        x = ttnn.add(residual, moe_out)
        residual.deallocate(True)
        moe_out.deallocate(True)
        return x

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class TtMiniMaxModel:
    """Full MiniMax-M2.5 inference model — Galaxy mesh (8,4).

    KV cache is device-resident inside each attention layer.
    Supports both paged and non-paged attention modes.
    Trace-safe decode via forward_decode_trace.
    """

    def __init__(
        self,
        device,
        state_dict: dict,
        config: MiniMaxM2TTConfig,
        max_seq_len: int = 4096,
        max_batch_size: int = 1,
        paged_attention_config: PagedAttentionConfig = None,
    ):
        self.config = config
        self.device = device
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        self.paged_attention_config = paged_attention_config
        self._is_mesh = isinstance(device, ttnn.MeshDevice)

        if self._is_mesh:
            self.mesh_config = make_mesh_config(device)
            num_links = 4 if device.shape[0] > 1 else 1
            self.ccl_manager = CCLManager(device, num_links=num_links)
        else:
            self.mesh_config = None
            self.ccl_manager = None

        rep = _mesh_mapper(device)

        # ---------- Embedding ----------
        embed_w = state_dict["model.embed_tokens.weight"].to(torch.bfloat16)
        self.embed_weight = ttnn.from_torch(
            embed_w.unsqueeze(0).unsqueeze(0),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=rep,
        )

        # ---------- Partial RoPE ----------
        self.rope = PartialRoPESetup(
            device,
            rotary_dim=config.rotary_dim,
            rope_theta=config.rope_theta,
            max_seq_len=max_seq_len,
        )

        # ---------- Decoder layers (with device-resident KV cache) ----------
        self.layers = [
            TtDecoderLayer(
                device,
                state_dict,
                config,
                i,
                mesh_config=self.mesh_config,
                ccl_manager=self.ccl_manager,
                max_seq_len=max_seq_len,
                max_batch_size=max_batch_size,
                paged_attention_config=paged_attention_config,
            )
            for i in range(config.num_hidden_layers)
        ]

        # ---------- Final norm ----------
        self.norm = TtRMSNorm(device, state_dict["model.norm.weight"], config.rms_norm_eps, mesh_mapper=rep)

        # ---------- LM head [H, V] — replicated -------
        lm_w = state_dict["lm_head.weight"].to(torch.bfloat16).T
        self.lm_head = ttnn.from_torch(
            lm_w,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=rep,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _embed_tt_ids(self, ids_tt: ttnn.Tensor, batch: int, seq: int) -> ttnn.Tensor:
        x = ttnn.embedding(ids_tt, self.embed_weight, layout=ttnn.TILE_LAYOUT)
        if len(x.shape) == 3:
            x = ttnn.unsqueeze_to_4D(x)
        return ttnn.reshape(x, (batch, seq, self.config.hidden_size))

    def _embed(self, input_ids: torch.Tensor) -> ttnn.Tensor:
        B, S = input_ids.shape
        rep = _mesh_mapper(self.device)
        ids_tt = ttnn.from_torch(
            input_ids.unsqueeze(0).unsqueeze(0),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            mesh_mapper=rep,
        )
        x = self._embed_tt_ids(ids_tt, B, S)
        ids_tt.deallocate(True)
        return x

    # ------------------------------------------------------------------
    # KV cache management
    # ------------------------------------------------------------------

    def clear_kv_caches(self):
        """Zero all device-resident KV caches in-place."""
        for layer in self.layers:
            layer.self_attn.clear_cache()

    # MoE routing is always on device (trace-safe) — no CPU fallback.

    # ------------------------------------------------------------------
    # Forward modes
    # ------------------------------------------------------------------

    def forward(self, input_ids: torch.Tensor) -> ttnn.Tensor:
        """Full-sequence forward without KV-cache (for unit tests)."""
        B, S = input_ids.shape
        x = self._embed(input_ids)
        cos, sin = self.rope.get_cos_sin(S)

        for layer in self.layers:
            x = layer(x, cos, sin)

        cos.deallocate(True)
        sin.deallocate(True)

        x = self.norm(x)
        return ttnn.linear(x, self.lm_head, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def forward_prefill(self, input_ids: torch.Tensor, user_id: int = 0, page_table: ttnn.Tensor = None):
        """Prefill: process prompt tokens, fill device-resident KV-cache.

        Args:
            input_ids: [B, S] token ids
            user_id: Batch index for non-paged attention
            page_table: [B, max_blocks_per_user] page table for paged attention
        """
        B, S = input_ids.shape
        x = self._embed(input_ids)
        cos, sin = self.rope.get_cos_sin(S)

        for layer in self.layers:
            x = layer.forward_prefill(x, cos, sin, user_id=user_id, page_table=page_table)

        cos.deallocate(True)
        sin.deallocate(True)

        x = self.norm(x)
        logits = ttnn.linear(x, self.lm_head, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return logits

    def forward_decode(self, input_ids: torch.Tensor, cur_pos: int):
        """Decode: single new token at cur_pos (not trace-safe)."""
        x = self._embed(input_ids)
        x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
        cos, sin = self.rope.get_single_position(cur_pos)

        for layer in self.layers:
            x = layer.forward_decode(x, cos, sin, cur_pos)

        cos.deallocate(True)
        sin.deallocate(True)

        x = self.norm(x)
        logits = ttnn.linear(x, self.lm_head, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return logits

    def forward_decode_tt(self, input_ids_tt: ttnn.Tensor, cur_pos: int, batch: int = 1):
        """Decode from persistent TT token tensor (not trace-safe, uses Python int pos)."""
        x = self._embed_tt_ids(input_ids_tt, batch, 1)
        x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
        cos, sin = self.rope.get_single_position(cur_pos)

        for layer in self.layers:
            x = layer.forward_decode(x, cos, sin, cur_pos)

        cos.deallocate(True)
        sin.deallocate(True)

        x = self.norm(x)
        logits = ttnn.linear(x, self.lm_head, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return logits

    def forward_decode_trace(
        self,
        input_ids_tt: ttnn.Tensor,
        position_idx: ttnn.Tensor,
        rope_ids: ttnn.Tensor,
        batch: int = 1,
        page_table: ttnn.Tensor = None,
    ):
        """Trace-safe decode: all operations use fixed tensor shapes.

        Args:
            input_ids_tt: [1, 1, B, 1] uint32 — persistent token buffer
            position_idx: [B] int32 — persistent position buffer for cache + SDPA
            rope_ids:     [B] uint32 — persistent position buffer for RoPE embedding
            batch:        batch size
            page_table:   [B, max_blocks_per_user] page table for paged attention

        Returns:
            logits tensor (persistent buffer during trace replay)
        """
        x = self._embed_tt_ids(input_ids_tt, batch, 1)
        x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)

        cos, sin = self.rope.get_cos_sin_decode(rope_ids)

        for layer in self.layers:
            x = layer.forward_decode_trace(x, cos, sin, position_idx, page_table=page_table)

        cos.deallocate(True)
        sin.deallocate(True)

        x = self.norm(x)
        logits = ttnn.linear(x, self.lm_head, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return logits

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
