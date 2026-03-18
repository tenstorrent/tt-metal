# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
MiniMax-M2.5 TTNN model: DecoderLayer + MiniMaxM2Model — Galaxy mesh (8,4).

Mesh parallelism:
  - Attention: TP=4 (column-parallel QKV, row-parallel O-proj, all-reduce)
  - MoE:       EP=8 + TP=4 (EP×TP sharded experts, EP+TP all-reduce)
  - Embeddings / norms / lm_head: replicated across all 32 devices

Modes:
  forward()         — full-sequence (no KV-cache), used for unit tests.
  forward_prefill() — fills KV-cache, returns hidden + updated caches.
  forward_decode()  — single-token decode using KV-cache.
"""

import torch

import ttnn
from models.demos.gpt_oss.tt.ccl import CCLManager

from .attention import TtMiniMaxAttention
from .model_config import MiniMaxM2TTConfig, make_mesh_config
from .moe import TtMiniMaxMoE
from .rms_norm import TtRMSNorm
from .rope import PartialRoPESetup


def _mesh_mapper(device):
    """Return ReplicateTensorToMesh for MeshDevice, else None (single device)."""
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
        )
        self.moe = TtMiniMaxMoE(
            device,
            state_dict,
            config,
            layer_idx,
            mesh_config=mesh_config,
            ccl_manager=ccl_manager,
        )

    def forward(
        self,
        x: ttnn.Tensor,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        attention_mask: ttnn.Tensor | None = None,
        is_causal: bool = True,
    ) -> ttnn.Tensor:
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

    def forward_prefill(self, x, cos, sin, k_cache, v_cache):
        residual = x
        normed = self.input_layernorm(x)
        attn_out, k_cache, v_cache = self.self_attn.forward_prefill(normed, cos, sin, k_cache, v_cache)
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
        return x, k_cache, v_cache

    def forward_decode(self, x, cos, sin, k_cache, v_cache, cur_pos):
        residual = x
        normed = self.input_layernorm(x)
        attn_out, k_cache, v_cache = self.self_attn.forward_decode(normed, cos, sin, k_cache, v_cache, cur_pos)
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
        return x, k_cache, v_cache

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class TtMiniMaxModel:
    """Full MiniMax-M2.5 inference model — Galaxy mesh (8,4)."""

    def __init__(
        self,
        device,
        state_dict: dict,
        config: MiniMaxM2TTConfig,
        max_seq_len: int = 4096,
    ):
        self.config = config
        self.device = device
        self.max_seq_len = max_seq_len
        self._is_mesh = isinstance(device, ttnn.MeshDevice)

        # Build mesh config and CCL manager
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

        # ---------- Decoder layers ----------
        self.layers = [
            TtDecoderLayer(
                device,
                state_dict,
                config,
                i,
                mesh_config=self.mesh_config,
                ccl_manager=self.ccl_manager,
            )
            for i in range(config.num_hidden_layers)
        ]

        # ---------- Final norm ----------
        self.norm = TtRMSNorm(device, state_dict["model.norm.weight"], config.rms_norm_eps, mesh_mapper=rep)

        # ---------- LM head [H, V] — replicated -------
        lm_w = state_dict["lm_head.weight"].to(torch.bfloat16).T  # [H, V]
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

    def _embed(self, input_ids: torch.Tensor) -> ttnn.Tensor:
        """Embed input_ids [B, S] → [B, S, H] on device."""
        B, S = input_ids.shape
        rep = _mesh_mapper(self.device)
        ids_tt = ttnn.from_torch(
            input_ids.unsqueeze(0).unsqueeze(0),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            mesh_mapper=rep,
        )
        x = ttnn.embedding(ids_tt, self.embed_weight, layout=ttnn.TILE_LAYOUT)
        ids_tt.deallocate(True)
        if len(x.shape) == 3:
            x = ttnn.unsqueeze_to_4D(x)
        return ttnn.reshape(x, (B, S, self.config.hidden_size))

    def allocate_kv_cache(self, batch: int = 1):
        """Allocate CPU KV-cache for all layers.

        Returns: list of (k_cache, v_cache) tuples.
        Each cache: [B, NK, max_seq_len, D] bfloat16 CPU tensor.
        """
        NK = self.config.num_key_value_heads
        D = self.config.head_dim
        shape = (batch, NK, self.max_seq_len, D)
        return [
            (torch.zeros(shape, dtype=torch.bfloat16), torch.zeros(shape, dtype=torch.bfloat16)) for _ in self.layers
        ]

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

    def forward_prefill(self, input_ids: torch.Tensor, kv_caches: list):
        """Prefill: process prompt tokens, fill KV-cache."""
        B, S = input_ids.shape
        x = self._embed(input_ids)
        cos, sin = self.rope.get_cos_sin(S)

        for i, layer in enumerate(self.layers):
            k_cache, v_cache = kv_caches[i]
            x, k_cache, v_cache = layer.forward_prefill(x, cos, sin, k_cache, v_cache)
            kv_caches[i] = (k_cache, v_cache)

        cos.deallocate(True)
        sin.deallocate(True)

        x = self.norm(x)
        logits = ttnn.linear(x, self.lm_head, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return logits, kv_caches

    def forward_decode(self, input_ids: torch.Tensor, kv_caches: list, cur_pos: int):
        """Decode: single new token at cur_pos using KV-cache."""
        x = self._embed(input_ids)
        x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
        cos, sin = self.rope.get_single_position(cur_pos)

        for i, layer in enumerate(self.layers):
            k_cache, v_cache = kv_caches[i]
            x, k_cache, v_cache = layer.forward_decode(x, cos, sin, k_cache, v_cache, cur_pos)
            kv_caches[i] = (k_cache, v_cache)

        cos.deallocate(True)
        sin.deallocate(True)

        x = self.norm(x)
        logits = ttnn.linear(x, self.lm_head, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return logits, kv_caches

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
