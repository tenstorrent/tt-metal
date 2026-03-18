# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
MiniMax-M2.5 GQA Attention with QK-norm and Partial RoPE — Galaxy mesh (8,4).

QK-norm design:
  MiniMax QK-norm computes RMS over ALL NQ*D=6144 elements jointly (not per-head).
  This is non-separable: splitting Q across TP shards gives different norms per shard.
  To be EXACT we replicate Q/K projection weights so every device has the full Q/K,
  applies the correct norm, then runs the full 48-head SDPA.

Weight layout on (8,4) mesh:
  wq, wk, wv, q_norm, k_norm:  ReplicateTensorToMesh  (every device has full weights)
  wo (O-proj):                  column-parallel         [NQ*D, H/TP] per device
  wo all-reduce:                reduce-scatter + all-gather across TP cols

All 32 devices run identical attention (full 48 Q heads, 8 KV heads).
O-proj benefits from TP=4 (H/TP = 768 per device), all-reduce restores full H.

KV-cache: CPU torch tensors (device upgrade is a follow-up).
"""

import torch

import ttnn
from models.demos.gpt_oss.config import MeshConfig
from models.demos.gpt_oss.tt.attention.operations import apply_allreduce
from models.demos.gpt_oss.tt.ccl import CCLManager

from .model_config import MiniMaxM2TTConfig
from .rms_norm import TtRMSNorm
from .rope import apply_partial_rope


def _extract_bsh(x: ttnn.Tensor, H: int):
    """Return (B, S, x_3d) from a 3D or 4D TTNN tensor using logical shape."""
    shape = list(x.shape)
    if len(shape) == 3:
        return shape[0], shape[1], x
    elif len(shape) == 4:
        B, S = shape[0] * shape[1], shape[2]
        return B, S, ttnn.reshape(x, (B, S, H))
    raise ValueError(f"Unexpected shape {shape}")


class TtMiniMaxAttention:
    """
    MiniMax-M2.5 attention — exact QK-norm via replicated Q/K projections.

    Weight layout on (8,4) mesh:
      wq, wk, wv:  Replicated — every device holds the full [H, N*D] weight.
                   Required so QK-norm is computed over the full head dimension.
      q_norm, k_norm: Replicated — exact RMS over NQ*D=6144 / NK*D=1024 elements.
      wo:          Column-parallel [NQ*D, H/TP] per device — TP=4 efficiency for output.
                   Followed by all-reduce across TP cols to restore full H.
    """

    def __init__(
        self,
        device,
        state_dict: dict,
        config: MiniMaxM2TTConfig,
        layer_idx: int,
        mesh_config: MeshConfig = None,
        ccl_manager: CCLManager = None,
    ):
        self.config = config
        self.device = device
        self.mesh_config = mesh_config
        self.ccl_manager = ccl_manager
        self._is_mesh = isinstance(device, ttnn.MeshDevice)

        NQ = config.num_attention_heads
        NK = config.num_key_value_heads
        D = config.head_dim
        H = config.hidden_size
        eps = config.rms_norm_eps
        tp = mesh_config.tp if mesh_config else 1

        prefix = f"model.layers.{layer_idx}.self_attn."

        rep_mapper = ttnn.ReplicateTensorToMesh(device) if self._is_mesh else None
        col_mapper = mesh_config.column_parallel(device) if (self._is_mesh and mesh_config) else None

        # ---------- Q/K/V projections: replicated so QK-norm is exact ----------
        # wq: [H, NQ*D] = [3072, 6144]  replicated
        # wk: [H, NK*D] = [3072, 1024]  replicated
        # wv: [H, NK*D] = [3072, 1024]  replicated
        def _load_rep(key):
            w = state_dict[prefix + key].T.to(torch.bfloat16)  # transpose: [in, out]
            return ttnn.from_torch(
                w,
                dtype=config.weight_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=rep_mapper,
            )

        self.wq = _load_rep("q_proj.weight")  # [H, NQ*D]
        self.wk = _load_rep("k_proj.weight")  # [H, NK*D]
        self.wv = _load_rep("v_proj.weight")  # [H, NK*D]

        # ---------- O-proj: column-parallel [NQ*D, H] → [NQ*D, H/TP] per device ----------
        # Column-parallel: each device outputs H/TP partial hidden; all-reduce restores H.
        wo_pt = state_dict[prefix + "o_proj.weight"].T.to(torch.bfloat16)  # [NQ*D, H]
        self.wo = ttnn.from_torch(
            wo_pt,
            dtype=config.weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=col_mapper,
        )

        # ---------- QK-norm: replicated, exact over full head dimension ----------
        self.q_norm = TtRMSNorm(device, state_dict[prefix + "q_norm.weight"], eps, mesh_mapper=rep_mapper)
        self.k_norm = TtRMSNorm(device, state_dict[prefix + "k_norm.weight"], eps, mesh_mapper=rep_mapper)

        self._NQ = NQ
        self._NK = NK
        self._D = D
        self._tp = tp

    # ------------------------------------------------------------------
    # Internal: QKV projection + exact QK-norm + reshape to heads + RoPE
    # ------------------------------------------------------------------

    def _qkv_rope(self, x: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor):
        """Full QKV projection + exact QK-norm + reshape to per-head + partial RoPE.

        All devices compute the same full Q/K/V (replicated weights).
        Exact QK-norm is applied over NQ*D=6144 and NK*D=1024 respectively.

        Returns q/k/v: [B, NQ, S, D], [B, NK, S, D], [B, NK, S, D]
        """
        cfg = self.config
        NQ, NK, D, H = self._NQ, self._NK, self._D, cfg.hidden_size

        B, S, x = _extract_bsh(x, H)

        # Separate projections (replicated weights)
        q = ttnn.linear(x, self.wq, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [B, S, NQ*D]
        k = ttnn.linear(x, self.wk, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [B, S, NK*D]
        v = ttnn.linear(x, self.wv, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [B, S, NK*D]

        # Exact QK-norm: RMS over full NQ*D and NK*D respectively
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Reshape to per-head: [B, N_heads, S, D]
        q = ttnn.permute(ttnn.reshape(q, (B, S, NQ, D)), (0, 2, 1, 3))
        k = ttnn.permute(ttnn.reshape(k, (B, S, NK, D)), (0, 2, 1, 3))
        v = ttnn.permute(ttnn.reshape(v, (B, S, NK, D)), (0, 2, 1, 3))

        # Partial RoPE (local, no CCL needed)
        q, k = apply_partial_rope(q, k, cos, sin, cfg.rotary_dim, D)
        return q, k, v, B, S

    # ------------------------------------------------------------------
    # Forward modes
    # ------------------------------------------------------------------

    def forward(
        self,
        x: ttnn.Tensor,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        attention_mask: ttnn.Tensor | None = None,
        is_causal: bool = True,
    ) -> ttnn.Tensor:
        """Full-sequence forward (no KV-cache). All devices run identical attention.

        O-proj is column-parallel → each device produces [B, S, H/TP].
        All-reduce across TP cols restores full [B, S, H].
        """
        cfg = self.config
        tp = self._tp
        NQ, D, H = self._NQ, self._D, cfg.hidden_size
        scale = D**-0.5

        q, k, v, B, S = self._qkv_rope(x, cos, sin)

        if attention_mask is not None:
            attn_out = ttnn.transformer.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attention_mask,
                is_causal=False,
                scale=scale,
            )
        else:
            attn_out = ttnn.transformer.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=is_causal,
                scale=scale,
            )

        # Reshape: [B, NQ, S, D] → [B, S, NQ*D]
        attn_out = ttnn.reshape(ttnn.permute(attn_out, (0, 2, 1, 3)), (B, S, NQ * D))

        # Column-parallel O-proj: [B, S, NQ*D] × [NQ*D, H/TP] → [B, S, H/TP] per device
        out = ttnn.linear(attn_out, self.wo, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attn_out.deallocate(True)

        # All-reduce across TP cols: sum partial H/TP outputs → full [B, S, H]
        # CCL ops require 4D; unsqueeze_to_4D returns a view — do not deallocate original
        if self._is_mesh and self.mesh_config and self.ccl_manager and tp > 1:
            out_4d = ttnn.unsqueeze_to_4D(out)
            out_4d = apply_allreduce(out_4d, self.mesh_config, self.ccl_manager, H)
            out = ttnn.reshape(out_4d, (B, S, H))

        return out

    def forward_prefill(self, x, cos, sin, k_cache, v_cache):
        """Prefill: process S tokens, fill CPU KV cache."""
        cfg = self.config
        tp = self._tp
        NQ, NK, D, H = self._NQ, self._NK, self._D, cfg.hidden_size
        scale = D**-0.5

        q, k, v, B, S = self._qkv_rope(x, cos, sin)

        # Save K/V to CPU cache
        extract = (
            lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0]).bfloat16()
            if self._is_mesh
            else ttnn.to_torch(t).bfloat16()
        )
        k_cache[:, :NK, :S, :] = extract(k)
        v_cache[:, :NK, :S, :] = extract(v)

        attn_out = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            scale=scale,
        )
        attn_out = ttnn.reshape(ttnn.permute(attn_out, (0, 2, 1, 3)), (B, S, NQ * D))
        out = ttnn.linear(attn_out, self.wo, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attn_out.deallocate(True)

        if self._is_mesh and self.mesh_config and self.ccl_manager and tp > 1:
            out_4d = ttnn.unsqueeze_to_4D(out)
            out_4d = apply_allreduce(out_4d, self.mesh_config, self.ccl_manager, H)
            out = ttnn.reshape(out_4d, (B, S, H))

        return out, k_cache, v_cache

    def forward_decode(self, x, cos, sin, k_cache, v_cache, cur_pos):
        """Decode: single token at cur_pos using CPU KV cache."""
        cfg = self.config
        tp = self._tp
        NQ, NK, D, H = self._NQ, self._NK, self._D, cfg.hidden_size
        scale = D**-0.5
        L1 = ttnn.L1_MEMORY_CONFIG

        q, k, v, B, _ = self._qkv_rope(x, cos, sin)

        extract = (
            lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0]).bfloat16()
            if self._is_mesh
            else ttnn.to_torch(t).bfloat16()
        )
        k_cache[:, :NK, cur_pos : cur_pos + 1, :] = extract(k)
        v_cache[:, :NK, cur_pos : cur_pos + 1, :] = extract(v)
        k.deallocate(True)
        v.deallocate(True)

        mesh_mapper = ttnn.ReplicateTensorToMesh(self.device) if self._is_mesh else None
        cache_mem = L1 if (cur_pos + 1) * NK * D * 2 < 256 * 1024 else ttnn.DRAM_MEMORY_CONFIG
        k_filled = ttnn.from_torch(
            k_cache[:, :NK, : cur_pos + 1, :].contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=cache_mem,
            mesh_mapper=mesh_mapper,
        )
        v_filled = ttnn.from_torch(
            v_cache[:, :NK, : cur_pos + 1, :].contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=cache_mem,
            mesh_mapper=mesh_mapper,
        )

        q_l1 = ttnn.to_memory_config(q, L1)
        q.deallocate(True)
        attn_out = ttnn.transformer.scaled_dot_product_attention(
            q_l1,
            k_filled,
            v_filled,
            is_causal=False,
            scale=scale,
        )
        q_l1.deallocate(True)
        k_filled.deallocate(True)
        v_filled.deallocate(True)

        attn_out = ttnn.reshape(ttnn.permute(attn_out, (0, 2, 1, 3)), (B, 1, NQ * D))
        out = ttnn.linear(attn_out, self.wo, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attn_out.deallocate(True)

        if self._is_mesh and self.mesh_config and self.ccl_manager and tp > 1:
            out_4d = ttnn.unsqueeze_to_4D(out)
            out_4d = apply_allreduce(out_4d, self.mesh_config, self.ccl_manager, H)
            out = ttnn.reshape(out_4d, (B, 1, H))

        return out, k_cache, v_cache

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
