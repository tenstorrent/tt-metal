# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TP=8 tensor-parallel pi0.5 action-expert (adaRMS Gemma) block for a 4x2 submesh.

Optimised variant — same kernel-level optimisations as the VLM TP block:
  - Fused K+V projection (replicated)
  - Cached `build_matmul_pcfg` per per-chip matmul shape
  - adaRMS norm uses the standard ttnn.rms_norm path with (1+scale) weight
    and shift bias — no sharded multi-core variant for adaRMS today (the
    weight/bias come from a runtime modulation Dense, so we can't pre-pad).

Sharding plan unchanged:
  Q col-parallel, KV fused replicated, O row-parallel + all_reduce,
  MLP gate/up col-parallel, down row-parallel + all_reduce.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import ttnn

from models.experimental.pi0_5.common.configs import GemmaConfig
from models.experimental.pi0_5.tt.ttnn_gemma import build_matmul_pcfg

from .tp_block import _shard_along, _replicate


class Pi0_5SubmeshTPAdaRMSBlock:
    """One TP=8 adaRMS Gemma block on a 4x2 submesh (optimised)."""

    def __init__(
        self,
        config: GemmaConfig,
        weights: Dict[str, torch.Tensor],
        layer_idx: int,
        submesh,
        cos_meta: "ttnn.Tensor",
        sin_meta: "ttnn.Tensor",
        tp_size: int = 8,
    ):
        if config.num_heads % tp_size != 0:
            raise ValueError(f"num_heads={config.num_heads} not divisible by tp_size={tp_size}")
        self.config = config
        self.layer_idx = layer_idx
        self.submesh = submesh
        self.cos_meta = cos_meta
        self.sin_meta = sin_meta
        self.tp_size = tp_size

        self.num_heads_per_chip = config.num_heads // tp_size
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5
        self.W = config.width

        gsz = submesh.compute_with_storage_grid_size()
        self.grid_x = gsz.x
        self.grid_y = gsz.y

        prefix = f"model.layers.{layer_idx}."

        # --- Q col-parallel, KV-fused replicated, O row-parallel ---------
        self.q_proj = _shard_along(
            weights[f"{prefix}self_attn.q_proj.weight"].T.contiguous(),
            submesh,
            dim=-1,
            dtype=ttnn.bfloat8_b,
        )
        kv_w = torch.cat(
            [
                weights[f"{prefix}self_attn.k_proj.weight"].T.contiguous(),
                weights[f"{prefix}self_attn.v_proj.weight"].T.contiguous(),
            ],
            dim=-1,
        ).contiguous()
        self.kv_proj = _replicate(kv_w, submesh, dtype=ttnn.bfloat8_b)
        self.o_proj = _shard_along(
            weights[f"{prefix}self_attn.o_proj.weight"].T.contiguous(),
            submesh,
            dim=0,
            dtype=ttnn.bfloat8_b,
        )

        # --- MLP gate/up col-parallel, down row-parallel -----------------
        self.gate_proj = _shard_along(
            weights[f"{prefix}mlp.gate_proj.weight"].T.contiguous(),
            submesh,
            dim=-1,
            dtype=ttnn.bfloat8_b,
        )
        self.up_proj = _shard_along(
            weights[f"{prefix}mlp.up_proj.weight"].T.contiguous(),
            submesh,
            dim=-1,
            dtype=ttnn.bfloat8_b,
        )
        self.down_proj = _shard_along(
            weights[f"{prefix}mlp.down_proj.weight"].T.contiguous(),
            submesh,
            dim=0,
            dtype=ttnn.bfloat8_b,
        )

        # --- Fused adaRMS modulation Dense (replicated) -------------------
        w_pre_attn = weights[f"{prefix}input_layernorm.dense.weight"]
        w_pre_ffw = weights[f"{prefix}post_attention_layernorm.dense.weight"]
        fused_w = torch.cat([w_pre_attn, w_pre_ffw], dim=0).T.contiguous()
        self.mod_weight = _replicate(fused_w, submesh, dtype=ttnn.bfloat8_b)

        b_pre_attn = weights.get(f"{prefix}input_layernorm.dense.bias")
        b_pre_ffw = weights.get(f"{prefix}post_attention_layernorm.dense.bias")
        if b_pre_attn is not None and b_pre_ffw is not None:
            fused_b = torch.cat([b_pre_attn, b_pre_ffw], dim=0).view(1, -1).contiguous()
            self.mod_bias = _replicate(fused_b, submesh, dtype=ttnn.bfloat16)
        else:
            self.mod_bias = None

        self._matmul_pcfgs: Dict[Tuple[int, int, int, int], object] = {}

        # All-reduce output lands in DRAM (see comment in tp_block).
        self._allreduce_kwargs = dict(
            num_links=1,
            topology=ttnn.Topology.Linear,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _matmul_pcfg(self, m_tiles: int, k_tiles: int, n_tiles: int, in0_block_w=None):
        key = (m_tiles, k_tiles, n_tiles, in0_block_w or 0)
        cached = self._matmul_pcfgs.get(key)
        if cached is not None:
            return cached
        pcfg = build_matmul_pcfg(m_tiles, k_tiles, n_tiles, self.grid_x, self.grid_y, in0_block_w=in0_block_w)
        self._matmul_pcfgs[key] = pcfg
        return pcfg

    def _all_reduce(self, t: "ttnn.Tensor") -> "ttnn.Tensor":
        return ttnn.all_reduce(t, cluster_axis=None, **self._allreduce_kwargs)

    # ------------------------------------------------------------------ #
    # Forward                                                             #
    # ------------------------------------------------------------------ #

    def _compute_modulation(self, adarms_cond: "ttnn.Tensor") -> Tuple["ttnn.Tensor", ...]:
        """linear(cond, [W, 6W]) → split into 6 W-wide tensors per chip."""
        # adarms_cond is [B, W] — m_tiles=1 (or whatever pads). Small mat.
        cond_m_tiles = max(1, (adarms_cond.shape[-2] + 31) // 32 if len(adarms_cond.shape) >= 2 else 1)
        mod_pcfg = self._matmul_pcfg(cond_m_tiles, self.W // 32, (6 * self.W) // 32, in0_block_w=4)
        mod = ttnn.linear(
            adarms_cond,
            self.mod_weight,
            bias=self.mod_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            program_config=mod_pcfg,
        )
        B = mod.shape[0]
        total = mod.shape[-1]
        W = total // 6
        mod3 = ttnn.reshape(mod, (B, 1, total))
        ttnn.deallocate(mod)
        chunks = tuple(mod3[:, :, i * W : (i + 1) * W] for i in range(6))
        return chunks  # (scale_a, shift_a, gate_a, scale_f, shift_f, gate_f)

    def _ada_rms(
        self,
        x: "ttnn.Tensor",
        scale: "ttnn.Tensor",
        shift: "ttnn.Tensor",
    ) -> "ttnn.Tensor":
        scale_plus_one = ttnn.add(scale, 1.0)
        out = ttnn.rms_norm(
            x,
            weight=scale_plus_one,
            bias=shift,
            epsilon=self.config.rms_norm_eps,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(scale_plus_one)
        return out

    def forward(
        self,
        hidden_states: "ttnn.Tensor",
        adarms_cond: "ttnn.Tensor",
        attention_mask: Optional["ttnn.Tensor"] = None,
        past_key_value: Optional[Tuple["ttnn.Tensor", "ttnn.Tensor"]] = None,
    ) -> "ttnn.Tensor":
        B = hidden_states.shape[0]
        S = hidden_states.shape[1] if len(hidden_states.shape) == 3 else hidden_states.shape[2]
        nq = self.num_heads_per_chip
        nkv = self.config.num_kv_heads
        D = self.head_dim
        m_tiles = (S + 31) // 32
        k_tiles_attn = self.W // 32

        # Modulation outputs
        sa, ta, ga, sf, tf, gf = self._compute_modulation(adarms_cond)

        # Pre-attn adaRMS
        x_pre = self._ada_rms(hidden_states, sa, ta)
        ttnn.deallocate(sa)
        ttnn.deallocate(ta)

        # Q (col-parallel) + fused KV (replicated)
        q_pcfg = self._matmul_pcfg(m_tiles, k_tiles_attn, (nq * D) // 32, in0_block_w=8)
        kv_pcfg = self._matmul_pcfg(m_tiles, k_tiles_attn, (2 * nkv * D) // 32, in0_block_w=8)
        q = ttnn.linear(
            x_pre,
            self.q_proj,
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            program_config=q_pcfg,
        )
        kv = ttnn.linear(
            x_pre,
            self.kv_proj,
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            program_config=kv_pcfg,
        )
        ttnn.deallocate(x_pre)
        kv_dim = nkv * D
        k = kv[:, :, 0:kv_dim]
        v = kv[:, :, kv_dim : 2 * kv_dim]
        ttnn.deallocate(kv)

        q4 = ttnn.reshape(q, (B, S, nq, D))
        k4 = ttnn.reshape(k, (B, S, nkv, D))
        v4 = ttnn.reshape(v, (B, S, nkv, D))
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        q4 = ttnn.permute(q4, (0, 2, 1, 3))
        k4 = ttnn.permute(k4, (0, 2, 1, 3))
        v4 = ttnn.permute(v4, (0, 2, 1, 3))

        q4 = ttnn.experimental.rotary_embedding(q4, self.cos_meta, self.sin_meta)
        k4 = ttnn.experimental.rotary_embedding(k4, self.cos_meta, self.sin_meta)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k_full = ttnn.concat([past_k, k4], dim=2, memory_config=ttnn.L1_MEMORY_CONFIG)
            v_full = ttnn.concat([past_v, v4], dim=2, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(k4)
            ttnn.deallocate(v4)
        else:
            k_full, v_full = k4, v4

        attn_out = ttnn.transformer.scaled_dot_product_attention(
            q4,
            k_full,
            v_full,
            attn_mask=attention_mask,
            is_causal=False,
            scale=self.scale,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(q4)
        ttnn.deallocate(k_full)
        ttnn.deallocate(v_full)

        attn_out = ttnn.permute(attn_out, (0, 2, 1, 3))
        attn_out = ttnn.reshape(attn_out, (B, S, nq * D))

        o_pcfg = self._matmul_pcfg(m_tiles, (nq * D) // 32, self.W // 32, in0_block_w=8)
        o_partial = ttnn.linear(
            attn_out,
            self.o_proj,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            program_config=o_pcfg,
        )
        ttnn.deallocate(attn_out)
        o_full = self._all_reduce(o_partial)
        ttnn.deallocate(o_partial)

        gated_attn = ttnn.multiply(o_full, ga)
        ttnn.deallocate(o_full)
        ttnn.deallocate(ga)
        h_post_attn = ttnn.add(hidden_states, gated_attn)
        ttnn.deallocate(gated_attn)

        # Pre-MLP adaRMS
        x_mlp = self._ada_rms(h_post_attn, sf, tf)
        ttnn.deallocate(sf)
        ttnn.deallocate(tf)

        mlp_n_tiles = (self.config.mlp_dim // self.tp_size) // 32
        gate_pcfg = self._matmul_pcfg(m_tiles, k_tiles_attn, mlp_n_tiles, in0_block_w=4)
        up_pcfg = self._matmul_pcfg(m_tiles, k_tiles_attn, mlp_n_tiles, in0_block_w=4)
        down_pcfg = self._matmul_pcfg(m_tiles, mlp_n_tiles, self.W // 32, in0_block_w=4)

        gate = ttnn.linear(
            x_mlp,
            self.gate_proj,
            activation="gelu",
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            program_config=gate_pcfg,
        )
        up = ttnn.linear(
            x_mlp,
            self.up_proj,
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            program_config=up_pcfg,
        )
        ttnn.deallocate(x_mlp)
        hidden_mlp = ttnn.multiply(gate, up)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)

        mlp_partial = ttnn.linear(
            hidden_mlp,
            self.down_proj,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            program_config=down_pcfg,
        )
        ttnn.deallocate(hidden_mlp)
        mlp_full = self._all_reduce(mlp_partial)
        ttnn.deallocate(mlp_partial)

        gated_mlp = ttnn.multiply(mlp_full, gf)
        ttnn.deallocate(mlp_full)
        ttnn.deallocate(gf)
        out = ttnn.add(h_post_attn, gated_mlp)
        ttnn.deallocate(h_post_attn)
        ttnn.deallocate(gated_mlp)
        return out
