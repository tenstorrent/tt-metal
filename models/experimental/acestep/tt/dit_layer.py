# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ACE-Step v1.5 DiT layer with AdaLN modulation (TTTv2-pattern composition module).

Reference: AceStepDiTLayer in modeling_acestep_v15_base.py. The core generative block:

    shift_msa, scale_msa, gate_msa, c_shift, c_scale, c_gate = (scale_shift_table + temb).chunk(6)

    # 1. self-attention with AdaLN
    n = self_attn_norm(x) * (1 + scale_msa) + shift_msa
    x = x + self_attn(n) * gate_msa

    # 2. cross-attention (optional, standard residual, no modulation)
    n = cross_attn_norm(x)
    x = x + cross_attn(n, encoder_hidden_states)

    # 3. MLP with AdaLN
    n = mlp_norm(x) * (1 + c_scale) + c_shift
    x = x + mlp(n) * c_gate

AdaLN pattern mirrors models/tt_dit/blocks/transformer_block.py (norm*(1+scale)+shift, gated
residual x + sub*gate). We reuse RMSNorm1D + AceStepAttention + MLP1D and express the modulation
with plain ttnn elementwise ops. Modulation params are [1, 6, B, dim] (broadcast over seq).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.modules.lazy_weight import LazyWeight
from models.common.modules.mlp.mlp_1d import MLP1D
from models.common.modules.rmsnorm.rmsnorm_1d import RMSNorm1D, RMSNorm1DConfig
from models.experimental.acestep.tt.attention import AceStepAttention, AceStepAttentionConfig


@dataclass
class AceStepDiTLayerConfig:
    # scale_shift_table parameter [1, 6, dim] as LazyWeight.
    scale_shift_table: LazyWeight

    # Norms.
    self_attn_norm_weight: LazyWeight
    mlp_norm_weight: LazyWeight
    cross_attn_norm_weight: LazyWeight | None = None

    # Self-attention.
    wq: LazyWeight = None
    wk: LazyWeight = None
    wv: LazyWeight = None
    wo: LazyWeight = None
    q_norm_weight: LazyWeight = None
    k_norm_weight: LazyWeight = None

    # Cross-attention (optional).
    c_wq: LazyWeight | None = None
    c_wk: LazyWeight | None = None
    c_wv: LazyWeight | None = None
    c_wo: LazyWeight | None = None
    c_q_norm_weight: LazyWeight | None = None
    c_k_norm_weight: LazyWeight | None = None

    # MLP.
    w1: LazyWeight = None
    w2: LazyWeight = None
    w3: LazyWeight = None

    n_heads: int = 16
    n_kv_heads: int = 8
    head_dim: int = 128
    dim: int = 2048
    eps: float = 1e-6
    sliding_window: int | None = None
    use_cross_attention: bool = True

    mesh_device: ttnn.MeshDevice | None = None


class AceStepDiTLayer(LightweightModule):
    """forward(hidden [1,1,seq,dim], cos, sin, temb [1,6,B,dim], encoder=None, attn_mask=None,
    cross_mask=None) -> [1,1,seq,dim]. temb is timestep_proj from TimestepEmbedding."""

    def __init__(self, config: AceStepDiTLayerConfig):
        self.config = config
        cfg = config
        if cfg.mesh_device is None:
            cfg.mesh_device = cfg.scale_shift_table.device

        self.scale_shift_table = cfg.scale_shift_table.get_device_weight()  # [1,6,dim] (padded)
        # Pre-increment the two SCALE rows (idx 1 = scale_msa, idx 4 = c_scale) by 1.0 so the AdaLN
        # apply is norm*scale_row + shift_row directly, instead of norm*(scale+1)+shift at runtime.
        # Mathematically identical (norm*(sst_scale+temb+1)+shift); saves a per-layer elementwise
        # add(scale,1.0) x2 (~0.05ms each). Built once here.
        _dim = self.scale_shift_table.shape[-1]
        _one_rows = torch.zeros(1, 6, 1, _dim)
        _one_rows[:, 1, :, :] = 1.0
        _one_rows[:, 4, :, :] = 1.0
        _one_tt = ttnn.from_torch(
            _one_rows, device=cfg.mesh_device, dtype=self.scale_shift_table.dtype, layout=ttnn.TILE_LAYOUT
        )
        self.scale_shift_table = ttnn.add(ttnn.reshape(self.scale_shift_table, (1, 6, 1, _dim)), _one_tt)
        self.scale_shift_table = ttnn.reshape(self.scale_shift_table, (1, 6, _dim))

        self.self_attn_norm = RMSNorm1D.from_config(RMSNorm1DConfig(weight=cfg.self_attn_norm_weight, eps=cfg.eps))
        self.mlp_norm = RMSNorm1D.from_config(RMSNorm1DConfig(weight=cfg.mlp_norm_weight, eps=cfg.eps))

        self.self_attn = AceStepAttention(
            AceStepAttentionConfig(
                wq=cfg.wq,
                wk=cfg.wk,
                wv=cfg.wv,
                wo=cfg.wo,
                q_norm_weight=cfg.q_norm_weight,
                k_norm_weight=cfg.k_norm_weight,
                n_heads=cfg.n_heads,
                n_kv_heads=cfg.n_kv_heads,
                head_dim=cfg.head_dim,
                eps=cfg.eps,
                is_cross_attention=False,
                sliding_window=cfg.sliding_window,
            )
        )

        self.use_cross_attention = cfg.use_cross_attention
        if self.use_cross_attention:
            self.cross_attn_norm = RMSNorm1D.from_config(
                RMSNorm1DConfig(weight=cfg.cross_attn_norm_weight, eps=cfg.eps)
            )
            self.cross_attn = AceStepAttention(
                AceStepAttentionConfig(
                    wq=cfg.c_wq,
                    wk=cfg.c_wk,
                    wv=cfg.c_wv,
                    wo=cfg.c_wo,
                    q_norm_weight=cfg.c_q_norm_weight,
                    k_norm_weight=cfg.c_k_norm_weight,
                    n_heads=cfg.n_heads,
                    n_kv_heads=cfg.n_kv_heads,
                    head_dim=cfg.head_dim,
                    eps=cfg.eps,
                    is_cross_attention=True,
                )
            )

        self.mlp = MLP1D(w1=cfg.w1, w2=cfg.w2, w3=cfg.w3)

    @classmethod
    def from_config(cls, config: AceStepDiTLayerConfig):
        return cls(config)

    def forward(self, hidden_states, cos, sin, temb, encoder_hidden_states=None, attn_mask=None, cross_mask=None):
        # Modulation: (scale_shift_table[1,6,dim] + temb[1,6,B,dim]) -> 6 chunks each [1,1,B,dim].
        # scale_shift_table broadcasts over batch; temb carries the per-sample timestep.
        sst = ttnn.reshape(self.scale_shift_table, (1, 6, 1, self.config.dim))
        mod = ttnn.add(sst, temb)  # [1,6,B,dim]
        shift_msa = mod[:, 0:1, :, :]
        scale_msa = mod[:, 1:2, :, :]
        gate_msa = mod[:, 2:3, :, :]
        c_shift = mod[:, 3:4, :, :]
        c_scale = mod[:, 4:5, :, :]
        c_gate = mod[:, 5:6, :, :]

        # 1. Self-attention with AdaLN: n = norm(x)*scale+shift (scale row is pre-incremented by 1.0
        # at construction, so this is the reference's norm*(1+scale)+shift). x = x + attn(n)*gate.
        n = self.self_attn_norm.forward(hidden_states, mode="prefill")
        n = ttnn.add(ttnn.mul(n, scale_msa), shift_msa)
        attn = self.self_attn.forward(n, cos=cos, sin=sin, attn_mask=attn_mask)
        hidden_states = ttnn.add(hidden_states, ttnn.mul(attn, gate_msa))

        # 2. Cross-attention (standard residual, no modulation).
        if self.use_cross_attention:
            n = self.cross_attn_norm.forward(hidden_states, mode="prefill")
            cattn = self.cross_attn.forward(n, encoder_hidden_states=encoder_hidden_states, attn_mask=cross_mask)
            hidden_states = ttnn.add(hidden_states, cattn)

        # 3. MLP with AdaLN (c_scale row pre-incremented by 1.0 at construction).
        n = self.mlp_norm.forward(hidden_states, mode="prefill")
        n = ttnn.add(ttnn.mul(n, c_scale), c_shift)
        ff = self.mlp.forward(n, mode="prefill")
        hidden_states = ttnn.add(hidden_states, ttnn.mul(ff, c_gate))
        return hidden_states
