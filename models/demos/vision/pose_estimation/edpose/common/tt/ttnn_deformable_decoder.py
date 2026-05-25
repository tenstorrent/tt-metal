# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Deformable Transformer Decoder Layer in ttnn.

Each layer:
  1. Self-attention: standard MHA(query=key=tgt+pos, value=tgt) → residual + LayerNorm
  2. Cross-attention: MSDeformAttn(tgt+pos, ref_pts, memory) → residual + LayerNorm
  3. FFN: Linear(256→2048) → ReLU → Linear(2048→256) → residual + LayerNorm

Self-attention runs on device using ttnn matmul ops. Cross-attention uses
TTMSDeformAttn (hybrid device/host). FFN and norms run entirely on device.

ED-Pose norm ordering: norm2 after self-attn, norm1 after cross-attn, norm3 after FFN.
"""

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.vision.pose_estimation.edpose.common.tt.ttnn_ms_deform_attn import TTMSDeformAttn


class TTDeformableDecoderLayer(LightweightModule):

    def __init__(self, device, state_dict, prefix, d_model=256, d_ffn=2048,
                 n_levels=5, n_heads=8, n_points=4, has_self_attn=True):
        super().__init__()
        self.device = device
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.has_self_attn = has_self_attn
        p = f"{prefix}." if prefix else ""

        self.cross_attn = TTMSDeformAttn(
            device, state_dict, f"{p}cross_attn", d_model, n_heads, n_levels, n_points
        )

        if has_self_attn:
            sa_p = f"{p}self_attn"
            in_proj_w = state_dict[f"{sa_p}.in_proj_weight"].float()
            in_proj_b = state_dict[f"{sa_p}.in_proj_bias"].float()
            qw, kw, vw = in_proj_w.chunk(3, dim=0)
            qb, kb, vb = in_proj_b.chunk(3, dim=0)

            self.sa_q_w = self._weight(qw)
            self.sa_q_b = self._bias(qb)
            self.sa_k_w = self._weight(kw)
            self.sa_k_b = self._bias(kb)
            self.sa_v_w = self._weight(vw)
            self.sa_v_b = self._bias(vb)
            self.sa_out_w = self._weight(state_dict[f"{sa_p}.out_proj.weight"])
            self.sa_out_b = self._bias(state_dict[f"{sa_p}.out_proj.bias"])

            self.norm2_w = self._param(state_dict[f"{p}norm2.weight"])
            self.norm2_b = self._param(state_dict[f"{p}norm2.bias"])

        self.norm1_w = self._param(state_dict[f"{p}norm1.weight"])
        self.norm1_b = self._param(state_dict[f"{p}norm1.bias"])
        self.norm3_w = self._param(state_dict[f"{p}norm3.weight"])
        self.norm3_b = self._param(state_dict[f"{p}norm3.bias"])

        self.ffn1_w = self._weight(state_dict[f"{p}linear1.weight"])
        self.ffn1_b = self._bias(state_dict[f"{p}linear1.bias"])
        self.ffn2_w = self._weight(state_dict[f"{p}linear2.weight"])
        self.ffn2_b = self._bias(state_dict[f"{p}linear2.bias"])

    def _weight(self, w):
        return ttnn.from_torch(
            w.T.contiguous().to(torch.bfloat16),
            layout=ttnn.TILE_LAYOUT, device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _bias(self, b):
        return ttnn.from_torch(
            b.unsqueeze(0).to(torch.bfloat16),
            layout=ttnn.TILE_LAYOUT, device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _param(self, t):
        return ttnn.from_torch(
            t.unsqueeze(0).to(torch.bfloat16),
            layout=ttnn.TILE_LAYOUT, device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _split_heads(self, x, batch, length):
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.reshape(x, (batch, length, self.n_heads, self.head_dim))
        x = ttnn.transpose(x, 1, 2)
        return ttnn.to_layout(x, ttnn.TILE_LAYOUT)

    def _merge_heads(self, x, batch, length):
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.transpose(x, 1, 2)
        x = ttnn.reshape(x, (batch, length, self.d_model))
        return ttnn.to_layout(x, ttnn.TILE_LAYOUT)

    def _self_attention_device(self, tgt_bf, qk_input_bf, self_attn_mask):
        """
        Self-attention fully on device.

        Args:
            tgt_bf: torch (N, Lq, C) bfloat16 — value input (batch-first)
            qk_input_bf: torch (N, Lq, C) bfloat16 — query/key input (tgt+pos, batch-first)
            self_attn_mask: ttnn (1, N_HEADS, Lq, Lq) float additive mask or None

        Returns:
            torch (N, Lq, C) float — self-attention output (batch-first)
        """
        N, Lq, C = qk_input_bf.shape

        qk_tt = ttnn.from_torch(
            qk_input_bf, layout=ttnn.TILE_LAYOUT, device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        v_tt = ttnn.from_torch(
            tgt_bf, layout=ttnn.TILE_LAYOUT, device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        q = ttnn.linear(qk_tt, self.sa_q_w, bias=self.sa_q_b)
        k = ttnn.linear(qk_tt, self.sa_k_w, bias=self.sa_k_b)
        v = ttnn.linear(v_tt, self.sa_v_w, bias=self.sa_v_b)
        ttnn.deallocate(qk_tt)
        ttnn.deallocate(v_tt)

        q = self._split_heads(q, N, Lq)
        k = self._split_heads(k, N, Lq)
        v = self._split_heads(v, N, Lq)

        scale = self.head_dim ** -0.5
        k_t = ttnn.transpose(k, -2, -1)
        ttnn.deallocate(k)
        attn = ttnn.matmul(q, k_t)
        ttnn.deallocate(q)
        ttnn.deallocate(k_t)
        old = attn
        attn = ttnn.multiply(attn, scale)
        ttnn.deallocate(old)

        if self_attn_mask is not None:
            old = attn
            attn = ttnn.add(attn, self_attn_mask)
            ttnn.deallocate(old)

        old = attn
        attn = ttnn.softmax(attn, dim=-1)
        ttnn.deallocate(old)
        out = ttnn.matmul(attn, v)
        ttnn.deallocate(attn)
        ttnn.deallocate(v)

        out = self._merge_heads(out, N, Lq)
        result = ttnn.linear(out, self.sa_out_w, bias=self.sa_out_b)
        ttnn.deallocate(out)

        result_t = ttnn.to_torch(result).float()
        ttnn.deallocate(result)
        return result_t

    def forward(
        self,
        tgt,
        tgt_query_pos=None,
        tgt_reference_points=None,
        memory=None,
        memory_spatial_shapes=None,
        memory_level_start_index=None,
        memory_key_padding_mask=None,
        self_attn_mask=None,
    ):
        """
        Args — all in sequence-first format (Lq, N, C) as torch tensors,
        EXCEPT memory which is ttnn (N, Len_in, C) in batch-first TILE_LAYOUT.

        tgt: torch (Lq, N, C)
        tgt_query_pos: torch (Lq, N, C) or None
        tgt_reference_points: torch (Lq, N, n_levels, 2|4)
        memory: ttnn (N, Len_in, C)
        memory_spatial_shapes: torch (n_levels, 2)
        memory_level_start_index: torch (n_levels,)
        memory_key_padding_mask: optional torch (N, Len_in)
        self_attn_mask: optional ttnn (1, N_HEADS, Lq, Lq) float additive, or None

        Returns: torch (Lq, N, C)
        """
        if self.has_self_attn:
            tgt_bf = tgt.to(torch.bfloat16).transpose(0, 1).contiguous()
            qk_input = tgt + tgt_query_pos if tgt_query_pos is not None else tgt
            qk_bf = qk_input.to(torch.bfloat16).transpose(0, 1).contiguous()

            sa_mask_tt = None
            if self_attn_mask is not None:
                sa_mask_tt = self_attn_mask
                if not isinstance(sa_mask_tt, ttnn.Tensor):
                    sa_mask_tt = ttnn.from_torch(
                        sa_mask_tt, layout=ttnn.TILE_LAYOUT, device=self.device,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )

            sa_out = self._self_attention_device(tgt_bf, qk_bf, sa_mask_tt)

            tgt_resid = tgt.float().transpose(0, 1).contiguous() + sa_out
            tgt_tt = ttnn.from_torch(
                tgt_resid.to(torch.bfloat16),
                layout=ttnn.TILE_LAYOUT, device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            old = tgt_tt
            tgt_tt = ttnn.layer_norm(tgt_tt, weight=self.norm2_w, bias=self.norm2_b)
            ttnn.deallocate(old)
            tgt = ttnn.to_torch(tgt_tt).transpose(0, 1).contiguous().float()
            ttnn.deallocate(tgt_tt)

        query_with_pos = tgt + tgt_query_pos if tgt_query_pos is not None else tgt
        query_bf = query_with_pos.transpose(0, 1).contiguous()
        ref_pts_bf = tgt_reference_points.transpose(0, 1).contiguous()

        cross_out = self.cross_attn(
            query_bf.to(torch.bfloat16),
            ref_pts_bf,
            memory,
            memory_spatial_shapes,
            memory_level_start_index,
            memory_key_padding_mask,
        )
        cross_out_torch = ttnn.to_torch(cross_out).transpose(0, 1).contiguous().float()
        ttnn.deallocate(cross_out)

        tgt = tgt + cross_out_torch
        tgt_tt = ttnn.from_torch(
            tgt.to(torch.bfloat16).transpose(0, 1).contiguous(),
            layout=ttnn.TILE_LAYOUT, device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        old = tgt_tt
        tgt_tt = ttnn.layer_norm(tgt_tt, weight=self.norm1_w, bias=self.norm1_b)
        ttnn.deallocate(old)

        ffn = ttnn.linear(tgt_tt, self.ffn1_w, bias=self.ffn1_b)
        old = ffn
        ffn = ttnn.relu(ffn)
        ttnn.deallocate(old)
        old = ffn
        ffn = ttnn.linear(ffn, self.ffn2_w, bias=self.ffn2_b)
        ttnn.deallocate(old)
        old = tgt_tt
        tgt_tt = ttnn.add(tgt_tt, ffn)
        ttnn.deallocate(old)
        ttnn.deallocate(ffn)
        old = tgt_tt
        tgt_tt = ttnn.layer_norm(tgt_tt, weight=self.norm3_w, bias=self.norm3_b)
        ttnn.deallocate(old)

        result = ttnn.to_torch(tgt_tt).transpose(0, 1).contiguous().float()
        ttnn.deallocate(tgt_tt)
        return result


class TTDeformableDecoder(LightweightModule):

    def __init__(self, device, state_dict, prefix, n_layers=6, d_model=256, d_ffn=2048,
                 n_levels=5, n_heads=8, n_points=4):
        super().__init__()
        p = f"{prefix}." if prefix else ""
        self.layers = []
        for i in range(n_layers):
            self.layers.append(
                TTDeformableDecoderLayer(
                    device, state_dict, f"{p}{i}",
                    d_model, d_ffn, n_levels, n_heads, n_points
                )
            )

    def forward(self, tgt, tgt_query_pos, tgt_reference_points, memory,
                memory_spatial_shapes, memory_level_start_index,
                memory_key_padding_mask=None, self_attn_mask=None):
        output = tgt
        for layer in self.layers:
            output = layer(
                output, tgt_query_pos, tgt_reference_points, memory,
                memory_spatial_shapes, memory_level_start_index,
                memory_key_padding_mask, self_attn_mask,
            )
        return output
