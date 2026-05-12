# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TTNN [`SeamlessM4Tv2SpeechEncoder`] — inference, pure device ops in ``forward``."""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple

import numpy as np
import ttnn
import torch


def _core_grid(device: ttnn.Device) -> ttnn.CoreGrid:
    grid = device.compute_with_storage_grid_size()
    return ttnn.CoreGrid(y=grid.y, x=grid.x)


# Match ``torch.finfo(torch.bfloat16).min`` used by HF attention masking.
_BF16_ATTN_MASK_MIN = float(torch.finfo(torch.bfloat16).min)


class TTSeamlessM4Tv2SpeechEncoder:
    """
    Device port of Hugging Face ``SeamlessM4Tv2SpeechEncoder`` (Conformer stack + adapter).

    Use ``create_speech_encoder_parameters`` for weights. ``forward`` uses only ``ttnn`` ops;
    host ``numpy`` is used only to build relative-position index tables (no PyTorch ops in
    ``forward``).
    """

    def __init__(
        self,
        device: ttnn.Device,
        parameters: Any,
        *,
        hidden_size: int,
        feature_projection_input_dim: int,
        speech_encoder_attention_heads: int,
        speech_encoder_intermediate_size: int,
        speech_encoder_layers: int,
        layer_norm_eps: float,
        speech_encoder_chunk_size: Optional[int],
        speech_encoder_left_chunk_num: int,
    ):
        self.device = device
        self.parameters = parameters
        self.hidden_size = hidden_size
        self.feature_projection_input_dim = feature_projection_input_dim
        self.speech_encoder_attention_heads = speech_encoder_attention_heads
        self.speech_encoder_intermediate_size = speech_encoder_intermediate_size
        self.speech_encoder_layers = speech_encoder_layers
        self.layer_norm_eps = layer_norm_eps
        self.speech_encoder_chunk_size = speech_encoder_chunk_size
        self.speech_encoder_left_chunk_num = speech_encoder_left_chunk_num
        self.has_adapter = parameters.adapter is not None

        self._compute = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def _linear(self, x: ttnn.Tensor, weight: ttnn.Tensor, bias: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.linear(
            x,
            weight,
            bias=bias,
            core_grid=_core_grid(self.device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self._compute,
        )

    def _layer_norm(self, x: ttnn.Tensor, *, weight: ttnn.Tensor, bias: ttnn.Tensor, eps: float) -> ttnn.Tensor:
        return ttnn.layer_norm(
            x,
            weight=weight,
            bias=bias,
            epsilon=eps,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _conv1d(
        self,
        x_nlc: ttnn.Tensor,
        *,
        weight: ttnn.Tensor,
        bias: Optional[ttnn.Tensor],
        batch: int,
        input_length: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        groups: int,
        dilation: int = 1,
    ) -> Tuple[ttnn.Tensor, int]:
        conv_config = ttnn.Conv1dConfig(
            weights_dtype=ttnn.bfloat16,
            shard_layout=None,
            deallocate_activation=False,
        )
        out, out_len = ttnn.conv1d(
            input_tensor=x_nlc,
            weight_tensor=weight,
            in_channels=in_channels,
            out_channels=out_channels,
            device=self.device,
            bias_tensor=bias,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            batch_size=batch,
            input_length=input_length,
            conv_config=conv_config,
            compute_config=self._compute,
            groups=groups,
            dilation=dilation,
            dtype=ttnn.bfloat16,
            return_output_dim=True,
        )
        out_len = int(out_len)
        if ttnn.is_sharded(out):
            out = ttnn.sharded_to_interleaved(out, ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.reshape(out, (batch, out_len, out_channels))
        if ttnn.is_sharded(out):
            out = ttnn.sharded_to_interleaved(out, ttnn.DRAM_MEMORY_CONFIG)
        return out, out_len

    @staticmethod
    def _heads(x: ttnn.Tensor, batch: int, seq: int, num_heads: int, head_dim: int) -> ttnn.Tensor:
        x = ttnn.reshape(x, (batch, seq, num_heads, head_dim))
        return ttnn.permute(x, (0, 2, 1, 3))

    @staticmethod
    def _merge_heads(
        x: ttnn.Tensor, batch: int, seq: int, num_heads: int, head_dim: int, hidden_size: int
    ) -> ttnn.Tensor:
        x = ttnn.permute(x, (0, 2, 1, 3))
        return ttnn.reshape(x, (batch, seq, hidden_size))

    def _relative_embedding_table(
        self,
        seq_len: int,
        *,
        distance_weight: ttnn.Tensor,
        left_max: int,
        right_max: int,
    ) -> ttnn.Tensor:
        r = np.arange(seq_len, dtype=np.int64)
        l = np.arange(seq_len, dtype=np.int64)
        dist = r[np.newaxis, :] - l[:, np.newaxis]
        dist = np.clip(dist, -left_max, right_max) + left_max
        dist_u32 = dist.astype(np.uint32).reshape(-1)
        idx_tt = ttnn.from_torch(
            dist_u32,
            device=self.device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        emb = ttnn.embedding(
            idx_tt,
            distance_weight,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(idx_tt)
        emb = ttnn.reshape(emb, (seq_len, seq_len, self.hidden_size // self.speech_encoder_attention_heads))
        if emb.get_layout() != ttnn.TILE_LAYOUT:
            return ttnn.to_layout(emb, ttnn.TILE_LAYOUT)
        return emb

    def _mh_attention(
        self,
        hidden_states: ttnn.Tensor,
        attn_module: Any,
        attention_mask_4d: Optional[ttnn.Tensor],
        *,
        batch: int,
        seq_len: int,
        use_relative: bool,
    ) -> ttnn.Tensor:
        num_heads = self.speech_encoder_attention_heads
        head_dim = self.hidden_size // num_heads
        hsz = self.hidden_size

        q = self._linear(hidden_states, attn_module.linear_q.weight, attn_module.linear_q.bias)
        k = self._linear(hidden_states, attn_module.linear_k.weight, attn_module.linear_k.bias)
        v = self._linear(hidden_states, attn_module.linear_v.weight, attn_module.linear_v.bias)

        qh = self._heads(q, batch, seq_len, num_heads, head_dim)
        kh = self._heads(k, batch, seq_len, num_heads, head_dim)
        vh = self._heads(v, batch, seq_len, num_heads, head_dim)
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        kh_t = ttnn.permute(kh, (0, 1, 3, 2))
        scores = ttnn.matmul(
            qh,
            kh_t,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(kh_t)
        ttnn.deallocate(kh)
        scale = 1.0 / math.sqrt(head_dim)
        scores = ttnn.multiply(scores, scale, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        if use_relative:
            pos_tab = self._relative_embedding_table(
                seq_len,
                distance_weight=attn_module.distance_embedding.weight,
                left_max=int(attn_module.left_max_position_embeddings),
                right_max=int(attn_module.right_max_position_embeddings),
            )
            qh_exp = ttnn.reshape(qh, (batch, num_heads, seq_len, 1, head_dim))
            pos_exp = ttnn.reshape(pos_tab, (1, 1, seq_len, seq_len, head_dim))
            prod = ttnn.multiply(qh_exp, pos_exp, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(qh_exp)
            ttnn.deallocate(pos_tab)
            rel_logits = ttnn.sum(prod, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(prod)
            rel_logits = ttnn.multiply(rel_logits, scale, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            scores = ttnn.add(scores, rel_logits, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(rel_logits)

        ttnn.deallocate(qh)

        if attention_mask_4d is not None:
            scores = ttnn.add(scores, attention_mask_4d, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        probs = ttnn.softmax(scores, dim=-1, numeric_stable=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(scores)

        attn_out = ttnn.matmul(probs, vh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(probs)
        ttnn.deallocate(vh)

        attn_out = ttnn.permute(attn_out, (0, 2, 1, 3))
        attn_out = ttnn.reshape(attn_out, (batch, seq_len, hsz))
        out = self._linear(attn_out, attn_module.linear_out.weight, attn_module.linear_out.bias)
        ttnn.deallocate(attn_out)
        return out

    def _glu_last_dim(self, x: ttnn.Tensor, *, batch: int, seq_len: int, width: int) -> ttnn.Tensor:
        half = width // 2
        a = ttnn.slice(x, [0, 0, 0], [batch, seq_len, half], [1, 1, 1])
        b = ttnn.slice(x, [0, 0, half], [batch, seq_len, width], [1, 1, 1])
        sig = ttnn.sigmoid(b, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(b)
        out = ttnn.mul(a, sig, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(sig)
        ttnn.deallocate(a)
        return out

    def _conformer_ffn(self, x: ttnn.Tensor, ffn: Any) -> ttnn.Tensor:
        h = self._linear(x, ffn.intermediate_dense.weight, ffn.intermediate_dense.bias)
        h = ttnn.silu(h, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return self._linear(h, ffn.output_dense.weight, ffn.output_dense.bias)

    def _relu_ffn(self, x: ttnn.Tensor, ffn: Any) -> ttnn.Tensor:
        h = self._linear(x, ffn.intermediate_dense.weight, ffn.intermediate_dense.bias)
        h = ttnn.relu(h, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return self._linear(h, ffn.output_dense.weight, ffn.output_dense.bias)

    def _conv_module(
        self,
        hidden: ttnn.Tensor,
        cm: Any,
        conv_mask_1d: Optional[ttnn.Tensor],
        *,
        batch: int,
        seq_len: int,
        hidden_size: int,
    ) -> ttnn.Tensor:
        h = self._layer_norm(
            hidden,
            weight=cm.layer_norm.weight,
            bias=cm.layer_norm.bias,
            eps=float(cm.layer_norm.eps),
        )
        if conv_mask_1d is not None:
            m = ttnn.reshape(conv_mask_1d, (batch, seq_len, 1))
            h = ttnn.mul(h, m, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        pc1 = cm.pointwise_conv1
        h, t1 = self._conv1d(
            h,
            weight=pc1.weight,
            bias=pc1.bias,
            batch=batch,
            input_length=seq_len,
            in_channels=pc1.in_channels,
            out_channels=pc1.out_channels,
            kernel_size=pc1.kernel_size,
            stride=pc1.stride,
            padding=pc1.padding,
            groups=pc1.groups,
        )
        h = self._glu_last_dim(h, batch=batch, seq_len=t1, width=pc1.out_channels)

        lp = int(cm.depthwise_conv.left_pad)
        if lp > 0:
            pad_tensor = ttnn.zeros(
                (batch, lp, hidden_size),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            h = ttnn.concat([pad_tensor, h], dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(pad_tensor)
            t_len = t1 + lp
        else:
            t_len = t1

        dw = cm.depthwise_conv
        h, t2 = self._conv1d(
            h,
            weight=dw.weight,
            bias=dw.bias,
            batch=batch,
            input_length=t_len,
            in_channels=dw.in_channels,
            out_channels=dw.out_channels,
            kernel_size=dw.kernel_size,
            stride=dw.stride,
            padding=dw.padding,
            groups=dw.groups,
            dilation=1,
        )
        dln = cm.depthwise_layer_norm
        h = self._layer_norm(h, weight=dln.weight, bias=dln.bias, eps=float(dln.eps))
        h = ttnn.silu(h, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        pc2 = cm.pointwise_conv2
        h, _ = self._conv1d(
            h,
            weight=pc2.weight,
            bias=pc2.bias,
            batch=batch,
            input_length=t2,
            in_channels=pc2.in_channels,
            out_channels=pc2.out_channels,
            kernel_size=pc2.kernel_size,
            stride=pc2.stride,
            padding=pc2.padding,
            groups=pc2.groups,
        )
        return h

    def _encoder_additive_mask(
        self,
        conv_mask_1d: Optional[ttnn.Tensor],
        *,
        batch: int,
        seq_len: int,
        dtype: ttnn.DataType,
    ) -> Optional[ttnn.Tensor]:
        bad_parts = []
        if conv_mask_1d is not None:
            m = ttnn.reshape(conv_mask_1d, (batch, 1, 1, seq_len))
            one = ttnn.ones(m.shape, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=self.device)
            inv = ttnn.subtract(one, m, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(one)
            row_bad = inv
            row_list = [row_bad] * seq_len
            pad_bad = ttnn.concat(row_list, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(row_bad)
            bad_parts.append(pad_bad)

        chunk_bad = self._chunk_attention_mask_float01(batch, seq_len, dtype)
        if chunk_bad is not None:
            bad_parts.append(chunk_bad)

        if not bad_parts:
            return None

        bad = bad_parts[0]
        for extra in bad_parts[1:]:
            s = ttnn.add(bad, extra, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(bad)
            ttnn.deallocate(extra)
            cap = ttnn.ones(s.shape, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=self.device)
            bad = ttnn.minimum(s, cap, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(s)
            ttnn.deallocate(cap)

        out = ttnn.multiply(bad, _BF16_ATTN_MASK_MIN, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(bad)
        return out

    def _chunk_attention_mask_float01(self, batch: int, seq_len: int, dtype: ttnn.DataType) -> Optional[ttnn.Tensor]:
        cs = self.speech_encoder_chunk_size
        if cs is None:
            return None
        lc = self.speech_encoder_left_chunk_num
        chunk_np = np.zeros((1, 1, seq_len, seq_len), dtype=np.float32)
        chunk_indices = np.arange(seq_len, dtype=np.int64) // cs
        start_indices = np.zeros(seq_len, dtype=np.int64)
        if lc >= 0:
            start_indices = np.clip(chunk_indices - lc, 0, None) * cs
        end_indices = np.minimum((chunk_indices + 1) * cs, seq_len)
        idx_cols = np.arange(seq_len, dtype=np.int64)
        for qi in range(seq_len):
            bad = (idx_cols < start_indices[qi]) | (idx_cols >= end_indices[qi])
            chunk_np[0, 0, qi, bad] = 1.0
        chunk_tt = ttnn.from_torch(
            torch.from_numpy(chunk_np).to(torch.bfloat16),
            device=self.device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if batch == 1:
            return chunk_tt
        out = ttnn.concat([chunk_tt for _ in range(batch)], dim=0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(chunk_tt)
        return out

    def _expand_attention_mask_2d_to_4d(self, mask_2d: ttnn.Tensor, *, batch: int, s: int) -> ttnn.Tensor:
        """HF ``AttentionMaskConverter._expand_mask`` — ``mask`` is 1 keep, 0 pad."""
        m = ttnn.reshape(mask_2d, (batch, 1, 1, s))
        one = ttnn.ones(m.shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
        inv = ttnn.subtract(one, m, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(one)
        inv_pos = ttnn.gt(inv, 0.0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        neg = ttnn.full(
            inv.shape, _BF16_ATTN_MASK_MIN, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )
        zero = ttnn.zeros(inv.shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
        add = ttnn.where(inv_pos, neg, zero, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(inv_pos)
        ttnn.deallocate(neg)
        ttnn.deallocate(zero)
        ttnn.deallocate(inv)
        rows = []
        for _ in range(s):
            rows.append(add)
        return ttnn.concat(rows, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _adapter_subsample_lengths(
        self, conv_mask_1d: ttnn.Tensor, *, kernel: int, stride: int, pad: int
    ) -> ttnn.Tensor:
        """Per-batch subsampled lengths after strided conv (HF adapter)."""
        s = ttnn.sum(conv_mask_1d, dim=1, keepdim=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        two_pad = float(2 * pad)
        padded = ttnn.add(s, two_pad, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(s)
        num = ttnn.subtract(padded, float(kernel), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(padded)
        scaled = ttnn.divide(num, float(stride), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(num)
        one = ttnn.ones(scaled.shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
        out = ttnn.add(scaled, one, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(scaled)
        ttnn.deallocate(one)
        return ttnn.floor(out, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _adapter_new_attention_mask(self, seq_len_out: int, seq_lens: ttnn.Tensor, *, batch: int) -> ttnn.Tensor:
        """``[B, seq_len_out]`` with 1 for valid positions (approximate floor parity via BF16)."""
        idx = ttnn.arange(0, seq_len_out, 1, device=self.device, dtype=ttnn.bfloat16)
        idx = ttnn.reshape(idx, (1, seq_len_out))
        lens = ttnn.reshape(seq_lens, (batch, 1))
        ok = ttnn.lt(idx, lens, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        one = ttnn.ones(ok.shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
        zero = ttnn.zeros(ok.shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
        mask = ttnn.where(ok, one, zero, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(ok)
        ttnn.deallocate(one)
        ttnn.deallocate(zero)
        ttnn.deallocate(idx)
        return mask

    def _conformer_encoder_layer(
        self,
        hidden: ttnn.Tensor,
        layer: Any,
        attention_mask_4d: Optional[ttnn.Tensor],
        conv_mask_1d: Optional[ttnn.Tensor],
        *,
        batch: int,
        seq_len: int,
    ) -> ttnn.Tensor:
        hsz = self.hidden_size

        res = hidden
        h = self._layer_norm(
            hidden,
            weight=layer.ffn1_layer_norm.weight,
            bias=layer.ffn1_layer_norm.bias,
            eps=self.layer_norm_eps,
        )
        ff = self._conformer_ffn(h, layer.ffn1)
        ttnn.deallocate(h)
        ff = ttnn.multiply(ff, 0.5, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        hidden = ttnn.add(res, ff, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(ff)
        ttnn.deallocate(res)

        res = hidden
        h = self._layer_norm(
            hidden,
            weight=layer.self_attn_layer_norm.weight,
            bias=layer.self_attn_layer_norm.bias,
            eps=self.layer_norm_eps,
        )
        attn = self._mh_attention(
            h,
            layer.self_attn,
            attention_mask_4d,
            batch=batch,
            seq_len=seq_len,
            use_relative=True,
        )
        ttnn.deallocate(h)
        hidden = ttnn.add(res, attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(attn)
        ttnn.deallocate(res)

        res = hidden
        conv = self._conv_module(hidden, layer.conv_module, conv_mask_1d, batch=batch, seq_len=seq_len, hidden_size=hsz)
        hidden = ttnn.add(res, conv, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(conv)
        ttnn.deallocate(res)

        res = hidden
        h = self._layer_norm(
            hidden,
            weight=layer.ffn2_layer_norm.weight,
            bias=layer.ffn2_layer_norm.bias,
            eps=self.layer_norm_eps,
        )
        ff2 = self._conformer_ffn(h, layer.ffn2)
        ttnn.deallocate(h)
        ff2 = ttnn.multiply(ff2, 0.5, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        hidden = ttnn.add(res, ff2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(ff2)
        ttnn.deallocate(res)

        return self._layer_norm(
            hidden,
            weight=layer.final_layer_norm.weight,
            bias=layer.final_layer_norm.bias,
            eps=self.layer_norm_eps,
        )

    def _adapter_layer(
        self,
        hidden: ttnn.Tensor,
        layer: Any,
        conv_mask_1d: Optional[ttnn.Tensor],
        *,
        batch: int,
        seq_len: int,
    ) -> ttnn.Tensor:
        hsz = self.hidden_size

        res_branch = self._layer_norm(
            hidden,
            weight=layer.residual_layer_norm.weight,
            bias=layer.residual_layer_norm.bias,
            eps=self.layer_norm_eps,
        )
        rc = layer.residual_conv
        res_out, lens_r = self._conv1d(
            res_branch,
            weight=rc.weight,
            bias=rc.bias,
            batch=batch,
            input_length=seq_len,
            in_channels=rc.in_channels,
            out_channels=rc.out_channels,
            kernel_size=rc.kernel_size,
            stride=rc.stride,
            padding=rc.padding,
            groups=rc.groups,
        )
        ttnn.deallocate(res_branch)
        res_out = self._glu_last_dim(res_out, batch=batch, seq_len=lens_r, width=rc.out_channels)

        h = self._layer_norm(
            hidden,
            weight=layer.self_attn_layer_norm.weight,
            bias=layer.self_attn_layer_norm.bias,
            eps=self.layer_norm_eps,
        )
        sc = layer.self_attn_conv
        attn_in, lens_a = self._conv1d(
            h,
            weight=sc.weight,
            bias=sc.bias,
            batch=batch,
            input_length=seq_len,
            in_channels=sc.in_channels,
            out_channels=sc.out_channels,
            kernel_size=sc.kernel_size,
            stride=sc.stride,
            padding=sc.padding,
            groups=sc.groups,
        )
        ttnn.deallocate(h)
        attn_in = self._glu_last_dim(attn_in, batch=batch, seq_len=lens_a, width=sc.out_channels)
        assert lens_a == lens_r

        attn_4d = None
        if conv_mask_1d is not None:
            sub_lens = self._adapter_subsample_lengths(
                conv_mask_1d, kernel=layer.kernel_size, stride=layer.stride, pad=layer.kernel_size // 2
            )
            mask_2d = self._adapter_new_attention_mask(lens_a, sub_lens, batch=batch)
            ttnn.deallocate(sub_lens)
            attn_4d = self._expand_attention_mask_2d_to_4d(mask_2d, batch=batch, s=lens_a)
            ttnn.deallocate(mask_2d)

        attn = self._mh_attention(
            attn_in,
            layer.self_attn,
            attn_4d,
            batch=batch,
            seq_len=lens_a,
            use_relative=False,
        )
        ttnn.deallocate(attn_in)
        if attn_4d is not None:
            ttnn.deallocate(attn_4d)

        out = ttnn.add(attn, res_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(attn)
        ttnn.deallocate(res_out)

        res2 = out
        h2 = self._layer_norm(
            out,
            weight=layer.ffn_layer_norm.weight,
            bias=layer.ffn_layer_norm.bias,
            eps=self.layer_norm_eps,
        )
        ff = self._relu_ffn(h2, layer.ffn)
        ttnn.deallocate(h2)
        return ttnn.add(res2, ff, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def forward(
        self,
        input_features: ttnn.Tensor,
        *,
        conv_attention_mask_1d: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        """
        Args:
            input_features: ``[batch, seq, feature_projection_input_dim]`` bfloat16 on device.
            conv_attention_mask_1d: optional ``[batch, seq]`` bfloat16 ``1``/``0`` mask (``1`` = real frame),
                same convention as HF ``attention_mask``.

        Returns:
            Last hidden state ``[batch, seq_out, hidden_size]`` (``seq_out`` may differ if adapter subsamples).
        """
        p = self.parameters
        batch = int(input_features.shape[0])
        seq = int(input_features.shape[1])

        fp = p.feature_projection
        h = self._layer_norm(
            input_features,
            weight=fp.layer_norm.weight,
            bias=fp.layer_norm.bias,
            eps=float(fp.layer_norm.eps),
        )
        h = self._linear(h, fp.projection.weight, fp.projection.bias)

        if conv_attention_mask_1d is not None:
            m1 = ttnn.reshape(conv_attention_mask_1d, (batch, seq, 1))
            h = ttnn.mul(h, m1, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        enc = p.encoder
        dtype = ttnn.bfloat16
        attn_4d = self._encoder_additive_mask(conv_attention_mask_1d, batch=batch, seq_len=seq, dtype=dtype)

        for i in range(self.speech_encoder_layers):
            layer = enc.layers[i]
            h = self._conformer_encoder_layer(h, layer, attn_4d, conv_attention_mask_1d, batch=batch, seq_len=seq)

        h = self._layer_norm(
            h,
            weight=enc.layer_norm.weight,
            bias=enc.layer_norm.bias,
            eps=self.layer_norm_eps,
        )

        if attn_4d is not None:
            ttnn.deallocate(attn_4d)

        im = p.intermediate_ffn
        exp = self._relu_ffn(h, im)
        exp_half = ttnn.multiply(exp, 0.5, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(exp)
        h = ttnn.add(h, exp_half, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(exp_half)

        if self.has_adapter:
            for ad_layer in p.adapter.layers:
                h = self._adapter_layer(h, ad_layer, conv_attention_mask_1d, batch=batch, seq_len=int(h.shape[1]))

        return self._layer_norm(
            h,
            weight=p.inner_layer_norm.weight,
            bias=p.inner_layer_norm.bias,
            eps=self.layer_norm_eps,
        )

    def __call__(
        self,
        input_features: ttnn.Tensor,
        *,
        conv_attention_mask_1d: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        return self.forward(input_features, conv_attention_mask_1d=conv_attention_mask_1d)
