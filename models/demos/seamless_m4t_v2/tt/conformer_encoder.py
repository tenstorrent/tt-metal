# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
TT port of the SeamlessM4Tv2 Conformer speech encoder (Phase 3).

Assembles, for batch=1 / unpadded clips (no attention mask needed — the chunk
mask is a no-op at realistic seq lengths):

  feature_projection (LN + Linear 160->1024)
  24 x ConformerEncoderLayer
      macaron FFN1 (x0.5 + res) -> self-attn(rel-pos) + res
      -> conv module + res -> FFN2 (x0.5 + res) -> final LN
  encoder.layer_norm
  intermediate_ffn (relu):  h = h + 0.5 * ffn(h)
  adapter.layers[0] (stride-8 downsample)
  inner_layer_norm

Reuses TtConformerConvModule (Phase 1) and TtConformerSelfAttention (Phase 2).
The two adapter Conv1d's (kernel 8, stride 8) run on host while
`config.conv_cpu_fallback` is set (Phase 7 moves them on-device).
"""

from __future__ import annotations

import torch

import ttnn
from models.demos.seamless_m4t_v2.tt.conformer_attention import TtConformerSelfAttention
from models.demos.seamless_m4t_v2.tt.conformer_conv import TtConformerConvModule


def _to_tt(t, device, dtype=ttnn.bfloat16):
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


class _LayerNorm:
    def __init__(self, weights, prefix, device, eps, dtype=ttnn.bfloat16):
        self.w = _to_tt(weights[prefix + "weight"], device, dtype)
        self.b = _to_tt(weights[prefix + "bias"], device, dtype)
        self.eps = eps

    def __call__(self, x):
        return ttnn.layer_norm(x, weight=self.w, bias=self.b, epsilon=self.eps)


class _FeedForward:
    """ConformerFeedForward: intermediate_dense -> act -> output_dense."""

    def __init__(self, weights, prefix, device, act, dtype=ttnn.bfloat16):
        self.w1 = _to_tt(weights[prefix + "intermediate_dense.weight"].t().contiguous(), device, dtype)
        self.b1 = _to_tt(weights[prefix + "intermediate_dense.bias"], device, dtype)
        self.w2 = _to_tt(weights[prefix + "output_dense.weight"].t().contiguous(), device, dtype)
        self.b2 = _to_tt(weights[prefix + "output_dense.bias"], device, dtype)
        self.act = ttnn.silu if act in ("swish", "silu") else ttnn.relu

    def __call__(self, x):
        x = ttnn.linear(x, self.w1, bias=self.b1)
        x = self.act(x)
        x = ttnn.linear(x, self.w2, bias=self.b2)
        return x


class TtConformerEncoderLayer:
    def __init__(self, weights, config, device, dtype=ttnn.bfloat16):
        eps = config.layer_norm_eps
        act = config.speech_encoder_hidden_act
        self.ffn1_ln = _LayerNorm(weights, "ffn1_layer_norm.", device, eps, dtype)
        self.ffn1 = _FeedForward(weights, "ffn1.", device, act, dtype)
        self.attn_ln = _LayerNorm(weights, "self_attn_layer_norm.", device, eps, dtype)
        self.attn = TtConformerSelfAttention(
            {k[len("self_attn."):]: v for k, v in weights.items() if k.startswith("self_attn.")},
            config, device, use_position_embeddings=True, dtype=dtype,
        )
        self.conv = TtConformerConvModule(
            {k[len("conv_module."):]: v for k, v in weights.items() if k.startswith("conv_module.")},
            config, device, dtype=dtype,
        )
        self.ffn2_ln = _LayerNorm(weights, "ffn2_layer_norm.", device, eps, dtype)
        self.ffn2 = _FeedForward(weights, "ffn2.", device, act, dtype)
        self.final_ln = _LayerNorm(weights, "final_layer_norm.", device, eps, dtype)

    def __call__(self, h):
        residual = h
        h = self.ffn1_ln(h)
        h = self.ffn1(h)
        h = ttnn.add(ttnn.multiply(h, 0.5), residual)

        residual = h
        h = self.attn_ln(h)
        h = self.attn(h)
        h = ttnn.add(h, residual)

        residual = h
        h = self.conv(h)
        h = ttnn.add(residual, h)

        residual = h
        h = self.ffn2_ln(h)
        h = self.ffn2(h)
        h = ttnn.add(ttnn.multiply(h, 0.5), residual)
        h = self.final_ln(h)
        return h


class TtConformerAdapterLayer:
    def __init__(self, weights, config, device, dtype=ttnn.bfloat16):
        eps = config.layer_norm_eps
        self.device = device
        self.dtype = dtype
        self.C = config.hidden_size
        self.kernel = config.adaptor_kernel_size
        self.stride = config.adaptor_stride
        self.pad = self.stride // 2
        self.cpu_fallback = config.conv_cpu_fallback

        self.residual_ln = _LayerNorm(weights, "residual_layer_norm.", device, eps, dtype)
        self.attn_ln = _LayerNorm(weights, "self_attn_layer_norm.", device, eps, dtype)
        self.ffn_ln = _LayerNorm(weights, "ffn_layer_norm.", device, eps, dtype)
        self.ffn = _FeedForward(weights, "ffn.", device, "relu", dtype)
        self.attn = TtConformerSelfAttention(
            {k[len("self_attn."):]: v for k, v in weights.items() if k.startswith("self_attn.")},
            config, device, use_position_embeddings=False, dtype=dtype,
        )
        # host copies (fallback path)
        self.res_conv_w = weights["residual_conv.weight"].float()
        self.res_conv_b = weights["residual_conv.bias"].float()
        self.attn_conv_w = weights["self_attn_conv.weight"].float()
        self.attn_conv_b = weights["self_attn_conv.bias"].float()
        # on-device path (Phase 7): ttnn weights/bias + conv/compute configs
        if not self.cpu_fallback:
            def _w(t):
                return ttnn.from_torch(t.float(), dtype=ttnn.float32)

            def _b(t):
                return ttnn.from_torch(t.float().reshape(1, 1, 1, -1), dtype=ttnn.float32)

            self.res_conv_w_tt, self.res_conv_b_tt = _w(self.res_conv_w), _b(self.res_conv_b)
            self.attn_conv_w_tt, self.attn_conv_b_tt = _w(self.attn_conv_w), _b(self.attn_conv_b)
            self.conv_config = ttnn.Conv1dConfig(weights_dtype=ttnn.bfloat16)
            self.conv_config.act_block_h_override = 32
            self.compute_config = ttnn.init_device_compute_kernel_config(
                device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
            )

    def _conv_glu_host(self, x_tt, w, b):
        """(1,seq,C) ttnn -> Conv1d(C,2C,k=8,s=8,p=4) -> GLU -> (1,seq_out,C) ttnn."""
        x = ttnn.to_torch(x_tt).float().transpose(1, 2)  # (1, C, seq)
        x = torch.nn.functional.conv1d(x, w, b, stride=self.stride, padding=self.pad)  # (1, 2C, seq_out)
        x = torch.nn.functional.glu(x, dim=1)  # (1, C, seq_out)
        x = x.transpose(1, 2).contiguous()  # (1, seq_out, C)
        return _to_tt(x, self.device, self.dtype)

    def _conv_glu_device(self, x_tt, w_tt, b_tt):
        """On-device Conv1d(C,2C,k=8,s=8,p=4) + GLU. (1,seq,C) -> (1,seq_out,C)."""
        seq = x_tt.shape[1]
        x = ttnn.to_layout(x_tt, ttnn.ROW_MAJOR_LAYOUT)
        out, out_len = ttnn.conv1d(
            input_tensor=x, weight_tensor=w_tt, bias_tensor=b_tt,
            in_channels=self.C, out_channels=2 * self.C, device=self.device,
            kernel_size=self.kernel, stride=self.stride, padding=self.pad, groups=1,
            batch_size=1, input_length=seq,
            conv_config=self.conv_config, compute_config=self.compute_config,
            dtype=self.dtype, return_output_dim=True,
        )
        out = ttnn.sharded_to_interleaved(out, ttnn.L1_MEMORY_CONFIG)
        out = ttnn.reshape(out, [1, out_len, 2 * self.C])
        out = ttnn.to_layout(out, ttnn.TILE_LAYOUT)
        a = out[..., : self.C]
        b = out[..., self.C :]
        return ttnn.multiply(a, ttnn.sigmoid(b))  # (1, seq_out, C)

    def _conv_glu(self, x_tt, host_w, host_b, w_tt, b_tt):
        if self.cpu_fallback:
            return self._conv_glu_host(x_tt, host_w, host_b)
        return self._conv_glu_device(x_tt, w_tt, b_tt)

    def __call__(self, h):
        residual = self.residual_ln(h)
        residual = self._conv_glu(
            residual, self.res_conv_w, self.res_conv_b,
            getattr(self, "res_conv_w_tt", None), getattr(self, "res_conv_b_tt", None),
        )

        hs = self.attn_ln(h)
        hs = self._conv_glu(
            hs, self.attn_conv_w, self.attn_conv_b,
            getattr(self, "attn_conv_w_tt", None), getattr(self, "attn_conv_b_tt", None),
        )
        hs = self.attn(hs)
        hs = ttnn.add(hs, residual)

        residual = hs
        hs = self.ffn_ln(hs)
        hs = ttnn.add(self.ffn(hs), residual)
        return hs


class TtSpeechEncoder:
    def __init__(self, state_dict, config, device, dtype=ttnn.bfloat16):
        """state_dict: HF `speech_encoder.` slice (prefix stripped)."""
        eps = config.layer_norm_eps
        self.device = device
        self.dtype = dtype

        def slice_pfx(pfx):
            return {k[len(pfx):]: v for k, v in state_dict.items() if k.startswith(pfx)}

        # feature projection
        self.fp_ln = _LayerNorm(slice_pfx("feature_projection."), "layer_norm.", device, eps, dtype)
        fp = slice_pfx("feature_projection.")
        self.fp_w = _to_tt(fp["projection.weight"].t().contiguous(), device, dtype)
        self.fp_b = _to_tt(fp["projection.bias"], device, dtype)

        # conformer layers
        self.layers = [
            TtConformerEncoderLayer(slice_pfx(f"encoder.layers.{i}."), config, device, dtype)
            for i in range(config.speech_encoder_layers)
        ]
        self.encoder_ln = _LayerNorm(slice_pfx("encoder."), "layer_norm.", device, eps, dtype)

        # intermediate ffn (relu) + adapter + inner norm
        self.intermediate_ffn = _FeedForward(slice_pfx("intermediate_ffn."), "", device, "relu", dtype)
        self.adapter = TtConformerAdapterLayer(slice_pfx("adapter.layers.0."), config, device, dtype)
        self.inner_ln = _LayerNorm(state_dict, "inner_layer_norm.", device, eps, dtype)

    def __call__(self, input_features):
        """input_features: (1, seq, 160) ttnn TILE. Returns (1, seq_out, 1024)."""
        h = self.fp_ln(input_features)
        h = ttnn.linear(h, self.fp_w, bias=self.fp_b)
        for layer in self.layers:
            h = layer(h)
        h = self.encoder_ln(h)
        expanded = self.intermediate_ffn(h)
        h = ttnn.add(h, ttnn.multiply(expanded, 0.5))
        h = self.adapter(h)
        h = self.inner_ln(h)
        return h
