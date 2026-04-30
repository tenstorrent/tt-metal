"""TTNN VAE Decoder model for Tongyi-MAI/Z-Image-Turbo.

Modular implementation that produces EXACTLY the same TTNN kernel sequence
as the flat 5624-line original. Module boundaries use NCHW BF16 format
(the divide-by-ones output), except where noted.

Key format conventions:
  - "conv format" = flat BF16 tensor from conv2d, shape varies
  - "NCHW BF16" = [1, C, H, W] BF16 after divide-by-ones
  - GroupNormSiLU always outputs conv format [1, 1, H*W, C] BF16
  - The first resnet after conv_in and shortcut resnets receive conv format
  - Normal resnets and attention receive NCHW BF16
"""

from vae.model_pt import SCALING_FACTOR, SHIFT_FACTOR, VaeDecoderPT
from vae.params import load_weights

import ttnn

DRAM = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)


class LightweightModule:
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


# ---------------------------------------------------------------------------
# GroupNormSiLU
# ---------------------------------------------------------------------------
# Two entry points matching the flat code exactly:
#   from_conv_format: conv_fmt → typecast F32 → reshape NHWC → permute NCHW
#                     → reshape GN → [norm core] → SiLU → typecast BF16
#                     → reshape NCHW → permute NHWC → reshape conv_fmt
#   from_nchw:        NCHW BF16 → typecast F32 → reshape GN
#                     → [norm core] → SiLU → typecast BF16
#                     → reshape NCHW → permute NHWC → reshape conv_fmt


class GroupNormSiLU(LightweightModule):
    """GroupNorm (32 groups) + SiLU.  Returns conv format [1, 1, H*W, C] BF16."""

    def __init__(self, weight, bias, gn_inv, gn_eps, channels, h, w):
        self.weight = weight
        self.bias = bias
        self.gn_inv = gn_inv
        self.gn_eps = gn_eps
        self.cpg = channels // 32
        self.spatial = h * w
        self.channels = channels
        self.h = h
        self.w = w

    def _norm_core(self, gn):
        """GN format [1,32,cpg,spatial] F32 → normed+scaled+biased [same shape] F32.
        Deallocates gn."""
        s = ttnn.sum(gn, [2, 3], True, memory_config=DRAM, compute_kernel_config=None)
        mean = ttnn.multiply(s, self.gn_inv, dtype=ttnn.DataType.FLOAT32, memory_config=DRAM)
        ttnn.deallocate(s, False)

        centered = ttnn.subtract(gn, mean, dtype=ttnn.DataType.FLOAT32, memory_config=DRAM)
        ttnn.deallocate(mean, False)
        ttnn.deallocate(gn, False)

        sq = ttnn.multiply(centered, centered, dtype=ttnn.DataType.FLOAT32, memory_config=DRAM)
        s2 = ttnn.sum(sq, [2, 3], True, memory_config=DRAM, compute_kernel_config=None)
        ttnn.deallocate(sq, False)
        var = ttnn.multiply(s2, self.gn_inv, dtype=ttnn.DataType.FLOAT32, memory_config=DRAM)
        ttnn.deallocate(s2, False)

        var_eps = ttnn.add(var, self.gn_eps, dtype=ttnn.DataType.FLOAT32, memory_config=DRAM)
        ttnn.deallocate(var, False)
        inv_std = ttnn.rsqrt(var_eps, fast_and_approximate_mode=False, memory_config=DRAM)
        ttnn.deallocate(var_eps, False)

        normed = ttnn.multiply(centered, inv_std, dtype=ttnn.DataType.FLOAT32, memory_config=DRAM)
        ttnn.deallocate(inv_std, False)
        ttnn.deallocate(centered, False)

        scaled = ttnn.multiply(normed, self.weight, dtype=ttnn.DataType.FLOAT32, memory_config=DRAM)
        ttnn.deallocate(normed, False)
        shifted = ttnn.add(scaled, self.bias, dtype=ttnn.DataType.FLOAT32, memory_config=DRAM)
        ttnn.deallocate(scaled, False)
        return shifted

    def _silu_to_conv_fmt(self, x_f32):
        """[1,32,cpg,spatial] F32 → SiLU → BF16 → NCHW → NHWC → conv_fmt. Deallocates x_f32."""
        activated = ttnn.silu(x_f32, memory_config=DRAM)
        ttnn.deallocate(x_f32, False)
        bf16 = ttnn.typecast(activated, ttnn.DataType.BFLOAT16, memory_config=DRAM)
        ttnn.deallocate(activated, False)

        nchw = ttnn.reshape(bf16, [1, self.channels, self.h, self.w], memory_config=DRAM)
        ttnn.deallocate(bf16, False)
        nhwc = ttnn.permute(nchw, [0, 2, 3, 1], memory_config=DRAM, pad_value=0.0)
        ttnn.deallocate(nchw, False)
        conv_fmt = ttnn.reshape(nhwc, [1, 1, self.spatial, self.channels], memory_config=DRAM)
        ttnn.deallocate(nhwc, False)
        return conv_fmt

    def from_conv_format(self, conv_fmt):
        """Conv format input → GN+SiLU → conv format output. Does NOT deallocate input."""
        f32 = ttnn.typecast(conv_fmt, ttnn.DataType.FLOAT32, memory_config=DRAM)
        nhwc = ttnn.reshape(f32, [1, self.h, self.w, self.channels], memory_config=DRAM)
        ttnn.deallocate(f32, False)
        nchw = ttnn.permute(nhwc, [0, 3, 1, 2], memory_config=DRAM, pad_value=0.0)
        ttnn.deallocate(nhwc, False)
        gn = ttnn.reshape(nchw, [1, 32, self.cpg, self.spatial], memory_config=DRAM)
        ttnn.deallocate(nchw, False)

        shifted = self._norm_core(gn)
        return self._silu_to_conv_fmt(shifted)

    def from_nchw(self, x_nchw):
        """NCHW BF16 input → GN+SiLU → conv format output. Does NOT deallocate input."""
        f32 = ttnn.typecast(x_nchw, ttnn.DataType.FLOAT32, memory_config=DRAM)
        gn = ttnn.reshape(f32, [1, 32, self.cpg, self.spatial], memory_config=DRAM)
        ttnn.deallocate(f32, False)

        shifted = self._norm_core(gn)
        return self._silu_to_conv_fmt(shifted)

    def forward(self, x_nchw):
        """Default: from NCHW BF16. Does NOT deallocate input."""
        return self.from_nchw(x_nchw)


# ---------------------------------------------------------------------------
# ResnetBlock2D
# ---------------------------------------------------------------------------


class ResnetBlock2D(LightweightModule):
    """Residual block matching the flat code exactly.

    Three modes:
      a) conv_format_input=True, has_conv_shortcut=True:
         Input is conv format (from upsample). Shortcut + norm from conv format.
      b) conv_format_input=True, has_conv_shortcut=False:
         Input is conv format (first resnet after conv_in). No shortcut.
      c) conv_format_input=False (default):
         Input is NCHW BF16. Norm from NCHW.

    Output: always NCHW BF16 (after divide-by-ones).
    """

    def __init__(
        self,
        weights,
        prefix,
        device,
        gn_inv_in,
        gn_inv_out,
        gn_eps,
        ones_4d,
        in_channels,
        out_channels,
        h,
        w,
        conv_act_block_h,
        conv1_slice_type,
        conv2_slice_type=None,
        has_conv_shortcut=False,
        conv_format_input=False,
    ):
        self.device = device
        self.ones_4d = ones_4d
        self.has_conv_shortcut = has_conv_shortcut
        self.conv_format_input = conv_format_input
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.h = h
        self.w = w

        self.norm1 = GroupNormSiLU(
            weights[f"{prefix}.norm1.weight"],
            weights[f"{prefix}.norm1.bias"],
            gn_inv_in,
            gn_eps,
            in_channels,
            h,
            w,
        )
        self.norm2 = GroupNormSiLU(
            weights[f"{prefix}.norm2.weight"],
            weights[f"{prefix}.norm2.bias"],
            gn_inv_out,
            gn_eps,
            out_channels,
            h,
            w,
        )

        self.conv1_weight = weights[f"{prefix}.conv1.weight"]
        self.conv1_bias = weights[f"{prefix}.conv1.bias"]
        self.conv2_weight = weights[f"{prefix}.conv2.weight"]
        self.conv2_bias = weights[f"{prefix}.conv2.bias"]
        self.conv_act_block_h = conv_act_block_h
        self.conv1_slice_type = conv1_slice_type
        self.conv2_slice_type = conv2_slice_type if conv2_slice_type is not None else conv1_slice_type

        if has_conv_shortcut:
            self.shortcut_weight = weights[f"{prefix}.conv_shortcut.weight"]
            self.shortcut_bias = weights[f"{prefix}.conv_shortcut.bias"]

    def _conv2d(self, conv_input, weight, bias, in_ch, out_ch, slice_type):
        """conv2d on conv-format input. Returns conv-format output."""
        return ttnn.conv2d(
            input_tensor=conv_input,
            weight_tensor=weight,
            device=self.device,
            in_channels=in_ch,
            out_channels=out_ch,
            batch_size=1,
            input_height=self.h,
            input_width=self.w,
            kernel_size=[3, 3],
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=[1, 1],
            groups=1,
            bias_tensor=bias,
            conv_config=ttnn.Conv2dConfig(
                weights_dtype=ttnn.DataType.BFLOAT16,
                deallocate_activation=True,
                config_tensors_in_dram=True,
                act_block_h_override=self.conv_act_block_h,
                enable_kernel_stride_folding=False,
            ),
            compute_config=None,
            slice_config=ttnn.Conv2dSliceConfig(slice_type=slice_type, num_slices=0),
            memory_config=DRAM,
        )

    def _conv_fmt_to_nchw_bf16(self, conv_fmt, channels):
        """Conv format BF16 → NCHW BF16. Deallocates input.
        Matches flat code: reshape NHWC → permute NCHW (no typecast)."""
        nhwc = ttnn.reshape(conv_fmt, [1, self.h, self.w, channels], memory_config=DRAM)
        ttnn.deallocate(conv_fmt, False)
        nchw = ttnn.permute(nhwc, [0, 3, 1, 2], memory_config=DRAM, pad_value=0.0)
        ttnn.deallocate(nhwc, False)
        return nchw

    def forward(self, x):
        if self.has_conv_shortcut:
            # x is conv format (from upsample output)
            # Shortcut: conv_shortcut on the conv format input
            shortcut = ttnn.conv2d(
                input_tensor=x,
                weight_tensor=self.shortcut_weight,
                device=self.device,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                batch_size=1,
                input_height=self.h,
                input_width=self.w,
                kernel_size=[1, 1],
                stride=[1, 1],
                padding=[0, 0, 0, 0],
                dilation=[1, 1],
                groups=1,
                bias_tensor=self.shortcut_bias,
                conv_config=ttnn.Conv2dConfig(
                    weights_dtype=ttnn.DataType.BFLOAT16,
                    deallocate_activation=True,
                    config_tensors_in_dram=True,
                    act_block_h_override=0,
                    enable_kernel_stride_folding=False,
                ),
                compute_config=None,
                slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
                memory_config=DRAM,
            )

            # norm1 from conv format (does NOT deallocate x, shortcut conv consumed it via deallocate_activation)
            # But wait - the flat code uses conv2d_18 for BOTH shortcut input AND norm1 input.
            # The shortcut conv has deallocate_activation=True so it consumed x.
            # But the flat code then does typecast(conv2d_18, F32) AFTER shortcut.
            # Actually looking at the flat code: conv2d_19 = conv_shortcut(conv2d_18),
            # then typecast_105 = typecast(conv2d_18, F32), then deallocate(conv2d_18).
            # So conv2d_18 is NOT deallocated by the shortcut conv! The deallocate_activation
            # only deallocates the reshaped input inside conv2d, not the outer tensor.
            # Actually, conv2d_19 uses conv2d_18 as input_tensor, and deallocate_activation=True
            # means the activation (input) is deallocated. But conv2d_18 is used again for
            # typecast. This means the flat code uses conv2d_18 twice: once for shortcut, once
            # for norm1. The deallocate_activation in the 1x1 conv must not actually free the
            # reference? Or perhaps it's a copy. Let me re-examine.
            #
            # In the flat code at line 3274-3305:
            #   conv2d_19 = conv2d(conv2d_18, shortcut_weight, ..., deallocate_activation=True)
            #   typecast_105 = typecast(conv2d_18, F32)   <-- conv2d_18 still alive!
            #   deallocate(conv2d_18)
            #
            # So deallocate_activation=True in conv2d deallocates the INTERNAL copy,
            # not the original tensor reference. The original tensor is still valid.
            # This means we can safely use x after the shortcut conv.

            # norm1 from conv format
            conv1_input = self.norm1.from_conv_format(x)
            ttnn.deallocate(x, False)

            conv1_out = self._conv2d(
                conv1_input,
                self.conv1_weight,
                self.conv1_bias,
                self.in_channels,
                self.out_channels,
                self.conv1_slice_type,
            )
            ttnn.deallocate(conv1_input, False)

            # norm2 from conv format
            conv2_input = self.norm2.from_conv_format(conv1_out)
            ttnn.deallocate(conv1_out, False)

            conv2_out = self._conv2d(
                conv2_input,
                self.conv2_weight,
                self.conv2_bias,
                self.out_channels,
                self.out_channels,
                self.conv2_slice_type,
            )
            ttnn.deallocate(conv2_input, False)

            # Residual add in conv format
            added = ttnn.add(shortcut, conv2_out, dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM)
            ttnn.deallocate(conv2_out, False)
            ttnn.deallocate(shortcut, False)

            # Conv format → NHWC → NCHW → divide → NCHW BF16
            nhwc = ttnn.reshape(added, [1, self.h, self.w, self.out_channels], memory_config=DRAM)
            ttnn.deallocate(added, False)
            nchw = ttnn.permute(nhwc, [0, 3, 1, 2], memory_config=DRAM, pad_value=0.0)
            ttnn.deallocate(nhwc, False)
            result = ttnn.divide(nchw, self.ones_4d, dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM)
            ttnn.deallocate(nchw, False)
            return result

        elif self.conv_format_input:
            # x is conv format (first resnet after conv_in)
            # No shortcut, residual is the conv format input itself

            # norm1 from conv format
            conv1_input = self.norm1.from_conv_format(x)

            conv1_out = self._conv2d(
                conv1_input,
                self.conv1_weight,
                self.conv1_bias,
                self.in_channels,
                self.out_channels,
                self.conv1_slice_type,
            )
            ttnn.deallocate(conv1_input, False)

            # norm2 from conv format
            conv2_input = self.norm2.from_conv_format(conv1_out)
            ttnn.deallocate(conv1_out, False)

            conv2_out = self._conv2d(
                conv2_input,
                self.conv2_weight,
                self.conv2_bias,
                self.out_channels,
                self.out_channels,
                self.conv2_slice_type,
            )
            ttnn.deallocate(conv2_input, False)

            # Residual add: conv_in_out (conv format) + conv2_out (conv format)
            added = ttnn.add(x, conv2_out, dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM)
            ttnn.deallocate(conv2_out, False)
            ttnn.deallocate(x, False)

            # Conv format → NHWC → NCHW → divide → NCHW BF16
            nhwc = ttnn.reshape(added, [1, self.h, self.w, self.out_channels], memory_config=DRAM)
            ttnn.deallocate(added, False)
            nchw = ttnn.permute(nhwc, [0, 3, 1, 2], memory_config=DRAM, pad_value=0.0)
            ttnn.deallocate(nhwc, False)
            result = ttnn.divide(nchw, self.ones_4d, dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM)
            ttnn.deallocate(nchw, False)
            return result

        else:
            # x is NCHW BF16 (normal case after divide-by-ones)

            # norm1 from NCHW
            conv1_input = self.norm1.from_nchw(x)

            conv1_out = self._conv2d(
                conv1_input,
                self.conv1_weight,
                self.conv1_bias,
                self.in_channels,
                self.out_channels,
                self.conv1_slice_type,
            )
            ttnn.deallocate(conv1_input, False)

            # norm2 from conv format (conv1_out is conv format)
            conv2_input = self.norm2.from_conv_format(conv1_out)
            ttnn.deallocate(conv1_out, False)

            conv2_out = self._conv2d(
                conv2_input,
                self.conv2_weight,
                self.conv2_bias,
                self.out_channels,
                self.out_channels,
                self.conv2_slice_type,
            )
            ttnn.deallocate(conv2_input, False)

            # conv2_out → NCHW BF16 for residual add (no typecast, matches flat code)
            conv2_nchw = self._conv_fmt_to_nchw_bf16(conv2_out, self.out_channels)

            # Residual add: NCHW BF16 + NCHW BF16 → NCHW BF16
            added = ttnn.add(x, conv2_nchw, dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM)
            ttnn.deallocate(conv2_nchw, False)
            ttnn.deallocate(x, False)

            # Divide → NCHW BF16
            result = ttnn.divide(added, self.ones_4d, dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM)
            ttnn.deallocate(added, False)
            return result


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class Attention(LightweightModule):
    """Self-attention with group norm, QKV projections, SDPA, and residual.

    Input/output: NCHW BF16. Consumes (deallocates) input.
    """

    SCALE = 0.04419417679309845  # 1 / sqrt(512)

    def __init__(self, weights, prefix, device, gn_inv, gn_eps, ones_4d, channels, h, w):
        self.device = device
        self.ones_4d = ones_4d
        self.channels = channels
        self.h = h
        self.w = w
        self.cpg = channels // 32
        self.spatial = h * w

        self.gn_weight = weights[f"{prefix}.group_norm.weight"]
        self.gn_bias = weights[f"{prefix}.group_norm.bias"]
        self.gn_inv = gn_inv
        self.gn_eps = gn_eps

        self.to_q_weight = weights[f"{prefix}.to_q.weight"]
        self.to_q_bias = weights[f"{prefix}.to_q.bias"]
        self.to_k_weight = weights[f"{prefix}.to_k.weight"]
        self.to_k_bias = weights[f"{prefix}.to_k.bias"]
        self.to_v_weight = weights[f"{prefix}.to_v.weight"]
        self.to_v_bias = weights[f"{prefix}.to_v.bias"]
        self.to_out_weight = weights[f"{prefix}.to_out.0.weight"]
        self.to_out_bias = weights[f"{prefix}.to_out.0.bias"]

    def _group_norm(self, x_nchw):
        """NCHW BF16 → GroupNorm (no SiLU) → 2D [spatial, channels] BF16.
        Does NOT deallocate input."""
        x_f32 = ttnn.typecast(x_nchw, ttnn.DataType.FLOAT32, memory_config=DRAM)
        gn = ttnn.reshape(x_f32, [1, 32, self.cpg, self.spatial], memory_config=DRAM)
        ttnn.deallocate(x_f32, False)

        s = ttnn.sum(gn, [2, 3], True, memory_config=DRAM, compute_kernel_config=None)
        mean = ttnn.multiply(s, self.gn_inv, dtype=ttnn.DataType.FLOAT32, memory_config=DRAM)
        ttnn.deallocate(s, False)
        centered = ttnn.subtract(gn, mean, dtype=ttnn.DataType.FLOAT32, memory_config=DRAM)
        ttnn.deallocate(mean, False)
        ttnn.deallocate(gn, False)

        sq = ttnn.multiply(centered, centered, dtype=ttnn.DataType.FLOAT32, memory_config=DRAM)
        s2 = ttnn.sum(sq, [2, 3], True, memory_config=DRAM, compute_kernel_config=None)
        ttnn.deallocate(sq, False)
        var = ttnn.multiply(s2, self.gn_inv, dtype=ttnn.DataType.FLOAT32, memory_config=DRAM)
        ttnn.deallocate(s2, False)
        var_eps = ttnn.add(var, self.gn_eps, dtype=ttnn.DataType.FLOAT32, memory_config=DRAM)
        ttnn.deallocate(var, False)
        inv_std = ttnn.rsqrt(var_eps, fast_and_approximate_mode=False, memory_config=DRAM)
        ttnn.deallocate(var_eps, False)

        normed = ttnn.multiply(centered, inv_std, dtype=ttnn.DataType.FLOAT32, memory_config=DRAM)
        ttnn.deallocate(inv_std, False)
        ttnn.deallocate(centered, False)

        scaled = ttnn.multiply(normed, self.gn_weight, dtype=ttnn.DataType.FLOAT32, memory_config=DRAM)
        ttnn.deallocate(normed, False)
        shifted = ttnn.add(scaled, self.gn_bias, dtype=ttnn.DataType.FLOAT32, memory_config=DRAM)
        ttnn.deallocate(scaled, False)

        bf16 = ttnn.typecast(shifted, ttnn.DataType.BFLOAT16, memory_config=DRAM)
        ttnn.deallocate(shifted, False)

        x_3d = ttnn.reshape(bf16, [1, self.channels, self.spatial], memory_config=DRAM)
        ttnn.deallocate(bf16, False)
        x_perm = ttnn.permute(x_3d, [0, 2, 1], memory_config=DRAM, pad_value=0.0)
        ttnn.deallocate(x_3d, False)
        x_2d = ttnn.reshape(x_perm, [self.spatial, self.channels], memory_config=DRAM)
        ttnn.deallocate(x_perm, False)
        return x_2d

    def _project(self, x_2d, weight, bias, deallocate_input=False):
        """Linear projection → F32 4D [1, 1, spatial, channels]."""
        proj = ttnn.linear(
            x_2d,
            weight,
            bias=bias,
            transpose_a=False,
            transpose_b=True,
            memory_config=DRAM,
            dtype=ttnn.DataType.BFLOAT16,
            program_config=None,
            activation=None,
            compute_kernel_config=None,
        )
        if deallocate_input:
            ttnn.deallocate(x_2d, False)
        proj_f32 = ttnn.typecast(proj, ttnn.DataType.FLOAT32, memory_config=DRAM)
        ttnn.deallocate(proj, False)
        proj_4d = ttnn.reshape(proj_f32, [1, 1, self.spatial, self.channels], memory_config=DRAM)
        ttnn.deallocate(proj_f32, False)
        return proj_4d

    def forward(self, x):
        normed_2d = self._group_norm(x)

        q_f32 = self._project(normed_2d, self.to_q_weight, self.to_q_bias)
        k_f32 = self._project(normed_2d, self.to_k_weight, self.to_k_bias)
        v_f32 = self._project(normed_2d, self.to_v_weight, self.to_v_bias, deallocate_input=True)

        q_bf16 = ttnn.typecast(q_f32, ttnn.DataType.BFLOAT16, memory_config=DRAM)
        ttnn.deallocate(q_f32, False)
        k_bf16 = ttnn.typecast(k_f32, ttnn.DataType.BFLOAT16, memory_config=DRAM)
        ttnn.deallocate(k_f32, False)
        v_bf16 = ttnn.typecast(v_f32, ttnn.DataType.BFLOAT16, memory_config=DRAM)
        ttnn.deallocate(v_f32, False)

        attn_out = ttnn.transformer.scaled_dot_product_attention(
            q_bf16,
            k_bf16,
            v_bf16,
            attn_mask=None,
            is_causal=False,
            scale=self.SCALE,
            sliding_window_size=None,
            memory_config=DRAM,
        )
        ttnn.deallocate(v_bf16, False)
        ttnn.deallocate(k_bf16, False)
        ttnn.deallocate(q_bf16, False)

        attn_f32 = ttnn.typecast(attn_out, ttnn.DataType.FLOAT32, memory_config=DRAM)
        ttnn.deallocate(attn_out, False)
        attn_bf16 = ttnn.typecast(attn_f32, ttnn.DataType.BFLOAT16, memory_config=DRAM)
        ttnn.deallocate(attn_f32, False)

        out_2d = ttnn.reshape(attn_bf16, [self.spatial, self.channels], memory_config=DRAM)
        ttnn.deallocate(attn_bf16, False)
        out_proj = ttnn.linear(
            out_2d,
            self.to_out_weight,
            bias=self.to_out_bias,
            transpose_a=False,
            transpose_b=True,
            memory_config=DRAM,
            dtype=ttnn.DataType.BFLOAT16,
            program_config=None,
            activation=None,
            compute_kernel_config=None,
        )
        ttnn.deallocate(out_2d, False)

        out_3d = ttnn.reshape(out_proj, [1, self.spatial, self.channels], memory_config=DRAM)
        ttnn.deallocate(out_proj, False)
        out_perm = ttnn.permute(out_3d, [0, 2, 1], memory_config=DRAM, pad_value=0.0)
        ttnn.deallocate(out_3d, False)
        out_nchw = ttnn.reshape(out_perm, [1, self.channels, self.h, self.w], memory_config=DRAM)
        ttnn.deallocate(out_perm, False)

        result = ttnn.add(out_nchw, x, dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM)
        ttnn.deallocate(out_nchw, False)
        ttnn.deallocate(x, False)

        output = ttnn.divide(result, self.ones_4d, dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM)
        ttnn.deallocate(result, False)
        return output


# ---------------------------------------------------------------------------
# UNetMidBlock2D
# ---------------------------------------------------------------------------


class UNetMidBlock2D(LightweightModule):
    """Mid block: ResnetBlock → Attention → ResnetBlock.
    Input: conv format (from conv_in). Output: NCHW BF16."""

    def __init__(self, resnet0, attention, resnet1):
        self.resnet0 = resnet0
        self.attention = attention
        self.resnet1 = resnet1

    def forward(self, x):
        x = self.resnet0(x)  # conv format → NCHW BF16
        x = self.attention(x)  # NCHW BF16 → NCHW BF16
        x = self.resnet1(x)  # NCHW BF16 → NCHW BF16
        return x


# ---------------------------------------------------------------------------
# Upsample2D
# ---------------------------------------------------------------------------


class Upsample2D(LightweightModule):
    """2x nearest-neighbor upsample via matmul + conv.

    Input: NCHW BF16. Output: conv format BF16 (NOT converted to NCHW).
    The first resnet of the next block receives this conv format.
    """

    def __init__(
        self, weights, prefix, device, upsample_matrix, ones_2d, channels, h_in, w_in, conv_act_block_h, conv_slice_type
    ):
        self.device = device
        self.upsample_matrix = upsample_matrix
        self.ones_2d = ones_2d
        self.channels = channels
        self.h_in = h_in
        self.w_in = w_in
        self.h_out = h_in * 2
        self.w_out = w_in * 2
        self.conv_weight = weights[f"{prefix}.conv.weight"]
        self.conv_bias = weights[f"{prefix}.conv.bias"]
        self.conv_act_block_h = conv_act_block_h
        self.conv_slice_type = conv_slice_type

    def forward(self, x):
        C, H, W = self.channels, self.h_in, self.w_in

        # NCHW → permute [0,1,3,2] → [1, C, W, H]
        x_t = ttnn.permute(x, [0, 1, 3, 2], memory_config=DRAM, pad_value=0.0)
        ttnn.deallocate(x, False)

        # Reshape to 2D [C*W, H]
        x_2d = ttnn.reshape(x_t, [C * W, H], memory_config=DRAM)
        ttnn.deallocate(x_t, False)

        # Divide by ones_2d
        x_div = ttnn.divide(x_2d, self.ones_2d, dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM)
        ttnn.deallocate(x_2d, False)

        # Matmul to upsample H dimension
        x_up_h = ttnn.matmul(
            x_div,
            self.upsample_matrix,
            transpose_a=False,
            transpose_b=False,
            memory_config=DRAM,
            dtype=ttnn.DataType.BFLOAT16,
            program_config=None,
            activation=None,
            compute_kernel_config=None,
        )
        ttnn.deallocate(x_div, False)

        # Reshape to 4D [1, C, W, H_out]
        x_4d = ttnn.reshape(x_up_h, [1, C, W, self.h_out], memory_config=DRAM)
        ttnn.deallocate(x_up_h, False)

        # Permute back [0,1,3,2] → [1, C, H_out, W]
        x_4d_t = ttnn.permute(x_4d, [0, 1, 3, 2], memory_config=DRAM, pad_value=0.0)
        ttnn.deallocate(x_4d, False)

        # Reshape to 2D [C*H_out, W]
        x_2d_2 = ttnn.reshape(x_4d_t, [C * self.h_out, W], memory_config=DRAM)
        ttnn.deallocate(x_4d_t, False)

        # Matmul to upsample W dimension
        x_up_w = ttnn.matmul(
            x_2d_2,
            self.upsample_matrix,
            transpose_a=False,
            transpose_b=False,
            memory_config=DRAM,
            dtype=ttnn.DataType.BFLOAT16,
            program_config=None,
            activation=None,
            compute_kernel_config=None,
        )
        ttnn.deallocate(x_2d_2, False)

        # Reshape to NCHW [1, C, H_out, W_out]
        x_nchw = ttnn.reshape(x_up_w, [1, C, self.h_out, self.w_out], memory_config=DRAM)
        ttnn.deallocate(x_up_w, False)

        # NCHW → NHWC → conv format
        nhwc = ttnn.permute(x_nchw, [0, 2, 3, 1], memory_config=DRAM, pad_value=0.0)
        ttnn.deallocate(x_nchw, False)
        conv_input = ttnn.reshape(nhwc, [1, 1, self.h_out * self.w_out, C], memory_config=DRAM)
        ttnn.deallocate(nhwc, False)

        # Conv2d
        conv_output = ttnn.conv2d(
            input_tensor=conv_input,
            weight_tensor=self.conv_weight,
            device=self.device,
            in_channels=C,
            out_channels=C,
            batch_size=1,
            input_height=self.h_out,
            input_width=self.w_out,
            kernel_size=[3, 3],
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=[1, 1],
            groups=1,
            bias_tensor=self.conv_bias,
            conv_config=ttnn.Conv2dConfig(
                weights_dtype=ttnn.DataType.BFLOAT16,
                deallocate_activation=True,
                config_tensors_in_dram=True,
                act_block_h_override=self.conv_act_block_h,
                enable_kernel_stride_folding=False,
            ),
            compute_config=None,
            slice_config=ttnn.Conv2dSliceConfig(slice_type=self.conv_slice_type, num_slices=0),
            memory_config=DRAM,
        )
        ttnn.deallocate(conv_input, False)

        # Return conv format (NOT converted to NCHW)
        return conv_output


# ---------------------------------------------------------------------------
# UpDecoderBlock2D
# ---------------------------------------------------------------------------


class UpDecoderBlock2D(LightweightModule):
    """Decoder up-block: 3 ResnetBlocks + optional Upsample2D.

    The first resnet may receive conv format (if preceded by upsample or if
    it has a shortcut). Subsequent resnets receive NCHW BF16.

    If there is an upsampler, it takes NCHW BF16 and outputs conv format,
    which becomes the input to the next block's first resnet.
    """

    def __init__(self, resnets, upsampler=None):
        self.resnets = resnets
        self.upsampler = upsampler

    def forward(self, x):
        for resnet in self.resnets:
            x = resnet(x)
        if self.upsampler is not None:
            x = self.upsampler(x)
        return x


# ---------------------------------------------------------------------------
# VaeDecoderTTNN
# ---------------------------------------------------------------------------


class VaeDecoderTTNN(LightweightModule):
    """TTNN VAE Decoder matching the diffusers AutoencoderKL decoder structure."""

    def __init__(self, mesh_device):
        self.device = mesh_device
        pt = VaeDecoderPT()
        self.weights = load_weights(pt.state_dict, self.device)
        del pt

        w = self.weights
        gn_eps = w["_gn_eps"]
        ones_4d = w["_ones_4d"]
        ones_2d = w["_ones_2d"]

        # GN inverse scalars: 1 / (channels_per_group * spatial_elements)
        gn_inv_512_64 = w["_gn_inv_512x64"]  # 512ch, 64x64: cpg=16, spatial=4096
        gn_inv_512_128 = w["_gn_inv_512x128"]  # 512ch, 128x128
        gn_inv_512_256 = w["_gn_inv_512x256"]  # 512ch, 256x256
        gn_inv_256_256 = w["_gn_eps_large"]  # 256ch, 256x256: cpg=8, spatial=65536
        gn_inv_256_512 = w["_gn_inv_256x512"]  # 256ch, 512x512
        gn_inv_128_512 = gn_inv_512_256  # 128ch, 512x512: cpg=4, spatial=262144

        d = mesh_device

        # -- Mid block --
        self.mid_block = UNetMidBlock2D(
            resnet0=ResnetBlock2D(
                w,
                "mid_block.resnets.0",
                d,
                gn_inv_512_64,
                gn_inv_512_64,
                gn_eps,
                ones_4d,
                512,
                512,
                64,
                64,
                0,
                ttnn.Conv2dL1Full,
                conv_format_input=True,  # First resnet after conv_in
            ),
            attention=Attention(
                w,
                "mid_block.attentions.0",
                d,
                gn_inv_512_64,
                gn_eps,
                ones_4d,
                512,
                64,
                64,
            ),
            resnet1=ResnetBlock2D(
                w,
                "mid_block.resnets.1",
                d,
                gn_inv_512_64,
                gn_inv_512_64,
                gn_eps,
                ones_4d,
                512,
                512,
                64,
                64,
                0,
                ttnn.Conv2dL1Full,
            ),
        )

        # -- Up blocks --
        self.up_blocks = [
            # up_blocks.0: 512→512 at 64x64, upsample to 128x128
            UpDecoderBlock2D(
                resnets=[
                    ResnetBlock2D(
                        w,
                        f"up_blocks.0.resnets.{i}",
                        d,
                        gn_inv_512_64,
                        gn_inv_512_64,
                        gn_eps,
                        ones_4d,
                        512,
                        512,
                        64,
                        64,
                        0,
                        ttnn.Conv2dL1Full,
                    )
                    for i in range(3)
                ],
                upsampler=Upsample2D(
                    w,
                    "up_blocks.0.upsamplers.0",
                    d,
                    w["_upsample_64_128"],
                    ones_2d,
                    512,
                    64,
                    64,
                    1024,
                    ttnn.Conv2dL1Full,
                ),
            ),
            # up_blocks.1: 512→512 at 128x128, upsample to 256x256
            UpDecoderBlock2D(
                resnets=[
                    ResnetBlock2D(
                        w,
                        "up_blocks.1.resnets.0",
                        d,
                        gn_inv_512_128,
                        gn_inv_512_128,
                        gn_eps,
                        ones_4d,
                        512,
                        512,
                        128,
                        128,
                        1024,
                        ttnn.Conv2dL1Full,
                        conv_format_input=True,  # First resnet after upsample
                    ),
                ]
                + [
                    ResnetBlock2D(
                        w,
                        f"up_blocks.1.resnets.{i}",
                        d,
                        gn_inv_512_128,
                        gn_inv_512_128,
                        gn_eps,
                        ones_4d,
                        512,
                        512,
                        128,
                        128,
                        1024,
                        ttnn.Conv2dL1Full,
                    )
                    for i in range(1, 3)
                ],
                upsampler=Upsample2D(
                    w,
                    "up_blocks.1.upsamplers.0",
                    d,
                    w["_upsample_128_256"],
                    ones_2d,
                    512,
                    128,
                    128,
                    1024,
                    ttnn.Conv2dDRAMSliceWidth,
                ),
            ),
            # up_blocks.2: 512→256 at 256x256 (shortcut), then 256→256, upsample to 512x512
            UpDecoderBlock2D(
                resnets=[
                    ResnetBlock2D(
                        w,
                        "up_blocks.2.resnets.0",
                        d,
                        gn_inv_512_256,
                        gn_inv_256_256,
                        gn_eps,
                        ones_4d,
                        512,
                        256,
                        256,
                        256,
                        1024,
                        conv1_slice_type=ttnn.Conv2dDRAMSliceWidth,  # 512→256 at 256x256
                        conv2_slice_type=ttnn.Conv2dL1Full,  # 256→256 at 256x256
                        has_conv_shortcut=True,
                        conv_format_input=True,  # First resnet after upsample (shortcut)
                    ),
                ]
                + [
                    ResnetBlock2D(
                        w,
                        f"up_blocks.2.resnets.{i}",
                        d,
                        gn_inv_256_256,
                        gn_inv_256_256,
                        gn_eps,
                        ones_4d,
                        256,
                        256,
                        256,
                        256,
                        1024,
                        ttnn.Conv2dL1Full,
                    )
                    for i in range(1, 3)
                ],
                upsampler=Upsample2D(
                    w,
                    "up_blocks.2.upsamplers.0",
                    d,
                    w["_upsample_256_512"],
                    ones_2d,
                    256,
                    256,
                    256,
                    1024,
                    ttnn.Conv2dDRAMSliceWidth,
                ),
            ),
            # up_blocks.3: 256→128 at 512x512 (shortcut), then 128→128, no upsample
            UpDecoderBlock2D(
                resnets=[
                    ResnetBlock2D(
                        w,
                        "up_blocks.3.resnets.0",
                        d,
                        gn_inv_256_512,
                        gn_inv_128_512,
                        gn_eps,
                        ones_4d,
                        256,
                        128,
                        512,
                        512,
                        1024,
                        conv1_slice_type=ttnn.Conv2dDRAMSliceWidth,  # 256→128 at 512x512
                        conv2_slice_type=ttnn.Conv2dDRAMSliceWidth,  # 128→128 at 512x512
                        has_conv_shortcut=True,
                        conv_format_input=True,  # First resnet after upsample (shortcut)
                    ),
                ]
                + [
                    ResnetBlock2D(
                        w,
                        f"up_blocks.3.resnets.{i}",
                        d,
                        gn_inv_128_512,
                        gn_inv_128_512,
                        gn_eps,
                        ones_4d,
                        128,
                        128,
                        512,
                        512,
                        1024,
                        conv1_slice_type=ttnn.Conv2dDRAMSliceWidth,
                        conv2_slice_type=ttnn.Conv2dDRAMSliceWidth,
                    )
                    for i in range(1, 3)
                ],
            ),
        ]

        # -- Final norm --
        self.conv_norm_out = GroupNormSiLU(
            w["conv_norm_out.weight"],
            w["conv_norm_out.bias"],
            gn_inv_128_512,
            gn_eps,
            128,
            512,
            512,
        )

    def forward(self, raw_latents):
        """Decode raw (pre-scaling) latents -> [1, 3, 512, 512] float32 CPU tensor."""
        z = (raw_latents.float() / SCALING_FACTOR) + SHIFT_FACTOR
        latent = ttnn.from_torch(
            z.bfloat16(),
            dtype=ttnn.DataType.BFLOAT16,
            layout=ttnn.Layout.ROW_MAJOR,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        )

        # conv_in: latent NCHW → conv format
        x = ttnn.to_layout(latent, ttnn.Layout.TILE, None, memory_config=None)
        ttnn.deallocate(latent, False)
        nhwc = ttnn.permute(x, [0, 2, 3, 1], memory_config=DRAM, pad_value=0.0)
        ttnn.deallocate(x, False)
        conv_in_input = ttnn.reshape(nhwc, [1, 1, 4096, 16], memory_config=DRAM)
        ttnn.deallocate(nhwc, False)

        conv_in_out = ttnn.conv2d(
            input_tensor=conv_in_input,
            weight_tensor=self.weights["conv_in.weight"],
            device=self.device,
            in_channels=16,
            out_channels=512,
            batch_size=1,
            input_height=64,
            input_width=64,
            kernel_size=[3, 3],
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=[1, 1],
            groups=1,
            bias_tensor=self.weights["conv_in.bias"],
            conv_config=ttnn.Conv2dConfig(
                weights_dtype=ttnn.DataType.BFLOAT16,
                deallocate_activation=True,
                config_tensors_in_dram=True,
                act_block_h_override=0,
                enable_kernel_stride_folding=False,
            ),
            compute_config=None,
            slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
            memory_config=DRAM,
        )
        ttnn.deallocate(conv_in_input, False)

        # mid_block: conv format → NCHW BF16
        h = self.mid_block(conv_in_out)

        # up_blocks: NCHW BF16 → ... → NCHW BF16
        # (upsample outputs conv format, which the next block's first resnet handles)
        for up_block in self.up_blocks:
            h = up_block(h)

        # conv_norm_out + conv_act (SiLU is inside GroupNormSiLU)
        # h is NCHW BF16 → norm produces conv format
        conv_norm_out = self.conv_norm_out.from_nchw(h)
        ttnn.deallocate(h, False)

        # conv_out
        conv_out = ttnn.conv2d(
            input_tensor=conv_norm_out,
            weight_tensor=self.weights["conv_out.weight"],
            device=self.device,
            in_channels=128,
            out_channels=3,
            batch_size=1,
            input_height=512,
            input_width=512,
            kernel_size=[3, 3],
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=[1, 1],
            groups=1,
            bias_tensor=self.weights["conv_out.bias"],
            conv_config=ttnn.Conv2dConfig(
                weights_dtype=ttnn.DataType.BFLOAT16,
                deallocate_activation=True,
                config_tensors_in_dram=True,
                act_block_h_override=192,
                enable_kernel_stride_folding=False,
            ),
            compute_config=None,
            slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dDRAMSliceWidth, num_slices=0),
            memory_config=DRAM,
        )
        ttnn.deallocate(conv_norm_out, False)

        # conv_out → NHWC → NCHW (match flat code: reshape → permute, no extra typecast)
        nhwc_out = ttnn.reshape(conv_out, [1, 512, 512, 3], memory_config=DRAM)
        ttnn.deallocate(conv_out, False)
        nchw_out = ttnn.permute(nhwc_out, [0, 3, 1, 2], memory_config=DRAM, pad_value=0.0)
        ttnn.deallocate(nhwc_out, False)

        out = ttnn.to_torch(
            ttnn.from_device(nchw_out),
            mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=0),
        )
        return out[: out.shape[0] // 4].float()
