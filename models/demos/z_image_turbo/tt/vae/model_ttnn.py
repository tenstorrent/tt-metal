# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""TTNN VAE Decoder model for Tongyi-MAI/Z-Image-Turbo.

Architecture (diffusers AutoencoderKL decoder):
  - conv_in (16→512, 3x3)
  - mid_block: resnet0 → attention → resnet1  (512ch, 64x64)
  - up_blocks[0]: 3 resnets (512ch, 64x64) + upsample → 128x128
  - up_blocks[1]: 3 resnets (512ch, 128x128) + upsample → 256x256
  - up_blocks[2]: 3 resnets (512→256ch, 256x256, first has conv_shortcut) + upsample → 512x512
  - up_blocks[3]: 3 resnets (256→128ch, 512x512, first has conv_shortcut)
  - conv_norm_out (GroupNorm+SiLU) + conv_out (128→3, 3x3)

Key format conventions:
  - "conv format" = flat BF16 tensor from conv2d
  - "NCHW BF16"   = [1, C, H, W] BF16 after divide-by-ones
  - GroupNorm+SiLU always outputs conv format
  - The first resnet after conv_in and conv_shortcut resnets receive conv format
  - Normal resnets and attention receive NCHW BF16
"""

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.z_image_turbo.tt.vae.model_pt import SCALING_FACTOR, SHIFT_FACTOR, VaeDecoderPT
from models.demos.z_image_turbo.tt.vae.params import load_weights


class VaeDecoderTTNN(LightweightModule):
    """TTNN VAE Decoder matching the diffusers AutoencoderKL decoder structure."""

    ATTN_SCALE = 0.04419417679309845  # 1 / sqrt(512)

    def __init__(self, mesh_device):
        self.device = mesh_device
        pt = VaeDecoderPT()
        self.weights = load_weights(pt.state_dict, self.device)
        del pt

        w = self.weights

        # GN inverse scalars: 1 / (channels_per_group * spatial_elements)
        self._gn_inv = {
            (512, 64): w["_gn_inv_512x64"],
            (512, 128): w["_gn_inv_512x128"],
            (512, 256): w["_gn_inv_512x256"],
            (256, 256): w["_gn_eps_large"],
            (256, 512): w["_gn_inv_256x512"],
            (128, 512): w["_gn_inv_512x256"],  # 4*262144 == 16*65536
        }

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(self, raw_latents):
        """Decode raw (pre-scaling) latents → [1, 3, 512, 512] float32 CPU tensor."""
        latent = self.preprocess(raw_latents)
        nchw = self.forward_device(latent)
        ttnn.deallocate(latent, False)
        out = ttnn.to_torch(
            ttnn.from_device(nchw),
            mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=0),
        )
        return out[: out.shape[0] // 4].float()

    def preprocess(self, raw_latents):
        """Scale + shift raw latents and upload to device.

        Returns a [1, 16, 64, 64] BF16 ROW_MAJOR TTNN tensor (replicated).
        """
        z = (raw_latents.float() / SCALING_FACTOR) + SHIFT_FACTOR
        return ttnn.from_torch(
            z.bfloat16(),
            dtype=ttnn.DataType.BFLOAT16,
            layout=ttnn.Layout.ROW_MAJOR,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        )

    def forward_device(self, latent):
        """Device-only VAE decode: on-device latent → on-device [1, 3, 512, 512] NCHW.

        Input:  [1, 16, 64, 64] BF16 ROW_MAJOR on device
        Output: [1, 3, 512, 512] BF16 TILE on device
        """
        w = self.weights

        # conv_in (caller owns latent — do not deallocate it here)
        x = ttnn.to_layout(latent, ttnn.Layout.TILE, None, memory_config=None)
        nhwc = ttnn.permute(x, [0, 2, 3, 1], memory_config=ttnn.DRAM_MEMORY_CONFIG, pad_value=0.0)
        ttnn.deallocate(x, False)
        x = ttnn.reshape(nhwc, [1, 1, 4096, 16], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(nhwc, False)
        x = self._conv2d(x, "conv_in", 16, 512, 64, 64, 0, ttnn.Conv2dL1Full)

        # mid_block
        x = self._resnet(
            "mid_block.resnets.0", x, 512, 512, 64, 0, ttnn.Conv2dL1Full, ttnn.Conv2dL1Full, conv_format_input=True
        )
        x = self._attention(x, 512, 64, 64)
        x = self._resnet("mid_block.resnets.1", x, 512, 512, 64, 0, ttnn.Conv2dL1Full, ttnn.Conv2dL1Full)

        # up_blocks.0: 512→512 at 64x64, upsample to 128x128
        for i in range(3):
            x = self._resnet(f"up_blocks.0.resnets.{i}", x, 512, 512, 64, 0, ttnn.Conv2dL1Full, ttnn.Conv2dL1Full)
        x = self._upsample(x, "up_blocks.0.upsamplers.0", 512, 64, w["_upsample_64_128"], 1024, ttnn.Conv2dL1Full)

        # up_blocks.1: 512→512 at 128x128, upsample to 256x256
        x = self._resnet(
            "up_blocks.1.resnets.0",
            x,
            512,
            512,
            128,
            1024,
            ttnn.Conv2dL1Full,
            ttnn.Conv2dL1Full,
            conv_format_input=True,
        )
        for i in range(1, 3):
            x = self._resnet(f"up_blocks.1.resnets.{i}", x, 512, 512, 128, 1024, ttnn.Conv2dL1Full, ttnn.Conv2dL1Full)
        x = self._upsample(
            x, "up_blocks.1.upsamplers.0", 512, 128, w["_upsample_128_256"], 1024, ttnn.Conv2dDRAMSliceWidth
        )

        # up_blocks.2: 512→256 at 256x256, upsample to 512x512
        x = self._resnet(
            "up_blocks.2.resnets.0",
            x,
            512,
            256,
            256,
            1024,
            ttnn.Conv2dDRAMSliceWidth,
            ttnn.Conv2dL1Full,
            has_conv_shortcut=True,
            conv_format_input=True,
        )
        for i in range(1, 3):
            x = self._resnet(f"up_blocks.2.resnets.{i}", x, 256, 256, 256, 1024, ttnn.Conv2dL1Full, ttnn.Conv2dL1Full)
        x = self._upsample(
            x, "up_blocks.2.upsamplers.0", 256, 256, w["_upsample_256_512"], 1024, ttnn.Conv2dDRAMSliceWidth
        )

        # up_blocks.3: 256→128 at 512x512, no upsample
        x = self._resnet(
            "up_blocks.3.resnets.0",
            x,
            256,
            128,
            512,
            1024,
            ttnn.Conv2dDRAMSliceWidth,
            ttnn.Conv2dDRAMSliceWidth,
            has_conv_shortcut=True,
            conv_format_input=True,
        )
        for i in range(1, 3):
            x = self._resnet(
                f"up_blocks.3.resnets.{i}", x, 128, 128, 512, 1024, ttnn.Conv2dDRAMSliceWidth, ttnn.Conv2dDRAMSliceWidth
            )

        # conv_norm_out + SiLU → conv_out
        x = self._group_norm_silu_from_nchw(x, "conv_norm_out", 128, 512)
        x = self._conv2d(x, "conv_out", 128, 3, 512, 512, 192, ttnn.Conv2dDRAMSliceWidth)

        nhwc = ttnn.reshape(x, [1, 512, 512, 3], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(x, False)
        nchw = ttnn.permute(nhwc, [0, 3, 1, 2], memory_config=ttnn.DRAM_MEMORY_CONFIG, pad_value=0.0)
        ttnn.deallocate(nhwc, False)
        return nchw

    # ── Conv2d ────────────────────────────────────────────────────────────────

    def _conv2d(self, conv_input, prefix, in_ch, out_ch, h, w, act_block_h, slice_type):
        return ttnn.conv2d(
            input_tensor=conv_input,
            weight_tensor=self.weights[f"{prefix}.weight"],
            device=self.device,
            in_channels=in_ch,
            out_channels=out_ch,
            batch_size=1,
            input_height=h,
            input_width=w,
            kernel_size=[3, 3],
            stride=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=[1, 1],
            groups=1,
            bias_tensor=self.weights[f"{prefix}.bias"],
            conv_config=ttnn.Conv2dConfig(
                weights_dtype=ttnn.DataType.BFLOAT16,
                deallocate_activation=True,
                config_tensors_in_dram=True,
                act_block_h_override=act_block_h,
                enable_kernel_stride_folding=False,
            ),
            compute_config=None,
            slice_config=ttnn.Conv2dSliceConfig(slice_type=slice_type, num_slices=0),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    # ── GroupNorm + SiLU ──────────────────────────────────────────────────────

    def _gn_core(self, gn, gn_inv, gn_eps, weight, bias):
        """GN layout [1,32,cpg,spatial] F32 → normed+affine+SiLU → BF16. Deallocates gn."""
        s = ttnn.sum(gn, [2, 3], True, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=None)
        mean = ttnn.multiply(s, gn_inv, dtype=ttnn.DataType.FLOAT32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(s, False)
        centered = ttnn.subtract(gn, mean, dtype=ttnn.DataType.FLOAT32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(mean, False)
        ttnn.deallocate(gn, False)

        sq = ttnn.multiply(centered, centered, dtype=ttnn.DataType.FLOAT32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        s2 = ttnn.sum(sq, [2, 3], True, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=None)
        ttnn.deallocate(sq, False)
        var = ttnn.multiply(s2, gn_inv, dtype=ttnn.DataType.FLOAT32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(s2, False)
        var_eps = ttnn.add(var, gn_eps, dtype=ttnn.DataType.FLOAT32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(var, False)
        inv_std = ttnn.rsqrt(var_eps, fast_and_approximate_mode=False, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(var_eps, False)

        normed = ttnn.multiply(centered, inv_std, dtype=ttnn.DataType.FLOAT32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(inv_std, False)
        ttnn.deallocate(centered, False)

        scaled = ttnn.multiply(normed, weight, dtype=ttnn.DataType.FLOAT32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(normed, False)
        shifted = ttnn.add(scaled, bias, dtype=ttnn.DataType.FLOAT32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(scaled, False)

        activated = ttnn.silu(shifted, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(shifted, False)
        bf16 = ttnn.typecast(activated, ttnn.DataType.BFLOAT16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(activated, False)
        return bf16

    def _gn_to_conv_fmt(self, bf16, channels, h, w):
        """GN-layout BF16 → reshape NCHW → permute NHWC → reshape conv format."""
        nchw = ttnn.reshape(bf16, [1, channels, h, w], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(bf16, False)
        nhwc = ttnn.permute(nchw, [0, 2, 3, 1], memory_config=ttnn.DRAM_MEMORY_CONFIG, pad_value=0.0)
        ttnn.deallocate(nchw, False)
        conv_fmt = ttnn.reshape(nhwc, [1, 1, h * w, channels], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(nhwc, False)
        return conv_fmt

    def _group_norm_silu_from_conv_format(self, conv_fmt, prefix, channels, hw):
        """Conv format → GN+SiLU → conv format. Does NOT deallocate input."""
        w = self.weights
        gn_inv = self._gn_inv[(channels, hw)]
        cpg = channels // 32
        spatial = hw * hw

        f32 = ttnn.typecast(conv_fmt, ttnn.DataType.FLOAT32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        nhwc = ttnn.reshape(f32, [1, hw, hw, channels], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(f32, False)
        nchw = ttnn.permute(nhwc, [0, 3, 1, 2], memory_config=ttnn.DRAM_MEMORY_CONFIG, pad_value=0.0)
        ttnn.deallocate(nhwc, False)
        gn = ttnn.reshape(nchw, [1, 32, cpg, spatial], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(nchw, False)

        bf16 = self._gn_core(gn, gn_inv, w["_gn_eps"], w[f"{prefix}.weight"], w[f"{prefix}.bias"])
        return self._gn_to_conv_fmt(bf16, channels, hw, hw)

    def _group_norm_silu_from_nchw(self, x_nchw, prefix, channels, hw):
        """NCHW BF16 → GN+SiLU → conv format. Does NOT deallocate input."""
        w = self.weights
        gn_inv = self._gn_inv[(channels, hw)]
        cpg = channels // 32
        spatial = hw * hw

        f32 = ttnn.typecast(x_nchw, ttnn.DataType.FLOAT32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        gn = ttnn.reshape(f32, [1, 32, cpg, spatial], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(f32, False)

        bf16 = self._gn_core(gn, gn_inv, w["_gn_eps"], w[f"{prefix}.weight"], w[f"{prefix}.bias"])
        return self._gn_to_conv_fmt(bf16, channels, hw, hw)

    # ── ResnetBlock2D ─────────────────────────────────────────────────────────

    def _resnet(
        self,
        prefix,
        x,
        in_ch,
        out_ch,
        hw,
        act_block_h,
        conv1_slice,
        conv2_slice,
        has_conv_shortcut=False,
        conv_format_input=False,
    ):
        """Residual block. Input: conv format or NCHW BF16. Output: NCHW BF16."""
        w = self.weights

        if has_conv_shortcut:
            shortcut = ttnn.conv2d(
                input_tensor=x,
                weight_tensor=w[f"{prefix}.conv_shortcut.weight"],
                device=self.device,
                in_channels=in_ch,
                out_channels=out_ch,
                batch_size=1,
                input_height=hw,
                input_width=hw,
                kernel_size=[1, 1],
                stride=[1, 1],
                padding=[0, 0, 0, 0],
                dilation=[1, 1],
                groups=1,
                bias_tensor=w[f"{prefix}.conv_shortcut.bias"],
                conv_config=ttnn.Conv2dConfig(
                    weights_dtype=ttnn.DataType.BFLOAT16,
                    deallocate_activation=True,
                    config_tensors_in_dram=True,
                    act_block_h_override=0,
                    enable_kernel_stride_folding=False,
                ),
                compute_config=None,
                slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

            conv1_in = self._group_norm_silu_from_conv_format(x, f"{prefix}.norm1", in_ch, hw)
            ttnn.deallocate(x, False)

        elif conv_format_input:
            conv1_in = self._group_norm_silu_from_conv_format(x, f"{prefix}.norm1", in_ch, hw)

        else:
            conv1_in = self._group_norm_silu_from_nchw(x, f"{prefix}.norm1", in_ch, hw)

        conv1_out = self._conv2d(conv1_in, f"{prefix}.conv1", in_ch, out_ch, hw, hw, act_block_h, conv1_slice)
        ttnn.deallocate(conv1_in, False)

        conv2_in = self._group_norm_silu_from_conv_format(conv1_out, f"{prefix}.norm2", out_ch, hw)
        ttnn.deallocate(conv1_out, False)

        conv2_out = self._conv2d(conv2_in, f"{prefix}.conv2", out_ch, out_ch, hw, hw, act_block_h, conv2_slice)
        ttnn.deallocate(conv2_in, False)

        if has_conv_shortcut or conv_format_input:
            residual = shortcut if has_conv_shortcut else x
            added = ttnn.add(residual, conv2_out, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(conv2_out, False)
            ttnn.deallocate(residual, False)

            nhwc = ttnn.reshape(added, [1, hw, hw, out_ch], memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(added, False)
            nchw = ttnn.permute(nhwc, [0, 3, 1, 2], memory_config=ttnn.DRAM_MEMORY_CONFIG, pad_value=0.0)
            ttnn.deallocate(nhwc, False)
        else:
            nhwc = ttnn.reshape(conv2_out, [1, hw, hw, out_ch], memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(conv2_out, False)
            nchw = ttnn.permute(nhwc, [0, 3, 1, 2], memory_config=ttnn.DRAM_MEMORY_CONFIG, pad_value=0.0)
            ttnn.deallocate(nhwc, False)
            added = ttnn.add(x, nchw, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(nchw, False)
            ttnn.deallocate(x, False)
            nchw = added

        result = ttnn.divide(nchw, w["_ones_4d"], dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(nchw, False)
        return result

    # ── Attention ──────────────────────────────────────────────────────────────

    def _attention(self, x, channels, h, w):
        """Self-attention with group norm, SDPA, residual. NCHW BF16 → NCHW BF16."""
        wt = self.weights
        prefix = "mid_block.attentions.0"
        gn_inv = self._gn_inv[(channels, h)]
        cpg = channels // 32
        spatial = h * w

        # Group norm (no SiLU) → 2D [spatial, channels]
        x_f32 = ttnn.typecast(x, ttnn.DataType.FLOAT32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        gn = ttnn.reshape(x_f32, [1, 32, cpg, spatial], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(x_f32, False)

        s = ttnn.sum(gn, [2, 3], True, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=None)
        mean = ttnn.multiply(s, gn_inv, dtype=ttnn.DataType.FLOAT32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(s, False)
        centered = ttnn.subtract(gn, mean, dtype=ttnn.DataType.FLOAT32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(mean, False)
        ttnn.deallocate(gn, False)

        sq = ttnn.multiply(centered, centered, dtype=ttnn.DataType.FLOAT32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        s2 = ttnn.sum(sq, [2, 3], True, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=None)
        ttnn.deallocate(sq, False)
        var = ttnn.multiply(s2, gn_inv, dtype=ttnn.DataType.FLOAT32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(s2, False)
        var_eps = ttnn.add(var, wt["_gn_eps"], dtype=ttnn.DataType.FLOAT32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(var, False)
        inv_std = ttnn.rsqrt(var_eps, fast_and_approximate_mode=False, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(var_eps, False)

        normed = ttnn.multiply(centered, inv_std, dtype=ttnn.DataType.FLOAT32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(inv_std, False)
        ttnn.deallocate(centered, False)

        scaled = ttnn.multiply(
            normed,
            wt[f"{prefix}.group_norm.weight"],
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(normed, False)
        shifted = ttnn.add(
            scaled, wt[f"{prefix}.group_norm.bias"], dtype=ttnn.DataType.FLOAT32, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(scaled, False)

        bf16 = ttnn.typecast(shifted, ttnn.DataType.BFLOAT16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(shifted, False)

        x_3d = ttnn.reshape(bf16, [1, channels, spatial], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(bf16, False)
        x_perm = ttnn.permute(x_3d, [0, 2, 1], memory_config=ttnn.DRAM_MEMORY_CONFIG, pad_value=0.0)
        ttnn.deallocate(x_3d, False)
        x_2d = ttnn.reshape(x_perm, [spatial, channels], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(x_perm, False)

        # QKV projections
        q = self._attn_project(x_2d, f"{prefix}.to_q", spatial, channels)
        k = self._attn_project(x_2d, f"{prefix}.to_k", spatial, channels)
        v = self._attn_project(x_2d, f"{prefix}.to_v", spatial, channels, deallocate_input=True)

        q_bf16 = ttnn.typecast(q, ttnn.DataType.BFLOAT16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(q, False)
        k_bf16 = ttnn.typecast(k, ttnn.DataType.BFLOAT16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(k, False)
        v_bf16 = ttnn.typecast(v, ttnn.DataType.BFLOAT16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(v, False)

        attn_out = ttnn.transformer.scaled_dot_product_attention(
            q_bf16,
            k_bf16,
            v_bf16,
            attn_mask=None,
            is_causal=False,
            scale=self.ATTN_SCALE,
            sliding_window_size=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(v_bf16, False)
        ttnn.deallocate(k_bf16, False)
        ttnn.deallocate(q_bf16, False)

        attn_f32 = ttnn.typecast(attn_out, ttnn.DataType.FLOAT32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(attn_out, False)
        attn_bf16 = ttnn.typecast(attn_f32, ttnn.DataType.BFLOAT16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(attn_f32, False)

        # Output projection
        out_2d = ttnn.reshape(attn_bf16, [spatial, channels], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(attn_bf16, False)
        out_proj = ttnn.linear(
            out_2d,
            wt[f"{prefix}.to_out.0.weight"],
            bias=wt[f"{prefix}.to_out.0.bias"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.DataType.BFLOAT16,
            program_config=None,
            activation=None,
            compute_kernel_config=None,
        )
        ttnn.deallocate(out_2d, False)

        out_3d = ttnn.reshape(out_proj, [1, spatial, channels], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(out_proj, False)
        out_perm = ttnn.permute(out_3d, [0, 2, 1], memory_config=ttnn.DRAM_MEMORY_CONFIG, pad_value=0.0)
        ttnn.deallocate(out_3d, False)
        out_nchw = ttnn.reshape(out_perm, [1, channels, h, w], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(out_perm, False)

        result = ttnn.add(out_nchw, x, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(out_nchw, False)
        ttnn.deallocate(x, False)

        output = ttnn.divide(
            result, wt["_ones_4d"], dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(result, False)
        return output

    def _attn_project(self, x_2d, prefix, spatial, channels, deallocate_input=False):
        """Linear projection → F32 4D [1, 1, spatial, channels]."""
        proj = ttnn.linear(
            x_2d,
            self.weights[f"{prefix}.weight"],
            bias=self.weights[f"{prefix}.bias"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.DataType.BFLOAT16,
            program_config=None,
            activation=None,
            compute_kernel_config=None,
        )
        if deallocate_input:
            ttnn.deallocate(x_2d, False)
        proj_f32 = ttnn.typecast(proj, ttnn.DataType.FLOAT32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(proj, False)
        proj_4d = ttnn.reshape(proj_f32, [1, 1, spatial, channels], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(proj_f32, False)
        return proj_4d

    # ── Upsample2D ────────────────────────────────────────────────────────────

    def _upsample(self, x, prefix, channels, hw_in, upsample_matrix, conv_act_block_h, conv_slice_type):
        """NCHW BF16 → 2x nearest-neighbor upsample + conv → conv format BF16."""
        C, H, W = channels, hw_in, hw_in
        H_out, W_out = H * 2, W * 2

        x_t = ttnn.permute(x, [0, 1, 3, 2], memory_config=ttnn.DRAM_MEMORY_CONFIG, pad_value=0.0)
        ttnn.deallocate(x, False)

        x_2d = ttnn.reshape(x_t, [C * W, H], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(x_t, False)

        x_div = ttnn.divide(
            x_2d, self.weights["_ones_2d"], dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(x_2d, False)

        x_up_h = ttnn.matmul(
            x_div,
            upsample_matrix,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.DataType.BFLOAT16,
            program_config=None,
            activation=None,
            compute_kernel_config=None,
        )
        ttnn.deallocate(x_div, False)

        x_4d = ttnn.reshape(x_up_h, [1, C, W, H_out], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(x_up_h, False)

        x_4d_t = ttnn.permute(x_4d, [0, 1, 3, 2], memory_config=ttnn.DRAM_MEMORY_CONFIG, pad_value=0.0)
        ttnn.deallocate(x_4d, False)

        x_2d_2 = ttnn.reshape(x_4d_t, [C * H_out, W], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(x_4d_t, False)

        x_up_w = ttnn.matmul(
            x_2d_2,
            upsample_matrix,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.DataType.BFLOAT16,
            program_config=None,
            activation=None,
            compute_kernel_config=None,
        )
        ttnn.deallocate(x_2d_2, False)

        x_nchw = ttnn.reshape(x_up_w, [1, C, H_out, W_out], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(x_up_w, False)

        nhwc = ttnn.permute(x_nchw, [0, 2, 3, 1], memory_config=ttnn.DRAM_MEMORY_CONFIG, pad_value=0.0)
        ttnn.deallocate(x_nchw, False)
        conv_input = ttnn.reshape(nhwc, [1, 1, H_out * W_out, C], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(nhwc, False)

        conv_output = self._conv2d(conv_input, f"{prefix}.conv", C, C, H_out, W_out, conv_act_block_h, conv_slice_type)
        ttnn.deallocate(conv_input, False)
        return conv_output
