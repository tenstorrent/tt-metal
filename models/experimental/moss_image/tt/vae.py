# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""MOSS FLUX.2 VAE decoder; all activation arithmetic remains on TTNN.

The encoder and quantization convolution are deliberately not loaded. Decoder
weights use the unmodified Diffusers checkpoint names. This decoder is replicated
on a mesh; model-level transformer tensor parallelism does not shard its channels.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import ttnn
from models.experimental.moss_image.tt.checkpoint import SafeWeights, validate_tensor_shapes


class TTMossVAE:
    """Decode normalized packed tokens [1, 1, H/16*W/16, 128].

    ``decode`` returns clamped [-1, 1] NHWC RGB, [1, H, W, 3]. Only
    ``upload`` and ``collect`` cross the host boundary. ``observe`` is an optional
    validation callback and is never called in ordinary integrated inference.
    """

    def __init__(self, checkpoint: Path, device):
        checkpoint = Path(checkpoint)
        vae_dir = checkpoint / "vae" if (checkpoint / "vae").is_dir() else checkpoint
        self.config = json.loads((vae_dir / "config.json").read_text())
        expected = {
            "_class_name": "AutoencoderKLFlux2",
            "latent_channels": 32,
            "decoder_block_out_channels": [96, 192, 384, 384],
            "patch_size": [2, 2],
            "layers_per_block": 2,
            "norm_num_groups": 32,
            "use_post_quant_conv": True,
            "mid_block_add_attention": True,
            "act_fn": "silu",
        }
        for key, value in expected.items():
            if self.config.get(key) != value:
                raise ValueError(f"Unsupported MOSS VAE {key}: {self.config.get(key)!r}")
        self.device = device
        self.state = {}
        tensors = SafeWeights(vae_dir)
        validate_tensor_shapes(tensors, "vae")
        for key in tensors.mapping:
            if key.startswith(("decoder.", "post_quant_conv.", "bn.")):
                self.state[key] = tensors.get(key).float()
        for key in ("bn.running_mean", "bn.running_var", "decoder.conv_in.weight"):
            if key not in self.state:
                raise ValueError(f"VAE checkpoint missing {key}")
        self.prepared = {}
        self.compute = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def upload(self, value: torch.Tensor, *, dtype=None):
        return ttnn.from_torch(
            value.contiguous(),
            dtype=ttnn.float32 if dtype is None else dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=(ttnn.ReplicateTensorToMesh(self.device) if self.device.get_num_devices() > 1 else None),
        )

    def collect(self, value):
        """Final output boundary, returning contiguous CPU float32 NCHW."""
        local = ttnn.get_device_tensors(value)[0]
        return ttnn.to_torch(local).float().permute(0, 3, 1, 2).contiguous()

    def _parameter(self, name):
        key = ("parameter", name)
        if key not in self.prepared:
            self.prepared[key] = self.upload(self.state[name].reshape(1, 1, 1, -1))
        return self.prepared[key]

    def _linear(self, x, name):
        key = ("linear", name)
        if key not in self.prepared:
            weight = self.state[name + ".weight"]
            self.prepared[key] = self.upload(weight.reshape(weight.shape[0], -1).T)
        x = ttnn.matmul(x, self.prepared[key], compute_kernel_config=self.compute)
        if name + ".bias" in self.state:
            x = ttnn.add(x, self._parameter(name + ".bias"))
        return x

    def conv(self, x, name, height, width):
        weight = self.state[name + ".weight"]
        out_channels, in_channels, kh, kw = weight.shape
        if kh == kw == 1:
            return self._linear(x, name)
        # Prepared convolution operands depend on shape/layout/dtype/memory.
        x = ttnn.typecast(x, ttnn.bfloat16)
        key = ("conv", name, height, width, str(x.dtype), str(x.layout), str(x.memory_config()))
        if key not in self.prepared:
            w = ttnn.from_torch(weight, dtype=ttnn.float32)
            b = ttnn.from_torch(self.state[name + ".bias"].reshape(1, 1, 1, -1), dtype=ttnn.float32)
        else:
            w, b = self.prepared[key]
        result, (oh, ow), (w, b) = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=w,
            bias_tensor=b,
            device=self.device,
            in_channels=in_channels,
            out_channels=out_channels,
            batch_size=1,
            input_height=height,
            input_width=width,
            kernel_size=(kh, kw),
            stride=(1, 1),
            padding=(kh // 2, kw // 2),
            dtype=ttnn.float32,
            conv_config=ttnn.Conv2dConfig(
                weights_dtype=ttnn.bfloat16,
                output_layout=ttnn.TILE_LAYOUT,
                act_block_h_override=32,
            ),
            compute_config=self.compute,
            return_output_dim=True,
            return_weights_and_bias=True,
        )
        if (oh, ow) != (height, width):
            raise RuntimeError(f"Unexpected convolution output shape {(oh, ow)}")
        self.prepared[key] = (w, b)
        return ttnn.reshape(
            ttnn.to_memory_config(result, ttnn.DRAM_MEMORY_CONFIG), (1, 1, height * width, out_channels)
        )

    def _group_average(self, x):
        """Spatial mean then channel-group mean, without expanding full activations."""
        channels = x.shape[-1]
        y = ttnn.mean(x, dim=2, keepdim=True)
        y = ttnn.reshape(ttnn.to_layout(y, ttnn.ROW_MAJOR_LAYOUT), (1, 1, 32, channels // 32))
        y = ttnn.mean(ttnn.to_layout(y, ttnn.TILE_LAYOUT), dim=3, keepdim=True)
        y = ttnn.repeat(y, (1, 1, 1, channels // 32))
        return ttnn.to_layout(
            ttnn.reshape(ttnn.to_layout(y, ttnn.ROW_MAJOR_LAYOUT), (1, 1, 1, channels)),
            ttnn.TILE_LAYOUT,
        )

    def norm(self, x, name):
        x = ttnn.typecast(x, ttnn.float32)
        centered = ttnn.subtract(x, self._group_average(x))
        variance = self._group_average(ttnn.multiply(centered, centered))
        y = ttnn.multiply(centered, ttnn.rsqrt(ttnn.add(variance, 1e-6)))
        return ttnn.add(ttnn.multiply(y, self._parameter(name + ".weight")), self._parameter(name + ".bias"))

    def residual(self, x, name, height, width, observe=None):
        shortcut = (
            self.conv(x, name + ".conv_shortcut", height, width) if name + ".conv_shortcut.weight" in self.state else x
        )
        for layer in (1, 2):
            x = self.norm(x, name + f".norm{layer}")
            if observe:
                observe(name + f".norm{layer}", x, height, width)
            x = self.conv(ttnn.silu(x), name + f".conv{layer}", height, width)
            if observe:
                observe(name + f".conv{layer}", x, height, width)
        x = ttnn.add(x, shortcut)
        if observe:
            observe(name, x, height, width)
        return x

    def attention(self, x, name, height, width, observe=None):
        residual = x
        x = self.norm(x, name + ".group_norm")
        q, k, v = (self._linear(x, name + ".to_" + label) for label in ("q", "k", "v"))
        # The VAE uses one head of width 384; SDPA avoids a dense S-by-S buffer.
        q, k, v = (ttnn.typecast(y, ttnn.bfloat16) for y in (q, k, v))
        x = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=False,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=self.device.compute_with_storage_grid_size(),
                q_chunk_size=128,
                k_chunk_size=128,
                exp_approx_mode=False,
            ),
            compute_kernel_config=self.compute,
        )
        x = self._linear(ttnn.typecast(x, ttnn.float32), name + ".to_out.0")
        x = ttnn.add(x, residual)
        if observe:
            observe(name, x, height, width)
        return x

    def preprocess(self, packed, height, width):
        """Inverse latent BN, then invert channel-major 2x2 packing on device."""
        h, w = height // 16, width // 16
        if height < 16 or width < 16 or height % 16 or width % 16:
            raise ValueError("Output height and width must be positive multiples of 16")
        if tuple(packed.shape) != (1, 1, h * w, 128):
            raise ValueError(f"Expected packed shape {(1, 1, h * w, 128)}, got {tuple(packed.shape)}")
        packed = ttnn.typecast(packed, ttnn.float32)
        std = ttnn.sqrt(ttnn.add(self._parameter("bn.running_var"), self.config["batch_norm_eps"]))
        x = ttnn.add(ttnn.multiply(packed, std), self._parameter("bn.running_mean"))
        x = ttnn.reshape(ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT), (1, h, w, 32, 2, 2))
        x = ttnn.permute(x, (0, 1, 4, 2, 5, 3))
        x = ttnn.reshape(x, (1, 1, 4 * h * w, 32))
        return ttnn.to_layout(x, ttnn.TILE_LAYOUT)

    def decode(self, packed, height: int, width: int, observe=None):
        x = self.preprocess(packed, height, width)
        h, w = height // 8, width // 8
        if observe:
            observe("vae_input", x, h, w)
        for name in ("post_quant_conv", "decoder.conv_in"):
            x = self.conv(x, name, h, w)
            if observe:
                observe(name, x, h, w)
        x = self.residual(x, "decoder.mid_block.resnets.0", h, w, observe)
        x = self.attention(x, "decoder.mid_block.attentions.0", h, w, observe)
        x = self.residual(x, "decoder.mid_block.resnets.1", h, w, observe)
        for block in range(4):
            for layer in range(3):
                x = self.residual(x, f"decoder.up_blocks.{block}.resnets.{layer}", h, w, observe)
            if block < 3:
                x = ttnn.reshape(ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT), (1, h, w, x.shape[-1]))
                x = ttnn.upsample(x, scale_factor=(2, 2))
                h, w = h * 2, w * 2
                x = ttnn.to_layout(ttnn.reshape(x, (1, 1, h * w, x.shape[-1])), ttnn.TILE_LAYOUT)
                name = f"decoder.up_blocks.{block}.upsamplers.0.conv"
                x = self.conv(x, name, h, w)
                if observe:
                    observe(name, x, h, w)
        x = self.norm(x, "decoder.conv_norm_out")
        if observe:
            observe("decoder.conv_norm_out", x, h, w)
        x = self.conv(ttnn.silu(x), "decoder.conv_out", h, w)
        if observe:
            observe("decoder.conv_out", x, h, w)
        return ttnn.reshape(ttnn.clamp(x, min=-1.0, max=1.0), (1, height, width, 3))
