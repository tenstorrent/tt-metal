# SPDX-FileCopyrightText: © 2026 Qwen Image 2.1 contributors
# SPDX-License-Identifier: Apache-2.0

"""Single-image Qwen-Image 2.1 VAE decoder using device-resident TTNN ops.

The pinned checkpoint uses 2D image convolutions and residual upsampling.
Only batch one / one frame is supported. No CUDA or host computation is used
between the packed latent input and decoded image output.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import ttnn
from safetensors import safe_open


class QwenImage21VAEDecoder:
    def __init__(self, checkpoint: Path, device):
        self.device = device
        self.config = json.loads((checkpoint / "vae/config.json").read_text())
        if (
            self.config["z_dim"],
            self.config["decoder_base_dim"],
            self.config["dim_mult"],
            self.config["is_residual"],
            self.config.get("patch_size"),
        ) != (64, 144, [1, 2, 4, 8, 8], True, None):
            raise ValueError("unsupported VAE architecture; expected the pinned Qwen-Image 2.1 checkpoint")
        with safe_open(checkpoint / "vae/diffusion_pytorch_model.safetensors", framework="pt") as weights:
            self.state = {
                key: weights.get_tensor(key)
                for key in weights.keys()
                if key.startswith(("decoder.", "post_quant_conv."))
            }
        self.prepared = {}
        self.compute = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def upload(self, value: torch.Tensor):
        return ttnn.from_torch(
            value.to(torch.bfloat16).contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device) if self.device.get_num_devices() > 1 else None,
        )

    def collect(self, value, height: int, width: int, channels: int):
        shards = ttnn.get_device_tensors(value)
        host = ttnn.to_torch(shards[0]) if len(shards) > 1 else ttnn.to_torch(value)
        return host.reshape(1, height, width, channels).permute(0, 3, 1, 2).unsqueeze(2).contiguous()

    def conv(self, x, name: str, height: int, width: int):
        weight = self.state[name + ".weight"]
        out_channels, in_channels, kh, kw = weight.shape
        if kh == kw == 1:
            if name not in self.prepared:
                self.prepared[name] = (
                    self.upload(weight[:, :, 0, 0].T),
                    self.upload(self.state[name + ".bias"].reshape(1, 1, 1, -1)),
                )
            w, bias = self.prepared[name]
            return ttnn.add(ttnn.matmul(x, w, compute_kernel_config=self.compute), bias)
        if name not in self.prepared:
            w = ttnn.from_torch(weight.float(), dtype=ttnn.float32)
            bias = ttnn.from_torch(self.state[name + ".bias"].float().reshape(1, 1, 1, -1), dtype=ttnn.float32)
        else:
            w, bias = self.prepared[name]
        result, (oh, ow), (w, bias) = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=w,
            bias_tensor=bias,
            in_channels=in_channels,
            out_channels=out_channels,
            device=self.device,
            batch_size=1,
            input_height=height,
            input_width=width,
            kernel_size=(kh, kw),
            stride=(1, 1),
            padding=(kh // 2, kw // 2),
            dtype=ttnn.bfloat16,
            conv_config=ttnn.Conv2dConfig(
                weights_dtype=ttnn.bfloat16, output_layout=ttnn.TILE_LAYOUT, act_block_h_override=32
            ),
            compute_config=self.compute,
            return_output_dim=True,
            return_weights_and_bias=True,
        )
        if (oh, ow) != (height, width):
            raise ValueError("VAE convolution unexpectedly changed spatial dimensions")
        self.prepared[name] = (w, bias)
        result = ttnn.to_memory_config(result, ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.reshape(result, (1, 1, height * width, out_channels))

    def norm(self, x, name: str):
        """Match F.normalize(fp32)->bf16, then two separately rounded products."""
        channels = x.shape[-1]
        key = name + ".gamma"
        if key not in self.prepared:
            self.prepared[key] = self.upload(self.state[key].reshape(1, 1, 1, channels))
        xf = ttnn.typecast(x, ttnn.float32)
        squared = ttnn.multiply(xf, xf)
        length = ttnn.sqrt(ttnn.sum(squared, dim=3, keepdim=True))
        inverse = ttnn.reciprocal(ttnn.maximum(length, 1e-12))
        normalized = ttnn.typecast(ttnn.multiply(xf, inverse), ttnn.bfloat16)
        return ttnn.multiply(ttnn.multiply(normalized, math.sqrt(channels)), self.prepared[key])

    def residual(self, x, name: str, height: int, width: int, observe=None):
        shortcut = (
            self.conv(x, name + ".conv_shortcut", height, width) if name + ".conv_shortcut.weight" in self.state else x
        )
        for index in (1, 2):
            x = self.norm(x, name + f".norm{index}")
            if observe:
                observe(name + f".norm{index}", x, height, width)
            x = ttnn.silu(x)
            x = self.conv(x, name + f".conv{index}", height, width)
            if observe:
                observe(name + f".conv{index}", x, height, width)
        return ttnn.add(x, shortcut)

    def attention(self, x, name: str, height: int, width: int, observe=None):
        residual = x
        channels = x.shape[-1]
        sequence = height * width
        x = self.norm(x, name + ".norm")
        if observe:
            observe(name + ".norm", x, height, width)
        qkv = self.conv(x, name + ".to_qkv", height, width)
        if observe:
            observe(name + ".to_qkv", qkv, height, width)
        q, k, v = (
            ttnn.slice(qkv, (0, 0, 0, index * channels), (1, 1, sequence, (index + 1) * channels)) for index in range(3)
        )
        scores = ttnn.matmul(q, ttnn.permute(k, (0, 1, 3, 2)), compute_kernel_config=self.compute)
        probabilities = ttnn.softmax(ttnn.multiply(scores, channels**-0.5), dim=-1)
        x = ttnn.matmul(probabilities, v, compute_kernel_config=self.compute)
        x = self.conv(x, name + ".proj", height, width)
        if observe:
            observe(name + ".proj", x, height, width)
        return ttnn.add(x, residual)

    def nearest(self, x, height: int, width: int):
        x = ttnn.reshape(ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT), (1, height, width, x.shape[-1]))
        x = ttnn.upsample(x, scale_factor=(2, 2))
        return ttnn.to_layout(ttnn.reshape(x, (1, 1, 4 * height * width, x.shape[-1])), ttnn.TILE_LAYOUT)

    def duplicate_shortcut(self, x, in_channels: int, out_channels: int, factor_t: int, height: int, width: int):
        """Wan DupUp3D first-frame shortcut expressed as channel gathers + upsample.

        The repeated-channel pixel shuffle can select different source channels
        at each of the four spatial offsets when channel counts decrease.
        """
        repeats = out_channels * factor_t * 4 // in_channels
        # For first_chunk only the last temporal offset is retained.
        outputs = []
        for dy in range(2):
            for dx in range(2):
                indices = (torch.arange(out_channels) * factor_t * 4 + (factor_t - 1) * 4 + dy * 2 + dx) // repeats
                # A sparse 0/1 projection performs the exact channel mapping on TT.
                key = ("duplicate", in_channels, out_channels, factor_t, dy, dx)
                if key not in self.prepared:
                    matrix = torch.zeros(in_channels, out_channels, dtype=torch.bfloat16)
                    matrix[indices, torch.arange(out_channels)] = 1
                    self.prepared[key] = self.upload(matrix)
                outputs.append(ttnn.matmul(x, self.prepared[key], compute_kernel_config=self.compute))
        top = ttnn.concat([ttnn.to_layout(y, ttnn.ROW_MAJOR_LAYOUT) for y in outputs[:2]], dim=3)
        bottom = ttnn.concat([ttnn.to_layout(y, ttnn.ROW_MAJOR_LAYOUT) for y in outputs[2:]], dim=3)
        top = ttnn.reshape(top, (1, height, width * 2, out_channels))
        bottom = ttnn.reshape(bottom, (1, height, width * 2, out_channels))
        rows = ttnn.concat([top, bottom], dim=2)
        return ttnn.to_layout(ttnn.reshape(rows, (1, 1, height * width * 4, out_channels)), ttnn.TILE_LAYOUT)

    def decode(self, packed, height: int, width: int, observe=None):
        if height % 32 or width % 32 or height < 32 or width < 32:
            raise ValueError("height and width must be positive multiples of 32")
        h, w = height // 16, width // 16
        if tuple(packed.shape) not in ((1, h * w, 64), (1, 1, h * w, 64)):
            raise ValueError("packed latent shape does not match output resolution")
        x = ttnn.reshape(packed, (1, 1, h * w, 64))
        x = ttnn.add(
            ttnn.multiply(x, self.upload(torch.tensor(self.config["latents_std"]).reshape(1, 1, 1, 64))),
            self.upload(torch.tensor(self.config["latents_mean"]).reshape(1, 1, 1, 64)),
        )
        if observe:
            observe("vae_input", x, h, w)
        x = self.conv(x, "post_quant_conv", h, w)
        if observe:
            observe("post_quant_conv", x, h, w)
        x = self.conv(x, "decoder.conv_in", h, w)
        if observe:
            observe("decoder.conv_in", x, h, w)
        x = self.residual(x, "decoder.mid_block.resnets.0", h, w, observe)
        if observe:
            observe("decoder.mid_block.resnets.0", x, h, w)
        x = self.attention(x, "decoder.mid_block.attentions.0", h, w, observe)
        if observe:
            observe("decoder.mid_block.attentions.0", x, h, w)
        x = self.residual(x, "decoder.mid_block.resnets.1", h, w)
        if observe:
            observe("decoder.mid_block", x, h, w)
        factors = self.config["temperal_downsample"][::-1]
        for block, out_channels in enumerate((1152, 1152, 576, 288, 144)):
            shortcut = x
            in_channels = x.shape[-1]
            for layer in range(self.config["num_res_blocks"] + 1):
                x = self.residual(x, f"decoder.up_blocks.{block}.resnets.{layer}", h, w)
            if block < 4:
                x = self.nearest(x, h, w)
                if observe:
                    observe(f"decoder.up_blocks.{block}.upsampler.resample.0", x, h * 2, w * 2)
                x = self.conv(x, f"decoder.up_blocks.{block}.upsampler.resample.1", h * 2, w * 2)
                shortcut = self.duplicate_shortcut(
                    shortcut, in_channels, out_channels, 2 if factors[block] else 1, h, w
                )
                if observe:
                    observe(f"decoder.up_blocks.{block}.avg_shortcut", shortcut, h * 2, w * 2)
                x = ttnn.add(x, shortcut)
                h, w = h * 2, w * 2
            if observe:
                observe(f"decoder.up_blocks.{block}", x, h, w)
        x = self.norm(x, "decoder.norm_out")
        if observe:
            observe("decoder.norm_out", x, h, w)
        x = self.conv(ttnn.silu(x), "decoder.conv_out", h, w)
        if observe:
            observe("decoder.conv_out", x, h, w)
        return ttnn.clamp(x, min=-1.0, max=1.0)
