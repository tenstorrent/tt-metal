# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""MOSS-Image DiT, batch one, BF16 HiFi4, replicated or two-card tensor parallel.

The logical sequence is image first, then true-length text. No learned operation
runs on the host. CPU work constructs checkpoint weights and positional metadata.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from models.experimental.moss_image.tt.checkpoint import SafeWeights, validate_tensor_shapes


def rotary_metadata(height: int, width: int, text_length: int, base=10000.0):
    """Official four-coordinate RoPE, retaining FP32 construction until BF16 cast."""
    if min(height, width, text_length) < 1:
        raise ValueError("RoPE dimensions and text length must be positive")
    h, w = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    image = torch.stack((torch.zeros_like(h), torch.full_like(h, text_length), h, w), -1).flatten(0, 1).float()
    image[:, 2] -= (height - 1) / 2
    image[:, 3] -= (width - 1) / 2
    text = torch.zeros(text_length, 4)
    text[:, 1] = torch.arange(text_length)
    bands = 1 / (base ** (torch.arange(16, dtype=torch.float32) / 16))
    phase = torch.cat((image, text), 0).unsqueeze(-1) * bands
    return torch.cat((phase.sin(), phase.cos()), -2).flatten(-2).unsqueeze(0).to(torch.bfloat16)


def time_metadata(timestep: float):
    freq = 10000 ** (-torch.arange(128, dtype=torch.float64) / 128)
    phase = torch.as_tensor(timestep, dtype=torch.float32).double().reshape(1, 1) * freq
    return torch.cat((phase.cos(), phase.sin()), -1).float().to(torch.bfloat16).reshape(1, 1, 256)


class DiTWeights(SafeWeights):
    def __init__(self, checkpoint):
        super().__init__(Path(checkpoint) / "transformer")
        validate_tensor_shapes(self, "transformer")


class TTMossDiT:
    """Head/channel TP for QKV/MLP, all-reduce before replicated residuals."""

    def __init__(self, checkpoint, device):
        import ttnn
        from models.experimental.moss_image.tt import precision
        from models.experimental.moss_image.tt.attention import AccurateAttention

        self.ops = precision
        self.t = ttnn
        self.device = device
        self.cards = device.get_num_devices()
        if self.cards not in (1, 2):
            raise ValueError("MOSS DiT supports one or two reserved cards")
        config = json.loads((Path(checkpoint) / "transformer/config.json").read_text())
        expected = dict(
            dim=3072, ffn_dim=8192, num_heads=24, num_layers=42, text_dim=4096, freq_dim=256, latent_channels=128
        )
        if any(config.get(k) != v for k, v in expected.items()):
            raise ValueError("Checkpoint differs from the qualified MOSS architecture")
        self.weights = DiTWeights(checkpoint)
        self.cache = {}
        self.compute = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.accurate_attention = AccurateAttention(device, self.compute, head_group_size=6)
        self.metadata_cache = {}

    def upload(self, value, *, dtype=None, shard=None):
        t = self.t
        mapper = None
        if self.cards == 2:
            mapper = (
                t.ShardTensorToMesh(self.device, dim=shard)
                if shard is not None
                else t.ReplicateTensorToMesh(self.device)
            )
        return t.from_torch(
            value.contiguous(),
            device=self.device,
            dtype=dtype or t.bfloat16,
            layout=t.TILE_LAYOUT,
            memory_config=t.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )

    def host(self, value):
        t = self.t
        return t.to_torch(t.get_device_tensors(value)[0]).contiguous()

    def parameter(self, name, *, transpose=False, shard=None, part=None):
        key = (name, transpose, shard, part)
        if key not in self.cache:
            value = self.weights.get(name)
            if part is not None:
                index, chunks = part
                value = value.chunk(chunks, dim=0)[index]
            if transpose:
                value = value.T
            if value.ndim == 1:
                value = value.reshape(1, 1, -1)
            self.cache[key] = self.upload(value, shard=shard)
        return self.cache[key]

    def linear(self, x, name, mode="replicate", *, part=None, bias=None):
        t = self.t
        # Matrix [in,out]: column sharding splits output, row sharding input.
        shard = 1 if mode == "column" else 0 if mode == "row" else None
        w = self.parameter(name + ".weight", transpose=True, shard=shard, part=part)
        if bias is None:
            bias = name + ".bias" in self.weights.mapping
        # Native F.linear rounds after the bias epilogue. A BF16 matmul followed
        # by a separate BF16 add introduces an additional rounding boundary.
        options = {"dtype": t.float32} if bias else {}
        y = t.matmul(x, w, compute_kernel_config=self.compute, memory_config=t.DRAM_MEMORY_CONFIG, **options)
        if mode == "row" and self.cards == 2:
            y = t.all_reduce(
                y, cluster_axis=1, num_links=1, topology=t.Topology.Linear, memory_config=t.DRAM_MEMORY_CONFIG
            )
        if bias:
            # Apply a replicated row-parallel bias once, after the FP32 reduction.
            b = self.parameter(name + ".bias", shard=2 if mode == "column" else None, part=part)
            y = t.add(y, t.typecast(b, t.float32))
            y = t.typecast(y, t.bfloat16)
        return y

    def norm(self, x, name):
        # Match official FP32 variance, BF16 normalized activation, BF16 scale.
        t = self.t
        xf = t.typecast(x, t.float32)
        var = t.mean(self.ops.multiply(xf, xf), dim=-1, keepdim=True)
        y = self.ops.multiply(xf, t.rsqrt(self.ops.add(var, 1e-6)))
        return self.ops.multiply(t.typecast(y, t.bfloat16), self.parameter(name + ".weight"))

    def situ(self, hidden, gate):
        # This bounded activation is part of MOSS; it is not ordinary SwiGLU.
        bounded_gate = self.ops.multiply(self.ops.tanh(self.ops.multiply(gate, 0.25)), 4.0)
        bounded_hidden = self.ops.multiply(self.ops.tanh(self.ops.multiply(hidden, 0.04)), 25.0)
        return self.ops.multiply(self.ops.multiply(bounded_gate, self.ops.sigmoid(gate)), bounded_hidden)

    def rope_and_mask(self, rope, sequence):
        t = self.t
        key = (sequence, rope.view(torch.int16).numpy().tobytes())
        if key not in self.metadata_cache:
            padded = (sequence + 31) // 32 * 32
            sin, cos = rope.chunk(2, dim=-1)
            sin = torch.cat((sin, sin), -1).unsqueeze(1)
            cos = torch.cat((cos, cos), -1).unsqueeze(1)
            if padded != sequence:
                sin = torch.nn.functional.pad(sin, (0, 0, 0, padded - sequence))
                cos = torch.nn.functional.pad(cos, (0, 0, 0, padded - sequence), value=1)
            mask = torch.full((1, 1, padded, padded), -10000.0, dtype=torch.bfloat16)
            mask[:, :, :, :sequence] = 0
            self.metadata_cache[key] = (self.upload(cos), self.upload(sin), self.upload(mask))
        return self.metadata_cache[key]

    def attention(self, x, prefix, rope, *, text=False, observe=None):
        t = self.t
        sequence = x.shape[-2]
        padded = (sequence + 31) // 32 * 32
        heads = 24 // self.cards
        cos, sin, mask = self.rope_and_mask(rope, sequence)

        def project(kind):
            if text:
                y = self.linear(x, prefix + ".qkv", "column", part=({"q": 0, "k": 1, "v": 2}[kind], 3))
            else:
                y = self.linear(x, prefix + ".proj." + kind, "column")
            y = t.reshape(y, (1, sequence, heads, 128))
            if kind in ("q", "k"):
                y = self.norm(y, prefix + f".{kind}_norm")
            y = t.permute(y, (0, 2, 1, 3))
            if sequence != padded:
                y = t.pad(y, ((0, 0), (0, 0), (0, padded - sequence), (0, 0)), 0.0)
            if kind in ("q", "k"):
                y = self.ops.rotary_split_half(y, cos, sin)
            return y

        q, k, v = project("q"), project("k"), project("v")
        y = self.accurate_attention(q, k, v, attn_mask=mask, is_causal=False, scale=128**-0.5)
        if observe:
            observe("attention", y)
        y = t.reshape(t.permute(y, (0, 2, 1, 3)), (1, padded, heads * 128))
        y = t.slice(y, (0, 0, 0), (1, sequence, heads * 128))
        return self.linear(y, prefix + (".proj" if text else ".proj.o"), "row")

    def text_block(self, x, index, rope, observe=None):
        p = f"text_embedder.blocks.{index}"
        x = self.ops.add(x, self.attention(self.norm(x, p + ".norm1"), p, rope, text=True, observe=observe))
        y = self.norm(x, p + ".norm2")
        hidden = self.linear(y, p + ".ffn_up", "column")
        gate = self.linear(y, p + ".ffn_gate", "column")
        return self.ops.add(x, self.linear(self.situ(hidden, gate), p + ".ffn_down", "row"))

    def block(self, x, index, rope, modulation, observe=None):
        p = f"layers.{index}"
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = modulation

        def emit(name, value):
            if observe:
                observe(name, value)
            return value

        y = emit(
            "attention_input",
            self.ops.add(self.ops.multiply(self.norm(x, p + ".norm1"), self.ops.add(scale_a, 1.0)), shift_a),
        )
        branch = self.attention(y, p, rope, observe=observe)
        x = emit("attention_residual", self.ops.add(x, self.ops.multiply(branch, gate_a)))
        y = emit(
            "ffn_input",
            self.ops.add(self.ops.multiply(self.norm(x, p + ".norm2"), self.ops.add(scale_f, 1.0)), shift_f),
        )
        hidden = self.linear(y, p + ".proj.ffn_up_gate", "column", part=(0, 2), bias=False)
        gate = self.linear(y, p + ".proj.ffn_up_gate", "column", part=(1, 2), bias=False)
        y = self.linear(self.situ(hidden, gate), p + ".proj.ffn_down", "row")
        return emit("output", self.ops.add(x, self.ops.multiply(y, gate_f)))

    def __call__(self, latents, timestep, context, height, width, *, skip_layers=(), observe=None):
        """Packed latents [1,N,128], context [1,L,4096] -> velocity [1,N,128]."""
        t = self.t
        if any(type(i) is not int or not 0 <= i < 42 for i in skip_layers):
            raise ValueError("Invalid MOSS skip layer")
        if tuple(latents.shape) != (1, height * width, 128) or context.shape[-1] != 4096:
            raise ValueError("Invalid packed latent/context shape")

        def emit(name, value):
            if observe:
                observe(name, value)
            return value

        time = self.linear(self.upload(time_metadata(timestep)), "time_embedding.0")
        time = emit("time_embedding", self.linear(self.ops.silu(time), "time_embedding.2"))
        mod = self.linear(time, "shared_block_modulation")
        head = self.linear(time, "video_head_modulation")

        def split(value, count):
            return tuple(t.slice(value, (0, 0, i * 3072), (1, 1, (i + 1) * 3072)) for i in range(count))

        modulation = split(mod, 6)
        shift, scale = split(head, 2)
        video = emit("video_embedder", self.linear(t.typecast(latents, t.bfloat16), "video_embedder"))
        text_length = context.shape[-2]
        rope = rotary_metadata(height, width, text_length)
        text_rope = rope[:, height * width :]
        text = emit("text_projection", self.linear(t.typecast(context, t.bfloat16), "text_embedder.proj"))
        for index in range(2):
            text = emit(f"text_block_{index:02d}", self.text_block(text, index, text_rope))
        x = t.concat((video, text), dim=1)
        for index in range(42):
            if index not in skip_layers:
                x = emit(f"block_{index:02d}", self.block(x, index, rope, modulation))
        x = t.slice(x, (0, 0, 0), (1, height * width, 3072))
        x = self.norm(x, "final_norm_video")
        x = emit("head_input", self.ops.add(self.ops.multiply(x, self.ops.add(scale, 1.0)), shift))
        return emit("velocity", self.linear(x, "final_linear_video"))
