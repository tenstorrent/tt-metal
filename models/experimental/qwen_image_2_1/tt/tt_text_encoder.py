# SPDX-FileCopyrightText: © 2026 Qwen Image 2.1 contributors
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-VL text-only prompt encoding for the pinned Qwen-Image 2.1 model.

Tokenizer and control metadata are host inputs. Embedding lookup and every
learned encoder operation execute on the caller's TT device. Image-conditioned
encoding and padded batches are deliberately outside this implementation.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import ttnn
from safetensors import safe_open
from safetensors.torch import load_file


class QwenImage21TextEncoder:
    def __init__(self, checkpoint: Path, device):
        self.root = checkpoint / "text_encoder"
        self.config = json.loads((self.root / "config.json").read_text())["text_config"]
        c = self.config
        if (
            c["hidden_size"],
            c["num_hidden_layers"],
            c["num_attention_heads"],
            c["num_key_value_heads"],
            c["head_dim"],
            c["intermediate_size"],
        ) != (4096, 36, 32, 8, 128, 12288):
            raise ValueError("expected the pinned Qwen-Image 2.1 text encoder")
        self.device = device
        self.compute = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def upload(self, value, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(
            value.contiguous(),
            dtype=dtype,
            layout=layout,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device) if self.device.get_num_devices() > 1 else None,
        )

    def checkpoint_state(self, filename: str, prefix: str):
        """Accept diagnostic exports or original Hugging Face shards on CPU."""
        if (self.root / filename).is_file():
            return load_file(self.root / filename)
        mapping = json.loads((self.root / "model.safetensors.index.json").read_text())["weight_map"]
        selected = [name for name in mapping if name.startswith(prefix)]
        if not selected:
            raise ValueError(f"missing text encoder weights: {prefix}")
        state = {}
        for shard in sorted({mapping[name] for name in selected}):
            with safe_open(self.root / shard, framework="pt", device="cpu") as source:
                for name in selected:
                    if mapping[name] == shard:
                        state[name] = source.get_tensor(name)
        return state

    def norm(self, x, weight):
        xf = ttnn.typecast(x, ttnn.float32)
        variance = ttnn.mean(ttnn.multiply(xf, xf), dim=-1, keepdim=True)
        inverse = ttnn.rsqrt(ttnn.add(variance, self.config["rms_norm_eps"]))
        normalized = ttnn.typecast(ttnn.multiply(xf, inverse), ttnn.bfloat16)
        return ttnn.multiply(normalized, self.upload(weight.reshape(1, 1, 1, -1)))

    def linear(self, x, weight):
        return ttnn.matmul(
            x,
            self.upload(weight.T.contiguous()),
            compute_kernel_config=self.compute,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def rope(self, padded_sequence):
        # Text-only positions agree in all three mRoPE axes. Frequency constants
        # are model metadata; phase multiplication and trig execute on TT.
        base = self.config.get("rope_theta", 5000000)
        inverse = 1 / (base ** (torch.arange(0, 128, 2).float() / 128))
        positions = self.upload(
            torch.arange(padded_sequence).float().reshape(1, 1, -1, 1),
            dtype=ttnn.float32,
        )
        inv = self.upload(inverse.reshape(1, 1, 1, 64), dtype=ttnn.float32)
        phase = ttnn.multiply(positions, inv)
        phase = ttnn.concat([phase, phase], dim=-1)
        return ttnn.typecast(ttnn.cos(phase), ttnn.bfloat16), ttnn.typecast(ttnn.sin(phase), ttnn.bfloat16)

    def layer(self, hidden, state, cos, sin, sequence, observe=None):
        def record(name, value):
            if observe:
                observe(name, value)

        x = self.norm(hidden, state["input_layernorm.weight"])
        record("input_layernorm", x)
        padded = (sequence + 31) // 32 * 32
        heads = []
        for name, count in (("q", 32), ("k", 8), ("v", 8)):
            y = self.linear(x, state[f"self_attn.{name}_proj.weight"])
            record(f"self_attn.{name}_proj", y)
            y = ttnn.reshape(y, (1, sequence, count, 128))
            if name != "v":
                y = self.norm(y, state[f"self_attn.{name}_norm.weight"])
                record(f"self_attn.{name}_norm", y)
            y = ttnn.permute(y, (0, 2, 1, 3))
            if padded != sequence:
                y = ttnn.pad(y, ((0, 0), (0, 0), (0, padded - sequence), (0, 0)), 0.0)
            if name != "v":
                y = ttnn.experimental.rotary_embedding_hf(y, cos, sin, is_decode_mode=False)
                record(
                    f"rotated_{name}",
                    ttnn.slice(y, (0, 0, 0, 0), (1, count, sequence, 128)),
                )
            heads.append(y)
        x = ttnn.transformer.scaled_dot_product_attention(
            *heads,
            is_causal=True,
            scale=128**-0.5,
            compute_kernel_config=self.compute,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        x = ttnn.slice(x, (0, 0, 0, 0), (1, 32, sequence, 128))
        x = ttnn.reshape(ttnn.permute(x, (0, 2, 1, 3)), (1, 1, sequence, 4096))
        record("attention_heads_merged", x)
        x = self.linear(x, state["self_attn.o_proj.weight"])
        record("self_attn.o_proj", x)
        hidden = ttnn.add(hidden, x)
        x = self.norm(hidden, state["post_attention_layernorm.weight"])
        record("post_attention_layernorm", x)
        gate = self.linear(x, state["mlp.gate_proj.weight"])
        up = self.linear(x, state["mlp.up_proj.weight"])
        record("mlp.gate_proj", gate)
        record("mlp.up_proj", up)
        x = ttnn.multiply(ttnn.silu(gate), up)
        record("mlp_product", x)
        x = self.linear(x, state["mlp.down_proj.weight"])
        record("mlp.down_proj", x)
        return ttnn.add(hidden, x)

    def encode(self, input_ids: torch.Tensor, drop_idx: int, observe=None, max_layers=None):
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] <= drop_idx or drop_idx < 0:
            raise ValueError("text-only batch one with a nonempty retained prompt is required")
        sequence = input_ids.shape[1]
        state = self.checkpoint_state("embedding.safetensors", "model.language_model.embed_tokens.")
        weight = state["model.language_model.embed_tokens.weight"]
        ids = self.upload(input_ids.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        embedding = self.upload(weight, layout=ttnn.ROW_MAJOR_LAYOUT)
        hidden = ttnn.embedding(ids, embedding, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        hidden = ttnn.reshape(hidden, (1, 1, sequence, 4096))
        del state, weight, embedding
        if observe:
            observe("embedding", hidden)
        cos, sin = self.rope((sequence + 31) // 32 * 32)
        if observe:
            observe("rotary_cos", ttnn.slice(cos, (0, 0, 0, 0), (1, 1, sequence, 128)))
            observe("rotary_sin", ttnn.slice(sin, (0, 0, 0, 0), (1, 1, sequence, 128)))
        count = self.config["num_hidden_layers"] if max_layers is None else max_layers
        if not 1 <= count <= self.config["num_hidden_layers"]:
            raise ValueError("invalid layer count")
        for index in range(count):
            prefix = f"model.language_model.layers.{index}."
            state = {
                key.removeprefix(prefix): value
                for key, value in self.checkpoint_state(f"layer_{index:03d}.safetensors", prefix).items()
            }
            callback = (
                (lambda name, value, i=index: observe(f"layer_{i:03d}/{name}", value))
                if observe and index == 0
                else None
            )
            hidden = self.layer(hidden, state, cos, sin, sequence, callback)
            if observe:
                observe(f"layer_{index:03d}", hidden)
        # QwenImage21 explicitly bypasses the final encoder RMSNorm.
        return ttnn.reshape(
            ttnn.slice(hidden, (0, 0, drop_idx, 0), (1, 1, sequence, 4096)),
            (1, sequence - drop_idx, 4096),
        )
