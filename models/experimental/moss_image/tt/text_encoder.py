# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5-9B text-only, batch-one TTNN prompt encoder for MOSS-Image.

Returns every token's final normalized hidden state, without an LM head. All
learned computation remains on TT. Tokenization, RoPE constants and checkpoint
loading are host preparation. Layer weights are streamed to bound device memory.
The single-device encoder can run before either a one- or two-card denoiser.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import torch
from models.experimental.moss_image.tt.checkpoint import SafeWeights, validate_tensor_shapes


def format_prompt(tokenizer, prompt: str, max_sequence_length: int = 4096):
    """Use the official MOSS chat template, retaining all special tokens."""
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")
    if not 1 <= max_sequence_length <= 4096:
        raise ValueError("max_sequence_length must be between 1 and 4096")
    formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    return tokenizer(formatted, return_tensors="pt", truncation=True, max_length=max_sequence_length)


class TextWeights:
    """Read the actual AutoModel safetensors prefix without guessing shard names."""

    def __init__(self, checkpoint: Path):
        self.root = Path(checkpoint)
        if (self.root / "text_encoder/config.json").is_file():
            self.root /= "text_encoder"
        self.config = json.loads((self.root / "config.json").read_text())["text_config"]
        self.tensors = SafeWeights(self.root)
        self.mapping = self.tensors.mapping
        candidates = [
            p
            for p in ("language_model.", "model.language_model.", "model.", "")
            if p + "embed_tokens.weight" in self.mapping
        ]
        if len(candidates) != 1:
            raise ValueError("Expected one unambiguous Qwen3.5 language-model weight prefix")
        self.prefix = candidates[0]
        validate_tensor_shapes(self.tensors, "text_encoder", self.prefix)

    def get(self, name):
        key = self.prefix + name
        if key not in self.mapping:
            raise KeyError(f"Missing text encoder weight {key}")
        return self.tensors.get(key)


class TTTextEncoder:
    """Stateless full-sequence encoder. No CUDA, injected activations or KV cache."""

    def __init__(self, checkpoint: Path, device):
        import ttnn

        from models.experimental.moss_image.tt import precision
        from models.experimental.moss_image.tt.attention import AccurateAttention

        self.precision = precision
        self.ttnn = ttnn
        self.device = device
        if device.get_num_devices() != 1:
            raise ValueError("Run the text encoder on one reserved device before opening the denoiser mesh")
        self.weights = TextWeights(checkpoint)
        self.config = self.weights.config
        c = self.config
        if (
            c["hidden_size"],
            c["num_hidden_layers"],
            c["num_attention_heads"],
            c["num_key_value_heads"],
            c["head_dim"],
            c["linear_num_key_heads"],
            c["linear_num_value_heads"],
            c["linear_key_head_dim"],
            c["linear_value_head_dim"],
        ) != (4096, 32, 16, 4, 256, 16, 32, 128, 128):
            raise ValueError("Only the pinned MOSS Qwen3.5-9B text architecture is supported")
        expected_layers = (["linear_attention"] * 3 + ["full_attention"]) * 8
        if (
            c["linear_conv_kernel_dim"] != 4
            or c["hidden_act"] != "silu"
            or not c["attn_output_gate"]
            or c["rope_parameters"]["partial_rotary_factor"] != 0.25
            or c["layer_types"] != expected_layers
        ):
            raise ValueError("Unsupported Qwen3.5 convolution, activation, gate or rotary configuration")
        self.compute = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        self.accurate_attention = AccurateAttention(device, self.compute, head_group_size=6)

    def upload(self, tensor, dtype=None, layout=None):
        t = self.ttnn
        return t.from_torch(
            tensor.contiguous(),
            device=self.device,
            dtype=dtype or t.bfloat16,
            layout=layout or t.TILE_LAYOUT,
            memory_config=t.DRAM_MEMORY_CONFIG,
        )

    def linear(self, x, state, name):
        t = self.ttnn
        weight = self.upload(state[name + ".weight"].T)
        return t.linear(x, weight, compute_kernel_config=self.compute, memory_config=t.DRAM_MEMORY_CONFIG)

    def norm(self, x, weight, zero_centered=True):
        t = self.ttnn
        # Qwen decoder/query/key norms store an offset; GDN output norm does not.
        xf = t.typecast(x, t.float32)
        variance = t.mean(self.precision.multiply(xf, xf), dim=-1, keepdim=True)
        normalized = self.precision.multiply(xf, t.rsqrt(self.precision.add(variance, self.config["rms_norm_eps"])))
        if zero_centered:
            # HF multiplies by (1 + weight.float()) before casting back. Folding
            # that offset into a BF16 weight would round it prematurely.
            scale = self.upload((weight.float() + 1).reshape(1, -1), dtype=t.float32)
            return t.typecast(self.precision.multiply(normalized, scale), x.dtype)
        normalized = t.typecast(normalized, x.dtype)
        return self.precision.multiply(normalized, self.upload(weight.reshape(1, -1)))

    def _rope(self, x):
        t = self.ttnn
        seq, dim = x.shape[-2], x.shape[-1]
        params = self.config["rope_parameters"]
        rotary = int(dim * params["partial_rotary_factor"])
        freq = params["rope_theta"] ** (-torch.arange(0, rotary, 2).float() / rotary)
        phase = torch.arange(seq).float()[:, None] * freq[None]
        phase = torch.cat([phase, phase], -1)
        cosine = self.upload(phase.cos().reshape(1, 1, seq, rotary))
        sine = self.upload(phase.sin().reshape(1, 1, seq, rotary))
        rotating = x[..., :rotary]
        rotated = self.precision.rotary_split_half(rotating, cosine, sine)
        return t.concat([rotated, x[..., rotary:]], dim=-1)

    def full_attention(self, x, state, record):
        t = self.ttnn
        seq = x.shape[1]
        qg = self.linear(x, state, "self_attn.q_proj")
        qg = t.reshape(qg, (1, seq, 16, 512))
        q, gate = qg[..., :256], qg[..., 256:]
        q = self.norm(q, state["self_attn.q_norm.weight"])
        k = t.reshape(self.linear(x, state, "self_attn.k_proj"), (1, seq, 4, 256))
        k = self.norm(k, state["self_attn.k_norm.weight"])
        v = t.reshape(self.linear(x, state, "self_attn.v_proj"), (1, seq, 4, 256))
        q = self._rope(t.permute(q, (0, 2, 1, 3)))
        k = self._rope(t.permute(k, (0, 2, 1, 3)))
        v = t.permute(v, (0, 2, 1, 3))
        y = self.accurate_attention(q, k, v, is_causal=True, scale=256**-0.5)
        y = t.permute(y, (0, 2, 1, 3))
        y = self.precision.multiply(y, self.precision.sigmoid(gate))
        record("attention_gated", y)
        y = t.reshape(y, (1, seq, 4096))
        return self.linear(y, state, "self_attn.o_proj")

    def linear_attention(self, x, state, valid_length, record):
        from .gated_delta import gated_delta_rule_recurrent

        t = self.ttnn
        seq = x.shape[1]
        p = "linear_attn."
        qkv = self.linear(x, state, p + "in_proj_qkv")
        record("linear_attn.in_proj_qkv", qkv)
        # Causal depthwise four-tap FIR. There is no history between prompts.
        zeros = t.zeros(
            (1, 3, 8192), dtype=t.bfloat16, layout=t.TILE_LAYOUT, device=self.device, memory_config=t.DRAM_MEMORY_CONFIG
        )
        padded = t.concat([zeros, qkv], dim=1)
        conv = state[p + "conv1d.weight"]
        out = None
        for index in range(4):
            window = t.to_layout(padded[:, index : index + seq, :], t.TILE_LAYOUT)
            tap = self.upload(conv[:, 0, index].float().reshape(1, 1, -1), dtype=t.float32)
            term = self.precision.multiply(t.typecast(window, t.float32), tap)
            out = term if out is None else self.precision.add(out, term)
        qkv = self.precision.silu(t.typecast(out, t.bfloat16))
        record("gdn_convolution", qkv)
        q = t.reshape(qkv[..., :2048], (1, seq, 16, 128))
        k = t.reshape(qkv[..., 2048:4096], (1, seq, 16, 128))
        v = t.reshape(qkv[..., 4096:], (1, seq, 32, 128))
        a = self.linear(x, state, p + "in_proj_a")
        record("linear_attn.in_proj_a", a)
        b = self.linear(x, state, p + "in_proj_b")
        record("linear_attn.in_proj_b", b)
        beta = self.precision.sigmoid(b)
        # HF calculates decay parameters in FP32.
        a = t.typecast(a, t.float32)
        dt = self.upload(state[p + "dt_bias"].float().reshape(1, 1, -1), dtype=t.float32)
        decay = self.upload(-state[p + "A_log"].float().exp().reshape(1, 1, -1), dtype=t.float32)
        g = self.precision.multiply(decay, t.softplus(self.precision.add(a, dt)))
        record("gdn_decay", g)
        record("gdn_beta", beta)
        y, final_state = gated_delta_rule_recurrent(
            t.typecast(q, t.float32),
            t.typecast(k, t.float32),
            v,
            beta,
            g,
            device=self.device,
            valid_length=valid_length,
        )
        t.deallocate(final_state)
        y = t.typecast(y, t.bfloat16)
        record("gdn_scan", y)
        gate = self.linear(x, state, p + "in_proj_z")
        record("linear_attn.in_proj_z", gate)
        gate = t.reshape(gate, (1, seq, 32, 128))
        y = self.norm(y, state[p + "norm.weight"], zero_centered=False)
        record("gdn_norm_before_gate", y)
        y = t.typecast(
            self.precision.multiply(t.typecast(y, t.float32), self.precision.silu(t.typecast(gate, t.float32))),
            t.bfloat16,
        )
        record("gdn_norm_output", y)
        y = t.reshape(y, (1, seq, 4096))
        y = self.linear(y, state, p + "out_proj")
        record("linear_attn.out_proj", y)
        return y

    def encode(self, input_ids: torch.Tensor, *, observe: Callable | None = None, max_layers: int | None = None):
        """Return TT [1,S,4096]; optional observer(name, tensor) is read-only.

        Right-padding to 128 tokens is device-kernel preparation. Causal/GDN
        masking retains the real length; padding rows are removed from output.
        max_layers is solely a component-validation option, never full inference.
        """
        t = self.ttnn
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or not 1 <= input_ids.shape[1] <= 4096:
            raise ValueError("input_ids must have shape [1,S], 1 <= S <= 4096")
        count = self.config["num_hidden_layers"] if max_layers is None else max_layers
        if not 1 <= count <= self.config["num_hidden_layers"]:
            raise ValueError("Invalid max_layers")
        valid = input_ids.shape[1]
        padded = (valid + 127) // 128 * 128
        ids = torch.nn.functional.pad(input_ids.to(torch.int32), (0, padded - valid))
        ids_tt = self.upload(ids, dtype=t.uint32, layout=t.ROW_MAJOR_LAYOUT)
        embedding = self.upload(self.weights.get("embed_tokens.weight"), layout=t.ROW_MAJOR_LAYOUT)
        x = t.embedding(ids_tt, embedding, layout=t.TILE_LAYOUT)
        t.deallocate(embedding)
        t.deallocate(ids_tt)
        x = t.reshape(x, (1, padded, 4096))
        if observe:
            observe("embedding", x)
        for layer in range(count):
            prefix = f"layers.{layer}."
            state = {
                name[len(self.weights.prefix + prefix) :]: self.weights.get(name[len(self.weights.prefix) :])
                for name in self.weights.mapping
                if name.startswith(self.weights.prefix + prefix)
            }
            record = lambda name, value: observe(f"layer_{layer:03d}.{name}", value) if observe else None
            normed = self.norm(x, state["input_layernorm.weight"])
            record("input_layernorm", normed)
            if self.config["layer_types"][layer] == "full_attention":
                y = self.full_attention(normed, state, record)
            else:
                y = self.linear_attention(normed, state, valid, record)
            x = self.precision.add(x, y)
            record("attention_residual", x)
            normed = self.norm(x, state["post_attention_layernorm.weight"])
            record("post_attention_layernorm", normed)
            gate = self.linear(normed, state, "mlp.gate_proj")
            record("mlp.gate_proj", gate)
            up = self.linear(normed, state, "mlp.up_proj")
            record("mlp.up_proj", up)
            y = self.linear(self.precision.multiply(self.precision.silu(gate), up), state, "mlp.down_proj")
            record("mlp.down_proj", y)
            x = self.precision.add(x, y)
            record("output", x)
            del state
        if count == self.config["num_hidden_layers"]:
            x = self.norm(x, self.weights.get("norm.weight"))
        x = x[:, :valid, :]
        if observe:
            observe("last_hidden_state" if count == 32 else "partial_hidden_state", x)
        return x

    def encode_prompt(self, prompt: str, tokenizer, *, max_sequence_length=4096, observe=None):
        inputs = format_prompt(tokenizer, prompt, max_sequence_length)
        hidden = self.encode(inputs["input_ids"], observe=observe)
        return hidden, torch.ones(inputs["input_ids"].shape, dtype=torch.bool)
