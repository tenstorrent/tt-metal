# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4 HCA compressor (TTNN prefill). Mirrors ``DeepseekV4HCACompressor``
(reference ``modeling_deepseek_v4.py``, paper §2.3.2) in stateless single-shot mode:
every complete window of ``compress_rate`` source tokens is softmax-pooled into one
compressed KV entry, RMS-normed, and RoPE'd at its window's absolute position."""

from __future__ import annotations

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3_d_p.tt.mla.rope import get_rot_transformation_mat


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    input_dtype = x.dtype
    x = x.to(torch.float32)
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return weight * x.to(input_dtype)


def compress_rope_cos_sin(
    n_windows: int, compress_rate: int, rope_head_dim: int, theta: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Interleaved cos/sin tables [1, 1, n_windows, rope_head_dim] at per-window positions ``i * compress_rate``."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, rope_head_dim, 2, dtype=torch.int64).float() / rope_head_dim))
    positions = torch.arange(n_windows, dtype=torch.float32) * compress_rate
    freqs = torch.outer(positions, inv_freq)
    cos = freqs.cos().repeat_interleave(2, dim=-1)
    sin = freqs.sin().repeat_interleave(2, dim=-1)
    return cos[None, None], sin[None, None]


def hca_block_bias(position_ids: torch.Tensor, compressed_len: int, compress_rate: int) -> torch.Tensor:
    """Per-query causal mask: query ``t`` may attend entry ``w`` only if ``t >= (w+1)*compress_rate``."""
    batch, seq_len = position_ids.shape
    entry_indices = torch.arange(compressed_len)
    causal_threshold = (position_ids + 1) // compress_rate
    block_bias = torch.zeros(batch, 1, seq_len, compressed_len)
    return block_bias.masked_fill(
        entry_indices.view(1, 1, 1, -1) >= causal_threshold.unsqueeze(1).unsqueeze(-1),
        float("-inf"),
    )


class TtHCACompressor(LightweightModule):
    def __init__(
        self,
        device,
        *,
        kv_proj_weight: torch.Tensor,
        gate_proj_weight: torch.Tensor,
        position_bias: torch.Tensor,
        kv_norm_weight: torch.Tensor,
        head_dim: int,
        compress_rate: int,
        rope_head_dim: int,
        compress_rope_theta: float,
        rms_norm_eps: float = 1e-6,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    ):
        self.device = device
        self.dtype = dtype
        self.memory_config = memory_config
        self.head_dim = int(head_dim)
        self.compress_rate = int(compress_rate)
        self.rope_head_dim = int(rope_head_dim)
        self.compress_rope_theta = float(compress_rope_theta)
        self.rms_norm_eps = float(rms_norm_eps)

        self.kv_norm_weight = kv_norm_weight.detach().float().contiguous()

        self.wkv = self._to_tt_linear_weight(kv_proj_weight)
        self.wgate = self._to_tt_linear_weight(gate_proj_weight)
        self.position_bias = self._from_torch(position_bias.detach().reshape(1, 1, self.compress_rate, self.head_dim))
        self.trans_mat = ttnn.from_torch(
            get_rot_transformation_mat(),
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            memory_config=memory_config,
        )

    @classmethod
    def from_reference(cls, device, reference, config, **kwargs) -> "TtHCACompressor":
        return cls(
            device,
            kv_proj_weight=reference.kv_proj.weight,
            gate_proj_weight=reference.gate_proj.weight,
            position_bias=reference.position_bias,
            kv_norm_weight=reference.kv_norm.weight,
            head_dim=config.head_dim,
            compress_rate=config.compress_rates["heavily_compressed_attention"],
            rope_head_dim=config.qk_rope_head_dim,
            compress_rope_theta=config.rope_parameters["compress"]["rope_theta"],
            rms_norm_eps=config.rms_norm_eps,
            **kwargs,
        )

    def _to_tt_linear_weight(self, weight: torch.Tensor):
        torch_weight = weight.detach().transpose(-2, -1).contiguous().unsqueeze(0).unsqueeze(0).to(torch.bfloat16)
        return ttnn.from_torch(
            torch_weight,
            device=self.device,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=self.memory_config,
        )

    def _from_torch(self, x: torch.Tensor):
        return ttnn.from_torch(
            x.to(torch.bfloat16),
            device=self.device,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=self.memory_config,
        )

    def forward(self, hidden_states, position_ids: torch.Tensor):
        """``hidden_states``: TTNN tensor [B, 1, S, hidden]. ``position_ids``: torch [B, S].
        Returns ``(compressed_kv, block_bias)`` — compressed_kv TTNN [B, 1, T, head_dim];
        block_bias host torch [B, 1, S, T] (or None)."""
        input_shape = tuple(hidden_states.shape)
        if len(input_shape) != 4 or input_shape[1] != 1:
            raise ValueError(f"Expected hidden_states shape [B, 1, S, hidden], got {input_shape}")
        batch, seq_len = input_shape[0], input_shape[2]

        kv = ttnn.linear(hidden_states, self.wkv, memory_config=self.memory_config)
        gate = ttnn.linear(hidden_states, self.wgate, memory_config=self.memory_config)
        usable = (seq_len // self.compress_rate) * self.compress_rate
        if usable > 0:
            n_windows = usable // self.compress_rate

            # device: +position_bias then softmax over the window axis (per channel)
            gate = ttnn.slice(gate, [0, 0, 0, 0], [batch, 1, usable, self.head_dim])
            gate = ttnn.reshape(gate, [batch, n_windows, self.compress_rate, self.head_dim])
            gate = ttnn.add(gate, self.position_bias)
            weights = ttnn.softmax(gate, dim=2, numeric_stable=True)

            # device: weighted sum over the window axis
            kv = ttnn.slice(kv, [0, 0, 0, 0], [batch, 1, usable, self.head_dim])
            kv = ttnn.reshape(kv, [batch, n_windows, self.compress_rate, self.head_dim])
            pooled = ttnn.sum(ttnn.multiply(kv, weights), dim=2)

            # host boundary: RMSNorm on host (fp32); RoPE on device
            pooled = ttnn.to_torch(pooled).float()
            compressed = rms_norm(pooled, self.kv_norm_weight.to(pooled.dtype), self.rms_norm_eps)

            # RoPE (device) on the trailing rope_head_dim channels only (op caps head_dim <= 256).
            compressed_dev = self._from_torch(compressed.unsqueeze(1))
            nope_dim = self.head_dim - self.rope_head_dim
            nope = ttnn.slice(compressed_dev, [0, 0, 0, 0], [batch, 1, n_windows, nope_dim])
            rope = ttnn.slice(compressed_dev, [0, 0, 0, nope_dim], [batch, 1, n_windows, self.head_dim])
            cos, sin = compress_rope_cos_sin(
                n_windows, self.compress_rate, self.rope_head_dim, self.compress_rope_theta
            )
            rope = ttnn.experimental.rotary_embedding_llama(
                rope, self._from_torch(cos), self._from_torch(sin), self.trans_mat, is_decode_mode=False
            )
            compressed_kv = ttnn.concat([nope, rope], dim=-1)
        else:
            n_windows = 0
            compressed_kv = self._from_torch(torch.zeros(batch, 1, 0, self.head_dim))

        block_bias = None
        if seq_len > 1 and n_windows > 0:
            block_bias = hca_block_bias(position_ids, n_windows, self.compress_rate)

        return compressed_kv, block_bias
