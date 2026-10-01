# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Exact Qwen4Exp Gated DeltaNet contract and TP4 checkpoint packing.

The CPU forward is the numerical oracle for the TTNN vertical slice.  The
packing contract matches the reusable Qwen3.6 TP GDN implementation: Q/K/V
heads are made contiguous per device, recurrence remains local to each value
head, and only the row-parallel output projection requires a four-device
reduction.  No tensor is replicated here except the 128-wide per-value-head
gated RMSNorm weight, which is semantically shared across all value heads.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from models.demos.blackhole.qwen38_flash_next.checkpoint import Qwen38Checkpoint
from models.demos.blackhole.qwen38_flash_next.reference import causal_depthwise_conv1d, gated_delta_recurrent

TP_SIZE = 4


@dataclass(frozen=True)
class Qwen38GDNDimensions:
    hidden_size: int
    q_heads: int
    value_heads: int
    key_head_dim: int
    value_head_dim: int
    conv_kernel: int
    tp_size: int = TP_SIZE

    def __post_init__(self) -> None:
        if self.tp_size != TP_SIZE:
            raise ValueError(f"Qwen3.8-Flash-Next GDN requires TP={TP_SIZE}, got {self.tp_size}")
        if self.q_heads % self.tp_size or self.value_heads % self.tp_size:
            raise ValueError("GDN head counts must divide exactly over four devices")
        if self.value_heads % self.q_heads:
            raise ValueError("GDN value heads must be an integer multiple of Q/K heads")

    @property
    def key_width(self) -> int:
        return self.q_heads * self.key_head_dim

    @property
    def value_width(self) -> int:
        return self.value_heads * self.value_head_dim

    @property
    def qkv_width(self) -> int:
        return 2 * self.key_width + self.value_width

    @property
    def q_heads_per_device(self) -> int:
        return self.q_heads // self.tp_size

    @property
    def value_heads_per_device(self) -> int:
        return self.value_heads // self.tp_size

    @property
    def key_width_per_device(self) -> int:
        return self.q_heads_per_device * self.key_head_dim

    @property
    def value_width_per_device(self) -> int:
        return self.value_heads_per_device * self.value_head_dim

    @property
    def qkv_width_per_device(self) -> int:
        return 2 * self.key_width_per_device + self.value_width_per_device

    @property
    def qk_repeat_factor(self) -> int:
        return self.value_heads // self.q_heads


@dataclass(frozen=True)
class Qwen38GDNDeviceShard:
    device_index: int
    dimensions: Qwen38GDNDimensions
    qkv: torch.Tensor
    z: torch.Tensor
    a: torch.Tensor
    b: torch.Tensor
    out: torch.Tensor
    conv: torch.Tensor
    dt_bias: torch.Tensor
    A_log: torch.Tensor
    norm: torch.Tensor

    def split_qkv(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key_width = self.dimensions.key_width_per_device
        return self.qkv[:key_width], self.qkv[key_width : 2 * key_width], self.qkv[2 * key_width :]

    def split_conv(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key_width = self.dimensions.key_width_per_device
        return self.conv[:key_width], self.conv[key_width : 2 * key_width], self.conv[2 * key_width :]

    def recurrent_state_shape(self, batch_size: int) -> tuple[int, int, int, int]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        return (
            batch_size,
            self.dimensions.value_heads_per_device,
            self.dimensions.key_head_dim,
            self.dimensions.value_head_dim,
        )

    @property
    def fused_qkvzab(self) -> torch.Tensor:
        """Column-parallel fused input projection in the TT execution order."""

        return torch.cat((self.qkv, self.z, self.a, self.b), dim=0)


@dataclass(frozen=True)
class Qwen38GDNWeights:
    layer_idx: int
    dimensions: Qwen38GDNDimensions
    rms_norm_eps: float
    output_gate: str
    qkv: torch.Tensor
    z: torch.Tensor
    a: torch.Tensor
    b: torch.Tensor
    out: torch.Tensor
    conv: torch.Tensor
    dt_bias: torch.Tensor
    A_log: torch.Tensor
    norm: torch.Tensor

    @classmethod
    def from_checkpoint(cls, checkpoint: Qwen38Checkpoint, layer_idx: int) -> "Qwen38GDNWeights":
        config = checkpoint.config
        if not 0 <= layer_idx < config.num_hidden_layers:
            raise ValueError(f"GDN layer index is outside [0, {config.num_hidden_layers}): {layer_idx}")
        if config.layer_types[layer_idx] != "linear_attention":
            raise ValueError(f"layer {layer_idx} is {config.layer_types[layer_idx]!r}, not linear_attention")
        dimensions = Qwen38GDNDimensions(
            hidden_size=config.hidden_size,
            q_heads=config.gdn_qk_heads,
            value_heads=config.gdn_value_heads,
            key_head_dim=config.gdn_key_head_dim,
            value_head_dim=config.gdn_value_head_dim,
            conv_kernel=config.gdn_conv_kernel,
        )
        prefix = f"model.language_model.layers.{layer_idx}.linear_attn."
        tensors = {
            field: checkpoint.tensor(prefix + checkpoint_name)
            for field, checkpoint_name in {
                "qkv": "in_proj_qkv.weight",
                "z": "in_proj_z.weight",
                "a": "in_proj_a.weight",
                "b": "in_proj_b.weight",
                "out": "out_proj.weight",
                "conv": "conv1d.weight",
                "dt_bias": "dt_bias",
                "A_log": "A_log",
                "norm": "norm.weight",
            }.items()
        }
        expected_shapes = {
            "qkv": (dimensions.qkv_width, dimensions.hidden_size),
            "z": (dimensions.value_width, dimensions.hidden_size),
            "a": (dimensions.value_heads, dimensions.hidden_size),
            "b": (dimensions.value_heads, dimensions.hidden_size),
            "out": (dimensions.hidden_size, dimensions.value_width),
            "conv": (dimensions.qkv_width, 1, dimensions.conv_kernel),
            "dt_bias": (dimensions.value_heads,),
            "A_log": (dimensions.value_heads,),
            "norm": (dimensions.value_head_dim,),
        }
        for name, expected in expected_shapes.items():
            tensor = tensors[name]
            if tuple(tensor.shape) != expected:
                raise ValueError(f"layer {layer_idx} GDN {name} shape must be {expected}, got {tuple(tensor.shape)}")
            if tensor.dtype != torch.bfloat16:
                raise ValueError(f"layer {layer_idx} GDN {name} must be BF16, got {tensor.dtype}")
        return cls(
            layer_idx=layer_idx,
            dimensions=dimensions,
            rms_norm_eps=config.rms_norm_eps,
            output_gate=config.gdn_output_gate,
            **tensors,
        )

    def device_shard(self, device_index: int) -> Qwen38GDNDeviceShard:
        if not 0 <= device_index < self.dimensions.tp_size:
            raise ValueError(f"device_index must be in [0, 4), got {device_index}")
        dims = self.dimensions
        q0 = device_index * dims.key_width_per_device
        q1 = q0 + dims.key_width_per_device
        v0 = device_index * dims.value_width_per_device
        v1 = v0 + dims.value_width_per_device
        qkv = torch.cat(
            (
                self.qkv[q0:q1],
                self.qkv[dims.key_width + q0 : dims.key_width + q1],
                self.qkv[2 * dims.key_width + v0 : 2 * dims.key_width + v1],
            ),
            dim=0,
        ).contiguous()
        conv = torch.cat(
            (
                self.conv[q0:q1],
                self.conv[dims.key_width + q0 : dims.key_width + q1],
                self.conv[2 * dims.key_width + v0 : 2 * dims.key_width + v1],
            ),
            dim=0,
        ).contiguous()
        return Qwen38GDNDeviceShard(
            device_index=device_index,
            dimensions=dims,
            qkv=qkv,
            z=self.z[v0:v1].contiguous(),
            a=self.a[
                device_index * dims.value_heads_per_device : (device_index + 1) * dims.value_heads_per_device
            ].contiguous(),
            b=self.b[
                device_index * dims.value_heads_per_device : (device_index + 1) * dims.value_heads_per_device
            ].contiguous(),
            out=self.out[:, v0:v1].contiguous(),
            conv=conv,
            dt_bias=self.dt_bias[
                device_index * dims.value_heads_per_device : (device_index + 1) * dims.value_heads_per_device
            ].contiguous(),
            A_log=self.A_log[
                device_index * dims.value_heads_per_device : (device_index + 1) * dims.value_heads_per_device
            ].contiguous(),
            norm=self.norm,
        )

    def transformers_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "in_proj_qkv.weight": self.qkv,
            "in_proj_z.weight": self.z,
            "in_proj_a.weight": self.a,
            "in_proj_b.weight": self.b,
            "out_proj.weight": self.out,
            "conv1d.weight": self.conv,
            "dt_bias": self.dt_bias,
            "A_log": self.A_log,
            "norm.weight": self.norm,
        }


@dataclass(frozen=True)
class Qwen38GDNState:
    """Mutable model state represented immutably at the component boundary."""

    conv: torch.Tensor
    recurrent: torch.Tensor


class Qwen38GDN:
    """Full exact CPU GDN forward used to qualify the TTNN implementation."""

    def __init__(self, weights: Qwen38GDNWeights):
        self.weights = weights

    def forward(
        self, hidden_states: torch.Tensor, state: Qwen38GDNState | None = None
    ) -> tuple[torch.Tensor, Qwen38GDNState]:
        dims = self.weights.dimensions
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != dims.hidden_size:
            raise ValueError(f"hidden_states must be [batch, sequence, {dims.hidden_size}]")
        if hidden_states.dtype != torch.bfloat16:
            raise ValueError(f"GDN oracle requires BF16 activations, got {hidden_states.dtype}")
        batch_size = hidden_states.shape[0]
        if state is not None:
            expected_conv = (batch_size, dims.qkv_width, dims.conv_kernel - 1)
            expected_recurrent = (batch_size, dims.value_heads, dims.key_head_dim, dims.value_head_dim)
            if tuple(state.conv.shape) != expected_conv:
                raise ValueError(f"GDN conv state must be {expected_conv}, got {tuple(state.conv.shape)}")
            if tuple(state.recurrent.shape) != expected_recurrent or state.recurrent.dtype != torch.float32:
                raise ValueError("GDN recurrent state shape/dtype does not match the exact FP32 contract")

        qkv = F.linear(hidden_states, self.weights.qkv).transpose(1, 2)
        qkv, next_conv = causal_depthwise_conv1d(
            qkv,
            self.weights.conv.squeeze(1),
            initial_state=None if state is None else state.conv,
            activation="silu",
        )
        qkv = qkv.transpose(1, 2)
        query, key, value = torch.split(qkv, (dims.key_width, dims.key_width, dims.value_width), dim=-1)
        sequence = hidden_states.shape[1]
        query = query.reshape(batch_size, sequence, dims.q_heads, dims.key_head_dim)
        key = key.reshape(batch_size, sequence, dims.q_heads, dims.key_head_dim)
        value = value.reshape(batch_size, sequence, dims.value_heads, dims.value_head_dim)
        query = query.repeat_interleave(dims.qk_repeat_factor, dim=2)
        key = key.repeat_interleave(dims.qk_repeat_factor, dim=2)

        a = F.linear(hidden_states, self.weights.a)
        b = F.linear(hidden_states, self.weights.b)
        beta = b.sigmoid()
        log_decay = -self.weights.A_log.float().exp() * F.softplus(a.float() + self.weights.dt_bias.float())
        output, next_recurrent = gated_delta_recurrent(
            query,
            key,
            value,
            log_decay,
            beta,
            None if state is None else state.recurrent,
            l2_normalize_qk=True,
        )

        z = F.linear(hidden_states, self.weights.z).reshape(batch_size, sequence, dims.value_heads, dims.value_head_dim)
        input_dtype = output.dtype
        normalized = output.float() * torch.rsqrt(
            output.float().square().mean(dim=-1, keepdim=True) + self.weights.rms_norm_eps
        )
        normalized = self.weights.norm * normalized.to(input_dtype)
        if self.weights.output_gate != "sigmoid":
            raise ValueError(f"unsupported pinned GDN output gate: {self.weights.output_gate!r}")
        gated = (normalized * torch.sigmoid(z.float())).to(input_dtype)
        projected = F.linear(gated.flatten(-2), self.weights.out)
        return projected, Qwen38GDNState(conv=next_conv, recurrent=next_recurrent)
