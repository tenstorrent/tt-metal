# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Optional

import torch

import ttnn

from .config import InformerConfig, get_ttnn_dtype
from .ops import apply_dropout, linear, max_pool1d, sinusoidal_position_encoding, slice_to_length
from .state_io import to_float_tensor


class LinearEmbedding:
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        rng: torch.Generator,
        *,
        device,
        dtype: ttnn.DataType,
        memory_config: Optional[ttnn.MemoryConfig] = None,
        weight_dtype: Optional[ttnn.DataType] = None,
        use_bias: bool = True,
    ):
        weight_dtype = weight_dtype or dtype
        self.device = device
        self.weight_torch = torch.randn((d_model, input_dim), generator=rng, dtype=torch.float32) * 0.02
        self.bias_torch = torch.zeros((d_model,), dtype=torch.float32) if use_bias else None
        self.weight = ttnn.from_torch(self.weight_torch, device=device, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT)
        self.bias = (
            ttnn.from_torch(self.bias_torch, device=device, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT)
            if self.bias_torch is not None
            else None
        )
        self.dtype = dtype
        self.memory_config = memory_config

    def load_hf_state_dict(self, state: dict[str, torch.Tensor], *, strict: bool = True) -> dict[str, list[str]]:
        used: set[str] = set()
        missing: list[str] = []

        weight = state.get("weight")
        if weight is None:
            missing.append("weight")
        else:
            used.add("weight")
            self.weight_torch = to_float_tensor(weight)
            self.weight = ttnn.from_torch(
                self.weight_torch,
                device=self.device,
                dtype=self.weight.dtype,
                layout=ttnn.TILE_LAYOUT,
            )

        bias = state.get("bias")
        if self.bias is None:
            if bias is not None:
                used.add("bias")
        else:
            if bias is None:
                missing.append("bias")
            else:
                used.add("bias")
                self.bias_torch = to_float_tensor(bias)
                self.bias = ttnn.from_torch(
                    self.bias_torch,
                    device=self.device,
                    dtype=self.bias.dtype,
                    layout=ttnn.TILE_LAYOUT,
                )

        unexpected = sorted(k for k in state if k not in used)
        if strict and missing:
            raise ValueError(f"Missing linear embedding weights: {missing}")
        return {"missing_keys": missing, "unexpected_keys": unexpected}

    def load_ttnn_state_dict(self, state: dict[str, torch.Tensor], *, strict: bool = True) -> dict[str, list[str]]:
        """Load TTNN-canonical linear embedding tensors."""
        return self.load_hf_state_dict(state, strict=strict)

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return linear(x, self.weight, self.bias, dtype=self.dtype, memory_config=self.memory_config)


# Keep compatibility names used by tests/callers without carrying wrapper classes.
ValueEmbedding = LinearEmbedding
TemporalEmbedding = LinearEmbedding


class PositionalEmbedding:
    def __init__(self, max_len: int, d_model: int, *, device, dtype: ttnn.DataType):
        self.pe = sinusoidal_position_encoding(max_len, d_model, device=device, dtype=dtype)

    def __call__(self, length: int) -> ttnn.Tensor:
        return slice_to_length(self.pe, dim=1, length=length)


class ConvDistillLayer:
    def __init__(
        self,
        d_model: int,
        rng: torch.Generator,
        *,
        device,
        dtype: ttnn.DataType,
        output_memory_config: Optional[ttnn.MemoryConfig] = None,
    ):
        self.kernel_size = 3
        self.stride = 2
        self.padding = 1
        self.dtype = dtype
        self.device = device
        self.output_memory_config = output_memory_config or ttnn.DRAM_MEMORY_CONFIG
        self.prepared_weight = None
        self.prepared_bias = None
        self.prepared_meta = None
        self.weight_torch = torch.randn((d_model, d_model, self.kernel_size), generator=rng, dtype=torch.float32) * 0.02
        self.bias_torch = torch.zeros((d_model,), dtype=torch.float32)
        self.conv_config = ttnn.Conv1dConfig(weights_dtype=dtype, config_tensors_in_dram=True)
        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4
        )

    def set_weights(self, weight: torch.Tensor, bias: torch.Tensor) -> None:
        self.weight_torch = weight.detach().float()
        self.bias_torch = bias.detach().float()
        self.prepared_weight = None
        self.prepared_bias = None
        self.prepared_meta = None

    def load_hf_state_dict(self, state: dict[str, torch.Tensor], *, strict: bool = True) -> dict[str, list[str]]:
        expected = (
            "downConv.weight",
            "downConv.bias",
            "norm.weight",
            "norm.bias",
            "norm.running_mean",
            "norm.running_var",
        )
        missing = [key for key in expected if key not in state]
        if strict and missing:
            raise ValueError(f"Missing distill conv weights: {missing}")
        if missing:
            return {"missing_keys": missing, "unexpected_keys": sorted(k for k in state if k not in expected)}

        eps = 1e-5
        conv_w = to_float_tensor(state["downConv.weight"])
        conv_b = to_float_tensor(state["downConv.bias"])
        bn_w = to_float_tensor(state["norm.weight"])
        bn_b = to_float_tensor(state["norm.bias"])
        bn_mean = to_float_tensor(state["norm.running_mean"])
        bn_var = to_float_tensor(state["norm.running_var"])
        scale = bn_w / torch.sqrt(bn_var + eps)
        fused_w = conv_w * scale.reshape(-1, 1, 1)
        fused_b = (conv_b - bn_mean) * scale + bn_b
        self.set_weights(fused_w, fused_b)
        return {"missing_keys": [], "unexpected_keys": sorted(k for k in state if k not in expected)}

    def prepare_conv_tensors(
        self,
        *,
        batch: int,
        length: int,
        channels: int,
        input_memory_config: ttnn.MemoryConfig,
        input_layout: ttnn.Layout,
        input_dtype: ttnn.DataType,
    ) -> None:
        meta = (batch, length, channels, input_memory_config, input_layout, input_dtype)
        if self.prepared_weight is not None and self.prepared_meta == meta:
            return
        weight_4d = self.weight_torch.unsqueeze(-1)
        weight = ttnn.from_torch(weight_4d, dtype=self.dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
        bias_4d = self.bias_torch.reshape(1, 1, 1, -1)
        bias = ttnn.from_torch(bias_4d, dtype=self.dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
        self.prepared_weight = ttnn.prepare_conv_weights(
            weight_tensor=weight,
            input_memory_config=input_memory_config,
            input_layout=input_layout,
            weights_format="OIHW",
            in_channels=channels,
            out_channels=channels,
            batch_size=batch,
            input_height=length,
            input_width=1,
            kernel_size=[self.kernel_size, 1],
            stride=[1, 1],
            padding=[0, 0],
            dilation=[1, 1],
            has_bias=True,
            groups=1,
            device=self.device,
            input_dtype=input_dtype,
            conv_config=self.conv_config,
            compute_config=self.compute_config,
        )
        self.prepared_bias = ttnn.prepare_conv_bias(
            bias_tensor=bias,
            input_memory_config=input_memory_config,
            input_layout=input_layout,
            in_channels=channels,
            out_channels=channels,
            batch_size=batch,
            input_height=length,
            input_width=1,
            kernel_size=[self.kernel_size, 1],
            stride=[1, 1],
            padding=[0, 0],
            dilation=[1, 1],
            device=self.device,
            input_dtype=input_dtype,
            groups=1,
            conv_config=self.conv_config,
            compute_config=self.compute_config,
        )
        if not ttnn.is_tensor_storage_on_device(self.prepared_weight):
            self.prepared_weight = ttnn.to_device(self.prepared_weight, self.device)
        if not ttnn.is_tensor_storage_on_device(self.prepared_bias):
            self.prepared_bias = ttnn.to_device(self.prepared_bias, self.device)
        self.prepared_meta = meta

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        batch, length, channels = x.shape
        x_rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        if self.padding > 0:
            left = ttnn.slice(x_rm, [0, length - self.padding, 0], [batch, length, channels])
            right = ttnn.slice(x_rm, [0, 0, 0], [batch, self.padding, channels])
            x_rm = ttnn.concat([left, x_rm, right], dim=1)
            length = length + 2 * self.padding
        x_rm = ttnn.reshape(x_rm, (batch, length, 1, channels))
        self.prepare_conv_tensors(
            batch=batch,
            length=length,
            channels=channels,
            input_memory_config=x_rm.memory_config(),
            input_layout=x_rm.layout,
            input_dtype=x_rm.dtype,
        )
        result = ttnn.conv1d(
            input_tensor=x_rm,
            weight_tensor=self.prepared_weight,
            in_channels=channels,
            out_channels=channels,
            device=self.device,
            bias_tensor=self.prepared_bias,
            kernel_size=self.kernel_size,
            stride=1,
            padding=0,
            batch_size=batch,
            input_length=length,
            dtype=self.dtype,
            conv_config=self.conv_config,
            compute_config=self.compute_config,
            groups=1,
            return_output_dim=True,
        )
        if isinstance(result, tuple):
            if len(result) == 3:
                output, out_length, _ = result
            else:
                output, out_length = result
        else:
            output = result
            out_length = output.shape[1]
        output = ttnn.reshape(output, (batch, out_length, channels))
        output = ttnn.to_memory_config(output, self.output_memory_config)
        output = ttnn.to_layout(output, ttnn.TILE_LAYOUT)
        output = ttnn.elu(output)
        return max_pool1d(
            output,
            kernel=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dtype=self.dtype,
        )


class InformerEmbedding:
    def __init__(self, config: InformerConfig, rng: torch.Generator, *, device):
        self.config = config
        self.dtype = get_ttnn_dtype(config.dtype)
        weight_dtype = ttnn.float32 if config.hf_compat else self.dtype
        memory_config = ttnn.L1_MEMORY_CONFIG if config.use_l1 else None
        self.value_embedding = ValueEmbedding(
            config.enc_in,
            config.d_model,
            rng,
            device=device,
            dtype=self.dtype,
            weight_dtype=weight_dtype,
            memory_config=memory_config,
        )
        self.temporal_embedding = TemporalEmbedding(
            config.time_feature_dim,
            config.d_model,
            rng,
            device=device,
            dtype=self.dtype,
            weight_dtype=weight_dtype,
            memory_config=memory_config,
        )
        max_len = config.seq_len + config.pred_len + config.label_len
        self.positional_embedding = PositionalEmbedding(max_len, config.d_model, device=device, dtype=self.dtype)
        self.dropout = config.dropout

    def _forward(self, values: ttnn.Tensor, time_features: ttnn.Tensor) -> ttnn.Tensor:
        pos = self.positional_embedding(values.shape[1])
        x = self.value_embedding(values) + self.temporal_embedding(time_features) + pos
        return apply_dropout(x, self.dropout)

    encoder = _forward
    decoder = _forward
