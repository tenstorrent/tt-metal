# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import torch

import ttnn
from models.demos.rvc.tt_impl.conv1d import TTConv1d
from models.demos.rvc.tt_impl.groupnorm import TTGroupNorm1D
from models.demos.rvc.tt_impl.layernorm import TTLayerNorm
from models.demos.rvc.tt_impl.linear import TTLinear


class MultiheadAttention:
    """TT port of huBERT MultiheadAttention used by TransformerSentenceEncoderLayer."""

    def __init__(self, device: ttnn.MeshDevice, embed_dim: int, num_heads: int, self_attention: bool = False) -> None:
        self.device = device
        self.embed_dim = embed_dim
        self.kdim = embed_dim
        self.vdim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.scaling = self.head_dim**-0.5
        self.self_attention = self_attention

        self.k_proj = TTLinear(device=device, in_features=self.kdim, out_features=embed_dim, dtype=ttnn.bfloat16)
        self.v_proj = TTLinear(device=device, in_features=self.vdim, out_features=embed_dim, dtype=ttnn.bfloat16)
        self.q_proj = TTLinear(device=device, in_features=embed_dim, out_features=embed_dim, dtype=ttnn.bfloat16)
        self.out_proj = TTLinear(device=device, in_features=embed_dim, out_features=embed_dim, dtype=ttnn.bfloat16)

    def load_parameters(self, parameters: dict[str, torch.Tensor], prefix: str = "") -> None:
        self.k_proj.load_parameters(parameters=parameters, key="k_proj", prefix=prefix)
        self.v_proj.load_parameters(parameters=parameters, key="v_proj", prefix=prefix)
        self.q_proj.load_parameters(parameters=parameters, key="q_proj", prefix=prefix)
        self.out_proj.load_parameters(parameters=parameters, key="out_proj", prefix=prefix)

    def __call__(self, query: ttnn.Tensor, key: ttnn.Tensor | None, value: ttnn.Tensor | None) -> ttnn.Tensor:
        # Input shape: T x B x C
        tgt_len, bsz, embed_dim = query.shape
        if embed_dim != self.embed_dim:
            raise ValueError(f"query dim {embed_dim} != {self.embed_dim}")

        if key is None or value is None:
            raise ValueError("key/value must be provided")

        src_len = key.shape[0]

        # TBC -> BTC
        q = ttnn.permute(query, (1, 0, 2))
        k = ttnn.permute(key, (1, 0, 2))
        v = ttnn.permute(value, (1, 0, 2))

        q = self.q_proj(q) * self.scaling
        k = self.k_proj(k)
        v = self.v_proj(v)

        q = ttnn.reshape(q, (bsz, tgt_len, self.num_heads, self.head_dim))
        k = ttnn.reshape(k, (bsz, src_len, self.num_heads, self.head_dim))
        v = ttnn.reshape(v, (bsz, src_len, self.num_heads, self.head_dim))

        q = ttnn.permute(q, (0, 2, 1, 3))
        k = ttnn.permute(k, (0, 2, 1, 3))
        v = ttnn.permute(v, (0, 2, 1, 3))

        q = ttnn.to_layout(q, ttnn.TILE_LAYOUT)
        k = ttnn.to_layout(k, ttnn.TILE_LAYOUT)
        v = ttnn.to_layout(v, ttnn.TILE_LAYOUT)

        attn_weights = ttnn.matmul(q, k, transpose_b=True)
        attn_probs = ttnn.softmax(attn_weights, dim=-1)
        attn = ttnn.matmul(attn_probs, v)

        attn = ttnn.permute(attn, (0, 2, 1, 3))
        attn = ttnn.to_layout(attn, ttnn.ROW_MAJOR_LAYOUT)
        attn = ttnn.reshape(attn, (bsz, tgt_len, self.embed_dim))
        attn = self.out_proj(attn)

        # BTC -> TBC
        return ttnn.permute(attn, (1, 0, 2))

    def deallocate(self) -> None:
        self.k_proj.deallocate()
        self.v_proj.deallocate()
        self.q_proj.deallocate()
        self.out_proj.deallocate()


class TransformerSentenceEncoderLayer:
    def __init__(
        self,
        device: ttnn.MeshDevice,
        embed_dim: int = 768,
        ffn_embed_dim: int = 3072,
        attention_heads: int = 8,
        activation_fn: str = "relu",
        layer_norm_first: bool = False,
    ) -> None:
        self.device = device
        self.embedding_dim = embed_dim
        self.activation_fn = activation_fn
        self.self_attn = MultiheadAttention(
            device=device, embed_dim=embed_dim, num_heads=attention_heads, self_attention=True
        )
        self.layer_norm_first = layer_norm_first
        self.fc1 = TTLinear(device=device, in_features=embed_dim, out_features=ffn_embed_dim, dtype=ttnn.bfloat16)
        self.fc2 = TTLinear(device=device, in_features=ffn_embed_dim, out_features=embed_dim, dtype=ttnn.bfloat16)

        self.self_attn_layer_norm_weight: ttnn.Tensor | None = None
        self.self_attn_layer_norm_bias: ttnn.Tensor | None = None
        self.final_layer_norm_weight: ttnn.Tensor | None = None
        self.final_layer_norm_bias: ttnn.Tensor | None = None

    def load_parameters(self, parameters: dict[str, torch.Tensor], prefix: str = "") -> None:
        self.self_attn.load_parameters(parameters, prefix=f"{prefix}self_attn.")
        self.fc1.load_parameters(parameters=parameters, key="fc1", prefix=prefix)
        self.fc2.load_parameters(parameters=parameters, key="fc2", prefix=prefix)

        for key_name, attr_w, attr_b in [
            ("self_attn_layer_norm", "self_attn_layer_norm_weight", "self_attn_layer_norm_bias"),
            ("final_layer_norm", "final_layer_norm_weight", "final_layer_norm_bias"),
        ]:
            w_key = f"{prefix}{key_name}.weight" if prefix else f"{key_name}.weight"
            b_key = f"{prefix}{key_name}.bias" if prefix else f"{key_name}.bias"
            if w_key not in parameters:
                raise KeyError(f"Missing required parameter: {w_key}")
            if b_key not in parameters:
                raise KeyError(f"Missing required parameter: {b_key}")
            w = ttnn.from_torch(
                parameters[w_key].reshape(1, 1, -1),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
            )
            b = ttnn.from_torch(
                parameters[b_key].reshape(1, 1, -1),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
            )
            setattr(self, attr_w, w)
            setattr(self, attr_b, b)

    def _apply_activation(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        if self.activation_fn == "relu":
            return ttnn.relu(x)
        if self.activation_fn == "gelu":
            return ttnn.gelu(x)
        if self.activation_fn == "tanh":
            return ttnn.tanh(x)
        if self.activation_fn == "linear":
            return x
        raise RuntimeError(f"Unsupported activation_fn in TT port: {self.activation_fn}")

    def _layer_norm(self, x: ttnn.Tensor, weight: ttnn.Tensor | None, bias: ttnn.Tensor | None) -> ttnn.Tensor:
        if weight is None or bias is None:
            raise ValueError("Layer norm parameters are not loaded.")
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        weight = ttnn.to_layout(weight, ttnn.TILE_LAYOUT)
        bias = ttnn.to_layout(bias, ttnn.TILE_LAYOUT)
        return ttnn.layer_norm(x, weight=weight, bias=bias, epsilon=1e-5)

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        residual = x

        if self.layer_norm_first:
            x = self._layer_norm(x, self.self_attn_layer_norm_weight, self.self_attn_layer_norm_bias)
            attn_out = self.self_attn(query=x, key=x, value=x)
            x = residual + attn_out

            residual = x
            x = self._layer_norm(x, self.final_layer_norm_weight, self.final_layer_norm_bias)
            x = self._apply_activation(self.fc1(x))
            x = self.fc2(x)
            x = residual + x
        else:
            x = self.self_attn(query=x, key=x, value=x)

            x = residual + x

            x = self._layer_norm(x, self.self_attn_layer_norm_weight, self.self_attn_layer_norm_bias)

            residual = x
            x = self._apply_activation(self.fc1(x))

            x = self.fc2(x)
            x = residual + x
            x = self._layer_norm(x, self.final_layer_norm_weight, self.final_layer_norm_bias)

        return x

    def deallocate(self) -> None:
        self.self_attn.deallocate()
        self.fc1.deallocate()
        self.fc2.deallocate()
        ttnn.deallocate(self.self_attn_layer_norm_weight)
        ttnn.deallocate(self.self_attn_layer_norm_bias)
        ttnn.deallocate(self.final_layer_norm_weight)
        ttnn.deallocate(self.final_layer_norm_bias)


class ConvFeatureExtractionModel:
    def __init__(
        self,
        device: ttnn.MeshDevice,
        conv_layers: list[tuple[int, int, int]],
        mode: str = "default",
        conv_bias: bool = False,
    ) -> None:
        if mode not in {"default", "layer_norm"}:
            raise ValueError("mode must be 'default' or 'layer_norm'")
        self.device = device
        self.mode = mode
        self.conv_layers_cfg = conv_layers
        self.conv_layers: list[TTConv1d] = []

        in_d = 1
        for i, cl in enumerate(conv_layers):
            if len(cl) != 3:
                raise ValueError(f"invalid conv definition: {cl}")
            dim, k, stride = cl
            self.conv_layers.append(
                TTConv1d(
                    device=device,
                    in_channels=in_d,
                    out_channels=dim,
                    kernel_size=k,
                    stride=stride,
                    padding=0,
                    dtype=ttnn.bfloat16,
                )
            )
            in_d = dim

        self.ln_weights: list[ttnn.Tensor | None] = [None for _ in conv_layers]
        self.ln_biases: list[ttnn.Tensor | None] = [None for _ in conv_layers]
        self.group_norms: list[TTGroupNorm1D | None] = [None for _ in conv_layers]
        if self.mode == "default" and conv_layers:
            self.group_norms[0] = TTGroupNorm1D(
                device=device, num_channels=conv_layers[0][0], num_groups=conv_layers[0][0]
            )

    def load_parameters(self, parameters: dict[str, torch.Tensor], prefix: str = "") -> None:
        for i, conv in enumerate(self.conv_layers):
            conv.load_parameters(parameters=parameters, key=f"conv_layers.{i}.0", prefix=prefix)

            if self.mode == "layer_norm":
                w_key = f"{prefix}conv_layers.{i}.1.1.weight" if prefix else f"conv_layers.{i}.1.1.weight"
                b_key = f"{prefix}conv_layers.{i}.1.1.bias" if prefix else f"conv_layers.{i}.1.1.bias"
                if w_key not in parameters:
                    raise KeyError(f"Missing required parameter: {w_key}")
                if b_key not in parameters:
                    raise KeyError(f"Missing required parameter: {b_key}")
                ln_w = ttnn.from_torch(
                    parameters[w_key].reshape(1, 1, -1),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=self.device,
                )
                ln_b = ttnn.from_torch(
                    parameters[b_key].reshape(1, 1, -1),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=self.device,
                )
                self.ln_weights[i] = ln_w
                self.ln_biases[i] = ln_b
            elif self.mode == "default" and i == 0:
                group_norm = self.group_norms[i]
                if group_norm is None:
                    raise ValueError("GroupNorm module is not initialized.")
                group_norm.load_parameters(parameters=parameters, key=f"conv_layers.{i}.1", prefix=prefix)

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        # Torch path takes BxT and does unsqueeze(1). TT conv expects BxLxC.
        batch_size = x.shape[0]
        current_length = x.shape[1]

        for i, conv in enumerate(self.conv_layers):
            x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
            x = conv(x)
            out_channels, kernel_size, stride = self.conv_layers_cfg[i]
            current_length = ((current_length - kernel_size) // stride) + 1
            x = ttnn.reshape(x, (batch_size, current_length, out_channels))

            if self.mode == "layer_norm":
                ln_w = self.ln_weights[i]
                ln_b = self.ln_biases[i]
                if ln_w is None or ln_b is None:
                    raise ValueError("LayerNorm parameters are not loaded.")
                x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
                x = ttnn.layer_norm(
                    x,
                    weight=ttnn.to_layout(ln_w, ttnn.TILE_LAYOUT),
                    bias=ttnn.to_layout(ln_b, ttnn.TILE_LAYOUT),
                    epsilon=1e-5,
                )
                x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
            elif self.mode == "default" and i == 0:
                group_norm = self.group_norms[i]
                if group_norm is None:
                    raise ValueError("GroupNorm parameters are not loaded.")
                # x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
                x = group_norm.gp_slice(x)
                x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

            x = ttnn.gelu(x)
        # Match Torch output shape: B x C x T
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        return x

    def deallocate(self) -> None:
        for conv in self.conv_layers:
            conv.deallocate()
        for group_norm in self.group_norms:
            if group_norm is not None:
                group_norm.deallocate()


class TTPositionalConvEmbedding:
    def __init__(self, device: ttnn.MeshDevice, embed_dim: int, kernel_size: int, groups: int) -> None:
        self.device = device
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        self.conv = TTConv1d(
            device=device,
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=groups,
            dtype=ttnn.bfloat16,
        )
        self.remove = 1 if kernel_size % 2 == 0 else 0

    def load_parameters(self, parameters: dict[str, torch.Tensor], prefix: str = "") -> None:
        self.conv.load_parameters(parameters=parameters, key="0", prefix=prefix)

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        batch_size = x.shape[0]
        input_length = x.shape[1]
        output_length = input_length + 2 * (self.kernel_size // 2) - self.kernel_size + 1
        x = self.conv(x)
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.reshape(x, (batch_size, output_length, self.embed_dim))
        if self.remove > 0:
            x = ttnn.slice(x, (0, 0, 0), (batch_size, output_length - self.remove, self.embed_dim))
        return ttnn.gelu(x)

    def deallocate(self) -> None:
        self.conv.deallocate()


class FeedForwardModule:
    """TT port of huBERT FeedForwardModule (conformer positionwise FFN)."""

    def __init__(
        self,
        device: ttnn.MeshDevice,
        input_feat: int,
        hidden_units: int,
        activation_fn: str = "swish",
        bias: bool = True,
    ) -> None:
        self.device = device
        self.input_feat = input_feat
        self.hidden_units = hidden_units
        self.activation_fn = activation_fn
        self.use_bias = bias

        self.layer_norm = TTLayerNorm(device=device, normalized_shape=input_feat, eps=1e-5, dtype=ttnn.bfloat16)
        self.w_1 = TTLinear(device=device, in_features=input_feat, out_features=hidden_units, dtype=ttnn.bfloat16)
        self.w_2 = TTLinear(device=device, in_features=hidden_units, out_features=input_feat, dtype=ttnn.bfloat16)

    def load_parameters(self, parameters: dict[str, torch.Tensor], prefix: str = "") -> None:
        self.layer_norm.load_parameters(parameters=parameters, key="layer_norm", prefix=prefix)
        self.w_1.load_parameters(parameters=parameters, key="w_1", prefix=prefix)
        self.w_2.load_parameters(parameters=parameters, key="w_2", prefix=prefix)

    def _apply_activation(self, x: ttnn.Tensor) -> ttnn.Tensor:
        if self.activation_fn == "relu":
            return ttnn.relu(x)
        if self.activation_fn == "gelu":
            return ttnn.gelu(x)
        if self.activation_fn == "tanh":
            return ttnn.tanh(x)
        if self.activation_fn == "linear":
            return x
        if self.activation_fn == "swish":
            return ttnn.silu(x)
        raise RuntimeError(f"Unsupported activation_fn in TT FeedForwardModule: {self.activation_fn}")

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        # x: T x B x C
        x = self.layer_norm(x)
        x = self.w_1(x)
        x = self._apply_activation(x)
        x = self.w_2(x)
        return x

    def deallocate(self) -> None:
        self.layer_norm.deallocate()
        self.w_1.deallocate()
        self.w_2.deallocate()


class ConvolutionModule:
    """TT port of huBERT ConvolutionModule used in conformer blocks."""

    def __init__(
        self,
        device: ttnn.MeshDevice,
        embed_dim: int,
        channels: int,
        depthwise_kernel_size: int,
        activation_fn: str = "swish",
        bias: bool = False,
    ) -> None:
        if (depthwise_kernel_size - 1) % 2 != 0:
            raise ValueError("depthwise_kernel_size should be odd for SAME padding")

        self.device = device
        self.embed_dim = embed_dim
        self.channels = channels
        self.depthwise_kernel_size = depthwise_kernel_size
        self.activation_fn = activation_fn
        self.use_bias = bias
        self.batch_norm_eps = 1e-5

        self.layer_norm = TTLayerNorm(device=device, normalized_shape=embed_dim, eps=1e-5, dtype=ttnn.bfloat16)
        self.pointwise_conv1 = TTConv1d(
            device=device,
            in_channels=embed_dim,
            out_channels=2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dtype=ttnn.bfloat16,
        )
        self.depthwise_conv = TTConv1d(
            device=device,
            in_channels=channels,
            out_channels=channels,
            kernel_size=depthwise_kernel_size,
            stride=1,
            padding=(depthwise_kernel_size - 1) // 2,
            groups=channels,
            dtype=ttnn.bfloat16,
        )
        self.pointwise_conv2 = TTConv1d(
            device=device,
            in_channels=channels,
            out_channels=embed_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            dtype=ttnn.bfloat16,
        )

        self.batch_norm_weight: ttnn.Tensor | None = None
        self.batch_norm_bias: ttnn.Tensor | None = None
        self.batch_norm_running_mean: ttnn.Tensor | None = None
        self.batch_norm_running_inv_std: ttnn.Tensor | None = None

    def _conv_output_to_nlc(
        self, x: ttnn.Tensor, batch_size: int, output_length: int, out_channels: int
    ) -> ttnn.Tensor:
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        return ttnn.reshape(x, (batch_size, output_length, out_channels))

    def _conv_output_length(
        self, input_length: int, kernel_size: int, stride: int, padding: int, dilation: int = 1
    ) -> int:
        return ((input_length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1

    def load_parameters(self, parameters: dict[str, torch.Tensor], prefix: str = "") -> None:
        self.layer_norm.load_parameters(parameters=parameters, key="layer_norm", prefix=prefix)
        self.pointwise_conv1.load_parameters(parameters=parameters, key="pointwise_conv1", prefix=prefix)
        self.depthwise_conv.load_parameters(parameters=parameters, key="depthwise_conv", prefix=prefix)
        self.pointwise_conv2.load_parameters(parameters=parameters, key="pointwise_conv2", prefix=prefix)

        bn_weight_key = f"{prefix}batch_norm.weight" if prefix else "batch_norm.weight"
        bn_bias_key = f"{prefix}batch_norm.bias" if prefix else "batch_norm.bias"
        bn_mean_key = f"{prefix}batch_norm.running_mean" if prefix else "batch_norm.running_mean"
        bn_var_key = f"{prefix}batch_norm.running_var" if prefix else "batch_norm.running_var"

        for required_key in [bn_weight_key, bn_bias_key, bn_mean_key, bn_var_key]:
            if required_key not in parameters:
                raise KeyError(f"Missing required parameter: {required_key}")

        running_inv_std = (parameters[bn_var_key] + self.batch_norm_eps).rsqrt()
        self.batch_norm_weight = ttnn.from_torch(
            parameters[bn_weight_key].reshape(1, 1, self.channels),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
        )
        self.batch_norm_bias = ttnn.from_torch(
            parameters[bn_bias_key].reshape(1, 1, self.channels),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
        )
        self.batch_norm_running_mean = ttnn.from_torch(
            parameters[bn_mean_key].reshape(1, 1, self.channels),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
        )
        self.batch_norm_running_inv_std = ttnn.from_torch(
            running_inv_std.reshape(1, 1, self.channels),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
        )

    def _apply_activation(self, x: ttnn.Tensor) -> ttnn.Tensor:
        if self.activation_fn == "relu":
            return ttnn.relu(x)
        if self.activation_fn == "gelu":
            return ttnn.gelu(x)
        if self.activation_fn == "tanh":
            return ttnn.tanh(x)
        if self.activation_fn == "linear":
            return x
        if self.activation_fn == "swish":
            return ttnn.silu(x)
        raise RuntimeError(f"Unsupported activation_fn in TT ConvolutionModule: {self.activation_fn}")

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        # x: B x T x C
        batch_size = x.shape[0]
        input_length = x.shape[1]
        x = self.layer_norm(x)

        # pointwise conv1: C -> 2*channels
        x = self.pointwise_conv1(x)
        x = self._conv_output_to_nlc(
            x=x,
            batch_size=batch_size,
            output_length=self._conv_output_length(input_length, kernel_size=1, stride=1, padding=0),
            out_channels=2 * self.channels,
        )

        # GLU on channel dim
        xa = ttnn.slice(x, (0, 0, 0), (x.shape[0], x.shape[1], self.channels))
        xb = ttnn.slice(x, (0, 0, self.channels), (x.shape[0], x.shape[1], 2 * self.channels))
        x = xa * ttnn.sigmoid(xb)

        # depthwise conv
        x = self.depthwise_conv(x)
        x = self._conv_output_to_nlc(
            x=x,
            batch_size=batch_size,
            output_length=self._conv_output_length(
                input_length,
                kernel_size=self.depthwise_kernel_size,
                stride=1,
                padding=(self.depthwise_kernel_size - 1) // 2,
            ),
            out_channels=self.channels,
        )

        # eval-mode BatchNorm1d over channels
        if (
            self.batch_norm_weight is None
            or self.batch_norm_bias is None
            or self.batch_norm_running_mean is None
            or self.batch_norm_running_inv_std is None
        ):
            raise ValueError("BatchNorm parameters are not loaded.")
        x = (x - self.batch_norm_running_mean) * self.batch_norm_running_inv_std
        x = (x * self.batch_norm_weight) + self.batch_norm_bias
        x = self._apply_activation(x)

        # pointwise conv2: channels -> embed_dim
        x = self.pointwise_conv2(x)
        x = self._conv_output_to_nlc(
            x=x,
            batch_size=batch_size,
            output_length=input_length,
            out_channels=self.embed_dim,
        )
        return x

    def deallocate(self) -> None:
        self.layer_norm.deallocate()
        self.pointwise_conv1.deallocate()
        self.depthwise_conv.deallocate()
        self.pointwise_conv2.deallocate()
        ttnn.deallocate(self.batch_norm_weight)
        ttnn.deallocate(self.batch_norm_bias)
        ttnn.deallocate(self.batch_norm_running_mean)
        ttnn.deallocate(self.batch_norm_running_inv_std)


class TransformerEncoder:
    def __init__(self, device: ttnn.MeshDevice, args: dict) -> None:
        self.device = device
        self.embedding_dim = args["encoder_embed_dim"]
        self.required_seq_len_multiple = args.get("required_seq_len_multiple", 2)
        self.pos_conv = TTPositionalConvEmbedding(
            device=device,
            embed_dim=self.embedding_dim,
            kernel_size=args["conv_pos"],
            groups=args["conv_pos_groups"],
        )
        self.layers = [self.build_encoder_layer(args) for _ in range(args["encoder_layers"])]
        self.layer_norm_first = args["layer_norm_first"]
        self.layer_norm = TTLayerNorm(device=device, normalized_shape=self.embedding_dim, eps=1e-5, dtype=ttnn.bfloat16)

    def build_encoder_layer(self, args: dict):
        return TransformerSentenceEncoderLayer(
            device=self.device,
            embed_dim=self.embedding_dim,
            ffn_embed_dim=args["encoder_ffn_embed_dim"],
            attention_heads=args["encoder_attention_heads"],
            activation_fn=args["activation_fn"],
            layer_norm_first=args["layer_norm_first"],
        )

    def load_parameters(self, parameters: dict[str, torch.Tensor], prefix: str = "") -> None:
        self.pos_conv.load_parameters(parameters=parameters, prefix=f"{prefix}pos_conv.")
        self.layer_norm.load_parameters(parameters=parameters, key="layer_norm", prefix=prefix)
        for i, layer in enumerate(self.layers):
            layer.load_parameters(parameters=parameters, prefix=f"{prefix}layers.{i}.")

    def __call__(self, x: ttnn.Tensor, tgt_layer: int) -> ttnn.Tensor:
        x = x + self.pos_conv(x)

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        pad_length = (
            self.required_seq_len_multiple - (x.shape[1] % self.required_seq_len_multiple)
        ) % self.required_seq_len_multiple
        if pad_length > 0:
            pad = ttnn.zeros(
                (x.shape[0], pad_length, x.shape[2]),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
            )
            x = ttnn.concat([x, pad], dim=1)

        x = ttnn.permute(x, (1, 0, 2))

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == tgt_layer:
                break

        x = ttnn.permute(x, (1, 0, 2))

        if pad_length > 0:
            x = ttnn.slice(x, (0, 0, 0), (x.shape[0], x.shape[1] - pad_length, x.shape[2]))
        return x

    def deallocate(self) -> None:
        self.pos_conv.deallocate()
        self.layer_norm.deallocate()
        for layer in self.layers:
            layer.deallocate()


class HubertModel:
    def __init__(self, device: ttnn.MeshDevice, cfg: dict, task_cfg) -> None:
        self.device = device
        self.cfg = cfg
        self._is_generation_fast = False

        feature_enc_layers = eval(cfg["conv_feature_layers"])  # noqa: S307
        self.embed = feature_enc_layers[-1][0]

        self.feature_extractor = ConvFeatureExtractionModel(
            device=device,
            conv_layers=feature_enc_layers,
            mode=cfg["extractor_mode"],
            conv_bias=cfg["conv_bias"],
        )
        feature_ds_rate = math.prod([stride for _, _, stride in feature_enc_layers])
        self.feat2tar_ratio = cfg["label_rate"] * feature_ds_rate / task_cfg.sample_rate

        self.post_extract_proj = None
        if self.embed != cfg["encoder_embed_dim"]:
            self.post_extract_proj = TTLinear(
                device=device,
                in_features=self.embed,
                out_features=cfg["encoder_embed_dim"],
                dtype=ttnn.bfloat16,
            )

        self.feature_grad_mult = cfg["feature_grad_mult"]
        self.logit_temp = cfg["logit_temp"]

        final_dim = cfg["final_dim"] if cfg["final_dim"] > 0 else cfg["encoder_embed_dim"]
        self.encoder = TransformerEncoder(device=device, args=cfg)
        self.layer_norm = TTLayerNorm(device=device, normalized_shape=self.embed, eps=1e-5, dtype=ttnn.bfloat16)
        self.untie_final_proj = cfg["untie_final_proj"]
        self.final_proj_linear = TTLinear(
            device=device,
            in_features=cfg["encoder_embed_dim"],
            out_features=final_dim,
            dtype=ttnn.bfloat16,
        )

    @classmethod
    def build_model(cls, cfg, task, device: ttnn.MeshDevice):
        return cls(device=device, cfg=cfg, task_cfg=task.cfg)

    def load_parameters(self, parameters: dict[str, torch.Tensor]) -> None:
        self.feature_extractor.load_parameters(parameters=parameters, prefix="feature_extractor.")
        self.layer_norm.load_parameters(parameters=parameters, key="layer_norm")
        if self.post_extract_proj is not None:
            self.post_extract_proj.load_parameters(parameters=parameters, key="post_extract_proj")
        self.encoder.load_parameters(parameters=parameters, prefix="encoder.")
        self.final_proj_linear.load_parameters(parameters=parameters, key="final_proj")

    def __call__(self, source: ttnn.Tensor, output_layer: int) -> ttnn.Tensor:
        x1 = self.feature_extractor(source)

        x2 = self.layer_norm(x1)

        if self.post_extract_proj is not None:
            x2 = self.post_extract_proj(x2)
        x4 = self.encoder(x2, tgt_layer=output_layer - 1)
        return x4

    def final_proj(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return self.final_proj_linear(x)

    def eval(self):
        return self

    def float(self):
        return self

    def to(self, *_args, **_kwargs):
        return self

    def deallocate(self) -> None:
        self.feature_extractor.deallocate()
        self.layer_norm.deallocate()
        if self.post_extract_proj is not None:
            self.post_extract_proj.deallocate()
        self.encoder.deallocate()
        self.final_proj_linear.deallocate()
