# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import torch

import ttnn
from models.demos.rvc.tt_impl.conv1d import Conv1d
from models.demos.rvc.tt_impl.convtranspose1d import ConvTranspose1d
from models.demos.rvc.tt_impl.layernorm import LayerNorm
from models.demos.rvc.tt_impl.linear import Linear

LRELU_SLOPE = 0.1


def _mesh_mapper_and_composer(device):
    if device.get_num_devices() > 1:
        return ttnn.ShardTensorToMesh(device, dim=0), ttnn.ConcatMeshToTensor(device, dim=0)
    return None, None


def ttnn_randn_fallback(shape, dtype, device):
    # Fallback random generator using PyTorch, since TTNN's random generation is not available in the current version.
    mesh_mapper, _ = _mesh_mapper_and_composer(device)
    return ttnn.from_torch(
        torch.randn(shape, dtype=torch.float32),
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        mesh_mapper=mesh_mapper,
    )


def ttnn_cumsum_fallback(x: ttnn.Tensor, dim: int) -> ttnn.Tensor:
    # Fallback implementation of cumsum using to_host, torch.cumsum, and from_torch.
    mesh_mapper, mesh_composer = _mesh_mapper_and_composer(x.device())
    x_torch = ttnn.to_torch(x, mesh_composer=mesh_composer)
    cumsum_torch = torch.cumsum(x_torch, dim=dim)
    cumsum = ttnn.from_torch(
        cumsum_torch,
        dtype=x.dtype,
        layout=x.layout,
        device=x.device(),
        memory_config=x.memory_config(),
        mesh_mapper=mesh_mapper,
    )
    return cumsum


def _interpolate_1d(
    x: ttnn.Tensor,
    scale_factor: int | float,
    mode: str = "nearest",
) -> ttnn.Tensor:
    # 1D upsample for [N, L, C] via 2D NHWC upsample with height fixed to 1.
    if mode not in ("nearest", "linear"):
        raise ValueError(f"Unsupported 1D interpolate mode: {mode}")
    upsample_mode = "nearest" if mode == "nearest" else "bilinear"
    x_nhwc = ttnn.unsqueeze(ttnn.unsqueeze(x, dim=1), dim=3)
    y_nhwc = ttnn.upsample(
        x_nhwc,
        [1, scale_factor],
        mode=upsample_mode,
    )
    return ttnn.squeeze(y_nhwc, dim=1)


def _flip_last_dim_ttnn(x: ttnn.Tensor) -> ttnn.Tensor:
    if x.layout != ttnn.TILE_LAYOUT:
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

    reverse_index = ttnn.arange(
        start=x.shape[-1] - 1,
        end=-1,
        step=-1,
        dtype=ttnn.int32,
        device=x.device(),
        layout=ttnn.TILE_LAYOUT,
    )
    reverse_index = ttnn.reshape(reverse_index, shape=(1,) * (len(x.shape) - 1) + (x.shape[-1],))
    reverse_index = ttnn.expand(reverse_index, tuple(x.shape))
    reverse_index = ttnn.typecast(reverse_index, ttnn.uint32)
    return ttnn.gather(x, dim=-1, index=reverse_index)


def ttnn_gather_fallback(x: ttnn.Tensor, dim: int, index: ttnn.Tensor, device) -> ttnn.Tensor:
    # Fallback implementation of gather using to_host, torch.gather, and from_torch.
    # needed since ttnn.gather is 4-8x slower than this fallback version
    mesh_mapper, mesh_composer = _mesh_mapper_and_composer(device)
    x = ttnn.reallocate(x)
    x_torch = ttnn.to_torch(x, mesh_composer=mesh_composer)
    index_torch = ttnn.to_torch(index, mesh_composer=mesh_composer).to(torch.int64)
    gathered_torch = torch.gather(x_torch, dim=dim, index=index_torch)
    gathered = ttnn.from_torch(gathered_torch, dtype=x.dtype, layout=x.layout, device=device, mesh_mapper=mesh_mapper)
    return gathered


class MultiHeadAttention:
    def __init__(
        self,
        device: ttnn.MeshDevice,
        in_features: int,
        out_features: int,
        num_heads: int,
        window_size: int | None = None,
    ) -> None:
        if in_features % num_heads != 0:
            raise ValueError("in_features must be divisible by num_heads")

        self.device = device
        self.num_heads = num_heads
        self.window_size = window_size
        self.features_per_head = in_features // num_heads

        self.linear_q = Linear(
            device=device,
            in_features=in_features,
            out_features=in_features,
        )
        self.linear_k = Linear(
            device=device,
            in_features=in_features,
            out_features=in_features,
        )
        self.linear_v = Linear(
            device=device,
            in_features=in_features,
            out_features=in_features,
        )
        self.linear_o = Linear(
            device=device,
            in_features=in_features,
            out_features=out_features,
        )
        self.emb_rel_k: ttnn.Tensor | None = None
        self.emb_rel_v: ttnn.Tensor | None = None
        self.relative_position_cache: dict[int : ttnn.Tensor] = {}
        self.index_cache: dict[int : ttnn.Tensor] = {}

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], module_prefix: str | None = None) -> None:
        if module_prefix is None:
            module_prefix = ""
        self.linear_q.load_state_dict(state_dict=state_dict, key="linear_q", module_prefix=module_prefix)
        self.linear_k.load_state_dict(state_dict=state_dict, key="linear_k", module_prefix=module_prefix)
        self.linear_v.load_state_dict(state_dict=state_dict, key="linear_v", module_prefix=module_prefix)
        self.linear_o.load_state_dict(state_dict=state_dict, key="linear_o", module_prefix=module_prefix)
        if self.window_size is not None:
            key_emb_rel_k = f"{module_prefix}emb_rel_k"
            key_emb_rel_v = f"{module_prefix}emb_rel_v"

            emb_rel_k = state_dict[key_emb_rel_k].reshape(1, 1, -1, self.features_per_head)
            emb_rel_v = state_dict[key_emb_rel_v].reshape(1, 1, -1, self.features_per_head)
            self.emb_rel_k = ttnn.from_torch(
                emb_rel_k,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
            )
            self.emb_rel_v = ttnn.from_torch(
                emb_rel_v,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
            )

    def __call__(self, x: ttnn.Tensor, context: ttnn.Tensor) -> ttnn.Tensor:
        q = self._reshape_to_heads(self.linear_q(x))
        k = self._reshape_to_heads(self.linear_k(context))
        v = self._reshape_to_heads(self.linear_v(context))
        q_scaled = ttnn.mul(q, 1.0 / math.sqrt(self.features_per_head), output_tensor=q)
        scores = ttnn.matmul(q_scaled, k, transpose_b=True)
        _, _, target_length, source_length = scores.shape
        if self.window_size is not None:
            if source_length != target_length:
                raise ValueError("Relative attention is only available for self-attention.")
            if self.emb_rel_k is None or self.emb_rel_v is None:
                raise ValueError("Relative embeddings are not loaded.")

            key_relative_embeddings = self._get_relative_embeddings(self.emb_rel_k, source_length)
            rel_logits = ttnn.matmul(q_scaled, key_relative_embeddings, transpose_b=True)
            scores_local = self._relative_to_absolute_position(rel_logits)
            scores = ttnn.add(scores, scores_local, output_tensor=scores)
        scores = ttnn.to_memory_config(scores, ttnn.L1_MEMORY_CONFIG)
        attn_weights = ttnn.softmax_in_place(scores, dim=-1)
        out = ttnn.matmul(attn_weights, v, memory_config=ttnn.L1_MEMORY_CONFIG)
        if self.window_size is not None:
            assert self.emb_rel_v is not None
            relative_weights = self._absolute_to_relative_position(attn_weights)
            value_relative_embeddings = self._get_relative_embeddings(self.emb_rel_v, source_length)
            out = ttnn.add(out, ttnn.matmul(relative_weights, value_relative_embeddings), output_tensor=out)
        out = ttnn.transformer.concatenate_heads(out)

        out = self.linear_o(out)
        return out

    def _reshape_to_heads(self, x: ttnn.Tensor) -> ttnn.Tensor:
        batch_size, length, channels = x.shape
        x = ttnn.reshape(x, (batch_size, length, self.num_heads, self.features_per_head))
        x_heads = ttnn.permute(x, (0, 2, 1, 3))
        return x_heads

    def _get_relative_embeddings(self, relative_embeddings: ttnn.Tensor, length: int) -> ttnn.Tensor:
        if self.window_size is None:
            raise ValueError("window_size must be set for relative attention.")
        pad_length: int = max(length - (self.window_size + 1), 0)
        slice_start_position = max((self.window_size + 1) - length, 0)
        slice_end_position = slice_start_position + 2 * length - 1

        embeddings = relative_embeddings
        if pad_length > 0:
            embeddings = ttnn.pad(
                embeddings,
                padding=((0, 0), (0, 0), (pad_length, pad_length), (0, 0)),
                value=0.0,
            )
        embeddings = ttnn.slice(
            embeddings,
            (0, 0, slice_start_position, 0),
            (1, 1, slice_end_position, self.features_per_head),
        )
        return ttnn.to_layout(embeddings, ttnn.TILE_LAYOUT)

    def _get_relative_position_index(self, length: int) -> ttnn.Tensor:
        if length in self.index_cache:
            return self.index_cache[length]
        idx_row = ttnn.unsqueeze(ttnn.arange(start=0, end=length, dtype=ttnn.uint32, device=self.device), dim=1)
        idx_col = ttnn.unsqueeze(
            ttnn.arange(start=length - 1, end=2 * length - 1, dtype=ttnn.uint32, device=self.device), dim=0
        )
        relative_position_index = idx_col - idx_row
        relative_position_index = ttnn.expand(
            ttnn.reshape(relative_position_index, shape=(1, 1, length, length)), (1, 1, length, length)
        )
        self.index_cache[length] = relative_position_index
        return relative_position_index

    def _relative_to_absolute_position(self, x: ttnn.Tensor) -> ttnn.Tensor:
        batch, heads, length, _ = x.shape
        relative_position_index = self._get_relative_position_index(length)
        out = ttnn_gather_fallback(x, dim=3, index=relative_position_index, device=self.device)
        return out

    def _absolute_to_relative_position(self, x: ttnn.Tensor) -> ttnn.Tensor:
        batch, heads, length, _ = x.shape
        relative_position_index = self._get_relative_position_index(length)
        if length in self.relative_position_cache:
            out = self.relative_position_cache[length]
        else:
            out = ttnn.zeros(
                (batch, heads, length, 2 * length - 1),
                dtype=x.dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            )
            self.relative_position_cache[length] = out
        out = ttnn.scatter(out, dim=3, index=relative_position_index, src=x)
        return out


class FFN:
    def __init__(
        self,
        device: ttnn.MeshDevice,
        in_channels: int,
        out_channels: int,
        filter_channels: int,
        kernel_size: int,
    ) -> None:
        self.conv_1 = Conv1d(
            device=device,
            in_channels=in_channels,
            out_channels=filter_channels,
            kernel_size=kernel_size,
            padding="same",
            activation="relu",
        )
        self.conv_2 = Conv1d(
            device=device,
            in_channels=filter_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding="same",
            deallocate_input=True,
        )

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], module_prefix: str | None = None) -> None:
        self.conv_1.load_state_dict(state_dict=state_dict, key="conv_1", module_prefix=module_prefix)
        self.conv_2.load_state_dict(state_dict=state_dict, key="conv_2", module_prefix=module_prefix)

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x = self.conv_1(x)
        out = self.conv_2(x)
        return out


class WN:
    def __init__(
        self,
        device: ttnn.MeshDevice,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        num_layers: int,
        gin_channels: int = 0,
    ) -> None:
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd")
        self.device = device
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.gin_channels = gin_channels
        self.in_layers: list[Conv1d] = []
        self.res_skip_layers: list[Linear] = []

        self.cond_layer: Linear | None = None
        if gin_channels != 0:
            self.cond_layer = Linear(
                device=device,
                in_features=gin_channels,
                out_features=2 * hidden_channels * num_layers,
            )

        for i in range(num_layers):
            dilation = dilation_rate**i
            self.in_layers.append(
                Conv1d(
                    device=device,
                    in_channels=hidden_channels,
                    out_channels=2 * hidden_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding="same",
                )
            )

            res_skip_channels = 2 * hidden_channels if i < num_layers - 1 else hidden_channels
            self.res_skip_layers.append(
                Linear(
                    device=device,
                    in_features=hidden_channels,
                    out_features=res_skip_channels,
                )
            )

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], module_prefix: str | None = None) -> None:
        if self.cond_layer is not None:
            self.cond_layer.load_state_dict(state_dict, key="cond_layer", module_prefix=module_prefix)
        for i, layer in enumerate(self.in_layers):
            layer.load_state_dict(state_dict, key=f"in_layers.{i}", module_prefix=module_prefix)
        for i, layer in enumerate(self.res_skip_layers):
            layer.load_state_dict(state_dict, key=f"res_skip_layers.{i}", module_prefix=module_prefix)

    def __call__(self, x: ttnn.Tensor, g: ttnn.Tensor | None = None) -> ttnn.Tensor:
        out = ttnn.zeros_like(x)
        g_proj = None
        if g is not None:
            if self.cond_layer is None:
                raise ValueError("g is provided but gin_channels is 0.")
            g_proj = self.cond_layer(g)

        for i, (in_layer, res_skip_layer) in enumerate(zip(self.in_layers, self.res_skip_layers, strict=True)):
            conv_out = in_layer(x)
            if g_proj is not None:
                cond_offset = i * 2 * self.hidden_channels
                layer_conditioning = ttnn.slice(
                    g_proj,
                    (0, 0, cond_offset),
                    (g_proj.shape[0], g_proj.shape[1], cond_offset + 2 * self.hidden_channels),
                )
            else:
                layer_conditioning = ttnn.zeros_like(conv_out)
            conv_out = ttnn.to_memory_config(conv_out, ttnn.L1_MEMORY_CONFIG)
            input_activation = ttnn.add(conv_out, layer_conditioning, output_tensor=conv_out)
            t_activation, s_activation = ttnn.chunk(input_activation, 2, dim=-1)
            gates_activations = ttnn.multiply(
                ttnn.sigmoid(s_activation, output_tensor=s_activation),
                ttnn.tanh(t_activation, output_tensor=t_activation),
                output_tensor=s_activation,
            )
            res_skip_out = res_skip_layer(gates_activations)
            if i < self.num_layers - 1:
                residual_out, skip_out = ttnn.chunk(res_skip_out, 2, dim=-1)
                x = ttnn.add(x, residual_out, output_tensor=x)
                out = ttnn.add(out, skip_out, output_tensor=out)
            else:
                out = ttnn.add(out, res_skip_out, output_tensor=out)

        return out


class ResBlock1:
    def __init__(
        self,
        device: ttnn.MeshDevice,
        channels: int,
        kernel_size: int = 3,
        dilation: tuple[int, int, int] = (1, 3, 5),
    ) -> None:
        self.convs1: list[Conv1d] = []
        self.convs2: list[Conv1d] = []
        self.lrelu_slope = LRELU_SLOPE
        for d_value in dilation:
            self.convs1.append(
                Conv1d(
                    device=device,
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=d_value,
                    padding="same",
                    activation=("leaky_relu", {"negative_slope": self.lrelu_slope}),
                    deallocate_input=True,
                )
            )
            self.convs2.append(
                Conv1d(
                    device=device,
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=1,
                    padding="same",
                    deallocate_input=True,
                )
            )

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], module_prefix: str | None = None) -> None:
        for i, conv in enumerate(self.convs1):
            conv.load_state_dict(state_dict, key=f"convs1.{i}", module_prefix=module_prefix)
        for i, conv in enumerate(self.convs2):
            conv.load_state_dict(state_dict, key=f"convs2.{i}", module_prefix=module_prefix)

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        # needed since x is modified in-place in the loop, and we want to keep the original x for the residual connection
        x = ttnn.clone(x)
        for c1, c2 in zip(self.convs1, self.convs2, strict=True):
            hidden = ttnn.leaky_relu(x, negative_slope=self.lrelu_slope)
            hidden = c1(hidden)
            hidden = c2(hidden)
            x = ttnn.add(hidden, x, output_tensor=x)
        return x


class ResBlock2:
    def __init__(
        self,
        device: ttnn.MeshDevice,
        channels: int,
        kernel_size: int = 3,
        dilation: tuple[int, int] = (1, 3),
    ) -> None:
        self.convs: list[Conv1d] = []
        self.lrelu_slope = LRELU_SLOPE
        for d_value in dilation:
            self.convs.append(
                Conv1d(
                    device=device,
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=d_value,
                    padding="same",
                    deallocate_input=True,
                )
            )

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], module_prefix: str | None = None) -> None:
        for i, conv in enumerate(self.convs):
            conv.load_state_dict(state_dict, key=f"convs.{i}", module_prefix=module_prefix)

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        # needed since x is modified in-place in the loop, and we want to keep the original x for the residual connection
        x = ttnn.clone(x)
        for conv in self.convs:
            xt = ttnn.leaky_relu(x, negative_slope=self.lrelu_slope)
            xt = conv(xt)
            x = ttnn.add(xt, x, output_tensor=x)
        return x


class ResidualCouplingLayer:
    def __init__(
        self,
        device: ttnn.MeshDevice,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        num_layers: int,
        gin_channels: int = 0,
    ) -> None:
        if channels % 2 != 0:
            raise ValueError("channels should be divisible by 2")
        self.half_channels = channels // 2
        self.pre_linear = Linear(
            device=device,
            in_features=self.half_channels,
            out_features=hidden_channels,
        )
        self.enc = WN(
            device=device,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            num_layers=num_layers,
            gin_channels=gin_channels,
        )
        self.post_linear = Linear(
            device=device,
            in_features=hidden_channels,
            out_features=self.half_channels,
        )

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], module_prefix: str | None = None) -> None:
        if module_prefix is None:
            module_prefix = ""
        enc_module_prefix = f"{module_prefix}enc."
        self.pre_linear.load_state_dict(state_dict, key="pre_linear", module_prefix=module_prefix)
        self.enc.load_state_dict(state_dict, module_prefix=enc_module_prefix)
        self.post_linear.load_state_dict(state_dict, key="post_linear", module_prefix=module_prefix)

    def __call__(self, x: ttnn.Tensor, g: ttnn.Tensor | None = None) -> ttnn.Tensor:
        x0, x1 = ttnn.chunk(x, 2, dim=-1)
        h = self.pre_linear(x0)
        h = self.enc(h, g=g)
        stats = self.post_linear(h)
        x1 = ttnn.subtract(x1, stats, output_tensor=x1)
        out = ttnn.concat([x0, x1], dim=-1, memory_config=x0.memory_config())
        return out


class Embedding:
    def __init__(self, device: ttnn.MeshDevice, num_embeddings: int, embedding_dim: int) -> None:
        self.device = device
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight: ttnn.Tensor | None = None

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], module_prefix: str | None = None) -> None:
        if module_prefix is None:
            module_prefix = ""
        weight_key = f"{module_prefix}weight"
        self.weight = ttnn.from_torch(
            state_dict[weight_key].detach(),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
        )

    def __call__(self, indices: ttnn.Tensor) -> ttnn.Tensor:
        if self.weight is None:
            raise ValueError("Embedding state_dict are not loaded.")
        return ttnn.embedding(indices, self.weight, layout=ttnn.TILE_LAYOUT)


class Encoder:
    def __init__(
        self,
        device: ttnn.MeshDevice,
        hidden_channels: int,
        filter_channels: int,
        num_heads: int,
        num_layers: int,
        kernel_size: int = 1,
        window_size: int = 10,
    ) -> None:
        self.device = device
        self.num_layers = int(num_layers)
        self.attn_layers = [
            MultiHeadAttention(
                device=device,
                in_features=hidden_channels,
                out_features=hidden_channels,
                num_heads=num_heads,
                window_size=window_size,
            )
            for _ in range(self.num_layers)
        ]
        self.norm_layers_1 = [LayerNorm(device, hidden_channels) for _ in range(self.num_layers)]
        self.ffn_layers = [
            FFN(
                device=device,
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                filter_channels=filter_channels,
                kernel_size=kernel_size,
            )
            for _ in range(self.num_layers)
        ]
        self.norm_layers_2 = [LayerNorm(device, hidden_channels) for _ in range(self.num_layers)]

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], module_prefix: str | None = None) -> None:
        if module_prefix is None:
            module_prefix = ""
        for i in range(self.num_layers):
            self.attn_layers[i].load_state_dict(state_dict, module_prefix=f"{module_prefix}attn_layers.{i}.")
            self.norm_layers_1[i].load_state_dict(state_dict, module_prefix=f"{module_prefix}norm_layers_1.{i}.")
            self.ffn_layers[i].load_state_dict(state_dict, module_prefix=f"{module_prefix}ffn_layers.{i}.")
            self.norm_layers_2[i].load_state_dict(state_dict, module_prefix=f"{module_prefix}norm_layers_2.{i}.")

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        for i in range(self.num_layers):
            y = self.attn_layers[i](x, x)
            x = self.norm_layers_1[i](ttnn.add(x, y, output_tensor=x))
            y = self.ffn_layers[i](x)
            x = self.norm_layers_2[i](ttnn.add(x, y, output_tensor=x))
        return x


class TextEncoder:
    def __init__(
        self,
        device: ttnn.MeshDevice,
        embedding_dims: int,
        out_channels: int,
        hidden_channels: int,
        filter_channels: int,
        num_heads: int,
        num_layers: int,
        kernel_size: int,
        f0: bool = True,
    ) -> None:
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.emb_phone = Linear(device, embedding_dims, hidden_channels)
        self.use_f0 = f0
        self.emb_pitch = Embedding(device, 256, hidden_channels) if f0 else None
        self.encoder = Encoder(
            device=device,
            hidden_channels=hidden_channels,
            filter_channels=filter_channels,
            num_heads=num_heads,
            num_layers=num_layers,
            kernel_size=kernel_size,
        )
        self.proj_linear = Linear(
            device=device,
            in_features=hidden_channels,
            out_features=out_channels * 2,
        )

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], module_prefix: str | None = None) -> None:
        if module_prefix is None:
            module_prefix = ""
        self.emb_phone.load_state_dict(state_dict, key="emb_phone", module_prefix=module_prefix)
        if self.use_f0 and self.emb_pitch is not None:
            self.emb_pitch.load_state_dict(state_dict, module_prefix=f"{module_prefix}emb_pitch.")
        self.encoder.load_state_dict(state_dict, module_prefix=f"{module_prefix}encoder.")
        proj_key = (
            "proj_linear"
            if (f"{module_prefix}proj_linear.weight" if module_prefix else "proj_linear.weight") in state_dict
            else "proj"
        )
        self.proj_linear.load_state_dict(state_dict, key=proj_key, module_prefix=module_prefix)

    def __call__(self, phone: ttnn.Tensor, pitch: ttnn.Tensor | None) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        x = self.emb_phone(phone)
        if self.use_f0 and pitch is not None and self.emb_pitch is not None:
            x = ttnn.add(x, self.emb_pitch(pitch), output_tensor=x)
        x = ttnn.multiply(x, math.sqrt(self.hidden_channels), output_tensor=x)
        x = ttnn.leaky_relu(x, negative_slope=0.1, output_tensor=x)
        encoded = self.encoder(x)
        stats = self.proj_linear(encoded)
        m, logs = ttnn.chunk(stats, chunks=2, dim=-1)
        return m, logs


class ResidualCouplingBlock:
    def __init__(
        self,
        device: ttnn.MeshDevice,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        num_layers: int,
        num_flows: int = 4,
        gin_channels: int = 0,
    ) -> None:
        self.flows = [
            ResidualCouplingLayer(
                device,
                channels,
                hidden_channels,
                kernel_size,
                dilation_rate,
                num_layers,
                gin_channels=gin_channels,
            )
            for _ in range(num_flows)
        ]
        self.device = device

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], module_prefix: str | None = None) -> None:
        if module_prefix is None:
            module_prefix = ""
        for i, flow in enumerate(self.flows):
            flow.load_state_dict(state_dict, module_prefix=f"{module_prefix}flows.{i}.")

    def __call__(self, x: ttnn.Tensor, g: ttnn.Tensor | None = None) -> ttnn.Tensor:
        for flow in self.flows:
            x = flow(_flip_last_dim_ttnn(x), g=g)
        return x


class Generator:
    def __init__(
        self,
        device: ttnn.MeshDevice,
        initial_channel: int,
        resblock: str,
        resblock_kernel_sizes: list[int],
        resblock_dilation_sizes: list[list[int]],
        upsample_rates: list[int],
        upsample_initial_channel: int,
        upsample_kernel_sizes: list[int],
        gin_channels: int = 0,
    ) -> None:
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        self.conv_pre = Conv1d(
            device=device,
            in_channels=initial_channel,
            out_channels=upsample_initial_channel,
            kernel_size=7,
            stride=1,
            padding="same",
            deallocate_input=True,
        )

        self.ups: list[ConvTranspose1d] = []
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes, strict=True)):
            self.ups.append(
                ConvTranspose1d(
                    device=device,
                    in_channels=upsample_initial_channel // (2**i),
                    out_channels=upsample_initial_channel // (2 ** (i + 1)),
                    kernel_size=k,
                    stride=u,
                    padding=(k - u) // 2,
                    deallocate_input=True,
                )
            )

        resblock_cls = ResBlock1 if resblock == "1" else ResBlock2
        self.resblocks = []
        for i in range(len(self.ups)):
            channels = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes, strict=True):
                self.resblocks.append(resblock_cls(device, channels, k, tuple(d)))

        self.conv_post = Conv1d(
            device=device,
            in_channels=channels,
            out_channels=1,
            kernel_size=7,
            stride=1,
            padding="same",
            activation="tanh",
            deallocate_input=True,
        )
        self.cond_linear = None
        if gin_channels != 0:
            self.cond_linear = Linear(
                device=device,
                in_features=gin_channels,
                out_features=upsample_initial_channel,
            )

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], module_prefix: str | None = None) -> None:
        if module_prefix is None:
            module_prefix = ""
        self.conv_pre.load_state_dict(state_dict, key="conv_pre", module_prefix=module_prefix)
        if self.cond_linear is not None:
            self.cond_linear.load_state_dict(state_dict, key="cond_linear", module_prefix=module_prefix)
        for i, up in enumerate(self.ups):
            up.load_state_dict(state_dict, key=f"ups.{i}", module_prefix=module_prefix)
        for i, rb in enumerate(self.resblocks):
            rb.load_state_dict(state_dict, module_prefix=f"{module_prefix}resblocks.{i}.")
        self.conv_post.load_state_dict(state_dict, key="conv_post", module_prefix=module_prefix)

    def __call__(self, x: ttnn.Tensor, conditioning: ttnn.Tensor | None = None) -> ttnn.Tensor:
        x = self.conv_pre(x)
        if conditioning is not None and self.cond_linear is not None:
            x = ttnn.add(x, self.cond_linear(conditioning), output_tensor=x)

        for i in range(self.num_upsamples):
            x = ttnn.leaky_relu(x, negative_slope=LRELU_SLOPE, output_tensor=x)
            x = self.ups[i](x)
            resblock_sum = self.resblocks[i * self.num_kernels](x)
            for j in range(1, self.num_kernels):
                resblock_sum = ttnn.add(
                    resblock_sum, self.resblocks[i * self.num_kernels + j](x), output_tensor=resblock_sum
                )
            x = ttnn.multiply(resblock_sum, 1.0 / self.num_kernels, output_tensor=resblock_sum)

        x = ttnn.leaky_relu(x, negative_slope=LRELU_SLOPE, output_tensor=x)
        x = self.conv_post(x)
        return x


class SineGen:
    def __init__(
        self,
        device: ttnn.MeshDevice,
        samp_rate: int,
        harmonic_num: int = 0,
        sine_amp: float = 0.1,
        noise_std: float = 0.003,
        voiced_threshold: float = 0,
    ) -> None:
        self.device = device
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold

    def __call__(self, f0: ttnn.Tensor, upp: int) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        # f0: [B, T]
        # Upsample f0 to full resolution first using TTNN wrapper.
        f0_up = _interpolate_1d(f0, scale_factor=upp, mode="nearest")
        # f0_up = ttnn.pad(f0_up, ((0, 0), (0, (32 - f0_up.shape[1] % 32) % 32), (0, 0)), value=0)
        f0_up = ttnn.reshape(f0_up, (1, int(f0_up.shape[1] / 32), 32))
        f0_up = ttnn.to_layout(f0_up, ttnn.TILE_LAYOUT)
        # Voiced/unvoiced mask.
        uv = ttnn.gt(f0_up, self.voiced_threshold)

        # Expand for harmonics: [B, T*upp, H].
        harmonics = ttnn.arange(
            start=1,
            end=self.harmonic_num + 2,
            dtype=f0_up.dtype,
            device=self.device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        f0_harm = f0_up * (harmonics / self.sampling_rate)

        # Accumulate phase and add random initial offset per harmonic.
        # phase = ttnn.cumsum(f0_harm, dim=1, out=f0_harm)
        # TODO: fallback is faster than native cumsum
        phase = ttnn_cumsum_fallback(f0_harm, dim=1)
        rand_ini = ttnn.rand(
            (f0_up.shape[0], self.harmonic_num + 1),
            dtype=ttnn.bfloat16,
            device=self.device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        phase = ttnn.add(phase, rand_ini, output_tensor=phase)
        phase = ttnn.multiply(phase, 2 * math.pi, output_tensor=phase)
        sine_waves = ttnn.multiply(ttnn.sin(phase, output_tensor=phase), self.sine_amp, output_tensor=phase)

        # Mix with noise based on voiced/unvoiced.
        noise_amp = uv * self.noise_std + ttnn.rsub(uv, 1) * self.sine_amp / 3
        noise_amp = ttnn.multiply(
            noise_amp,
            ttnn_randn_fallback(tuple(sine_waves.shape), dtype=ttnn.bfloat16, device=self.device),
            output_tensor=noise_amp,
        )
        out = ttnn.add(sine_waves, noise_amp, output_tensor=sine_waves)
        out = ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.reshape(out, (out.shape[0], out.shape[1] * out.shape[2], 1))
        return out


class SourceModuleHnNSF:
    def __init__(
        self,
        device: ttnn.MeshDevice,
        sampling_rate: int,
        harmonic_num: int = 0,
        sine_amp: float = 0.1,
        add_noise_std: float = 0.003,
        voiced_threshod: float = 0,
    ) -> None:
        self.device = device
        self.l_sin_gen = SineGen(device, sampling_rate, harmonic_num, sine_amp, add_noise_std, voiced_threshod)
        self.l_linear = Linear(device=device, in_features=harmonic_num + 1, out_features=1, activation="tanh")

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], module_prefix: str | None = None) -> None:
        self.l_linear.load_state_dict(state_dict=state_dict, key="l_linear", module_prefix=module_prefix)

    def __call__(self, x: ttnn.Tensor, upp: int = 1) -> ttnn.Tensor:
        sine_wavs = self.l_sin_gen(x, upp)
        tt_linear = self.l_linear(sine_wavs)
        # this is needed since last dim is 1, padding due to tile layout would cause huge unnecessary memory usage
        tt_linear = ttnn.to_layout(tt_linear, ttnn.ROW_MAJOR_LAYOUT)
        return tt_linear


class GeneratorNSF:
    def __init__(
        self,
        device: ttnn.MeshDevice,
        initial_channel: int,
        resblock: str,
        resblock_kernel_sizes: list[int],
        resblock_dilation_sizes: list[list[int]],
        upsample_rates: list[int],
        upsample_initial_channel: int,
        upsample_kernel_sizes: list[int],
        gin_channels: int,
        sr: int,
    ) -> None:
        self.device = device
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.m_source = SourceModuleHnNSF(device=device, sampling_rate=sr, harmonic_num=0)
        self.upp = math.prod(upsample_rates)
        self.lrelu_slope = LRELU_SLOPE

        self.conv_pre = Conv1d(
            device=device,
            in_channels=initial_channel,
            out_channels=upsample_initial_channel,
            kernel_size=7,
            stride=1,
            padding="same",
            deallocate_input=True,
        )

        self.ups: list[ConvTranspose1d] = []
        self.noise_convs: list[Conv1d | Linear] = []
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes, strict=True)):
            current_channels = upsample_initial_channel // (2 ** (i + 1))
            self.ups.append(
                ConvTranspose1d(
                    device=device,
                    in_channels=upsample_initial_channel // (2**i),
                    out_channels=current_channels,
                    kernel_size=k,
                    stride=u,
                    padding=(k - u) // 2,
                )
            )
            if i + 1 < len(upsample_rates):
                stride_f0 = math.prod(upsample_rates[i + 1 :])
                self.noise_convs.append(
                    Conv1d(
                        device=device,
                        in_channels=1,
                        out_channels=current_channels,
                        kernel_size=stride_f0 * 2,
                        stride=stride_f0,
                        padding=stride_f0 // 2,
                        deallocate_input=True,
                    )
                )
            else:
                self.noise_convs.append(
                    Linear(
                        device=device,
                        in_features=1,
                        out_features=current_channels,
                    )
                )

        resblock_cls = ResBlock1 if resblock == "1" else ResBlock2
        self.resblocks = []
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes, strict=True):
                self.resblocks.append(resblock_cls(device, ch, k, tuple(d)))

        self.conv_post = Conv1d(
            device=device,
            in_channels=ch,
            out_channels=1,
            kernel_size=7,
            stride=1,
            padding="same",
            activation="tanh",
            deallocate_input=True,
        )
        self.cond_linear = Linear(device=device, in_features=gin_channels, out_features=upsample_initial_channel)

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], module_prefix: str | None = None) -> None:
        if module_prefix is None:
            module_prefix = ""
        self.conv_pre.load_state_dict(state_dict, key="conv_pre", module_prefix=module_prefix)
        self.cond_linear.load_state_dict(state_dict, key="cond_linear", module_prefix=module_prefix)
        self.m_source.load_state_dict(state_dict, module_prefix=f"{module_prefix}m_source.")
        for i, up in enumerate(self.ups):
            up.load_state_dict(state_dict, key=f"ups.{i}", module_prefix=module_prefix)
        for i, nc in enumerate(self.noise_convs):
            nc.load_state_dict(state_dict, key=f"noise_convs.{i}", module_prefix=module_prefix)
        for i, rb in enumerate(self.resblocks):
            rb.load_state_dict(state_dict, module_prefix=f"{module_prefix}resblocks.{i}.")
        self.conv_post.load_state_dict(state_dict, key="conv_post", module_prefix=module_prefix)

    def __call__(self, x: ttnn.Tensor, f0: ttnn.Tensor, conditioning: ttnn.Tensor | None = None) -> ttnn.Tensor:
        harmonic_source = self.m_source(f0, self.upp)
        x = self.conv_pre(x)
        if conditioning is not None:
            x = ttnn.add(x, self.cond_linear(conditioning), output_tensor=x)
        # before upscalin they neede to loaded to DRAM since up scaling creates large size tensors that cannot fit in L1
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        for i, (ups, noise_convs) in enumerate(zip(self.ups, self.noise_convs, strict=True)):
            x = ttnn.leaky_relu(x, negative_slope=self.lrelu_slope, output_tensor=x)
            x = ups(x)
            # the layout conversion happens inside noise_convs because doign it here causes oom for some reason
            # TODO: investigate the reasoning behind this
            source_features = noise_convs(harmonic_source)
            x = ttnn.add(x, source_features, output_tensor=x)
            resblock_sum = self.resblocks[i * self.num_kernels](x)
            ttnn.deallocate(source_features)
            for j in range(i * self.num_kernels + 1, (i + 1) * self.num_kernels):
                resblock_sum = ttnn.add(resblock_sum, self.resblocks[j](x), output_tensor=resblock_sum)
            x = ttnn.multiply(resblock_sum, 1.0 / self.num_kernels, output_tensor=resblock_sum)

        x = ttnn.leaky_relu(x, negative_slope=self.lrelu_slope, output_tensor=x)
        x = self.conv_post(x)
        return x


sr2sr = {
    "32k": 32000,
    "40k": 40000,
    "48k": 48000,
}


class SynthesizerTrnMsNSF:
    def __init__(
        self,
        device: ttnn.MeshDevice,
        embedding_dims: int,
        inter_channels: int,
        hidden_channels: int,
        filter_channels: int,
        num_heads: int,
        num_layers: int,
        kernel_size: int,
        resblock: str,
        resblock_kernel_sizes: list[int],
        resblock_dilation_sizes: list[list[int]],
        upsample_rates: list[int],
        upsample_initial_channel: int,
        upsample_kernel_sizes: list[int],
        spk_embed_dim: int,
        gin_channels: int,
        sr: int | str,
    ) -> None:
        if isinstance(sr, str):
            sr = sr2sr[sr]
        self.device = device
        self.enc_p = TextEncoder(
            device,
            embedding_dims,
            inter_channels,
            hidden_channels,
            filter_channels,
            num_heads,
            num_layers,
            kernel_size,
        )
        self.dec = GeneratorNSF(
            device,
            inter_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
            sr=sr,
        )
        self.flow = ResidualCouplingBlock(device, inter_channels, hidden_channels, 5, 1, 3, gin_channels=gin_channels)
        self.embedding = Embedding(device, spk_embed_dim, gin_channels)

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], module_prefix: str | None = None) -> None:
        if module_prefix is None:
            module_prefix = ""
        self.enc_p.load_state_dict(state_dict, module_prefix=f"{module_prefix}enc_p.")
        self.dec.load_state_dict(state_dict, module_prefix=f"{module_prefix}dec.")
        self.flow.load_state_dict(state_dict, module_prefix=f"{module_prefix}flow.")
        self.embedding.load_state_dict(state_dict, module_prefix=f"{module_prefix}emb_g.")

    def __call__(
        self, phone: ttnn.Tensor, pitch: ttnn.Tensor, nsf_f0: ttnn.Tensor, speaker_id: ttnn.Tensor
    ) -> ttnn.Tensor:
        conditioning = self.embedding(speaker_id)
        conditioning = ttnn.unsqueeze(conditioning, dim=1)
        prior_mean, prior_log = self.enc_p(phone, pitch)
        latent = (
            prior_mean
            + ttnn.exp(prior_log, output_tensor=prior_log)
            * ttnn_randn_fallback(tuple(prior_mean.shape), dtype=ttnn.bfloat16, device=self.device)
            * 0.66666
        )
        latent_flow = self.flow(latent, conditioning)
        out = self.dec(latent_flow, nsf_f0, conditioning)
        return out


class SynthesizerTrnMsNSF_nono:
    def __init__(
        self,
        device: ttnn.MeshDevice,
        embedding_dims: int,
        inter_channels: int,
        hidden_channels: int,
        filter_channels: int,
        num_heads: int,
        num_layers: int,
        kernel_size: int,
        resblock: str,
        resblock_kernel_sizes: list[int],
        resblock_dilation_sizes: list[list[int]],
        upsample_rates: list[int],
        upsample_initial_channel: int,
        upsample_kernel_sizes: list[int],
        spk_embed_dim: int,
        gin_channels: int,
        sr: int | None = None,
    ) -> None:
        self.device = device
        self.enc_p = TextEncoder(
            device,
            embedding_dims,
            inter_channels,
            hidden_channels,
            filter_channels,
            num_heads,
            num_layers,
            kernel_size,
            f0=False,
        )
        self.dec = Generator(
            device,
            inter_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
        )
        self.flow = ResidualCouplingBlock(device, inter_channels, hidden_channels, 5, 1, 3, gin_channels=gin_channels)
        self.embedding = Embedding(device, spk_embed_dim, gin_channels)

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], module_prefix: str | None = None) -> None:
        if module_prefix is None:
            module_prefix = ""
        self.enc_p.load_state_dict(state_dict, module_prefix=f"{module_prefix}enc_p.")
        self.dec.load_state_dict(state_dict, module_prefix=f"{module_prefix}dec.")
        self.flow.load_state_dict(state_dict, module_prefix=f"{module_prefix}flow.")
        self.embedding.load_state_dict(state_dict, module_prefix=f"{module_prefix}emb_g.")

    def __call__(self, phone: ttnn.Tensor, speaker_id: ttnn.Tensor) -> ttnn.Tensor:
        conditioning = self.embedding(speaker_id)
        conditioning = ttnn.unsqueeze(conditioning, dim=1)
        prior_mean, prior_log = self.enc_p(phone, None)
        latent = (
            prior_mean
            + ttnn.exp(prior_log)
            * ttnn_randn_fallback(tuple(m_p.shape), dtype=ttnn.bfloat16, device=self.device)
            * 0.66666
        )
        latent_flow = self.flow(latent, conditioning)
        out = self.dec(latent_flow, conditioning)
        return out
