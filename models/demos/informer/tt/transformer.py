# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Optional

import torch

import ttnn

from .attention import MultiHeadAttention
from .config import (
    InformerConfig,
    align_to_tile,
    create_sharded_memory_config,
    get_compute_dtype,
    get_shard_shape,
    get_shard_strategy,
    get_ttnn_dtype,
)
from .embeddings import ConvDistillLayer
from .ops import apply_dropout, linear, max_pool1d, slice_sequence
from .state_io import to_float_tensor


class LayerNorm:
    def __init__(
        self,
        d_model: int,
        rng: torch.Generator,
        *,
        device,
        dtype: ttnn.DataType,
        compute_dtype: Optional[ttnn.DataType] = None,
        eps: float = 1e-5,
    ):
        self.output_dtype = dtype
        self.compute_dtype = compute_dtype or dtype
        self.device = device
        self.weight_torch = torch.ones((d_model,), dtype=torch.float32)
        self.bias_torch = torch.zeros((d_model,), dtype=torch.float32)
        weight = self.weight_torch.reshape(1, -1)
        bias = self.bias_torch.reshape(1, -1)
        self.weight = ttnn.from_torch(weight, device=device, dtype=self.compute_dtype, layout=ttnn.TILE_LAYOUT)
        self.bias = ttnn.from_torch(bias, device=device, dtype=self.compute_dtype, layout=ttnn.TILE_LAYOUT)
        self.eps = eps

    def load_hf_state_dict(self, state: dict[str, torch.Tensor], *, strict: bool = True) -> dict[str, list[str]]:
        mapping = (("weight", "weight", True), ("bias", "bias", True))
        used: set[str] = set()
        missing: list[str] = []
        for key, attr, reshape in mapping:
            tensor = state.get(key)
            if tensor is None:
                missing.append(key)
                continue
            used.add(key)
            value = to_float_tensor(tensor)
            setattr(self, f"{attr}_torch", value)
            upload = value.reshape(1, -1) if reshape else value
            ref = getattr(self, attr)
            setattr(
                self,
                attr,
                ttnn.from_torch(upload, device=self.device, dtype=ref.dtype, layout=ttnn.TILE_LAYOUT),
            )
        unexpected = sorted(k for k in state if k not in used)
        if strict and missing:
            raise ValueError(f"Missing layer norm weights: {missing}")
        return {"missing_keys": missing, "unexpected_keys": unexpected}

    def load_ttnn_state_dict(self, state: dict[str, torch.Tensor], *, strict: bool = True) -> dict[str, list[str]]:
        """Load TTNN-canonical layer-norm tensors."""
        return self.load_hf_state_dict(state, strict=strict)

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        if x.dtype != self.compute_dtype:
            x = ttnn.typecast(x, self.compute_dtype)
        out = ttnn.layer_norm(x, weight=self.weight, bias=self.bias, epsilon=self.eps)
        if self.output_dtype != self.compute_dtype:
            out = ttnn.typecast(out, self.output_dtype)
        return out


class FeedForward:
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float,
        rng: torch.Generator,
        *,
        device,
        dtype: ttnn.DataType,
        compute_dtype: Optional[ttnn.DataType] = None,
        weight_dtype: Optional[ttnn.DataType] = None,
        memory_config: Optional[ttnn.MemoryConfig] = None,
        use_sharded: bool = False,
        shard_strategy: str = "height",
    ):
        self.output_dtype = dtype
        self.compute_dtype = compute_dtype or dtype
        self.weight_dtype = weight_dtype or dtype
        self.w1_torch = torch.randn((d_ff, d_model), generator=rng, dtype=torch.float32) * 0.02
        self.b1_torch = torch.zeros((d_ff,), dtype=torch.float32)
        self.w2_torch = torch.randn((d_model, d_ff), generator=rng, dtype=torch.float32) * 0.02
        self.b2_torch = torch.zeros((d_model,), dtype=torch.float32)
        self.w1 = ttnn.from_torch(self.w1_torch, device=device, dtype=self.weight_dtype, layout=ttnn.TILE_LAYOUT)
        self.b1 = ttnn.from_torch(self.b1_torch, device=device, dtype=self.weight_dtype, layout=ttnn.TILE_LAYOUT)
        self.w2 = ttnn.from_torch(self.w2_torch, device=device, dtype=self.weight_dtype, layout=ttnn.TILE_LAYOUT)
        self.b2 = ttnn.from_torch(self.b2_torch, device=device, dtype=self.weight_dtype, layout=ttnn.TILE_LAYOUT)
        self.dropout = dropout
        self.dtype = dtype
        self.memory_config = memory_config
        self.use_sharded = use_sharded
        self.shard_strategy = get_shard_strategy(shard_strategy) if use_sharded else None
        self.device = device
        self.core_grid = None
        self.interleaved_mem_config = None
        if self.use_sharded:
            grid_size = device.compute_with_storage_grid_size()
            self.core_grid = ttnn.CoreGrid(y=grid_size.y, x=grid_size.x)
            self.interleaved_mem_config = ttnn.L1_MEMORY_CONFIG if memory_config else ttnn.DRAM_MEMORY_CONFIG

    def load_hf_state_dict(self, state: dict[str, torch.Tensor], *, strict: bool = True) -> dict[str, list[str]]:
        mapping = (
            ("fc1.weight", "w1"),
            ("fc1.bias", "b1"),
            ("fc2.weight", "w2"),
            ("fc2.bias", "b2"),
        )
        used: set[str] = set()
        missing: list[str] = []
        for key, attr in mapping:
            tensor = state.get(key)
            if tensor is None:
                missing.append(key)
                continue
            used.add(key)
            value = to_float_tensor(tensor)
            setattr(self, f"{attr}_torch", value)
            ref = getattr(self, attr)
            setattr(
                self,
                attr,
                ttnn.from_torch(value, device=self.device, dtype=ref.dtype, layout=ttnn.TILE_LAYOUT),
            )
        unexpected = sorted(k for k in state if k not in used)
        if strict and missing:
            raise ValueError(f"Missing feed-forward weights: {missing}")
        return {"missing_keys": missing, "unexpected_keys": unexpected}

    def load_ttnn_state_dict(self, state: dict[str, torch.Tensor], *, strict: bool = True) -> dict[str, list[str]]:
        """Load TTNN-canonical feed-forward tensors."""
        mapping = (
            ("w1", "w1"),
            ("b1", "b1"),
            ("w2", "w2"),
            ("b2", "b2"),
        )
        used: set[str] = set()
        missing: list[str] = []
        for key, attr in mapping:
            tensor = state.get(key)
            if tensor is None:
                missing.append(key)
                continue
            used.add(key)
            value = to_float_tensor(tensor)
            setattr(self, f"{attr}_torch", value)
            ref = getattr(self, attr)
            setattr(
                self,
                attr,
                ttnn.from_torch(value, device=self.device, dtype=ref.dtype, layout=ttnn.TILE_LAYOUT),
            )
        unexpected = sorted(k for k in state if k not in used)
        if strict and missing:
            raise ValueError(f"Missing feed-forward weights: {missing}")
        return {"missing_keys": missing, "unexpected_keys": unexpected}

    def linear_output_memcfg(self, x: ttnn.Tensor, output_width: int) -> ttnn.MemoryConfig:
        shape = list(get_shard_shape(x))
        shape[-1] = align_to_tile(output_width)
        return create_sharded_memory_config(tuple(shape), device=self.device, strategy=self.shard_strategy)

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        if x.dtype != self.compute_dtype:
            x = ttnn.typecast(x, self.compute_dtype)
        # Keep sharded FFN on common sequence lengths, but avoid known non-finite instability on very long rows.
        use_sharded = self.use_sharded and int(x.shape[1]) <= 256
        memcfg = self.memory_config
        if use_sharded:
            memcfg = create_sharded_memory_config(get_shard_shape(x), device=self.device, strategy=self.shard_strategy)
            x = ttnn.to_memory_config(x, memcfg)
            memcfg = self.linear_output_memcfg(x, self.w1_torch.shape[0])
        x = linear(x, self.w1, self.b1, dtype=self.compute_dtype, memory_config=memcfg)
        if use_sharded:
            x = ttnn.to_memory_config(x, self.interleaved_mem_config)
        x = ttnn.gelu(x)
        x = apply_dropout(x, self.dropout)
        memcfg = self.memory_config
        if use_sharded:
            memcfg = create_sharded_memory_config(get_shard_shape(x), device=self.device, strategy=self.shard_strategy)
            x = ttnn.to_memory_config(x, memcfg)
            memcfg = self.linear_output_memcfg(x, self.w2_torch.shape[0])
        x = linear(x, self.w2, self.b2, dtype=self.compute_dtype, memory_config=memcfg)
        if use_sharded:
            x = ttnn.to_memory_config(x, self.interleaved_mem_config)
        x = apply_dropout(x, self.dropout)
        if x.dtype != self.output_dtype:
            x = ttnn.typecast(x, self.output_dtype)
        return x


class EncoderLayer:
    def __init__(self, config: InformerConfig, rng: torch.Generator, *, device, dtype: ttnn.DataType):
        memory_config = ttnn.L1_MEMORY_CONFIG if config.use_l1 else None
        compute_dtype = get_compute_dtype(config, dtype)
        weight_dtype = compute_dtype if config.hf_compat else dtype
        self.attn = MultiHeadAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            dropout=config.dropout,
            rng=rng,
            device=device,
            dtype=dtype,
            prob_sparse=config.attention_type == "prob",
            factor=config.factor,
            mask_value=config.attn_mask_value,
            use_sdpa=config.use_sdpa,
            random_sampling=config.hf_compat,
            is_decoder=False,
            compute_dtype=compute_dtype,
            memory_config=memory_config,
            use_sharded=config.use_sharded,
            shard_strategy=config.shard_strategy,
        )
        ffn_dim = config.encoder_ffn_dim if config.encoder_ffn_dim is not None else config.d_ff
        self.ffn = FeedForward(
            config.d_model,
            ffn_dim,
            config.dropout,
            rng,
            device=device,
            dtype=dtype,
            compute_dtype=compute_dtype,
            weight_dtype=weight_dtype,
            memory_config=memory_config,
            use_sharded=config.use_sharded,
            shard_strategy=config.shard_strategy,
        )
        self.norm1 = LayerNorm(config.d_model, rng, device=device, dtype=dtype, compute_dtype=compute_dtype)
        self.norm2 = LayerNorm(config.d_model, rng, device=device, dtype=dtype, compute_dtype=compute_dtype)

    def __call__(
        self,
        x: ttnn.Tensor,
        mask: ttnn.Tensor | None,
        valid_length: int,
    ) -> ttnn.Tensor:
        attn_out = self.attn(
            x,
            x,
            x,
            mask,
            q_valid_len=valid_length,
            k_valid_len=valid_length,
        )
        x = self.norm1(x + attn_out)
        ff = self.ffn(x)
        x = self.norm2(x + ff)
        return x


class Encoder:
    def __init__(self, config: InformerConfig, rng: torch.Generator, *, device):
        self.dtype = get_ttnn_dtype(config.dtype)
        compute_dtype = get_compute_dtype(config, self.dtype)
        self.layers = [EncoderLayer(config, rng, device=device, dtype=self.dtype) for _ in range(config.e_layers)]
        self.distil = config.distil
        self.use_conv_distil = self.distil.enabled and self.distil.use_conv
        if self.use_conv_distil:
            output_memory_config = ttnn.L1_MEMORY_CONFIG if config.use_l1 else None
            self.conv_layers = [
                (
                    ConvDistillLayer(
                        config.d_model,
                        rng,
                        device=device,
                        dtype=self.dtype,
                        output_memory_config=output_memory_config,
                    )
                    if i < len(self.layers) - 1
                    else None
                )
                for i in range(len(self.layers))
            ]
            self.distil_norm = None
        else:
            self.conv_layers = None
            self.distil_norm = (
                LayerNorm(config.d_model, rng, device=device, dtype=self.dtype, compute_dtype=compute_dtype)
                if self.distil.enabled
                else None
            )
        self.mask_value = config.attn_mask_value

    def __call__(
        self,
        x: ttnn.Tensor,
        mask: ttnn.Tensor | None,
    ) -> tuple[ttnn.Tensor, int]:
        valid_length = x.shape[1]
        for i, layer in enumerate(self.layers):
            x = layer(x, mask, valid_length)
            if self.use_conv_distil and self.conv_layers is not None and self.conv_layers[i] is not None:
                x = self.conv_layers[i](x)
                valid_length = max(
                    1,
                    (valid_length + 2 * self.distil.padding - self.distil.kernel_size) // self.distil.stride + 1,
                )
                x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
            elif self.distil.enabled and i < len(self.layers) - 1:
                x = max_pool1d(
                    x,
                    kernel=self.distil.kernel_size,
                    stride=self.distil.stride,
                    padding=self.distil.padding,
                    dtype=self.dtype,
                )
                valid_length = max(
                    1,
                    (valid_length + 2 * self.distil.padding - self.distil.kernel_size) // self.distil.stride + 1,
                )
                x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
                x = self.distil_norm(x)
        return x, valid_length


class DecoderLayer:
    def __init__(self, config: InformerConfig, rng: torch.Generator, *, device, dtype: ttnn.DataType):
        memory_config = ttnn.L1_MEMORY_CONFIG if config.use_l1 else None
        compute_dtype = get_compute_dtype(config, dtype)
        weight_dtype = compute_dtype if config.hf_compat else dtype
        self.self_attn = MultiHeadAttention(
            config.d_model,
            config.n_heads,
            config.dropout,
            rng,
            device=device,
            dtype=dtype,
            prob_sparse=False,
            factor=config.factor,
            mask_value=config.attn_mask_value,
            use_sdpa=config.use_sdpa,
            random_sampling=config.hf_compat,
            is_decoder=True,
            compute_dtype=compute_dtype,
            memory_config=memory_config,
            use_sharded=config.use_sharded,
            shard_strategy=config.shard_strategy,
        )
        self.cross_attn = MultiHeadAttention(
            config.d_model,
            config.n_heads,
            config.dropout,
            rng,
            device=device,
            dtype=dtype,
            prob_sparse=False,
            mask_value=config.attn_mask_value,
            use_sdpa=config.use_sdpa,
            random_sampling=False,
            is_decoder=False,
            compute_dtype=compute_dtype,
            memory_config=memory_config,
            use_sharded=config.use_sharded,
            shard_strategy=config.shard_strategy,
        )
        ffn_dim = config.decoder_ffn_dim if config.decoder_ffn_dim is not None else config.d_ff
        self.ffn = FeedForward(
            config.d_model,
            ffn_dim,
            config.dropout,
            rng,
            device=device,
            dtype=dtype,
            compute_dtype=compute_dtype,
            weight_dtype=weight_dtype,
            memory_config=memory_config,
            use_sharded=config.use_sharded,
            shard_strategy=config.shard_strategy,
        )
        self.norm1 = LayerNorm(config.d_model, rng, device=device, dtype=dtype, compute_dtype=compute_dtype)
        self.norm2 = LayerNorm(config.d_model, rng, device=device, dtype=dtype, compute_dtype=compute_dtype)
        self.norm3 = LayerNorm(config.d_model, rng, device=device, dtype=dtype, compute_dtype=compute_dtype)

    def __call__(
        self,
        x: ttnn.Tensor,
        enc_out: ttnn.Tensor,
        self_mask: ttnn.Tensor | None,
        cross_mask: ttnn.Tensor | None,
        enc_valid_len: int,
    ) -> ttnn.Tensor:
        attn1 = self.self_attn(
            x,
            x,
            x,
            self_mask,
            q_valid_len=x.shape[1],
            k_valid_len=x.shape[1],
        )
        x = self.norm1(x + attn1)

        attn2 = self.cross_attn(
            x,
            enc_out,
            enc_out,
            cross_mask,
            q_valid_len=x.shape[1],
            k_valid_len=enc_valid_len,
        )
        x = self.norm2(x + attn2)

        ff = self.ffn(x)
        x = self.norm3(x + ff)
        return x

    def forward_streaming(
        self,
        x: ttnn.Tensor,
        enc_out: ttnn.Tensor,
        enc_valid_len: int,
        *,
        cache: dict[str, ttnn.Tensor | int],
        cross_mask: ttnn.Tensor | None = None,
    ) -> tuple[ttnn.Tensor, dict[str, ttnn.Tensor | int]]:
        attn1, cache = self.self_attn.call_with_cache(
            x,
            x,
            x,
            None,
            kv_cache=cache,
            q_valid_len=x.shape[1],
            k_valid_len=x.shape[1],
            is_causal=True,
        )
        x = self.norm1(x + attn1)

        attn2 = self.cross_attn(
            x,
            enc_out,
            enc_out,
            cross_mask,
            q_valid_len=x.shape[1],
            k_valid_len=enc_valid_len,
        )
        x = self.norm2(x + attn2)

        ff = self.ffn(x)
        x = self.norm3(x + ff)
        return x, cache


class Decoder:
    def __init__(self, config: InformerConfig, rng: torch.Generator, *, device):
        self.dtype = get_ttnn_dtype(config.dtype)
        self.layers = [DecoderLayer(config, rng, device=device, dtype=self.dtype) for _ in range(config.d_layers)]

    def __call__(
        self,
        x: ttnn.Tensor,
        enc_out: ttnn.Tensor,
        self_mask: ttnn.Tensor | None,
        cross_mask: ttnn.Tensor | None,
        enc_valid_len: int,
    ) -> ttnn.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(
                x,
                enc_out,
                self_mask,
                cross_mask,
                enc_valid_len,
            )
        return x

    def forward_streaming(
        self,
        x: ttnn.Tensor,
        enc_out: ttnn.Tensor,
        enc_valid_len: int,
        *,
        chunk_size: int,
        caches: list[dict[str, ttnn.Tensor | int]] | None = None,
    ) -> tuple[ttnn.Tensor, list[dict[str, ttnn.Tensor | int]]]:
        if caches is None:
            caches = [{"k": None, "v": None, "valid_len": 0} for _ in self.layers]
        outputs: list[ttnn.Tensor] = []
        total_len = x.shape[1]
        for start in range(0, total_len, chunk_size):
            end = min(total_len, start + chunk_size)
            chunk = slice_sequence(x, dim=1, start=start, end=end)
            for i, layer in enumerate(self.layers):
                chunk, caches[i] = layer.forward_streaming(
                    chunk,
                    enc_out,
                    enc_valid_len,
                    cache=caches[i],
                )
            outputs.append(chunk)
        out = outputs[0] if len(outputs) == 1 else ttnn.concat(outputs, dim=1)
        return out, caches
