# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch

import ttnn

TILE_SIZE = 32


def align_to_tile(value: int) -> int:
    return int(math.ceil(value / TILE_SIZE) * TILE_SIZE)


@dataclass
class DistilConfig:
    enabled: bool = True
    use_conv: bool = False
    kernel_size: int = 3
    stride: int = 2
    padding: int = 1


@dataclass
class InformerConfig:
    # Input / output
    enc_in: int = 7
    dec_in: int = 7
    c_out: int = 7
    seq_len: int = 96
    label_len: int = 48
    pred_len: int = 24

    # Model dims
    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 2048

    # Depth
    e_layers: int = 2
    d_layers: int = 1

    # Attention / sparsity
    factor: int = 5
    dropout: float = 0.0

    # Positional / temporal embedding
    embed_dim: Optional[int] = None
    time_feature_dim: int = 4

    # TTNN runtime
    device_id: int = 0
    dtype: str = "bfloat16"
    attn_mask_value: float = -1e4
    use_l1: bool = True
    use_sharded: bool = True
    shard_strategy: str = "height"
    use_sdpa: bool = True
    use_program_cache: bool = True
    use_trace: bool = True

    # Distilling
    distil: DistilConfig = field(default_factory=DistilConfig)

    # HuggingFace compatibility
    hf_compat: bool = False
    hf_compute_dtype: Optional[str] = None
    feature_size: Optional[int] = None
    input_size: Optional[int] = None
    context_length: Optional[int] = None
    prediction_length: Optional[int] = None
    lags_sequence: tuple[int, ...] = field(default_factory=tuple)
    num_time_features: int = 0
    num_dynamic_real_features: int = 0
    num_static_real_features: int = 0
    num_static_categorical_features: int = 0
    cardinality: tuple[int, ...] = field(default_factory=tuple)
    embedding_dimension: tuple[int, ...] = field(default_factory=tuple)
    attention_type: str = "prob"
    attention_dropout: float = 0.0
    activation_dropout: float = 0.0
    encoder_ffn_dim: Optional[int] = None
    decoder_ffn_dim: Optional[int] = None
    distribution_output: str = "student_t"
    scaling: str = "mean"
    minimum_scale: float = 1e-10
    default_scale: Optional[float] = None

    def __post_init__(self):
        if self.embed_dim is None:
            self.embed_dim = self.d_model
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads for head splits.")
        if self.d_model % TILE_SIZE != 0:
            raise ValueError("d_model must be a multiple of 32 for TTNN tile matmul.")
        if self.hf_compat:
            if self.context_length is None:
                self.context_length = self.seq_len
            if self.prediction_length is None:
                self.prediction_length = self.pred_len
            if self.encoder_ffn_dim is None:
                self.encoder_ffn_dim = self.d_ff
            if self.decoder_ffn_dim is None:
                self.decoder_ffn_dim = self.d_ff
            if self.feature_size is None and self.input_size is not None and self.lags_sequence:
                static_dim = self.num_static_real_features + sum(self.embedding_dimension) + 2
                time_dim = self.num_time_features + self.num_dynamic_real_features
                self.feature_size = self.input_size * len(self.lags_sequence) + time_dim + static_dim


def get_ttnn_dtype(dtype: str) -> ttnn.DataType:
    if dtype == "bfloat16":
        return ttnn.bfloat16
    if dtype == "float32":
        return ttnn.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def get_compute_dtype(config: "InformerConfig", base_dtype: ttnn.DataType) -> ttnn.DataType:
    if not config.hf_compat:
        return base_dtype
    if config.hf_compute_dtype is not None:
        return get_ttnn_dtype(config.hf_compute_dtype)
    return ttnn.float32


def get_shard_strategy(name: str) -> ttnn.ShardStrategy:
    if name == "height":
        return ttnn.ShardStrategy.HEIGHT
    if name == "width":
        return ttnn.ShardStrategy.WIDTH
    if name == "block":
        return ttnn.ShardStrategy.BLOCK
    raise ValueError(f"Unsupported shard strategy: {name}")


def create_sharded_memory_config(shape: tuple[int, ...], *, device, strategy: ttnn.ShardStrategy) -> ttnn.MemoryConfig:
    shape = tuple(int(dim) for dim in shape)
    grid_size = device.compute_with_storage_grid_size()
    if len(shape) < 2:
        raise ValueError(f"Sharded config expects at least 2 dims, got {shape}.")
    if len(shape) == 2:
        height = shape[0]
        width = shape[1]
    else:
        height = 1
        for dim in shape[:-1]:
            height *= dim
        width = shape[-1]
    if strategy in (ttnn.ShardStrategy.HEIGHT, ttnn.ShardStrategy.WIDTH):
        max_cores = grid_size.x * grid_size.y
        if strategy == ttnn.ShardStrategy.HEIGHT:
            target = int(math.ceil(height / max_cores))
            shard_height = int(math.ceil(target / TILE_SIZE) * TILE_SIZE)
            num_cores = max(1, int(math.ceil(height / shard_height)))
            shard_shape = (shard_height, width)
        else:
            target = int(math.ceil(width / max_cores))
            shard_width = int(math.ceil(target / TILE_SIZE) * TILE_SIZE)
            num_cores = max(1, int(math.ceil(width / shard_width)))
            shard_shape = (height, shard_width)
        core_grid = ttnn.num_cores_to_corerangeset(num_cores, grid_size, row_wise=True)
        return ttnn.create_sharded_memory_config(
            shard_shape,
            core_grid=core_grid,
            strategy=strategy,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    core_grid = ttnn.CoreGrid(y=grid_size.y, x=grid_size.x)
    return ttnn.create_sharded_memory_config(
        shape,
        core_grid=core_grid,
        strategy=strategy,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )


def get_shard_shape(x: ttnn.Tensor) -> tuple[int, ...]:
    if hasattr(x, "padded_shape"):
        return tuple(int(dim) for dim in x.padded_shape)
    return tuple(int(dim) for dim in x.shape)


def informer_config_from_hf(
    hf_config,
    *,
    device_id: int = 0,
    dtype: str = "bfloat16",
    hf_mask_value: Optional[float] = None,
    hf_compute_dtype: Optional[str] = None,
) -> InformerConfig:
    input_size = int(getattr(hf_config, "input_size", None) or 1)
    context_length = int(
        getattr(hf_config, "context_length", None) or getattr(hf_config, "prediction_length", None) or 24
    )
    prediction_length = int(getattr(hf_config, "prediction_length", None) or 24)
    d_model = int(hf_config.d_model)
    encoder_ffn_dim = int(getattr(hf_config, "encoder_ffn_dim", None) or d_model)
    decoder_ffn_dim = int(getattr(hf_config, "decoder_ffn_dim", None) or d_model)
    distil_enabled = bool(getattr(hf_config, "distil", True))
    mask_value = float(torch.finfo(torch.float32).min) if hf_mask_value is None else float(hf_mask_value)

    return InformerConfig(
        enc_in=input_size,
        dec_in=input_size,
        c_out=input_size,
        seq_len=context_length,
        label_len=max(1, context_length // 2),
        pred_len=prediction_length,
        d_model=d_model,
        n_heads=int(getattr(hf_config, "encoder_attention_heads", hf_config.decoder_attention_heads)),
        d_ff=encoder_ffn_dim,
        e_layers=int(hf_config.encoder_layers),
        d_layers=int(hf_config.decoder_layers),
        time_feature_dim=int(getattr(hf_config, "num_time_features", None) or 0),
        device_id=device_id,
        dtype=dtype,
        factor=int(hf_config.factor),
        dropout=0.0,
        distil=DistilConfig(enabled=distil_enabled, use_conv=distil_enabled),
        attention_type=str(hf_config.attention_type),
        attention_dropout=0.0,
        activation_dropout=0.0,
        attn_mask_value=mask_value,
        encoder_ffn_dim=encoder_ffn_dim,
        decoder_ffn_dim=decoder_ffn_dim,
        distribution_output=str(hf_config.distribution_output),
        scaling=str(hf_config.scaling),
        default_scale=getattr(hf_config, "default_scale", None),
        hf_compat=True,
        hf_compute_dtype=hf_compute_dtype,
        feature_size=int(hf_config.feature_size),
        input_size=input_size,
        context_length=int(getattr(hf_config, "context_length", None) or context_length),
        prediction_length=int(getattr(hf_config, "prediction_length", None) or prediction_length),
        lags_sequence=tuple(getattr(hf_config, "lags_sequence", None) or ()),
        num_time_features=int(getattr(hf_config, "num_time_features", None) or 0),
        num_dynamic_real_features=int(getattr(hf_config, "num_dynamic_real_features", None) or 0),
        num_static_real_features=int(getattr(hf_config, "num_static_real_features", None) or 0),
        num_static_categorical_features=int(getattr(hf_config, "num_static_categorical_features", None) or 0),
        cardinality=tuple(getattr(hf_config, "cardinality", None) or ()),
        embedding_dimension=tuple(getattr(hf_config, "embedding_dimension", None) or ()),
    )
