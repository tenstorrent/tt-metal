# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Configuration and TTNN runtime helpers for the LLVC bring-up."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import ttnn

TILE_SIZE = 32


def align_to_tile(value: int) -> int:
    return int(math.ceil(value / TILE_SIZE) * TILE_SIZE)


@dataclass
class ConvNetConfig:
    """Cached (streaming) convolution prenet config."""

    convnet_prenet: bool = True
    out_channels: tuple[int, ...] = field(default_factory=lambda: tuple([1] * 12))
    kernel_sizes: tuple[int, ...] = field(default_factory=lambda: tuple([3] * 12))
    dilations: tuple[int, ...] = field(default_factory=lambda: tuple([1] * 12))
    dropout: float = 0.5
    combine_residuals: str | None = None
    skip_connection: str = "add"
    use_residual_blocks: bool = True


@dataclass
class LLVCConfig:
    """LLVC generator config. Defaults match the official KoeAI checkpoint."""

    # Architecture (KoeAI experiments/llvc/config.json ``model_params``)
    label_len: int = 1
    L: int = 16
    enc_dim: int = 512
    num_enc_layers: int = 8
    dec_dim: int = 256
    num_dec_layers: int = 1
    dec_buf_len: int = 13
    dec_chunk_size: int = 13
    out_buf_len: int = 4
    nhead: int = 8
    use_pos_enc: bool = True
    skip_connection: bool = True
    proj: bool = True
    lookahead: bool = True
    decoder_dropout: float = 0.0  # inference: dropout is identity

    convnet: ConvNetConfig = field(default_factory=ConvNetConfig)

    # Audio
    sample_rate: int = 16000

    # TTNN runtime
    device_id: int = 0
    dtype: str = "bfloat16"
    math_fidelity: str = "LoFi"
    use_l1: bool = True
    use_sharded: bool = True
    use_program_cache: bool = True
    use_trace: bool = True  # capture forward_chunk once and replay it per streaming chunk
    attn_mask_value: float = -1e4

    def __post_init__(self):
        if self.dec_dim % self.nhead != 0:
            raise ValueError("dec_dim must be divisible by nhead for head splits.")

    @property
    def kernel_size_in_conv(self) -> int:
        return 3 * self.L if self.lookahead else self.L

    @property
    def enc_buf_len(self) -> int:
        """Total context length held by the dilated causal encoder."""
        return (3 - 1) * (2**self.num_enc_layers - 1)


def get_ttnn_dtype(dtype: str) -> ttnn.DataType:
    if dtype == "bfloat16":
        return ttnn.bfloat16
    if dtype == "float32":
        return ttnn.float32
    if dtype == "bfloat8_b":
        return ttnn.bfloat8_b
    raise ValueError(f"Unsupported dtype: {dtype}")


def get_math_fidelity(name: str) -> ttnn.MathFidelity:
    mapping = {
        "LoFi": ttnn.MathFidelity.LoFi,
        "HiFi2": ttnn.MathFidelity.HiFi2,
        "HiFi3": ttnn.MathFidelity.HiFi3,
        "HiFi4": ttnn.MathFidelity.HiFi4,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported math fidelity: {name}")
    return mapping[name]


def default_memory_config(config: LLVCConfig) -> ttnn.MemoryConfig:
    return ttnn.L1_MEMORY_CONFIG if config.use_l1 else ttnn.DRAM_MEMORY_CONFIG


def llvc_config_from_json(model_params: dict, *, device_id: int = 0, dtype: str = "bfloat16") -> LLVCConfig:
    """Build an ``LLVCConfig`` from a KoeAI ``config.json`` ``model_params`` dict."""
    convnet_raw = model_params.get("convnet_config") or {}
    convnet = ConvNetConfig(
        convnet_prenet=bool(convnet_raw.get("convnet_prenet", True)),
        out_channels=tuple(convnet_raw.get("out_channels", [1] * 12)),
        kernel_sizes=tuple(convnet_raw.get("kernel_sizes", [3] * 12)),
        dilations=tuple(convnet_raw.get("dilations", [1] * 12)),
        dropout=float(convnet_raw.get("dropout", 0.5)),
        combine_residuals=convnet_raw.get("combine_residuals", None),
        skip_connection=convnet_raw.get("skip_connection", "add"),
        use_residual_blocks=bool(convnet_raw.get("use_residual_blocks", True)),
    )
    return LLVCConfig(
        label_len=int(model_params.get("label_len", 1)),
        L=int(model_params.get("L", 16)),
        enc_dim=int(model_params.get("enc_dim", 512)),
        num_enc_layers=int(model_params.get("num_enc_layers", 8)),
        dec_dim=int(model_params.get("dec_dim", 256)),
        num_dec_layers=int(model_params.get("num_dec_layers", 1)),
        dec_buf_len=int(model_params.get("dec_buf_len", 13)),
        dec_chunk_size=int(model_params.get("dec_chunk_size", 13)),
        out_buf_len=int(model_params.get("out_buf_len", 4)),
        use_pos_enc=bool(model_params.get("use_pos_enc", True)),
        skip_connection=bool(model_params.get("skip_connection", True)),
        proj=bool(model_params.get("proj", True)),
        lookahead=bool(model_params.get("lookahead", True)),
        decoder_dropout=0.0,
        convnet=convnet,
        device_id=device_id,
        dtype=dtype,
    )
