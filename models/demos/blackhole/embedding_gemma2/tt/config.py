# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Config dataclasses for the EmbeddingGemma 2 text tower.

Every dataclass is built from the HF `EmbeddingGemma2TextConfig` (or a plain dict with the same
fields, e.g. `config.json["text_config"]`), so the TT modules never import transformers.

Shapes follow `google/embeddinggemma-2` (hidden 512, 24 layers, 4 query heads, sliding layers
head_dim 256 / 2 KV heads / rope theta 1e4, full layers 5/11/17/23 head_dim 512 / 1 KV head /
rope theta 1e6, sliding radius 512, PLE width 512, output 768).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import ttnn

SLIDING = "sliding_attention"
FULL = "full_attention"

# Physical sequence buckets. A logical length L runs in the smallest bucket >= L; the
# padding keys are masked and padded rows are never read back.
SEQ_BUCKETS = (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768)

# Additive mask value for keys a query must not see (finite: -inf * 0 style NaNs are avoided).
MASK_NEG = -1.0e9


def _get(cfg: Any, name: str, default=None):
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def _per_layer_override(cfg: Any, layer_idx: int) -> dict:
    """Return the `per_layer_config` override for `layer_idx` ({} when the layer uses the defaults).

    HF stores it as {"05": {...}} in config.json and as int-keyed sub-configs after __post_init__.
    """
    plc = _get(cfg, "per_layer_config", None) or {}
    for key in (layer_idx, f"{layer_idx:02d}", str(layer_idx)):
        if isinstance(plc, dict) and key in plc:
            entry = plc[key]
            if isinstance(entry, dict):
                return entry
            return {"head_dim": _get(entry, "head_dim"), "num_key_value_heads": _get(entry, "num_key_value_heads")}
    if not isinstance(plc, dict):
        # HF sometimes exposes a list-like / mapping object indexed by layer
        try:
            entry = plc[layer_idx]
            return {"head_dim": _get(entry, "head_dim"), "num_key_value_heads": _get(entry, "num_key_value_heads")}
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"cannot read per_layer_config for layer {layer_idx}") from e
    return {}


def compute_kernel_config_hifi4():
    """HiFi4 + fp32 dest accumulation: functional-stage default for matmuls, norms and SDPA."""
    return ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


def compute_kernel_config_norm():
    return ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )


def default_weight_cache_dir() -> Path:
    root = os.environ.get("EMBEDDING_GEMMA2_CACHE", os.path.expanduser("~/.cache/ttnn/models/embedding_gemma2"))
    return Path(root)


@dataclass(frozen=True)
class RMSNormConfig:
    dim: int
    eps: float = 1e-6
    with_scale: bool = True  # False: HF `with_scale=False` (v_norm); weight is used as-is, not (1 + w)


@dataclass(frozen=True)
class AttentionConfig:
    layer_idx: int
    layer_type: str  # SLIDING or FULL
    hidden_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rope_theta: float
    sliding_window: int | None  # inclusive radius for sliding layers, None for full layers
    rms_norm_eps: float
    scaling: float = 1.0

    @property
    def q_width(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def kv_width(self) -> int:
        return self.num_kv_heads * self.head_dim


@dataclass(frozen=True)
class MLPConfig:
    hidden_size: int
    intermediate_size: int
    activation: str = "gelu_pytorch_tanh"


@dataclass(frozen=True)
class PerLayerInputConfig:
    hidden_size: int
    num_layers: int
    per_layer_dim: int
    rms_norm_eps: float

    @property
    def projection_scale(self) -> float:
        return self.hidden_size**-0.5


@dataclass(frozen=True)
class EncoderLayerConfig:
    layer_idx: int
    hidden_size: int
    rms_norm_eps: float
    attention: AttentionConfig
    mlp: MLPConfig
    per_layer_dim: int


@dataclass(frozen=True)
class TextModelConfig:
    vocab_size: int
    hidden_size: int
    num_layers: int
    embedding_dim: int
    pad_token_id: int
    rms_norm_eps: float
    max_position_embeddings: int
    sliding_window: int
    layer_types: tuple[str, ...]
    rope_theta: dict
    layers: tuple[EncoderLayerConfig, ...] = field(repr=False)
    per_layer_input: PerLayerInputConfig = field(repr=False)
    max_seq_len: int = 8192  # largest physical bucket this instance prepares RoPE tables for

    @property
    def embed_scale(self) -> float:
        return self.hidden_size**0.5

    @classmethod
    def from_hf(cls, hf_text_config: Any, max_seq_len: int = 8192) -> "TextModelConfig":
        cfg = hf_text_config
        num_layers = _get(cfg, "num_hidden_layers")
        hidden = _get(cfg, "hidden_size")
        eps = _get(cfg, "rms_norm_eps")
        layer_types = tuple(_get(cfg, "layer_types"))
        rope_params = _get(cfg, "rope_parameters")
        rope_theta = {}
        for lt in (SLIDING, FULL):
            p = rope_params[lt]
            if _get(p, "rope_type", "default") != "default":
                raise ValueError(f"unsupported rope_type for {lt}: {_get(p, 'rope_type')}")
            rope_theta[lt] = float(_get(p, "rope_theta"))
        if _get(cfg, "attention_bias", False):
            raise ValueError("attention_bias=True is not supported (embeddinggemma-2 has none)")
        if _get(cfg, "hidden_activation") != "gelu_pytorch_tanh":
            raise ValueError(f"unsupported hidden_activation {_get(cfg, 'hidden_activation')}")
        if max_seq_len not in SEQ_BUCKETS:
            raise ValueError(f"max_seq_len must be one of {SEQ_BUCKETS}, got {max_seq_len}")

        mlp = MLPConfig(hidden_size=hidden, intermediate_size=_get(cfg, "intermediate_size"))
        per_layer_dim = _get(cfg, "hidden_size_per_layer_input")
        layers = []
        for i in range(num_layers):
            lt = layer_types[i]
            override = _per_layer_override(cfg, i)
            head_dim = override.get("head_dim") or _get(cfg, "head_dim")
            n_kv = override.get("num_key_value_heads") or _get(cfg, "num_key_value_heads")
            attn = AttentionConfig(
                layer_idx=i,
                layer_type=lt,
                hidden_size=hidden,
                num_heads=_get(cfg, "num_attention_heads"),
                num_kv_heads=n_kv,
                head_dim=head_dim,
                rope_theta=rope_theta[lt],
                sliding_window=_get(cfg, "sliding_window") if lt == SLIDING else None,
                rms_norm_eps=eps,
            )
            layers.append(
                EncoderLayerConfig(
                    layer_idx=i,
                    hidden_size=hidden,
                    rms_norm_eps=eps,
                    attention=attn,
                    mlp=mlp,
                    per_layer_dim=per_layer_dim,
                )
            )
        ple = PerLayerInputConfig(
            hidden_size=hidden, num_layers=num_layers, per_layer_dim=per_layer_dim, rms_norm_eps=eps
        )
        return cls(
            vocab_size=_get(cfg, "vocab_size"),
            hidden_size=hidden,
            num_layers=num_layers,
            embedding_dim=_get(cfg, "embedding_dim"),
            pad_token_id=_get(cfg, "pad_token_id"),
            rms_norm_eps=eps,
            max_position_embeddings=_get(cfg, "max_position_embeddings"),
            sliding_window=_get(cfg, "sliding_window"),
            layer_types=layer_types,
            rope_theta=rope_theta,
            layers=tuple(layers),
            per_layer_input=ple,
            max_seq_len=max_seq_len,
        )

    def head_dim_for(self, layer_type: str) -> int:
        for lc in self.layers:
            if lc.attention.layer_type == layer_type:
                return lc.attention.head_dim
        raise KeyError(layer_type)


def bucket_for(seq_len: int, max_seq_len: int) -> int:
    """Smallest physical bucket >= seq_len (raises if seq_len exceeds max_seq_len)."""
    if seq_len < 1:
        raise ValueError(f"seq_len must be >= 1, got {seq_len}")
    for b in SEQ_BUCKETS:
        if b >= seq_len and b <= max_seq_len:
            return b
    raise ValueError(f"seq_len {seq_len} exceeds the supported max_seq_len {max_seq_len}")
