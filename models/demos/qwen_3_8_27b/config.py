# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.8-27B constants and the binding prefill spec.

Two sources, kept apart on purpose:

* ``Qwen38Config`` — everything readable from the checkpoint ``config.json`` (``text_config``).
  The values are pinned here as constants and ``tests/torch/test_config_and_reference.py`` asserts
  every one of them against the vendored ``configs/Qwen3.8-27B/config.json`` and, when available,
  the real checkpoint's ``config.json``.
* ``PrefillSpec`` — the binding prefill spec (mesh, chunk size, dataformats, PCC thresholds),
  read from ``$PREFILL_SPEC``. Where anything in this package conflicts with it, the spec wins.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
VENDORED_CONFIG = PACKAGE_DIR / "configs" / "Qwen3.8-27B" / "config.json"

FULL_ATTENTION = "full_attention"
LINEAR_ATTENTION = "linear_attention"


@dataclass(frozen=True)
class Qwen38Config:
    """Text-model constants of Qwen3.8-27B (``model_type == qwen3_5_text``)."""

    hidden_size: int = 5120
    intermediate_size: int = 17408
    num_hidden_layers: int = 64
    vocab_size: int = 248320
    rms_norm_eps: float = 1e-6
    hidden_act: str = "silu"
    tie_word_embeddings: bool = False
    full_attention_interval: int = 4
    # gated full attention (every 4th layer)
    num_attention_heads: int = 24
    num_key_value_heads: int = 4
    head_dim: int = 256
    attention_bias: bool = False
    attn_output_gate: bool = True
    rope_theta: float = 10_000_000.0
    partial_rotary_factor: float = 0.25
    mrope_section: tuple = (11, 11, 10)
    mrope_interleaved: bool = True
    max_position_embeddings: int = 262144
    # Gated DeltaNet (linear attention, the other 3 of every 4 layers)
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 48
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    layer_types: tuple = field(default=None)

    def __post_init__(self):
        if self.layer_types is None:
            types = tuple(
                FULL_ATTENTION if (i + 1) % self.full_attention_interval == 0 else LINEAR_ATTENTION
                for i in range(self.num_hidden_layers)
            )
            object.__setattr__(self, "layer_types", types)

    # ---- derived ----
    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def linear_key_dim(self) -> int:
        return self.linear_num_key_heads * self.linear_key_head_dim

    @property
    def linear_value_dim(self) -> int:
        return self.linear_num_value_heads * self.linear_value_head_dim

    @property
    def conv_dim(self) -> int:
        return 2 * self.linear_key_dim + self.linear_value_dim

    @property
    def full_attention_layers(self) -> list[int]:
        return [i for i, t in enumerate(self.layer_types) if t == FULL_ATTENTION]

    def is_full_attention(self, layer_idx: int) -> bool:
        return self.layer_types[layer_idx] == FULL_ATTENTION

    def reduced(self, **overrides) -> "Qwen38Config":
        """A reduced copy for diagnostics / random-weight tests. Never the graded result."""
        kw = {k: getattr(self, k) for k in self.__dataclass_fields__ if k != "layer_types"}
        kw.update(overrides)
        return Qwen38Config(**kw)

    @classmethod
    def from_hf_json(cls, path) -> "Qwen38Config":
        cfg = json.loads(Path(path).read_text())
        tc = cfg.get("text_config", cfg)
        rp = tc["rope_parameters"]
        return cls(
            hidden_size=tc["hidden_size"],
            intermediate_size=tc["intermediate_size"],
            num_hidden_layers=tc["num_hidden_layers"],
            vocab_size=tc["vocab_size"],
            rms_norm_eps=tc["rms_norm_eps"],
            hidden_act=tc["hidden_act"],
            tie_word_embeddings=cfg.get("tie_word_embeddings", tc.get("tie_word_embeddings", False)),
            full_attention_interval=tc["full_attention_interval"],
            num_attention_heads=tc["num_attention_heads"],
            num_key_value_heads=tc["num_key_value_heads"],
            head_dim=tc["head_dim"],
            attention_bias=tc["attention_bias"],
            attn_output_gate=tc["attn_output_gate"],
            rope_theta=float(rp["rope_theta"]),
            partial_rotary_factor=rp["partial_rotary_factor"],
            mrope_section=tuple(rp["mrope_section"]),
            mrope_interleaved=rp["mrope_interleaved"],
            max_position_embeddings=tc["max_position_embeddings"],
            linear_num_key_heads=tc["linear_num_key_heads"],
            linear_num_value_heads=tc["linear_num_value_heads"],
            linear_key_head_dim=tc["linear_key_head_dim"],
            linear_value_head_dim=tc["linear_value_head_dim"],
            linear_conv_kernel_dim=tc["linear_conv_kernel_dim"],
            layer_types=tuple(tc["layer_types"]),
        )


QWEN38 = Qwen38Config()


# ---------------------------------------------------------------------------------------------
# Binding prefill spec
# ---------------------------------------------------------------------------------------------
_DEFAULT_SPEC = {
    "model_name": "qwen_3_8_27b",
    "target_hw": "bh_galaxy",
    "parallelism": {"tp": 4, "sp": 8},
    "shapes": {"max_seq_len": 262144, "chunk_size": 5120},
    "dataformats": {
        "activations": {"default": "bfloat16"},
        "kv_cache": {"default": "bfloat8_b"},
        "weights": {"default": "bfloat8_b", "attention": "", "mlp": {"up": "", "gate": "", "down": ""}},
    },
    "acceptance": {"pcc_target": 0.99, "pcc_lower_bound": 0.87},
}


@dataclass(frozen=True)
class PrefillSpec:
    raw: dict
    path: str | None

    @classmethod
    def load(cls, path: str | None = None) -> "PrefillSpec":
        path = path or os.environ.get("PREFILL_SPEC")
        if path:
            return cls(json.loads(Path(path).read_text()), path)
        return cls(_DEFAULT_SPEC, None)

    @property
    def tp(self) -> int:
        return int(self.raw["parallelism"]["tp"])

    @property
    def sp(self) -> int:
        return int(self.raw["parallelism"]["sp"])

    @property
    def mesh_shape(self) -> tuple[int, int]:
        # rows = SP, cols = TP (the galaxy is 8x4)
        return (self.sp, self.tp)

    @property
    def target_hw(self) -> str:
        return self.raw["target_hw"]

    @property
    def max_seq_len(self) -> int:
        return int(self.raw["shapes"]["max_seq_len"])

    @property
    def chunk_size(self) -> int:
        return int(self.raw["shapes"]["chunk_size"])

    @property
    def pcc_target(self) -> float:
        return float(self.raw["acceptance"]["pcc_target"])

    @property
    def pcc_lower_bound(self) -> float:
        return float(self.raw["acceptance"]["pcc_lower_bound"])

    def _df(self, group: str, sub: str | None = None, leaf: str | None = None) -> str:
        g = self.raw["dataformats"][group]
        val = ""
        if sub is not None:
            s = g.get(sub, "")
            val = (
                (s.get(leaf, "") if isinstance(s, dict) else s)
                if leaf is not None
                else (s if isinstance(s, str) else "")
            )
        return val or g["default"]

    # resolved dtype names (empty override -> the group's default)
    @property
    def activation_dtype(self) -> str:
        return self._df("activations")

    @property
    def kv_cache_dtype(self) -> str:
        return self._df("kv_cache")

    @property
    def weight_dtype_default(self) -> str:
        return self._df("weights")

    @property
    def weight_dtype_attention(self) -> str:
        return self._df("weights", "attention")

    def weight_dtype_mlp(self, which: str) -> str:
        return self._df("weights", "mlp", which)

    def validate(self):
        a = 32 * self.sp
        assert self.max_seq_len % a == 0, f"max_seq_len {self.max_seq_len} % (32*sp) != 0"
        assert self.chunk_size % a == 0, f"chunk_size {self.chunk_size} % (32*sp) != 0"
        assert self.target_hw == "bh_galaxy", f"this package targets bh_galaxy, spec says {self.target_hw}"
        return self


def ttnn_dtype(name: str):
    import ttnn

    return {
        "bfloat16": ttnn.bfloat16,
        "bfloat8_b": ttnn.bfloat8_b,
        "bfloat4_b": ttnn.bfloat4_b,
        "float32": ttnn.float32,
    }[name]
