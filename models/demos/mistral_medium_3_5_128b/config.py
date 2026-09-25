# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Model constants, prefill spec and resolved dataformats for Mistral-Medium-3.5-128B prefill.

``MistralMediumConfig`` mirrors ``text_config`` of the vendored ``configs/config.json`` (the Mistral3
checkpoint's language model; the vision tower is out of scope). ``from_json`` refuses a config that
leaves the envelope this package implements: dense GQA, SiLU-gated MLP, YaRN RoPE with an identity
llama-4 query scale, no sliding window, untied embeddings and per-tensor fp8 weights.

The prefill spec (``PREFILL_SPEC``, else the vendored snapshot) is binding: mesh, chunk size,
dataformats and the two PCC thresholds all come from it.
"""

import dataclasses
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

PACKAGE_DIR = Path(__file__).resolve().parent
VENDORED_CONFIG = PACKAGE_DIR / "configs" / "config.json"
VENDORED_SPEC = PACKAGE_DIR / "configs" / "prefill_spec.json"


@dataclass(frozen=True)
class MistralMediumConfig:
    vocab_size: int = 131072
    hidden_size: int = 12288
    intermediate_size: int = 28672
    num_hidden_layers: int = 88
    num_attention_heads: int = 96
    num_key_value_heads: int = 8
    head_dim: int = 128
    rms_norm_eps: float = 1e-5
    max_position_embeddings: int = 262144
    hidden_act: str = "silu"
    tie_word_embeddings: bool = False
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 11
    # RoPE: YaRN (transformers modeling_rope_utils._compute_yarn_parameters branch).
    rope_type: str = "yarn"
    rope_theta: float = 1_000_000.0
    rope_factor: float = 64.0
    original_max_position_embeddings: int = 4096
    beta_fast: float = 4.0
    beta_slow: float = 1.0
    mscale: float = 1.0
    mscale_all_dim: float = 0.0
    rope_truncate: bool = True
    # Ministral3Attention scales q by 1 + beta * log(1 + floor(pos / original_max)); beta 0 = identity.
    llama_4_scaling_beta: float = 0.0
    # Checkpoint quantization: fp8_e4m3 weights with one rank-0 weight_scale_inv per tensor.
    quant_method: str = "fp8"
    weight_block_size: Optional[int] = None

    @property
    def num_kv_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def rope_parameters(self) -> dict:
        return {
            "rope_type": self.rope_type,
            "rope_theta": self.rope_theta,
            "factor": self.rope_factor,
            "original_max_position_embeddings": self.original_max_position_embeddings,
            "beta_fast": self.beta_fast,
            "beta_slow": self.beta_slow,
            "mscale": self.mscale,
            "mscale_all_dim": self.mscale_all_dim,
            "llama_4_scaling_beta": self.llama_4_scaling_beta,
        }

    def reduced(self, **overrides) -> "MistralMediumConfig":
        """A reduced copy for diagnostics (depth/width cut). Never the graded configuration."""
        return dataclasses.replace(self, **overrides)

    @classmethod
    def from_json(cls, path=VENDORED_CONFIG) -> "MistralMediumConfig":
        with open(path) as f:
            raw = json.load(f)
        tc = raw["text_config"]
        rp = tc["rope_parameters"]
        qc = raw.get("quantization_config") or {}
        cfg = cls(
            vocab_size=tc["vocab_size"],
            hidden_size=tc["hidden_size"],
            intermediate_size=tc["intermediate_size"],
            num_hidden_layers=tc["num_hidden_layers"],
            num_attention_heads=tc["num_attention_heads"],
            num_key_value_heads=tc["num_key_value_heads"],
            head_dim=tc["head_dim"],
            rms_norm_eps=tc["rms_norm_eps"],
            max_position_embeddings=tc["max_position_embeddings"],
            hidden_act=tc["hidden_act"],
            tie_word_embeddings=tc["tie_word_embeddings"],
            bos_token_id=tc["bos_token_id"],
            eos_token_id=tc["eos_token_id"],
            pad_token_id=tc["pad_token_id"],
            rope_type=rp["rope_type"],
            rope_theta=rp["rope_theta"],
            rope_factor=rp["factor"],
            original_max_position_embeddings=rp["original_max_position_embeddings"],
            beta_fast=rp["beta_fast"],
            beta_slow=rp["beta_slow"],
            mscale=rp["mscale"],
            mscale_all_dim=rp["mscale_all_dim"],
            rope_truncate=rp.get("truncate", True),
            llama_4_scaling_beta=rp.get("llama_4_scaling_beta", 0.0) or 0.0,
            quant_method=qc.get("quant_method", ""),
            weight_block_size=qc.get("weight_block_size"),
        )
        # The envelope this package implements; anything else must fail here, not as a PCC drop.
        assert tc["model_type"] == "ministral3", tc["model_type"]
        assert tc.get("sliding_window") is None, "sliding-window attention is not implemented"
        assert cfg.hidden_act == "silu", cfg.hidden_act
        assert cfg.rope_type == "yarn", cfg.rope_type
        assert cfg.llama_4_scaling_beta == 0, "device path implements the identity llama-4 query scale only"
        assert not cfg.tie_word_embeddings
        assert cfg.num_attention_heads % cfg.num_key_value_heads == 0
        assert cfg.quant_method == "fp8" and cfg.weight_block_size is None, "expects per-tensor fp8 weights"
        return cfg


def load_prefill_spec(path=None) -> dict:
    """The binding prefill spec: ``path``, else ``PREFILL_SPEC``, else the vendored snapshot."""
    path = path or os.environ.get("PREFILL_SPEC") or VENDORED_SPEC
    with open(path) as f:
        return json.load(f)


def pcc_thresholds(spec=None):
    """``(pcc_target, pcc_lower_bound)`` from the spec's acceptance block."""
    acc = (spec or load_prefill_spec())["acceptance"]
    return float(acc["pcc_target"]), float(acc["pcc_lower_bound"])


def resolve_dataformats(spec=None) -> dict:
    """Spec dataformats with empty overrides inheriting their group default (dtype names)."""
    df = (spec or load_prefill_spec())["dataformats"]
    w = df["weights"]
    w_default = w["default"]
    mlp = w.get("mlp") or {}
    return {
        "activations": df["activations"]["default"],
        "kv_cache": df["kv_cache"]["default"],
        "weights_default": w_default,
        "attention": w.get("attention") or w_default,
        "mlp_gate": mlp.get("gate") or w_default,
        "mlp_up": mlp.get("up") or w_default,
        "mlp_down": mlp.get("down") or w_default,
    }


def ttnn_dtype(name: str):
    import ttnn

    return {
        "bfloat16": ttnn.bfloat16,
        "bfloat8_b": ttnn.bfloat8_b,
        "bfloat4_b": ttnn.bfloat4_b,
        "float32": ttnn.float32,
    }[name]
