# SPDX-FileCopyrightText: © 2026 Abror Shopulatov

# SPDX-License-Identifier: Apache-2.0
#
# Model configuration, supported precisions, device options and precision
# policies for the amazon/chronos-2 TTNN port.
#
# Configuration-driven: every architectural dimension comes from the pinned
# checkpoint's config.json (`chronos_config` + T5-style fields). No hardcoded
# sizes.

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

# fp32 is the default. bf16 runs a mixed-precision graph (see
# PRECISION_POLICIES["bf16"]); bfp8_b stores linear weights as BFP8_B and saves
# memory without improving latency. See docs/validation.md for measurements.
SUPPORTED_PRECISIONS = ("fp32", "bf16", "bfp8_b")
DEFAULT_PRECISION = "fp32"

# ttnn dtype names per precision: (activations, linear weight matrices).
# Biases, RoPE tables, masks and the REG token follow the activation dtype.
# Under bf16 the executor additionally keeps the residual stream, the final
# norm and the quantile head in FP32 (see PRECISION_POLICIES["bf16"]).
PRECISION_DTYPES = {
    "bf16": ("bfloat16", "bfloat16"),
    "fp32": ("float32", "float32"),
    "bfp8_b": ("bfloat16", "bfloat8_b"),
}

# Options to pass to ttnn.open_device. 64 MiB of trace region covers every
# supported shape (batch <= 4, context <= 512, horizon <= 64).
DEVICE_OPTIONS = {"trace_region_size": 64 * 1024 * 1024}
DEVICE_OPTION_BOUNDS = {
    "trace_region_size": (0, 128 * 1024 * 1024),
    "l1_small_size": (0, 128 * 1024),
    "num_hw_cqs": (1, 2),
}


@dataclass(frozen=True)
class Chronos2Config:
    """All model dimensions, parsed from the checkpoint's config.json."""

    d_model: int
    d_ff: int
    d_kv: int
    num_heads: int
    num_layers: int
    layer_norm_epsilon: float
    dense_act_fn: str
    dropout_rate: float
    initializer_factor: float
    rope_theta: float
    # chronos_config:
    context_length: int
    input_patch_size: int
    input_patch_stride: int
    output_patch_size: int
    max_output_patches: int
    use_reg_token: bool
    use_arcsinh: bool
    quantiles: tuple
    time_encoding_scale: int
    # derived
    num_quantiles: int = field(init=False)
    input_feature_dim: int = field(init=False)  # 3 * input_patch_size

    def __post_init__(self):
        object.__setattr__(self, "num_quantiles", len(self.quantiles))
        object.__setattr__(self, "input_feature_dim", 3 * self.input_patch_size)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Chronos2Config":
        chronos = dict(raw["chronos_config"])
        context_length = int(chronos["context_length"])
        # Chronos2Model.__init__ defaults time_encoding_scale to context_length.
        tes = chronos.get("time_encoding_scale", context_length)
        return cls(
            d_model=int(raw["d_model"]),
            d_ff=int(raw["d_ff"]),
            d_kv=int(raw["d_kv"]),
            num_heads=int(raw["num_heads"]),
            num_layers=int(raw["num_layers"]),
            layer_norm_epsilon=float(raw["layer_norm_epsilon"]),
            dense_act_fn=str(raw["dense_act_fn"]),
            dropout_rate=float(raw.get("dropout_rate", 0.0)),
            initializer_factor=float(raw.get("initializer_factor", 1.0)),
            rope_theta=float(raw.get("rope_theta", 10000.0)),
            context_length=context_length,
            input_patch_size=int(chronos["input_patch_size"]),
            input_patch_stride=int(chronos["input_patch_stride"]),
            output_patch_size=int(chronos["output_patch_size"]),
            max_output_patches=int(chronos["max_output_patches"]),
            use_reg_token=bool(chronos.get("use_reg_token", True)),
            use_arcsinh=bool(chronos.get("use_arcsinh", True)),
            quantiles=tuple(float(q) for q in chronos["quantiles"]),
            time_encoding_scale=int(tes),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "Chronos2Config":
        with open(path) as f:
            return cls.from_dict(json.load(f))


def resolve_config(config: Any, weights_path: str | Path) -> Chronos2Config:
    """Accept an explicit dict, a path to config.json, or None (search near weights)."""
    if isinstance(config, Chronos2Config):
        return config
    if isinstance(config, Mapping):
        return Chronos2Config.from_dict(config)
    if isinstance(config, (str, Path)):
        return Chronos2Config.from_json(config)
    wp = Path(weights_path)
    candidates = [wp / "config.json", wp.parent / "config.json"] if wp.suffix else [wp / "config.json"]
    for c in candidates:
        if c.exists():
            return Chronos2Config.from_json(c)
    raise FileNotFoundError(f"could not locate config.json near {weights_path}")


def validate_device_options(opts: Mapping[str, int]) -> dict:
    """Bounds-check the options the caller opened the device with."""
    out = {}
    for key, value in opts.items():
        if key not in DEVICE_OPTION_BOUNDS:
            raise ValueError(f"unknown device option '{key}'")
        lo, hi = DEVICE_OPTION_BOUNDS[key]
        if not (lo <= value <= hi):
            raise ValueError(f"device option '{key}'={value} outside [{lo}, {hi}]")
        out[key] = value
    return out


# PRECISION_POLICIES documents what each precision actually runs: the storage
# and activation formats, plus every place where the graph deviates from them.

# Exceptions shared by every precision's graph (host math and manual ops).
_COMMON_EXCEPTIONS = [
    "instance-norm scaling, arcsinh, patch value prep and time encoding: host FP32",
    "attention softmax: manual stable max-sub/exp/sum/reciprocal computed in FP32 on device (more accurate than ttnn.softmax for the masked, non-tile-aligned sequence lengths used here)",
    "rms-norm: composed from ttnn mul/sum/rsqrt; the norm weight is folded (FP32 host math) into the following linear before upload",
    "RoPE rotate_half folded into q/k weight columns (FP32 host math) before upload",
    "quantile-head unscale (sinh/scale/loc): host FP32",
]

# The bf16 graph keeps the numerically sensitive parts in FP32 (see
# tt/executor.py, `_ACCURATE_BF16`). Plain bf16 lost accuracy on short,
# high-dynamic-range contexts; this policy costs about 11 % latency.
_BF16_EXCEPTIONS = [
    "matmul math: HiFi4 fidelity with fp32_dest_acc_en and packer_l1_acc on every encoder/embedding linear and on q@k^T and probs@v",
    "fp32 residual stream: embedding outputs, REG token and every residual add stay FP32; attention-o, fused group-attention and FFN-wo linears emit FP32; the embedding output/residual linears (with bias) run bf16 and are typecast to FP32",
    "rms-norm: computed on the FP32 residual stream, then typecast to bf16 as the matmul input; matmul inputs (normed tokens, q/k/v, probs, FFN hidden) are bf16",
    "final rms-norm and quantile head in FP32: FP32 output-embedding weights and biases (default compute config), FP32 head output",
    *_COMMON_EXCEPTIONS,
]

PRECISION_POLICIES = {
    "bf16": {
        "mode": "bf16",
        "weights": "bf16",
        "activations": "bf16",
        "accumulation": "fp32",
        "exceptions": list(_BF16_EXCEPTIONS),
    },
    "fp32": {
        "mode": "fp32",
        "weights": "fp32",
        "activations": "fp32",
        "accumulation": "fp32",
        "exceptions": [
            "matmuls use the default compute kernel configuration",
            "attention softmax and rms-norm composed manually; norm weights and RoPE rotate_half folded into linear weights (same rationale as bf16)",
        ],
    },
    "bfp8_b": {
        "mode": "bfp8_b",
        "weights": "bfp8_b",
        "activations": "bf16",
        "accumulation": "fp32",
        "exceptions": [
            "only linear weight matrices (embedding/head residual blocks, fused q|k|v and rotated q|k, o, fused group-attention, FFN wi/wo) are BFP8_B; biases, RoPE tables, mask and REG token stay bf16",
            "attention q@k^T and probs@v are activation-activation matmuls in bf16",
            "default compute config, bf16 residual stream and bf16 head (the bf16 FP32-residual policy is not applied)",
            *_COMMON_EXCEPTIONS,
        ],
    },
}
