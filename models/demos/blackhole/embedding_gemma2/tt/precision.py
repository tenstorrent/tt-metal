# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision policy (weight dtypes, compute fidelities, activation / residual / mask dtypes).

The text construction path (`TextModel.from_state_dict`, hence `TextEncoder.from_pretrained` and
`EmbeddingGemma2.from_pretrained`) reads this policy and builds every text weight, compute-kernel
config, SDPA program config and mask with it. With no explicit `precision=` argument the policy is
loaded from `doc/datatype_sweep/selected_precision_config.json` (the E6 sweep's selected config),
or from the file named by `EMBEDDING_GEMMA2_PRECISION` when that variable is set.

Schema (`text` section; every key is required, unknown keys are rejected):

    weights           {group: {"dtype": bf16|bfp8|bfp4, "fidelity": LoFi|HiFi2|HiFi3|HiFi4,
                               "fp32_acc": bool}} for the groups in TEXT_WEIGHT_GROUPS
    layer_exceptions  [{"layers": [i, ...], "weights": {group: {...}}, "sdpa": {...}}]  later
                      entries win; a key left out keeps the global value
    sdpa              {"fidelity", "fp32_acc", "exp_approx"}
    norms             {"fidelity", "fp32_acc"}             every text RMSNorm
    activation_dtype  bf16|bfp8   dtype of every text projection output
    residual_dtype    bf16|bfp8   dtype of the residual stream (h1, h2, layer output)
    mask_dtype        bf16|bfp8|bfp4   additive attention masks

`vision` and `audio` sections describe the current tower policies. In E6a they are recorded,
validated for presence only and not consumed (both towers keep their hard-coded E5b / E5c
policies); E6b wires them. KV-cache, CCL and logits fields are "n/a" for this single-chip encoder.
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import ttnn

MODEL_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PRECISION_PATH = MODEL_DIR / "doc" / "datatype_sweep" / "selected_precision_config.json"
PRECISION_ENV = "EMBEDDING_GEMMA2_PRECISION"

# Text matmul weight groups:
#   attn_qkv     fused q|k|v projection, every layer (512 -> 2048 sliding / 3072 full)
#   attn_o       attention output projection (2048 -> 512)
#   mlp_gate_up  MLP gate and up projections (512 -> 2048 each)
#   mlp_down     MLP down projection (2048 -> 512)
#   ple_proj     model-level PLE projection (512 -> 24 x 512), once per forward
#   ple_block    per-layer PLE gate (512 -> 512) and projection (512 -> 512)
#   final_proj   embedding projection after the final norm (512 -> 768)
TEXT_WEIGHT_GROUPS = ("attn_qkv", "attn_o", "mlp_gate_up", "mlp_down", "ple_proj", "ple_block", "final_proj")
# groups that exist in every layer (layer exceptions may override these)
PER_LAYER_GROUPS = ("attn_qkv", "attn_o", "mlp_gate_up", "mlp_down", "ple_block")

_DTYPES = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}
_FIDELITIES = {
    "LoFi": ttnn.MathFidelity.LoFi,
    "HiFi2": ttnn.MathFidelity.HiFi2,
    "HiFi3": ttnn.MathFidelity.HiFi3,
    "HiFi4": ttnn.MathFidelity.HiFi4,
}
_ACT_DTYPES = ("bf16", "bfp8")
_MASK_DTYPES = ("bf16", "bfp8", "bfp4")
NA_FIELDS = ("kv_cache_dtype", "ccl_dtype", "logits_dtype", "sampling")


def _require(d: dict, keys, where: str) -> None:
    if not isinstance(d, dict):
        raise ValueError(f"{where}: expected an object, got {type(d).__name__}")
    missing = [k for k in keys if k not in d]
    extra = [k for k in d if k not in keys]
    if missing or extra:
        raise ValueError(f"{where}: missing keys {missing}, unknown keys {extra}")


def _check(value, allowed, where: str):
    if value not in allowed:
        raise ValueError(f"{where}: {value!r} not in {list(allowed)}")
    return value


def _bool(value, where: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{where}: expected true/false, got {value!r}")
    return value


@lru_cache(maxsize=None)
def compute_kernel_config(fidelity: str, fp32_acc: bool, packer_l1_acc: bool, approx: bool = False):
    return ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=_FIDELITIES[fidelity],
        math_approx_mode=approx,
        fp32_dest_acc_en=fp32_acc,
        packer_l1_acc=packer_l1_acc,
    )


@dataclass(frozen=True)
class MatmulPolicy:
    dtype: str
    fidelity: str
    fp32_acc: bool

    @classmethod
    def from_dict(cls, d: dict, where: str) -> "MatmulPolicy":
        _require(d, ("dtype", "fidelity", "fp32_acc"), where)
        return cls(
            _check(d["dtype"], _DTYPES, f"{where}.dtype"),
            _check(d["fidelity"], _FIDELITIES, f"{where}.fidelity"),
            _bool(d["fp32_acc"], f"{where}.fp32_acc"),
        )

    @property
    def ttnn_dtype(self):
        return _DTYPES[self.dtype]

    @property
    def compute_kernel_config(self):
        return compute_kernel_config(self.fidelity, self.fp32_acc, True)

    def to_dict(self) -> dict:
        return {"dtype": self.dtype, "fidelity": self.fidelity, "fp32_acc": self.fp32_acc}


@dataclass(frozen=True)
class SDPAPolicy:
    fidelity: str
    fp32_acc: bool
    exp_approx: bool

    @classmethod
    def from_dict(cls, d: dict, where: str) -> "SDPAPolicy":
        _require(d, ("fidelity", "fp32_acc", "exp_approx"), where)
        return cls(
            _check(d["fidelity"], _FIDELITIES, f"{where}.fidelity"),
            _bool(d["fp32_acc"], f"{where}.fp32_acc"),
            _bool(d["exp_approx"], f"{where}.exp_approx"),
        )

    @property
    def compute_kernel_config(self):
        return compute_kernel_config(self.fidelity, self.fp32_acc, False)

    def to_dict(self) -> dict:
        return {"fidelity": self.fidelity, "fp32_acc": self.fp32_acc, "exp_approx": self.exp_approx}


@dataclass(frozen=True)
class NormPolicy:
    fidelity: str
    fp32_acc: bool

    @classmethod
    def from_dict(cls, d: dict, where: str) -> "NormPolicy":
        _require(d, ("fidelity", "fp32_acc"), where)
        return cls(_check(d["fidelity"], _FIDELITIES, f"{where}.fidelity"), _bool(d["fp32_acc"], f"{where}.fp32_acc"))

    @property
    def compute_kernel_config(self):
        return compute_kernel_config(self.fidelity, self.fp32_acc, False)

    def to_dict(self) -> dict:
        return {"fidelity": self.fidelity, "fp32_acc": self.fp32_acc}


@dataclass(frozen=True)
class LayerException:
    layers: tuple[int, ...]
    weights: tuple[tuple[str, MatmulPolicy], ...]
    sdpa: SDPAPolicy | None

    @classmethod
    def from_dict(cls, d: dict, where: str) -> "LayerException":
        if not isinstance(d, dict) or "layers" not in d:
            raise ValueError(f"{where}: needs a `layers` list")
        extra = [k for k in d if k not in ("layers", "weights", "sdpa", "note")]
        if extra:
            raise ValueError(f"{where}: unknown keys {extra}")
        layers = tuple(int(i) for i in d["layers"])
        weights = []
        for g, p in (d.get("weights") or {}).items():
            _check(g, PER_LAYER_GROUPS, f"{where}.weights group")
            weights.append((g, MatmulPolicy.from_dict(p, f"{where}.weights.{g}")))
        sdpa = SDPAPolicy.from_dict(d["sdpa"], f"{where}.sdpa") if d.get("sdpa") is not None else None
        return cls(layers, tuple(weights), sdpa)


_TEXT_KEYS = (
    "weights",
    "layer_exceptions",
    "sdpa",
    "norms",
    "activation_dtype",
    "residual_dtype",
    "mask_dtype",
)


@dataclass(frozen=True)
class TextPrecision:
    weights: tuple[tuple[str, MatmulPolicy], ...]
    layer_exceptions: tuple[LayerException, ...]
    sdpa: SDPAPolicy
    norms: NormPolicy
    activation_dtype: str
    residual_dtype: str
    mask_dtype: str

    @classmethod
    def from_dict(cls, d: dict) -> "TextPrecision":
        _require(d, _TEXT_KEYS, "text")
        _require(d["weights"], TEXT_WEIGHT_GROUPS, "text.weights")
        weights = tuple((g, MatmulPolicy.from_dict(d["weights"][g], f"text.weights.{g}")) for g in TEXT_WEIGHT_GROUPS)
        if not isinstance(d["layer_exceptions"], list):
            raise ValueError("text.layer_exceptions: expected a list")
        exceptions = tuple(
            LayerException.from_dict(e, f"text.layer_exceptions[{i}]") for i, e in enumerate(d["layer_exceptions"])
        )
        return cls(
            weights=weights,
            layer_exceptions=exceptions,
            sdpa=SDPAPolicy.from_dict(d["sdpa"], "text.sdpa"),
            norms=NormPolicy.from_dict(d["norms"], "text.norms"),
            activation_dtype=_check(d["activation_dtype"], _ACT_DTYPES, "text.activation_dtype"),
            residual_dtype=_check(d["residual_dtype"], _ACT_DTYPES, "text.residual_dtype"),
            mask_dtype=_check(d["mask_dtype"], _MASK_DTYPES, "text.mask_dtype"),
        )

    def matmul(self, group: str, layer_idx: int | None = None) -> MatmulPolicy:
        """Policy of `group` in layer `layer_idx` (layer exceptions applied in order)."""
        policy = dict(self.weights)[group]
        if layer_idx is not None:
            for exc in self.layer_exceptions:
                if layer_idx in exc.layers:
                    policy = dict(exc.weights).get(group, policy)
        return policy

    def sdpa_for(self, layer_idx: int) -> SDPAPolicy:
        policy = self.sdpa
        for exc in self.layer_exceptions:
            if layer_idx in exc.layers and exc.sdpa is not None:
                policy = exc.sdpa
        return policy

    @property
    def activation_ttnn_dtype(self):
        return _DTYPES[self.activation_dtype]

    @property
    def residual_ttnn_dtype(self):
        return _DTYPES[self.residual_dtype]

    @property
    def mask_ttnn_dtype(self):
        return _DTYPES[self.mask_dtype]


@dataclass(frozen=True)
class PrecisionConfig:
    config_id: str
    text: TextPrecision
    raw: dict  # the parsed JSON (vision / audio / n/a fields kept as recorded)
    source: str  # file path, or "<dict>"

    @classmethod
    def from_dict(cls, d: dict, source: str = "<dict>") -> "PrecisionConfig":
        for k in ("config_id", "text", "vision", "audio", *NA_FIELDS):
            if k not in d:
                raise ValueError(f"precision config {source}: missing top-level key {k!r}")
        for k in NA_FIELDS:
            if not str(d[k]).startswith("n/a"):
                raise ValueError(f"precision config {source}: {k} must be 'n/a: ...' for this encoder, got {d[k]!r}")
        return cls(str(d["config_id"]), TextPrecision.from_dict(d["text"]), copy.deepcopy(d), source)


def default_precision_path() -> Path:
    env = os.environ.get(PRECISION_ENV)
    return Path(env) if env else DEFAULT_PRECISION_PATH


def load_precision(path: str | Path | None = None) -> PrecisionConfig:
    """Parse and validate a precision config file (default: `default_precision_path()`)."""
    p = Path(path) if path is not None else default_precision_path()
    with open(p) as f:
        d = json.load(f)
    return PrecisionConfig.from_dict(d, source=str(p))


def deep_merge(base: dict, delta: dict) -> dict:
    """Recursive dict merge (lists and scalars in `delta` replace those in `base`)."""
    out = copy.deepcopy(base)
    for k, v in delta.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def resolve_precision(precision) -> PrecisionConfig:
    """None -> default file; str / Path -> that file; dict -> parsed; PrecisionConfig as is."""
    if precision is None:
        return load_precision()
    if isinstance(precision, PrecisionConfig):
        return precision
    if isinstance(precision, dict):
        return PrecisionConfig.from_dict(precision)
    return load_precision(precision)
