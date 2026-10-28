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
                               "fp32_acc": bool, "out_dtype": bf16|bfp8}} for the groups in
                      TEXT_WEIGHT_GROUPS; `out_dtype` is the dtype of the projection's output
                      activation (the activation dtype, per group)
    layer_exceptions  [{"layers": [i, ...], "weights": {group: {...}}, "sdpa": {...}}]  later
                      entries win; a key left out keeps the global value
    sdpa              {"fidelity", "fp32_acc", "exp_approx"}
    norms             {"fidelity", "fp32_acc"}             every text RMSNorm
    residual_dtype    bf16|bfp8   dtype of the residual stream (h1, h2, layer output)
    mask_dtype        bf16|bfp8|bfp4   additive attention masks

`vision` section (E6b; consumed by `VisionModel.from_state_dict`, hence `VisionEncoder.from_pretrained`
and `EmbeddingGemma2.from_pretrained`; every key required, unknown keys rejected, "note" allowed):

    weights           {group: {"dtype", "fidelity", "fp32_acc", "out_dtype": bf16|bfp8|fp32}} for
                      VISION_WEIGHT_GROUPS
    layer_exceptions  as text, over VISION_PER_LAYER_GROUPS and `sdpa`
    sdpa              {"fidelity", "fp32_acc", "exp_approx"}
    norms             {"fidelity", "fp32_acc"}             every vision RMSNorm (layers, q/k/v, embed_vision)
    mask_dtype        bf16|bfp8|bfp4   key-padding mask

`audio` section (E6b; consumed by `AudioModel.from_state_dict`, hence `AudioEncoder.from_pretrained`
and `EmbeddingGemma2.from_pretrained`):

    weights           {group: {...}} for AUDIO_WEIGHT_GROUPS; `out_dtype` bf16|bfp8|fp32 is the dtype
                      of the projection output (fp32 on the activation groups = the E3 fp32-activation
                      variant); `subsample_conv.out_dtype` must be bf16 (conv2d output)
    layer_exceptions  as text, over AUDIO_PER_LAYER_GROUPS
    attention         {"fidelity", "fp32_acc"}  band-attention block matmuls (rel term, QK^T, PV) and
                      the softmax sum
    lconv_shift       {"fidelity", "fp32_acc"}  the lconv 0/1 causal-shift / selection matmuls
    norms             {"fidelity", "fp32_acc", "sharded"}  every audio norm (conformer RMSNorms, the
                      subsampling LayerNorms, embed_audio); `sharded`: block-sharded RMSNorm round trip
                      for <= 24 tile rows on the conformer norms (E5c c21)

KV-cache, CCL and logits fields are "n/a" for this single-chip encoder.
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

# Vision matmul weight groups (E6b):
#   attn_qkv     fused q|k|v projection, every layer (768 -> 2304)
#   attn_o       attention output projection (768 -> 768)
#   mlp_gate_up  MLP gate and up projections (768 -> 3072 each)
#   mlp_down     MLP down projection (3072 -> 768)
#   patch_embed  patch input projection (768 -> 768), once per forward
#   projector    embed_vision projection (768 -> 512), once per forward
# Fixed, not swept: the pooler matmul (0/1 pooling matrix, HiFi4, fp32 output) and the bf16 position tables.
VISION_WEIGHT_GROUPS = ("attn_qkv", "attn_o", "mlp_gate_up", "mlp_down", "patch_embed", "projector")
VISION_PER_LAYER_GROUPS = ("attn_qkv", "attn_o", "mlp_gate_up", "mlp_down")
# Audio matmul weight groups (E6b):
#   subsample_conv  the two subsampling conv2d (weights dtype, compute config; output bf16)
#   subsample_proj  input_proj_linear (1024 -> 1024), once per forward
#   attn_qkv        fused q|k|v ClippableLinear (1024 -> 3072), every layer
#   attn_o          attention `post` ClippableLinear (1024 -> 1024)
#   ffn_up          ffw_layer_1 of both Macaron FFNs (1024 -> 4096)
#   ffn_down        ffw_layer_2 of both Macaron FFNs (4096 -> 1024)
#   lconv_start     lconv linear_start (1024 -> 2048)
#   lconv_end       lconv linear_end (1024 -> 1024)
#   output_proj     audio tower output_proj (1024 -> 1536, bias)
#   projector       embed_audio projection (1536 -> 512)
# Fixed, not swept: the depthwise conv taps (fp32), the relative-key matrix and the q scale (fp32).
AUDIO_WEIGHT_GROUPS = (
    "subsample_conv",
    "subsample_proj",
    "attn_qkv",
    "attn_o",
    "ffn_up",
    "ffn_down",
    "lconv_start",
    "lconv_end",
    "output_proj",
    "projector",
)
AUDIO_PER_LAYER_GROUPS = ("attn_qkv", "attn_o", "ffn_up", "ffn_down", "lconv_start", "lconv_end")

_DTYPES = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}
# projection output dtypes: text allows bf16 / bfp8; the vision / audio towers also fp32
_OUT_DTYPES = {**_DTYPES, "fp32": ttnn.float32}
_TOWER_OUT_DTYPES = ("bf16", "bfp8", "fp32")
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
    out_dtype: str

    @classmethod
    def from_dict(cls, d: dict, where: str, out_dtypes=_ACT_DTYPES) -> "MatmulPolicy":
        _require(d, ("dtype", "fidelity", "fp32_acc", "out_dtype"), where)
        return cls(
            _check(d["dtype"], _DTYPES, f"{where}.dtype"),
            _check(d["fidelity"], _FIDELITIES, f"{where}.fidelity"),
            _bool(d["fp32_acc"], f"{where}.fp32_acc"),
            _check(d["out_dtype"], out_dtypes, f"{where}.out_dtype"),
        )

    @property
    def ttnn_dtype(self):
        return _DTYPES[self.dtype]

    @property
    def out_ttnn_dtype(self):
        return _OUT_DTYPES[self.out_dtype]

    def kernel_config(self, packer_l1_acc: bool):
        """Compute config with an explicit packer_l1_acc (ops that the E5 code ran without it)."""
        return compute_kernel_config(self.fidelity, self.fp32_acc, packer_l1_acc)

    @property
    def compute_kernel_config(self):
        return compute_kernel_config(self.fidelity, self.fp32_acc, True)

    def to_dict(self) -> dict:
        return {"dtype": self.dtype, "fidelity": self.fidelity, "fp32_acc": self.fp32_acc, "out_dtype": self.out_dtype}


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

    def kernel_config(self, packer_l1_acc: bool):
        return compute_kernel_config(self.fidelity, self.fp32_acc, packer_l1_acc)

    def to_dict(self) -> dict:
        return {"fidelity": self.fidelity, "fp32_acc": self.fp32_acc}


@dataclass(frozen=True)
class AudioNormPolicy:
    fidelity: str
    fp32_acc: bool
    sharded: bool

    @classmethod
    def from_dict(cls, d: dict, where: str) -> "AudioNormPolicy":
        _require(d, ("fidelity", "fp32_acc", "sharded"), where)
        return cls(
            _check(d["fidelity"], _FIDELITIES, f"{where}.fidelity"),
            _bool(d["fp32_acc"], f"{where}.fp32_acc"),
            _bool(d["sharded"], f"{where}.sharded"),
        )

    @property
    def compute_kernel_config(self):
        return compute_kernel_config(self.fidelity, self.fp32_acc, False)

    def to_dict(self) -> dict:
        return {"fidelity": self.fidelity, "fp32_acc": self.fp32_acc, "sharded": self.sharded}


@dataclass(frozen=True)
class LayerException:
    layers: tuple[int, ...]
    weights: tuple[tuple[str, MatmulPolicy], ...]
    sdpa: SDPAPolicy | None

    @classmethod
    def from_dict(
        cls, d: dict, where: str, groups=PER_LAYER_GROUPS, out_dtypes=_ACT_DTYPES, allow_sdpa: bool = True
    ) -> "LayerException":
        if not isinstance(d, dict) or "layers" not in d:
            raise ValueError(f"{where}: needs a `layers` list")
        keys = ("layers", "weights", "sdpa", "note") if allow_sdpa else ("layers", "weights", "note")
        extra = [k for k in d if k not in keys]
        if extra:
            raise ValueError(f"{where}: unknown keys {extra}")
        layers = tuple(int(i) for i in d["layers"])
        weights = []
        for g, p in (d.get("weights") or {}).items():
            _check(g, groups, f"{where}.weights group")
            weights.append((g, MatmulPolicy.from_dict(p, f"{where}.weights.{g}", out_dtypes)))
        sdpa = SDPAPolicy.from_dict(d["sdpa"], f"{where}.sdpa") if d.get("sdpa") is not None else None
        return cls(layers, tuple(weights), sdpa)


_TEXT_KEYS = (
    "weights",
    "layer_exceptions",
    "sdpa",
    "norms",
    "residual_dtype",
    "mask_dtype",
)


@dataclass(frozen=True)
class TextPrecision:
    weights: tuple[tuple[str, MatmulPolicy], ...]
    layer_exceptions: tuple[LayerException, ...]
    sdpa: SDPAPolicy
    norms: NormPolicy
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
    def residual_ttnn_dtype(self):
        return _DTYPES[self.residual_dtype]

    @property
    def mask_ttnn_dtype(self):
        return _DTYPES[self.mask_dtype]


def _tower_section(d: dict, name: str, keys: tuple) -> dict:
    """The section without its optional free-text `note`, checked for exactly `keys`."""
    if not isinstance(d, dict):
        raise ValueError(f"{name}: expected an object, got {type(d).__name__}")
    body = {k: v for k, v in d.items() if k != "note"}
    _require(body, keys, name)
    return body


def _tower_weights(d: dict, name: str, groups: tuple) -> tuple:
    _require(d, groups, f"{name}.weights")
    return tuple((g, MatmulPolicy.from_dict(d[g], f"{name}.weights.{g}", _TOWER_OUT_DTYPES)) for g in groups)


def _tower_exceptions(d, name: str, groups: tuple, allow_sdpa: bool) -> tuple:
    if not isinstance(d, list):
        raise ValueError(f"{name}.layer_exceptions: expected a list")
    return tuple(
        LayerException.from_dict(e, f"{name}.layer_exceptions[{i}]", groups, _TOWER_OUT_DTYPES, allow_sdpa)
        for i, e in enumerate(d)
    )


class _TowerMatmuls:
    weights: tuple
    layer_exceptions: tuple

    def matmul(self, group: str, layer_idx: int | None = None) -> MatmulPolicy:
        """Policy of `group` in layer `layer_idx` (layer exceptions applied in order)."""
        policy = dict(self.weights)[group]
        if layer_idx is not None:
            for exc in self.layer_exceptions:
                if layer_idx in exc.layers:
                    policy = dict(exc.weights).get(group, policy)
        return policy


@dataclass(frozen=True)
class VisionPrecision(_TowerMatmuls):
    weights: tuple[tuple[str, MatmulPolicy], ...]
    layer_exceptions: tuple[LayerException, ...]
    sdpa: SDPAPolicy
    norms: NormPolicy
    mask_dtype: str

    @classmethod
    def from_dict(cls, d: dict) -> "VisionPrecision":
        d = _tower_section(d, "vision", ("weights", "layer_exceptions", "sdpa", "norms", "mask_dtype"))
        return cls(
            weights=_tower_weights(d["weights"], "vision", VISION_WEIGHT_GROUPS),
            layer_exceptions=_tower_exceptions(d["layer_exceptions"], "vision", VISION_PER_LAYER_GROUPS, True),
            sdpa=SDPAPolicy.from_dict(d["sdpa"], "vision.sdpa"),
            norms=NormPolicy.from_dict(d["norms"], "vision.norms"),
            mask_dtype=_check(d["mask_dtype"], _MASK_DTYPES, "vision.mask_dtype"),
        )

    def sdpa_for(self, layer_idx: int) -> SDPAPolicy:
        policy = self.sdpa
        for exc in self.layer_exceptions:
            if layer_idx in exc.layers and exc.sdpa is not None:
                policy = exc.sdpa
        return policy

    @property
    def mask_ttnn_dtype(self):
        return _DTYPES[self.mask_dtype]


@dataclass(frozen=True)
class AudioPrecision(_TowerMatmuls):
    weights: tuple[tuple[str, MatmulPolicy], ...]
    layer_exceptions: tuple[LayerException, ...]
    attention: NormPolicy
    lconv_shift: NormPolicy
    norms: AudioNormPolicy

    @classmethod
    def from_dict(cls, d: dict) -> "AudioPrecision":
        d = _tower_section(d, "audio", ("weights", "layer_exceptions", "attention", "lconv_shift", "norms"))
        weights = _tower_weights(d["weights"], "audio", AUDIO_WEIGHT_GROUPS)
        if dict(weights)["subsample_conv"].out_dtype != "bf16":
            raise ValueError("audio.weights.subsample_conv.out_dtype: the conv2d output is bf16 (only 'bf16')")
        return cls(
            weights=weights,
            layer_exceptions=_tower_exceptions(d["layer_exceptions"], "audio", AUDIO_PER_LAYER_GROUPS, False),
            attention=NormPolicy.from_dict(d["attention"], "audio.attention"),
            lconv_shift=NormPolicy.from_dict(d["lconv_shift"], "audio.lconv_shift"),
            norms=AudioNormPolicy.from_dict(d["norms"], "audio.norms"),
        )


@dataclass(frozen=True)
class PrecisionConfig:
    config_id: str
    text: TextPrecision
    raw: dict  # the parsed JSON (n/a fields kept as recorded)
    source: str  # file path, or "<dict>"
    vision: VisionPrecision
    audio: AudioPrecision

    @classmethod
    def from_dict(cls, d: dict, source: str = "<dict>") -> "PrecisionConfig":
        for k in ("config_id", "text", "vision", "audio", *NA_FIELDS):
            if k not in d:
                raise ValueError(f"precision config {source}: missing top-level key {k!r}")
        for k in NA_FIELDS:
            if not str(d[k]).startswith("n/a"):
                raise ValueError(f"precision config {source}: {k} must be 'n/a: ...' for this encoder, got {d[k]!r}")
        return cls(
            str(d["config_id"]),
            TextPrecision.from_dict(d["text"]),
            copy.deepcopy(d),
            source,
            VisionPrecision.from_dict(d["vision"]),
            AudioPrecision.from_dict(d["audio"]),
        )


def _mm_dict(dtype="bf16", fidelity="HiFi2", fp32_acc=True, out_dtype="bf16") -> dict:
    return {"dtype": dtype, "fidelity": fidelity, "fp32_acc": fp32_acc, "out_dtype": out_dtype}


# The hard-coded E5b vision / E5c audio policies in this schema (the E6b sweep baselines). Built with
# them, both towers are bit-identical to the E5 code.
E5_VISION_POLICY = {
    "weights": {
        "attn_qkv": _mm_dict(),
        "attn_o": _mm_dict(),
        "mlp_gate_up": _mm_dict(),
        "mlp_down": _mm_dict(),
        "patch_embed": _mm_dict(fidelity="HiFi4"),
        "projector": _mm_dict(fidelity="HiFi4"),
    },
    "layer_exceptions": [],
    "sdpa": {"fidelity": "HiFi2", "fp32_acc": True, "exp_approx": False},
    "norms": {"fidelity": "HiFi4", "fp32_acc": True},
    "mask_dtype": "bfp4",
}
E5_AUDIO_POLICY = {
    "weights": {
        "subsample_conv": _mm_dict(fidelity="HiFi4"),
        "subsample_proj": _mm_dict(fidelity="HiFi4"),
        "attn_qkv": _mm_dict(fidelity="HiFi4", out_dtype="fp32"),
        "attn_o": _mm_dict(fidelity="HiFi4"),
        "ffn_up": _mm_dict(fidelity="HiFi4"),
        "ffn_down": _mm_dict(fidelity="HiFi4"),
        "lconv_start": _mm_dict(fidelity="HiFi4"),
        "lconv_end": _mm_dict(fidelity="HiFi4"),
        "output_proj": _mm_dict(fidelity="HiFi4", out_dtype="fp32"),
        "projector": _mm_dict(fidelity="HiFi4"),
    },
    "layer_exceptions": [],
    "attention": {"fidelity": "HiFi4", "fp32_acc": True},
    "lconv_shift": {"fidelity": "HiFi4", "fp32_acc": True},
    "norms": {"fidelity": "HiFi4", "fp32_acc": True, "sharded": False},
}
# audio groups whose output is the conformer activation stream (fp32 here = the E3 `act_dtype=float32`
# variant, `AudioModel.from_state_dict(act_dtype=ttnn.float32)`)
AUDIO_ACT_GROUPS = ("subsample_proj", "attn_o", "ffn_up", "ffn_down", "lconv_start", "lconv_end")


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
