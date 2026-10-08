# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in 8-bit (bfloat8_b) precision for the MiniMax-H3 transformer blocks' matmuls.

Off by default; `FAST_H3_FP8=1` (or a preset name) turns it on. The four block linears (`to_qkv`, `to_out`, `ff1`,
`ff2`) carry essentially all of the denoiser's matmul work, so they are the only ones quantized. Everything else keeps
its dtype on purpose: the adaLN projections and the float32 time embedders (every block reads them and a rounding
there biases the whole trajectory), the per-request input projections, the token refiner and the output heads.

The conversion is applied to the loaded transformer in place -- weights are typecast on device after the checkpoint
(and any fused adapter delta) has landed, so the weight cache stays bf16 and adapter-independent. The declared
`Parameter.dtype` stays bf16 so a reload after eviction still passes the cache's dtype check; the pipeline re-applies
the config after every load, and the typecast is skipped for a weight that already has the target dtype.

Two constraints shape the defaults (see MiniMaxH3.md):
  * `to_out` fuses `residual + gate * out` into its matmul epilogue, which requires the weight tile format to match
    the bf16 residual; its weight therefore stays bf16 while its activation is still cast. `FAST_H3_FP8_OUT_WEIGHT=1`
    un-fuses that epilogue and quantizes the weight too.
  * The norms take bf16, so every linear feeding a norm or the residual stream pins its output back to bf16; only
    `ff1`'s output (the SwiGLU intermediate that feeds `ff2`) may be produced directly in bfloat8_b.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace

from loguru import logger

import ttnn

ENV_FLAG = "FAST_H3_FP8"
LINEARS = ("qkv", "out", "ff1", "ff2")
PRESETS = ("w8", "w8a8", "w8_lofi", "w8a8_lofi")

_TRUE = ("1", "true", "yes", "on")
_FALSE = ("0", "false", "no", "off", "")


@dataclass(frozen=True)
class LinearQuant:
    """Precision of one block linear. `None` dtypes mean "leave as built" (bf16)."""

    weight_dtype: ttnn.DataType | None = None
    activation_dtype: ttnn.DataType | None = None
    math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi2
    fp32_dest_acc: bool = True

    @property
    def quantized(self) -> bool:
        return self.weight_dtype is not None or self.activation_dtype is not None


@dataclass(frozen=True)
class MiniMaxH3QuantConfig:
    """What each block linear runs at, plus the two attention-side knobs."""

    qkv: LinearQuant = LinearQuant()
    out: LinearQuant = LinearQuant()
    ff1: LinearQuant = LinearQuant()
    ff2: LinearQuant = LinearQuant()
    #: Q, K, V (and the joint dummies) are typecast to this before the ring SDPA; None keeps them bf16.
    sdpa_input_dtype: ttnn.DataType | None = None
    #: Quantize `to_out`'s weight as well, at the cost of un-fusing its addcmul epilogue.
    out_weight: bool = False

    def linear(self, name: str) -> LinearQuant:
        return getattr(self, name)

    @property
    def active(self) -> bool:
        return any(self.linear(n).quantized for n in LINEARS) or self.sdpa_input_dtype is not None

    @staticmethod
    def default() -> MiniMaxH3QuantConfig:
        return MiniMaxH3QuantConfig()

    @staticmethod
    def preset(
        name: str,
        *,
        linears: tuple[str, ...] = LINEARS,
        activations: bool | None = None,
        fidelity: ttnn.MathFidelity | None = None,
        fp32_dest_acc: bool | None = None,
        sdpa: bool = False,
        out_weight: bool = False,
    ) -> MiniMaxH3QuantConfig:
        """One of `PRESETS`, narrowed to `linears` and with any explicit overrides applied.

        The FPU consumes the weight (SrcA) 1+4 mantissa bits per pass and the activation (SrcB) 1+6 bits, so HiFi2
        (two passes) is exact for bfloat8_b operands and LoFi (one pass, twice the throughput) keeps the activation's
        7 bits but rounds the weight to 5. The presets are the four corners of that trade:

        w8: bfloat8_b weights, bf16 activations, HiFi2 (halves the weight bytes; the matmul math is unchanged).
        w8a8: bfloat8_b weights and activations, HiFi2 (also halves what the TP all-gathers move).
        w8_lofi: bfloat8_b weights, bf16 activations, LoFi (the compute lever without quantizing activations).
        w8a8_lofi: bfloat8_b weights and activations, LoFi (the Wan / LTX bf8 tier).
        fp32 destination accumulation stays on in every preset; `fp32_dest_acc=False` is a separate, measured knob.
        """
        if name not in PRESETS:
            raise ValueError(f"unknown preset {name!r}, expected one of {PRESETS}")
        unknown = sorted(set(linears) - set(LINEARS))
        if unknown:
            raise ValueError(f"unknown linears {unknown}, expected a subset of {LINEARS}")
        act = name in ("w8a8", "w8a8_lofi") if activations is None else activations
        lofi = name.endswith("_lofi")
        fid = fidelity if fidelity is not None else (ttnn.MathFidelity.LoFi if lofi else ttnn.MathFidelity.HiFi2)
        acc = True if fp32_dest_acc is None else fp32_dest_acc
        quant = LinearQuant(
            weight_dtype=ttnn.bfloat8_b,
            activation_dtype=ttnn.bfloat8_b if act else None,
            math_fidelity=fid,
            fp32_dest_acc=acc,
        )
        # to_out's fused epilogue pins its weight to bf16 unless the caller un-fuses it.
        out_quant = quant if out_weight else replace(quant, weight_dtype=None)
        fields = {n: (out_quant if n == "out" else quant) for n in linears}
        return MiniMaxH3QuantConfig(
            **fields,
            sdpa_input_dtype=ttnn.bfloat8_b if sdpa else None,
            out_weight=out_weight and "out" in linears,
        )

    def describe(self) -> str:
        parts = []
        for n in LINEARS:
            q = self.linear(n)
            if not q.quantized:
                continue
            w = "w8" if q.weight_dtype == ttnn.bfloat8_b else "w16"
            a = "a8" if q.activation_dtype == ttnn.bfloat8_b else "a16"
            fid = str(q.math_fidelity).replace("MathFidelity.", "")
            parts.append(f"{n}:{w}{a}/{fid}{'' if q.fp32_dest_acc else '/no-fp32-acc'}")
        if self.sdpa_input_dtype is not None:
            parts.append("sdpa:in8")
        return " ".join(parts) if parts else "off"


# --------------------------------------------------------------------------------------------- environment


def _flag(var: str, default: bool | None = None) -> bool | None:
    value = os.environ.get(var)
    if value is None:
        return default
    if value.lower() in _TRUE:
        return True
    if value.lower() in _FALSE:
        return False
    raise ValueError(f"{var}={value!r}: expected 0 or 1")


def math_fidelity_from_env(var: str) -> ttnn.MathFidelity | None:
    name = os.environ.get(var)
    if not name:
        return None
    valid = [key for key in ttnn.MathFidelity.__members__ if key != "Invalid"]
    if name not in valid:
        raise ValueError(f"{var}={name!r}: expected one of {valid}")
    return ttnn.MathFidelity.__members__[name]


def quant_config_from_env() -> MiniMaxH3QuantConfig:
    """`FAST_H3_FP8` = 0 (default, off) | 1 (the `w8a8` preset) | a preset name, refined by the optional
    FAST_H3_FP8_LINEARS (subset of qkv,out,ff1,ff2), FAST_H3_FP8_ACTIVATIONS, FAST_H3_FP8_FIDELITY,
    FAST_H3_FP8_FP32_ACC, FAST_H3_FP8_SDPA and FAST_H3_FP8_OUT_WEIGHT."""
    raw = os.environ.get(ENV_FLAG, "0").strip()
    if raw.lower() in _FALSE:
        return MiniMaxH3QuantConfig.default()
    name = "w8a8" if raw.lower() in _TRUE else raw
    if name not in PRESETS:
        raise ValueError(f"{ENV_FLAG}={raw!r}: expected 0, 1 or one of {PRESETS}")
    linears = tuple(n for n in os.environ.get("FAST_H3_FP8_LINEARS", ",".join(LINEARS)).split(",") if n)
    return MiniMaxH3QuantConfig.preset(
        name,
        linears=linears,
        activations=_flag("FAST_H3_FP8_ACTIVATIONS"),
        fidelity=math_fidelity_from_env("FAST_H3_FP8_FIDELITY"),
        fp32_dest_acc=_flag("FAST_H3_FP8_FP32_ACC"),
        sdpa=_flag("FAST_H3_FP8_SDPA", False),
        out_weight=_flag("FAST_H3_FP8_OUT_WEIGHT", False),
    )


# --------------------------------------------------------------------------------------------- application


def _typecast_parameter(param, dtype: ttnn.DataType) -> None:
    """Cast the live device tensor only. The declared dtype stays as cached (see the module docstring)."""
    if param._data is not None and param._data.dtype != dtype:
        param._data = ttnn.typecast(param._data, dtype)


def _compute_config(arch, quant: LinearQuant):
    return ttnn.init_device_compute_kernel_config(
        arch,
        math_fidelity=quant.math_fidelity,
        math_approx_mode=True,
        fp32_dest_acc_en=quant.fp32_dest_acc,
        packer_l1_acc=True,
    )


def _apply_linear(linear, quant: LinearQuant, *, cast_input: bool, pin_output: bool) -> None:
    if quant.weight_dtype is not None:
        _typecast_parameter(linear.weight, quant.weight_dtype)
        if getattr(linear, "bias", None) is not None:
            _typecast_parameter(linear.bias, quant.weight_dtype)
    if cast_input:
        linear.activation_dtype = quant.activation_dtype
        linear.pin_output_bf16 = pin_output and quant.activation_dtype is not None


def apply_quant_config(model, config: MiniMaxH3QuantConfig) -> None:
    """Apply `config` to the transformer (or a single block) in place. Idempotent; safe to re-run after a reload."""
    blocks = list(getattr(model, "transformer_blocks", [model]))
    if not blocks:
        return
    arch = blocks[0].mesh_device.arch()
    for block in blocks:
        attn, ff = block.attn, block.ff
        _apply_linear(attn.to_qkv, config.qkv, cast_input=True, pin_output=True)
        _apply_linear(attn.to_out, config.out, cast_input=True, pin_output=True)
        _apply_linear(ff.ff1, config.ff1, cast_input=True, pin_output=False)
        _apply_linear(ff.ff2, config.ff2, cast_input=False, pin_output=False)
        # ff1's SwiGLU output is ff2's input: the matmul writes it in bfloat8_b directly. (A bf16 output typecast
        # afterwards was tried and dropped: the fused SwiGLU returns non-finite values when its block-float input
        # is paired with a bf16 output override.)
        ff.ff1_output_dtype = config.ff2.activation_dtype
        attn.qkv_compute_kernel_config = _compute_config(arch, config.qkv)
        attn.out_compute_kernel_config = _compute_config(arch, config.out)
        block.ff_compute_kernel_config = _compute_config(arch, config.ff1)
        attn.sdpa_input_dtype = config.sdpa_input_dtype
        attn.fuse_out_addcmul = not config.out_weight
    if config.active:
        logger.info(f"minimax-h3 8-bit matmuls: {config.describe()} on {len(blocks)} block(s)")


def apply_env_quant_config(model) -> MiniMaxH3QuantConfig:
    """Read the environment and apply it; returns the config for logging."""
    config = quant_config_from_env()
    apply_quant_config(model, config)
    return config


__all__ = [
    "ENV_FLAG",
    "LINEARS",
    "PRESETS",
    "LinearQuant",
    "MiniMaxH3QuantConfig",
    "apply_env_quant_config",
    "apply_quant_config",
    "math_fidelity_from_env",
    "quant_config_from_env",
]
