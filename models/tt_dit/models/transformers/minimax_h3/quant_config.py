# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in bfloat8_b precision for the four matmuls of each MiniMax-H3 transformer block (`FAST_H3_FP8`).

Applied in place once the weights and any fused adapter are on device; the weight cache and the declared parameter
dtypes stay bf16. Design and measurements: models/tt_dit/models/MiniMaxH3_fp8.md.
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
    """Precision of one block linear; `None` dtypes keep bf16."""

    weight_dtype: ttnn.DataType | None = None
    activation_dtype: ttnn.DataType | None = None
    math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi2
    fp32_dest_acc: bool = True

    @property
    def quantized(self) -> bool:
        return self.weight_dtype is not None or self.activation_dtype is not None


@dataclass(frozen=True)
class MiniMaxH3QuantConfig:
    """Per-linear precision, an optional bfloat8_b cast of the ring SDPA inputs, and the block range it covers."""

    qkv: LinearQuant = LinearQuant()
    out: LinearQuant = LinearQuant()
    ff1: LinearQuant = LinearQuant()
    ff2: LinearQuant = LinearQuant()
    sdpa_input_dtype: ttnn.DataType | None = None
    blocks: tuple[int, int] | None = None

    def linear(self, name: str) -> LinearQuant:
        return getattr(self, name)

    @property
    def active(self) -> bool:
        return any(self.linear(n).quantized for n in LINEARS) or self.sdpa_input_dtype is not None

    @property
    def fuse_out_addcmul(self) -> bool:
        """The fused addcmul epilogue needs a bf16 weight; a quantized `to_out` weight un-fuses it."""
        return self.out.weight_dtype is None

    @staticmethod
    def preset(
        name: str,
        *,
        linears: tuple[str, ...] = LINEARS,
        activations: bool | None = None,
        fidelity: ttnn.MathFidelity | None = None,
        fp32_dest_acc: bool | None = None,
        sdpa: bool = False,
        out_weight: bool | None = None,
        blocks: tuple[int, int] | None = None,
    ) -> MiniMaxH3QuantConfig:
        """One of `PRESETS` (w8 / w8a8 at HiFi2, w8_lofi / w8a8_lofi at LoFi), narrowed to `linears`, with explicit
        overrides. `out_weight` quantizes `to_out`'s weight and un-fuses its epilogue; it defaults to on at LoFi."""
        if name not in PRESETS:
            raise ValueError(f"unknown preset {name!r}, expected one of {PRESETS}")
        unknown = sorted(set(linears) - set(LINEARS))
        if unknown:
            raise ValueError(f"unknown linears {unknown}, expected a subset of {LINEARS}")
        act = name in ("w8a8", "w8a8_lofi") if activations is None else activations
        lofi = name.endswith("_lofi")
        fid = fidelity if fidelity is not None else (ttnn.MathFidelity.LoFi if lofi else ttnn.MathFidelity.HiFi2)
        acc = True if fp32_dest_acc is None else fp32_dest_acc
        out_weight = lofi if out_weight is None else out_weight
        quant = LinearQuant(
            weight_dtype=ttnn.bfloat8_b,
            activation_dtype=ttnn.bfloat8_b if act else None,
            math_fidelity=fid,
            fp32_dest_acc=acc,
        )
        out_quant = quant if out_weight else replace(quant, weight_dtype=None)
        fields = {n: (out_quant if n == "out" else quant) for n in linears}
        return MiniMaxH3QuantConfig(**fields, sdpa_input_dtype=ttnn.bfloat8_b if sdpa else None, blocks=blocks)

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
        if self.blocks is not None:
            parts.append(f"blocks:{self.blocks[0]}-{self.blocks[1]}")
        return " ".join(parts) if parts else "off"

    def covers(self, block_index: int) -> bool:
        return self.blocks is None or self.blocks[0] <= block_index <= self.blocks[1]


def _flag(var: str, default: bool | None = None) -> bool | None:
    value = os.environ.get(var)
    if value is None:
        return default
    if value.lower() in _TRUE:
        return True
    if value.lower() in _FALSE:
        return False
    raise ValueError(f"{var}={value!r}: expected 0 or 1")


def _math_fidelity_from_env(var: str) -> ttnn.MathFidelity | None:
    name = os.environ.get(var)
    if not name:
        return None
    valid = [key for key in ttnn.MathFidelity.__members__ if key != "Invalid"]
    if name not in valid:
        raise ValueError(f"{var}={name!r}: expected one of {valid}")
    return ttnn.MathFidelity.__members__[name]


def _block_range(var: str) -> tuple[int, int] | None:
    value = os.environ.get(var)
    if not value:
        return None
    lo, sep, hi = value.partition("-")
    if not sep or not lo.strip().isdigit() or not hi.strip().isdigit() or int(lo) > int(hi):
        raise ValueError(f"{var}={value!r}: expected an inclusive range like 2-46")
    return int(lo), int(hi)


def quant_config_from_env() -> MiniMaxH3QuantConfig:
    """`FAST_H3_FP8` = 0 (off) | 1 (`w8a8_lofi`) | a preset name, refined by `FAST_H3_FP8_{LINEARS, ACTIVATIONS,
    FIDELITY, FP32_ACC, SDPA, OUT_WEIGHT, BLOCKS}` (see MiniMaxH3.md)."""
    raw = os.environ.get(ENV_FLAG, "0").strip()
    if raw.lower() in _FALSE:
        return MiniMaxH3QuantConfig()
    name = "w8a8_lofi" if raw.lower() in _TRUE else raw
    if name not in PRESETS:
        raise ValueError(f"{ENV_FLAG}={raw!r}: expected 0, 1 or one of {PRESETS}")
    linears = tuple(n for n in os.environ.get("FAST_H3_FP8_LINEARS", ",".join(LINEARS)).split(",") if n)
    return MiniMaxH3QuantConfig.preset(
        name,
        linears=linears,
        activations=_flag("FAST_H3_FP8_ACTIVATIONS"),
        fidelity=_math_fidelity_from_env("FAST_H3_FP8_FIDELITY"),
        fp32_dest_acc=_flag("FAST_H3_FP8_FP32_ACC"),
        sdpa=_flag("FAST_H3_FP8_SDPA", False),
        out_weight=_flag("FAST_H3_FP8_OUT_WEIGHT"),
        blocks=_block_range("FAST_H3_FP8_BLOCKS"),
    )


def _typecast_parameter(param, dtype: ttnn.DataType) -> None:
    if param._data is not None and param._data.dtype != dtype:
        param._data = ttnn.typecast(param._data, dtype)


def _compute_config(arch, quant: LinearQuant):
    if quant == LinearQuant():
        return None
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
    """Apply `config` to the transformer (or one block) in place; idempotent, so it re-runs after every reload."""
    blocks = list(getattr(model, "transformer_blocks", [model]))
    if not blocks:
        return
    arch = blocks[0].mesh_device.arch()
    skipped = 0
    for index, block in enumerate(blocks):
        if not config.covers(index):
            skipped += 1
            continue
        attn, ff = block.attn, block.ff
        _apply_linear(attn.to_qkv, config.qkv, cast_input=True, pin_output=True)
        _apply_linear(attn.to_out, config.out, cast_input=True, pin_output=True)
        _apply_linear(ff.ff1, config.ff1, cast_input=True, pin_output=False)
        _apply_linear(ff.ff2, config.ff2, cast_input=False, pin_output=False)
        if config.ff2.activation_dtype is not None:
            ff.ff1_output_dtype = config.ff2.activation_dtype
        else:
            ff.ff1_output_dtype = ttnn.bfloat16 if config.ff1.activation_dtype is not None else None
        qkv_config = _compute_config(arch, config.qkv)
        out_config = _compute_config(arch, config.out)
        attn.qkv_compute_kernel_config = attn.mm_compute_kernel_config if qkv_config is None else qkv_config
        attn.out_compute_kernel_config = attn.mm_compute_kernel_config if out_config is None else out_config
        ff.ff1_compute_kernel_config = _compute_config(arch, config.ff1)
        ff.ff2_compute_kernel_config = _compute_config(arch, config.ff2)
        attn.sdpa_input_dtype = config.sdpa_input_dtype
        attn.fuse_out_addcmul = config.fuse_out_addcmul
    if config.active:
        logger.info(f"minimax-h3 8-bit matmuls: {config.describe()} on {len(blocks) - skipped} of {len(blocks)} block(s)")
