# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

# Scalar FP8 / FP4 scale compat for Hugging Face ``Fp8Dequantize`` across ``transformers`` versions. Devstral checkpoints can ship a **scalar** ``weight_scale_inv`` (0-D tensor). Older ``transformers`` exposed ``Fp8Dequantize._dequantize_one`` for per-weight dequant; newer releases (e.g. 5.7) removed that hook and perform dequant in ``Fp8Dequantize.convert`` with a dict-based API. Call :func:`apply_fp8_dequantize_compat` once at process startup (before loading weights). The patch is idempotent.

from __future__ import annotations

from typing import Any, Callable

import torch

_APPLIED = False
_ORIGINAL_DEQUANTIZE_ONE: Callable[..., Any] | None = None
_ORIGINAL_CONVERT: Callable[..., Any] | None = None


def _unpack_fp4_safe(self: Any, quantized: torch.Tensor) -> torch.Tensor:
    unpack = getattr(self, "_unpack_fp4", None)
    if unpack is not None:
        return unpack(quantized)
    return quantized.to(torch.float32)


def _scalar_scale_dequantize(self: Any, quantized: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
    if quantized.dtype == torch.int8 or (fp4_dtype is not None and quantized.dtype == fp4_dtype):
        quantized_fp32 = _unpack_fp4_safe(self, quantized)
    else:
        quantized_fp32 = quantized.to(torch.float32)
    out_dtype = scales.dtype if scales.dtype.is_floating_point and scales.element_size() >= 2 else torch.bfloat16
    scale = scales.to(torch.float32)
    return (quantized_fp32 * scale).to(out_dtype)


def _apply_legacy_dequantize_one_patch() -> None:
    global _ORIGINAL_DEQUANTIZE_ONE
    from transformers.integrations.finegrained_fp8 import Fp8Dequantize

    _ORIGINAL_DEQUANTIZE_ONE = Fp8Dequantize._dequantize_one

    def _dequantize_one_compat(self: Any, quantized: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        if scales.ndim == 0:
            return _scalar_scale_dequantize(self, quantized, scales)
        return _ORIGINAL_DEQUANTIZE_ONE(self, quantized, scales)

    Fp8Dequantize._dequantize_one = _dequantize_one_compat  # type: ignore[method-assign]


def _apply_convert_patch() -> None:
    global _ORIGINAL_CONVERT
    from transformers.integrations.finegrained_fp8 import Fp8Dequantize

    _ORIGINAL_CONVERT = Fp8Dequantize.convert

    def convert_compat(self: Any, input_dict: dict[str, Any], full_layer_name: str | None = None, **kwargs: Any):
        if len(input_dict) < 2:
            return _ORIGINAL_CONVERT(self, input_dict, full_layer_name=full_layer_name, **kwargs)
        try:
            quantized = input_dict["weight$"][0]
            scales = input_dict["weight_scale_inv"][0]
        except (KeyError, IndexError, TypeError):
            return _ORIGINAL_CONVERT(self, input_dict, full_layer_name=full_layer_name, **kwargs)
        if getattr(scales, "ndim", None) != 0:
            return _ORIGINAL_CONVERT(self, input_dict, full_layer_name=full_layer_name, **kwargs)
        deq = _scalar_scale_dequantize(self, quantized, scales)
        out_key = full_layer_name if full_layer_name is not None else "weight"
        return {out_key: deq}

    Fp8Dequantize.convert = convert_compat  # type: ignore[method-assign]


def apply_fp8_dequantize_compat() -> None:
    """Apply scalar-scale FP8 dequant workaround if not already applied."""
    global _APPLIED
    if _APPLIED:
        return
    from transformers.integrations.finegrained_fp8 import Fp8Dequantize

    _deq_one = getattr(Fp8Dequantize, "_dequantize_one", None)
    if callable(_deq_one):
        _apply_legacy_dequantize_one_patch()
    else:
        _apply_convert_patch()
    _APPLIED = True


__all__ = ["apply_fp8_dequantize_compat"]
