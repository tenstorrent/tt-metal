"""Dequantization helpers for the DeepSeek-V4-Flash checkpoint.

The V4-Flash weights ship in two mixed-precision formats (see the checkpoint's
``quantization_config`` and the per-tensor ``.scale`` companions exposed by
:class:`DeepseekV4WeightLoader`). Shapes here are host (torch) ``[R, C]``: ``R`` rows and ``C``
logical columns, where a packed MXFP4 weight stores ``C / 2``.

* **Block FP8** (``fmt=e4m3``, ``scale_fmt=ue8m0``, ``weight_block_size=128x128``)
  for the dense projections and the *shared* expert MLP. Each ``128x128`` block
  of the e4m3 weight shares one power-of-two (e8m0) scale.
* **MXFP4** for the *routed* experts: the weight is packed 2x fp4 (e2m1) per
  ``int8`` byte (so the stored last dim is half the logical one), and every run
  of 32 logical values along the contracted dim shares one e8m0 scale.

The fp4 value table and the nibble-unpacking order match HuggingFace
``transformers`` (``transformers/integrations/mxfp4.py``) so the dequantized
weights are bit-for-bit what the reference forward pass would consume.
"""

from __future__ import annotations

import torch


# e2m1 (fp4) code -> value, matching transformers' ``FP4_VALUES`` (OCP MXFP4).
# Index is the 4-bit code: top bit = sign, next two = exponent, low = mantissa.
FP4_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)

# e8m0 stores a pure biased exponent (bias 127); value == 2 ** (byte - 127).
_E8M0_BIAS = 127


def _e8m0_to_exponent(scale: torch.Tensor) -> torch.Tensor:
    """Reinterpret an e8m0 (or raw uint8) scale ``[R, C]`` as its int32 exponents ``[R, C]``.

    ``2 ** (byte - 127)`` is the represented value; we return ``byte - 127`` so callers
    can fold it in with :func:`torch.ldexp`.
    """
    if scale.dtype == torch.uint8:
        raw = scale
    else:
        # float8_e8m0fnu (and any other 1-byte dtype) -> reinterpret the byte.
        raw = scale.view(torch.uint8)
    return raw.to(torch.int32) - _E8M0_BIAS


def dequantize_mxfp4(
    weight: torch.Tensor,
    scale: torch.Tensor,
    block_size: int = 32,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Dequantize an MXFP4 weight ``[R, C/2]`` int8 -> ``[R, C]`` in ``dtype``.

    Two fp4 nibbles per byte, the low nibble holding the lower logical index (matching
    transformers). ``scale`` is the e8m0 companion ``[R, C/block_size]`` -- uint8 or
    float8_e8m0fnu -- and ``block_size`` (32) is the number of logical values sharing one scale
    along the last dim. Raises if the expanded scales do not span exactly ``C``.
    """
    packed = weight.view(torch.uint8)
    rows, cols_b = packed.shape
    lut = torch.tensor(FP4_VALUES, dtype=dtype)

    lo = (packed & 0x0F).to(torch.long)
    hi = (packed >> 4).to(torch.long)
    out = torch.empty(rows, cols_b * 2, dtype=dtype)
    out[:, 0::2] = lut[lo]
    out[:, 1::2] = lut[hi]

    exponent = _e8m0_to_exponent(scale).repeat_interleave(block_size, dim=1)
    if exponent.shape[1] != out.shape[1]:
        raise ValueError(f"mxfp4 scale span {exponent.shape[1]} != unpacked width {out.shape[1]}")
    return torch.ldexp(out, exponent)


def dequantize_fp8_block(
    weight: torch.Tensor,
    scale: torch.Tensor,
    block: tuple[int, int] = (128, 128),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Dequantize a block-FP8 weight ``[R, C]`` -> ``[R, C]`` in ``dtype``.

    ``weight`` is ``float8_e4m3fn``; ``scale`` is the e8m0 companion
    ``[ceil(R/bh), ceil(C/bw)]`` for the ``(bh, bw)`` block size (``128x128`` for V4-Flash), and
    is cropped to the weight's own extent, so a padded scale grid is fine.
    """
    w = weight.to(dtype)
    bh, bw = block
    exponent = _e8m0_to_exponent(scale)
    exponent = exponent.repeat_interleave(bh, dim=0).repeat_interleave(bw, dim=1)
    exponent = exponent[: w.shape[0], : w.shape[1]].contiguous()
    return torch.ldexp(w, exponent)


def dequantize_weight(
    weight: torch.Tensor, scale: torch.Tensor | None, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Dequantize ``weight`` given its companion ``scale`` (or pass through).

    ``weight`` is ``[R, C]`` -- or ``[R, C/2]`` when packed MXFP4 -- and the result is ``[R, C]``
    in ``dtype``. Dispatches on the *weight* dtype: ``int8`` -> MXFP4, ``float8_e4m3fn`` ->
    block-FP8. Anything else (bf16/fp32 unquantized) is returned cast to ``dtype`` and the scale,
    which should be ``None``, is ignored.
    """
    if scale is None:
        return weight.to(dtype)
    if weight.dtype == torch.int8:
        return dequantize_mxfp4(weight, scale, dtype=dtype)
    if weight.dtype == torch.float8_e4m3fn:
        return dequantize_fp8_block(weight, scale, dtype=dtype)
    return weight.to(dtype)


__all__ = [
    "FP4_VALUES",
    "dequantize_mxfp4",
    "dequantize_fp8_block",
    "dequantize_weight",
]
