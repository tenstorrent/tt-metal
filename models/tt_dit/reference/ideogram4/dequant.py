# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Dequantize the Ideogram 4.0 weight-only FP8 checkpoints to bf16.

The fp8 checkpoints store each quantized Linear as an e4m3 ``<name>.weight``
(float8_e4m3fn, shape [out, in]) plus a per-output-row ``<name>.weight_scale``
(float32, shape [out]); the reference dequantizes as
``w ≈ weight_fp8.to(dtype) * weight_scale[:, None]`` (see the reference
quantized_loading.py). Non-quantized tensors are passed through (cast to bf16 if
floating point). This yields a plain bf16 state dict that loads into the standard
(un-quantized) reference modules, to serve as the bringup golden.
"""

from __future__ import annotations

import torch

_SCALE_SUFFIX = ".weight_scale"


def dequant_fp8_state_dict(
    sd: dict[str, torch.Tensor], *, dtype: torch.dtype = torch.bfloat16
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    consumed: set[str] = set()

    for key in sd:
        if not key.endswith(_SCALE_SUFFIX):
            continue
        base = key[: -len(_SCALE_SUFFIX)]
        weight_key = base + ".weight"
        w = sd[weight_key].to(torch.float32)  # float8 -> float32
        scale = sd[key].to(torch.float32)
        out[weight_key] = (w * scale[:, None]).to(dtype)
        consumed.add(weight_key)
        consumed.add(key)

    for key, val in sd.items():
        if key in consumed:
            continue
        out[key] = val.to(dtype) if val.is_floating_point() else val

    return out
