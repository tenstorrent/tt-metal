# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PER-TENSOR fp8 dequantization for the Mistral-Medium-3.5 checkpoint.

Donor: ``deepseek_v3/utils/hf_model_utils.py`` (``dequantize_state_dict`` /
``dequantize_weight_tensor``) — the donor map's own weakest entry, and it is weak for a concrete
reason: **no donor in the repo dequantizes per-tensor fp8.** Every fp8 donor is DeepSeek BLOCKWISE
(a ``<name>_scale_inv`` sibling plus ``get_weight_block_shape(hf_config)`` from
``quantization_config.weight_block_size``), and the only other quantized donor is Kimi's packed
INT4. So the scale application itself is written fresh here; what is reused is the donor's structure,
which is the part that actually prevents bugs:

  * the sorted key walk, so the output order is deterministic;
  * "an fp8 tensor with no matching scale" is a HARD FAILURE, never a silent pass-through — a
    forgotten scale would otherwise leave weights ~e4m3-scaled and produce garbage;
  * the ``.to(bfloat16).contiguous()`` exit, so every consumer sees one dtype and layout.

What differs, from ``config.json``'s ``quantization_config``:

    {"quant_method": "fp8", "activation_scheme": "static", "weight_block_size": null,
     "modules_to_not_convert": ["model.vision_tower", "model.multi_modal_projector", "lm_head"]}

``weight_block_size: null`` means compressed-tensors / vLLM PER-TENSOR fp8, so the sibling tensors
are ``<name>.weight`` (float8_e4m3fn) + ``<name>.weight_scale`` (a scalar) + ``<name>.input_scale``,
and the donor's block-broadcast reshape becomes a scalar multiply. ``input_scale`` is an ACTIVATION
scale under the static scheme; the device computes activations in bf16, so those keys are dropped
rather than applied — applying one to a weight would scale it twice.
``modules_to_not_convert`` are stored unquantized and pass through untouched.

Cross-checked against ``models/demos/blackhole/qwen36/tt/tp_common.py`` (``dequant_fp8_block``) as a
second reading of fp8 scale application.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from loguru import logger

from models.demos.mistral_3_5_d_p.reference.mistral_config import MistralMedium35Config as C

# Suffixes the checkpoint uses for the quantization side-channel.
WEIGHT_SCALE_SUFFIX = "weight_scale"
INPUT_SCALE_SUFFIX = "input_scale"
# Suffixes that are metadata, not weights: never emitted by the dequantizer.
_SCALE_SUFFIXES = (WEIGHT_SCALE_SUFFIX, INPUT_SCALE_SUFFIX, "weight_scale_inv")

FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


def is_scale_key(name: str) -> bool:
    return any(name.endswith(f".{suffix}") or name == suffix for suffix in _SCALE_SUFFIXES)


def weight_scale_key(weight_name: str) -> str:
    """``...q_proj.weight`` -> ``...q_proj.weight_scale`` (the per-tensor scale's key)."""
    assert weight_name.endswith(".weight"), f"expected a '.weight' key, got {weight_name!r}"
    return f"{weight_name[: -len('.weight')]}.{WEIGHT_SCALE_SUFFIX}"


def is_unquantized_module(name: str, modules_to_not_convert=C.MODULES_TO_NOT_CONVERT) -> bool:
    """True for the modules the checkpoint leaves in bf16 (vision tower, projector, lm_head)."""
    return any(module in name for module in modules_to_not_convert)


def dequantize_weight_tensor(tensor: torch.Tensor, weight_scale: torch.Tensor, *, dtype=torch.bfloat16):
    """Apply a PER-TENSOR fp8 scale: ``w_bf16 = w_fp8.float() * weight_scale``.

    ``weight_scale`` is a scalar (shape ``()`` or ``(1,)``); anything with more than one element
    means the checkpoint is actually blockwise-quantized and this function is the wrong tool, so it
    fails rather than broadcasting something plausible.
    """
    if weight_scale.numel() != 1:
        raise ValueError(
            f"per-tensor fp8 expects a scalar weight_scale, got shape {tuple(weight_scale.shape)}. "
            "A multi-element scale means a BLOCKWISE checkpoint — use the DeepSeek blockwise path "
            "(dequantize_weight_tensor in deepseek_v3/utils/hf_model_utils.py) instead."
        )
    # fp8 -> fp32 before the multiply: e4m3 has 3 mantissa bits, and scaling in fp8 would round
    # twice. The result is cast once, at the end.
    return (tensor.float() * weight_scale.float().reshape(())).to(dtype).contiguous()


def dequantize_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    *,
    dtype: torch.dtype = torch.bfloat16,
    modules_to_not_convert=C.MODULES_TO_NOT_CONVERT,
) -> dict[str, torch.Tensor]:
    """Dequantize a per-tensor fp8 state dict to ``dtype``, dropping the scale keys.

    Every fp8 tensor MUST have a matching ``weight_scale``; one without is an error, because the
    alternative is emitting an unscaled weight that trains no alarm and produces garbage output.
    Non-fp8 tensors are cast (floating point) or copied (integer) so the caller gets one dtype.
    """
    out: dict[str, torch.Tensor] = {}
    n_dequantized = 0
    n_passthrough = 0

    for name in sorted(key for key in state_dict if not is_scale_key(key)):
        tensor = state_dict[name]
        if tensor is None:
            raise ValueError(f"expected tensor {name} to exist in the state dict but it was None")

        if tensor.dtype in FP8_DTYPES:
            scale_name = weight_scale_key(name)
            scale = state_dict.get(scale_name)
            if scale is None:
                raise ValueError(
                    f"found an fp8 tensor {name!r} with no matching per-tensor scale {scale_name!r}. "
                    "Emitting it unscaled would be silently wrong, so this is fatal."
                )
            out[name] = dequantize_weight_tensor(tensor, scale, dtype=dtype)
            n_dequantized += 1
            continue

        if is_unquantized_module(name, modules_to_not_convert):
            # Declared unquantized by the checkpoint; nothing to apply.
            out[name] = tensor.to(dtype).contiguous() if tensor.is_floating_point() else tensor.clone()
            n_passthrough += 1
            continue

        out[name] = tensor.to(dtype).contiguous() if tensor.is_floating_point() else tensor.clone()
        n_passthrough += 1

    logger.info(
        f"[fp8] dequantized {n_dequantized} per-tensor fp8 weights, passed through {n_passthrough} "
        f"(dropped {len(state_dict) - len(out)} scale keys)"
    )
    return out
