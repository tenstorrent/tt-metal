# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
MaskFormer with Swin-Base backbone (experimental).

This package hosts the TT-NN reference implementation and demo entrypoints for
running MaskFormer-Swin-B on Tenstorrent devices. Modules are intentionally
split by sub-component to mirror the high-level architecture; see
``README.md`` for the engineering plan.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "MaskFormerSwinBackbone",
    "SwinBackboneConfig",
    "MaskFormerPixelDecoder",
    "PixelDecoderConfig",
    "MaskFormerTransformerDecoder",
    "TransformerDecoderConfig",
    "MaskFormerHeads",
    "MaskFormerHeadsConfig",
    "MaskFormerFallbackPipeline",
    "parity",
]


_LAZY_IMPORTS = {
    "MaskFormerSwinBackbone": ".backbone_swin",
    "SwinBackboneConfig": ".backbone_swin",
    "MaskFormerPixelDecoder": ".pixel_decoder",
    "PixelDecoderConfig": ".pixel_decoder",
    "MaskFormerTransformerDecoder": ".transformer_decoder",
    "TransformerDecoderConfig": ".transformer_decoder",
    "MaskFormerHeads": ".heads",
    "MaskFormerHeadsConfig": ".heads",
    "MaskFormerFallbackPipeline": ".fallback",
}


def __getattr__(name: str) -> Any:  # pragma: no cover - exercised implicitly via imports
    if name == "parity":
        module = import_module(".parity", __name__)
        globals()[name] = module
        return module

    module_path = _LAZY_IMPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    module = import_module(module_path, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
