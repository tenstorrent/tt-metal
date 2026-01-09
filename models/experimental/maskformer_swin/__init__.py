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
    "MaskFormerSwinBackbone": ".tt.backbone_swin",
    "SwinBackboneConfig": ".tt.backbone_swin",
    "MaskFormerPixelDecoder": ".tt.pixel_decoder",
    "PixelDecoderConfig": ".tt.pixel_decoder",
    "MaskFormerTransformerDecoder": ".tt.transformer_decoder",
    "TransformerDecoderConfig": ".tt.transformer_decoder",
    "MaskFormerHeads": ".tt.heads",
    "MaskFormerHeadsConfig": ".tt.heads",
    "MaskFormerFallbackPipeline": ".tt.fallback",
}


def __getattr__(name: str) -> Any:  # pragma: no cover - exercised implicitly via imports
    if name == "parity":
        module = import_module(".tt.parity", __name__)
        globals()[name] = module
        return module

    module_path = _LAZY_IMPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    module = import_module(module_path, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
