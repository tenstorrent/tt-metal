# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Offline weight preprocessing for native TT BFP inference."""
from .calibration import HessianAccumulator, capture_linear_inputs
from .quantize import HessianFactor, factor_hessian, gptq_search, search_linear, search_packed, to_bf16_exact
from .validation import validate_repacking
from .checkpoint import export_checkpoint

__version__ = "0.2.0"
__all__ = [
    "HessianAccumulator",
    "capture_linear_inputs",
    "HessianFactor",
    "factor_hessian",
    "gptq_search",
    "search_linear",
    "search_packed",
    "to_bf16_exact",
    "validate_repacking",
    "export_checkpoint",
]
