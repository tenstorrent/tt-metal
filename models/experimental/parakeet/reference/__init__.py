# SPDX-FileCopyrightText: © 2026 Abror Shopulatov

# SPDX-License-Identifier: Apache-2.0
"""CPU FP32 PyTorch reference for nvidia/parakeet-tdt-0.6b-v3 (transformers ParakeetForTDT)."""
from .torch_parakeet import ParakeetReference, row_nrmse, strip_pad

__all__ = ["ParakeetReference", "row_nrmse", "strip_pad"]
