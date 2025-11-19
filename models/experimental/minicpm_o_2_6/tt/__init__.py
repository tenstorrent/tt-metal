# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
MiniCPM-o-2_6 TTNN Implementation

This package contains TTNN implementations of MiniCPM-o-2_6 multimodal model components.
"""

from .common import (
    get_weights_memory_config,
    get_activations_memory_config,
    torch_to_ttnn,
    ttnn_to_torch,
)

from .test_utils import (
    compute_pcc,
    validate_pcc,
)

try:
    from .minicpm_qwen_model import MiniCPMQwenModel
    from .minicpm_transformer import MiniCPMTransformer

    _minicpm_available = True
except ImportError:
    _minicpm_available = False

__all__ = [
    "get_weights_memory_config",
    "get_activations_memory_config",
    "torch_to_ttnn",
    "ttnn_to_torch",
    "compute_pcc",
    "validate_pcc",
]

if _minicpm_available:
    __all__.extend(["MiniCPMQwenModel", "MiniCPMTransformer"])
