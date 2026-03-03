# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
DeepSeek-V3 PyTorch reference implementations.
"""

from .moe_gate import MoEGate, ReferenceMoEGate
from .routed_experts import SingleExpert, SimplifiedRoutedExperts, RoutedExpertsReference, RoutedExperts
from .moe_block import ReferenceMoEBlock, SimplifiedMoEBlock
from .moe_reference import DeepseekV3MoE

__all__ = [
    "MoEGate",
    "ReferenceMoEGate",
    "SingleExpert",
    "SimplifiedRoutedExperts",
    "RoutedExpertsReference",
    "RoutedExperts",
    "ReferenceMoEBlock",
    "SimplifiedMoEBlock",
    "DeepseekV3MoE",
]
