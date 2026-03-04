# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility layer - MoEGate is now GroupedTopKRouter in tt_moe"""
from models.tt_moe.components.routers.grouped_topk_router import GroupedTopKRouter as MoEGate

__all__ = ["MoEGate"]
