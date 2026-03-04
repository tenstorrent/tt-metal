# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility layer - Experts is now RoutedExperts in tt_moe"""
from models.tt_moe.components.experts.routed_experts import RoutedExperts as Experts

__all__ = ["Experts"]
