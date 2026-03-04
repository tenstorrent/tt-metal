# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility layer - MoE has been moved to tt_moe"""
from models.tt_moe.moe_block import MoEBlock as MoE

__all__ = ["MoE"]
