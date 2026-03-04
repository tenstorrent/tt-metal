# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility layer - CCL has been moved to tt_moe"""
from models.tt_moe.collectives.ccl import CCL

__all__ = ["CCL"]
