# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Compatibility layer - test_moe has been moved to tt_moe for the modular implementation.
The original test with full git history is now at models/tt_moe/tests/test_moe_block.py

This file provides backwards compatibility for existing references.
"""

# Re-export everything from the new location to ensure fixtures and tests work
from models.tt_moe.tests.test_moe_block import *  # noqa: F401,F403
