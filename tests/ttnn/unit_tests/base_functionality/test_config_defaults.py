# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Locks the shipped defaults of ttnn.CONFIG that nothing else would notice being changed."""

import ttnn


def test_validate_program_args_defaults_on():
    # Turning validation off only makes the runtime more permissive, so a revert of this default
    # would not fail any other test — assert it here instead.
    assert ttnn.CONFIG.validate_program_args is True

    original = ttnn.CONFIG.validate_program_args
    try:
        ttnn.CONFIG.validate_program_args = False
        assert ttnn.CONFIG.validate_program_args is False
    finally:
        ttnn.CONFIG.validate_program_args = original
    assert ttnn.CONFIG.validate_program_args is True
