# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fail closed when the native scheduling assumptions no longer hold."""

import pytest

from models.demos.qwen38_27b_qb2.tests.gdn_resident_schedule import LOOP, MODES, compute_variant
from models.demos.qwen38_27b_qb2.tt.gdn_step.op import kernel_source


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("source", ["", LOOP * 2])
def test_changed_compute_is_rejected(mode, source, expect_error):
    with expect_error(ValueError, "anchor changed or is ambiguous"):
        compute_variant(source, mode)


@pytest.mark.parametrize("mode", [None, "", "approximate"])
def test_unknown_mode_is_rejected(mode, expect_error):
    with expect_error(ValueError, "Unknown resident scheduling mode"):
        compute_variant(kernel_source("compute_resident.cpp"), mode)


def test_only_initial_load_schedule_changes():
    source = kernel_source("compute_resident.cpp")
    assert compute_variant(source, "native") == source
    before, after = source.split(LOOP)
    for mode in MODES[1:]:
        result = compute_variant(source, mode)
        assert result.startswith(before) and result.endswith(after)
        assert result.count("tile_regs_acquire();") == source.count("tile_regs_acquire();")
        assert result.count("tile_regs_release();") == source.count("tile_regs_release();")
