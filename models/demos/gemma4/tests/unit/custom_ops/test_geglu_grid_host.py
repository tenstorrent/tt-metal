# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host-only: custom_ops/geglu_routed_expert's grid rule. The kernel family is Blackhole-only and runs on an 11x8
rectangle taken from the device grid; a grid that cannot host it (the Wormhole 8x8) must be reported, so callers
and the device tests skip instead of failing. No device, no conftest (run with --noconftest)."""

from models.demos.gemma4.custom_ops.geglu_routed_expert.op import GegluRoutedExpert


def test_blackhole_grid_hosts_the_rectangle():
    assert GegluRoutedExpert.grid_reason(13, 10, True) is None
    assert GegluRoutedExpert.grid_reason(11, 8, True) is None


def test_small_grids_and_other_archs_report_blackhole_only():
    for gx, gy, bh in ((8, 8, True), (10, 8, True), (11, 7, True), (13, 10, False), (8, 8, False)):
        reason = GegluRoutedExpert.grid_reason(gx, gy, bh)
        assert reason is not None and reason.startswith("Blackhole-only"), (gx, gy, bh, reason)
