# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Test parametrization for the 4x4 prefill stage (a Galaxy tray half, tests/run_4x4_stage_tests.sh).

A 4x4 stage runs 2560-token chunks at SP=4, the same 640 rows per chip as the 8x4 stage's 5120 at SP=8, so
the per-chip depths the 8x4 rows already use cover it; only cases a 4x4 stage cannot host are deselected.
"""


def is_4x4(**params) -> bool:
    """``uncollect_if`` predicate: the row runs on the 4x4 stage mesh."""
    return tuple(params.get("mesh_device") or ()) == (4, 4)
