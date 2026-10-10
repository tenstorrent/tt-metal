# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Frozen compact GDN coverage, independent of device imports."""

MODES = ((False, False, 0), (False, True, 0), (True, False, 0), (True, True, 0), (True, True, 2560))
CASES = [(b, m, mode) for b in (16, 32) for m in ("l1", "dram") for mode in MODES]
CASES += [(b, m, MODES[-1]) for b in (1, 17, 31) for m in ("l1", "dram")]
BASELINE = "single_step_flat_prepare_epilogue"
CANDIDATE = "single_step_compact_gdn"
