# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""On-device determinism helpers; they live in tests/measure/determinism.py."""

from models.demos.deepseek_v3_d_p.tests.measure.determinism import (  # noqa: F401
    deterministic_input,
    device_mismatch_marker,
    device_mismatch_marker_is_set,
    fold_mismatches,
    merge_device_mismatch_markers,
    mismatch_report,
)
