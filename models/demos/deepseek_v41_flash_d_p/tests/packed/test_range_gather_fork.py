# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""PB.6L: the fork test ttnn/ttnn/bringup/high_bw_all_gather_ttnn/tests/unit/test_range_gather.py (gathered_dim_start)
on the resident runner's mesh (only test files under models/ share its process: core/batch.shareable). Mesh only."""

import importlib.util
from pathlib import Path

BATCH_SHARED = True
BATCH_LAYERS = "executed"

_F = Path(__file__).resolve().parents[5] / "ttnn/ttnn/bringup/high_bw_all_gather_ttnn/tests/unit/test_range_gather.py"
_spec = importlib.util.spec_from_file_location("pb6l_test_range_gather", _F)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
test_range_gather = _mod.test_range_gather
