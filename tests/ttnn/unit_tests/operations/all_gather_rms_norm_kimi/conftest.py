# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

# Reuse the golden suite's focus-ring mesh fixture (1x4 ring, FABRIC_2D_TORUS_XY, production payload).
from eval.golden_tests.all_gather_rms_norm_kimi.conftest import _fabric_mesh_cache, mesh_device  # noqa: F401
