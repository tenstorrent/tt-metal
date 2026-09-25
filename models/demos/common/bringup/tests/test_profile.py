# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Performance review, measurement: warm per-section, per-chip device time of one long chunk (testing/profile.py).
Needs TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1."""

from models.demos.common.bringup.testing.harness import mesh_parametrize, spec
from models.demos.common.bringup.testing.profile import run_profile

S = spec()


@mesh_parametrize
def test_profile(mesh_device):
    out = run_profile(S, mesh_device)
    assert out["sections_ms"]
