# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Performance review, measurement: warm per-section, per-chip device time of one long chunk (testing/profile.py).
Device time from the program real-time profiler (F155; no environment). BRINGUP_PROFILER=device for the device
profiler, with TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1."""

import os

from models.demos.common.bringup.testing.harness import device_timeout, mesh_parametrize, spec
from models.demos.common.bringup.testing.profile import run_profile

S = spec()
pytestmark = device_timeout(S)


@mesh_parametrize
def test_profile(mesh_device):
    out = run_profile(S, mesh_device)
    assert out["wall_ms"] > 0 if os.environ.get("BRINGUP_PROFILE_WALL") == "1" else out["sections_ms"]  # F114
