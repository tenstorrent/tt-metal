# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Performance: warm time of one chunk at several start positions (testing/positions.py), spec BRINGUP_SPEC."""

from models.demos.common.bringup.testing.harness import device_timeout, mesh_parametrize, spec
from models.demos.common.bringup.testing.positions import run_positions

S = spec()
pytestmark = device_timeout(S)


@mesh_parametrize
def test_positions(mesh_device):
    rows = run_positions(S, mesh_device)
    assert rows and all(ms > 0 for _, ms in rows)
