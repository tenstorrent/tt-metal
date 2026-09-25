# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Integrate step: every ladder rung of the spec (BRINGUP_SPEC) on device. Select a rung with -k <rung name>."""

import pytest

from models.demos.common.bringup.testing.harness import mesh_parametrize, spec
from models.demos.common.bringup.testing.ladder import run_ladder

S = spec()
RUNGS = [r["name"] for r in S.data["ladder"]]


@mesh_parametrize
@pytest.mark.parametrize("rung", RUNGS, ids=RUNGS)
def test_ladder(mesh_device, rung):
    out = run_ladder(S, rung, mesh_device)
    assert not out["failed"], out["failed"]
