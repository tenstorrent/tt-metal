# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-core timeline of the PERF FOCUS call from device zones (not a correctness test; no test_ prefix).

Runs the focus case a few times with the device profiler dumping to generated/profiler/.logs, then
summarize with tests/ttnn/unit_tests/operations/q_stem/zone_summary.py.

    TT_METAL_DEVICE_PROFILER=1 scripts/run_safe_pytest.sh --no-precompile \\
        tests/ttnn/unit_tests/operations/q_stem/zone_probe.py -s
"""

from __future__ import annotations

import os

import pytest
import ttnn

from eval.golden_tests.q_stem.conftest import _fabric_mesh_cache, focus_mesh_or_skip  # noqa: F401
from eval.golden_tests.q_stem.helpers import prepare_q_stem
from eval.golden_tests.q_stem.test_golden import _loose_cases
from ttnn.operations.q_stem import q_stem  # type: ignore
from ttnn.operations.q_stem import q_stem_program_descriptor as _pd

# QSTEM_KNOBS="X_INJECTORS=1;...": program-descriptor knob overrides (as in perf_ablate.py)
for _kv in filter(None, os.environ.get("QSTEM_KNOBS", "").split(";")):
    _k, _v = _kv.split("=")
    assert hasattr(_pd, _k), _k
    setattr(_pd, _k, type(getattr(_pd, _k))(eval(_v)))


@pytest.fixture
def zmesh(request, _fabric_mesh_cache):  # noqa: F811
    extras = request.node.callspec.params["extras"]
    return _fabric_mesh_cache.get(focus_mesh_or_skip(), payload_bytes=extras.get("fabric_payload_bytes"))


@pytest.mark.parametrize("inputs,axes,extras", _loose_cases()[:1])
def test_zone_probe(inputs, axes, extras, zmesh):
    call, _ = prepare_q_stem(inputs, mesh_device=zmesh, extras=extras, op=q_stem, **axes)
    for _ in range(3):
        call()
    ttnn.synchronize_device(zmesh)
    ttnn.ReadDeviceProfiler(zmesh)
