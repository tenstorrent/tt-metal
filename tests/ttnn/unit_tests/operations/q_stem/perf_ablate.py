# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Ablation twin of eval/golden_tests/q_stem/perf_measure.py (same ccl_perf protocol, PERF FOCUS case).

Not collected by default (no test_ prefix). With QSTEM_ABLATE=<STAGE,...> set, the program descriptor stubs stage
payloads (wrong output by design), so the golden check is skipped; otherwise it runs like perf_measure.py.

    QSTEM_ABLATE=OUT_WRITE CCL_PERF_OUT=<dir> scripts/run_safe_pytest.sh --no-precompile \\
        tests/ttnn/unit_tests/operations/q_stem/perf_ablate.py -s
"""

from __future__ import annotations

import os
from pathlib import Path

from eval import ccl_perf

for _k, _v in ccl_perf.PROFILER_ENV.items():
    os.environ.setdefault(_k, _v)

import pytest  # noqa: E402

from eval.golden_tests.q_stem.conftest import _fabric_mesh_cache, focus_mesh_or_skip  # noqa: E402,F401
from eval.golden_tests.q_stem.helpers import prepare_q_stem  # noqa: E402
from eval.golden_tests.q_stem.test_golden import _loose_cases  # noqa: E402
from ttnn.operations.q_stem import q_stem  # type: ignore  # noqa: E402
from ttnn.operations.q_stem import q_stem_program_descriptor as _pd  # noqa: E402

# QSTEM_KNOBS="S1_BLOCK_M_TILES=5;X_INJECTOR_NOCS=(0, 0)": override program-descriptor knobs for a sweep run (fresh process
# per variant, so the program cache never mixes variants).
for _kv in filter(None, os.environ.get("QSTEM_KNOBS", "").split(";")):
    _k, _v = _kv.split("=")
    assert hasattr(_pd, _k), _k
    setattr(_pd, _k, type(getattr(_pd, _k))(eval(_v)))

EXPERIMENTS = int(os.environ.get("CCL_PERF_EXPERIMENTS", str(ccl_perf.CHECK_EXPERIMENTS)))
OUT = Path(os.environ.get("CCL_PERF_OUT", ".eval/ccl_perf_ablate"))
ABLATE = os.environ.get("QSTEM_ABLATE", "")


@pytest.fixture
def perf_mesh(request, _fabric_mesh_cache):  # noqa: F811
    extras = request.node.callspec.params["extras"]
    return _fabric_mesh_cache.get(
        focus_mesh_or_skip(),
        trace_region_size=ccl_perf.TRACE_REGION_SIZE,
        payload_bytes=extras.get("fabric_payload_bytes"),
    )


@pytest.mark.timeout(7200)
@pytest.mark.parametrize("inputs,axes,extras", _loose_cases()[:1])
def test_perf_ablate(inputs, axes, extras, perf_mesh):
    call, check = prepare_q_stem(inputs, mesh_device=perf_mesh, extras=extras, op=q_stem, **axes)
    out = call()
    if not ABLATE:
        check(out)
    knobs = os.environ.get("QSTEM_KNOBS", "")
    label = f"ablate={ABLATE or 'none'} knobs={knobs or 'default'}"
    ccl_perf.run(
        perf_mesh,
        call,
        experiments=EXPERIMENTS,
        out_path=OUT
        / f"{(ABLATE or 'none').replace(',', '_')}{('__' + knobs.replace(';', '_').replace('=', '').replace(' ', '').replace(',', '-').replace('(', '').replace(')', '')) if knobs else ''}.json",
        label=label,
        meta={"op": "q_stem", "ablate": ABLATE, "knobs": knobs, "experiments": EXPERIMENTS},
    )
