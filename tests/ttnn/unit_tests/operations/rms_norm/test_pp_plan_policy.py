# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Driver shim for the `plan_policy` perf experiment (evidence, never a gate).

All logic lives beside the experiment:
`ttnn/ttnn/operations/rms_norm/perf_experiments/plan_policy/`.  A test module
physically inside `ttnn/ttnn/...` does not collect in this tree (pytest re-imports
the ttnn package under a second name and the C++ op registry refuses it), hence
this shim — same pattern as test_fs_graduation_ab.py.

    PP_MODE=probe  scripts/run_safe_pytest.sh           tests/.../test_pp_plan_policy.py -s
    PP_MODE=groups scripts/run_safe_pytest.sh --profile tests/.../test_pp_plan_policy.py -s
"""

import pytest


@pytest.mark.timeout(3600)
def test_pp_plan_policy(device):
    from ttnn.operations.rms_norm.perf_experiments.plan_policy import pp_run

    pp_run.run(device)
