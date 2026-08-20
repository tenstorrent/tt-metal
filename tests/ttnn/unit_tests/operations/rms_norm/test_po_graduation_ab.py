# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Driver shim for the pipeline_overlap graduation A/B (perf evidence, not a gate).

All of the logic lives beside the experiment it belongs to:
`ttnn/ttnn/operations/rms_norm/perf_experiments/pipeline_overlap/graduation_ab.py`.
It cannot be a pytest file itself: a test module physically inside `ttnn/ttnn/...`
no longer collects in this tree (pytest re-imports the ttnn package under a second
name and the C++ op registry refuses the duplicate).

    scripts/run_safe_pytest.sh --profile --run-all \\
        tests/ttnn/unit_tests/operations/rms_norm/test_po_graduation_ab.py -s
"""

import os
import sys

import pytest

_LAB = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "..",
    "..",
    "..",
    "..",
    "ttnn",
    "ttnn",
    "operations",
    "rms_norm",
    "perf_experiments",
    "pipeline_overlap",
)
sys.path.insert(0, os.path.normpath(_LAB))


@pytest.mark.timeout(3600)
def test_po_graduation_ab(device):
    import graduation_ab

    graduation_ab.run(device)
