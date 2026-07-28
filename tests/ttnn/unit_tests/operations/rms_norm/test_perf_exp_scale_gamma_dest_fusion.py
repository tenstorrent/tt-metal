# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""pytest entry point for the `scale_gamma_dest_fusion` perf experiment.

All logic lives in the experiment dir:
    tests/ttnn/unit_tests/operations/rms_norm/perf_experiments/scale_gamma_dest_fusion/
This file only exists because a pytest module placed INSIDE `ttnn/ttnn/` is given
the dotted name `ttnn.ttnn....` by pytest's importlib mode, which re-executes
`ttnn/__init__.py` a second time ("Operation with name ... is already registered").

    scripts/run_safe_pytest.sh --run-all \
        tests/ttnn/unit_tests/operations/rms_norm/test_perf_exp_scale_gamma_dest_fusion.py
"""

import importlib.util
from pathlib import Path

_EXP = Path(__file__).resolve().parent / "perf_experiments" / "scale_gamma_dest_fusion" / "bench_test.py"
_spec = importlib.util.spec_from_file_location("sgdf_bench_test", _EXP)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

test_option_menu = _mod.test_option_menu
test_regime_sweep = _mod.test_regime_sweep
test_debug_expand = _mod.test_debug_expand
test_op_reference_zones = _mod.test_op_reference_zones
