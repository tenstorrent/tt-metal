# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F56 device check: the built-in component checks (checks="auto") on the device for every component test of a model
(STEP / LAYER read from its frozen test files). Must pass: a failure is a false alarm of the built-in checks on a
device implementation that passed its own (reviewed) test.

    BRINGUP_SPEC=models/demos/<model>/bringup/spec.yaml scripts/run_safe_pytest.sh --run-all \\
        models/demos/common/bringup/dev/f56_device_check.py [-k <step>]
"""
import pytest

from models.demos.common.bringup.dev.f49_mutation_proof import install_caches
from models.demos.common.bringup.dev.f56_mutation_proof import component_tests
from models.demos.common.bringup.testing.component import run_component_test
from models.demos.common.bringup.testing.harness import mesh_parametrize, spec

S = spec()
install_caches(S)  # one reference per layer; the golden source top-k for shared layers (models without swap_context)
CASES = [(t["step"], t["layer"], t["compare"], t["thr"]) for t in component_tests(S)]


@pytest.mark.parametrize("step,layer,mode,thr", CASES, ids=[f"{c[0]}_L{c[1]}" for c in CASES])
@mesh_parametrize
def test_f56_device(mesh_device, step, layer, mode, thr):
    assert run_component_test(S, step, layer, mesh_device, mode, thr, checks="auto")
