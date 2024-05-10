# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from importlib.machinery import SourceFileLoader
from .sweeps import (
    SWEEP_SOURCES_DIR,
    permutations,
    run_single_test,
)
from dataclasses import dataclass
import pytest
import os


@dataclass
class SweepTest:
    file_name: str
    sweep_test_index: int
    parameter_list: dict

    def __str__(self):
        return f"{os.path.basename(self.file_name)}-{self.sweep_test_index}-{self.parameter_list}"


sweep_tests = []
for file_name in sorted(SWEEP_SOURCES_DIR.glob("**/*.py")):
    sweep_tests.append(file_name)


def create_test_function(file_name):
    base_name = os.path.basename(file_name)
    base_name = os.path.splitext(base_name)[0]
    sweep_module = SourceFileLoader(f"sweep_module_{base_name}", str(file_name)).load_module()
    base_name = base_name + ".csv"
    sweep_tests = []
    for sweep_test_index, parameter_list in enumerate(permutations(sweep_module.parameters)):
        sweep_tests.append(SweepTest(file_name, sweep_test_index, parameter_list))

    @pytest.mark.parametrize("sweep_test", sweep_tests, ids=str)
    def test_sweep(device, sweep_test):
        status, message = run_single_test(
            sweep_test.file_name,
            sweep_test.sweep_test_index,
            device=device,
        )
        assert status not in {"failed", "crashed"}, f"{message}"

    test_sweep.__name__ = f"test_{os.path.basename(file_name).replace('.py', '')}"
    return test_sweep


for file_name in sweep_tests:
    test_func = create_test_function(file_name)
    globals()[test_func.__name__] = test_func
