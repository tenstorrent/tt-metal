# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


from dataclasses import dataclass
import pytest
import os
import pathlib
import glob
from tests.ttnn.python_api_testing.sweep_tests.op_map import op_map

from tests.tt_eager.python_api_testing.sweep_tests.run_pytorch_test import (
    generate_test_sweep_parameters,
    run_sweep_test,
)


@dataclass
class SweepTest:
    file_name: str
    sweep_test_index: int
    parameters: dict

    def __str__(self):
        return f"{os.path.basename(self.file_name)}-{self.sweep_test_index}"


def create_test_function(file_name):
    test_sweep_parameters, _ = generate_test_sweep_parameters(file_name)

    sweep_tests = []
    for sweep_test_index, parameters in enumerate(test_sweep_parameters):
        sweep_tests.append(SweepTest(file_name, sweep_test_index, parameters))

    @pytest.mark.parametrize("sweep_test", sweep_tests, ids=str)
    def test_sweep(sweep_test, device):
        test_pass = run_sweep_test(sweep_test.parameters, op_map, device)
        assert test_pass

    splitted = file_name.split("/")
    basename = os.path.basename(file_name).replace(".yaml", "")
    working_splitted = splitted[-3].split("_")[-1]
    test_name = f"test_{working_splitted}_{splitted[-2]}_{basename}"

    if test_name.endswith("_test"):
        test_name = test_name[: -len("_test")]

    test_sweep.__name__ = test_name
    return test_sweep


SWEEP_TEST_DIR = pathlib.Path(__file__).parent / "**" / "*.yaml"

for file_name in glob.glob(str(SWEEP_TEST_DIR), recursive=True):
    test_func = create_test_function(file_name)
    globals()[test_func.__name__] = test_func
