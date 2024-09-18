# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from tests.tt_eager.python_api_testing.sweep_tests.run_pytorch_test import (
    generate_test_sweep_parameters,
    run_sweep_tests,
)
from tests.ttnn.python_api_testing.sweep_tests.op_map import op_map
from loguru import logger


def test_run_sweep(device, input_path, user_input):
    """
    Example for running sweep test:

    pytest tests/ttnn/python_api_testing/sweep_tests/run_sweep_test.py --input-path tests/ttnn/python_api_testing/sweep_tests/test_configs/ci_sweep_tests_working/grayskull/ttnn_eltwise_add_test.yaml --input-method cli --cli-input results_ttnn_add
    """

    if len(user_input) < 1:
        logger.error(f"Please pass user input. First user input should be output_folder_path")
        return

    output_folder_path = user_input[0]

    test_sweep_parameters, output_file = generate_test_sweep_parameters(input_path, env="")
    run_sweep_tests(
        test_sweep_parameters, output_folder_path, output_file, run_tests_for_ci=False, op_map=op_map, device=device
    )
