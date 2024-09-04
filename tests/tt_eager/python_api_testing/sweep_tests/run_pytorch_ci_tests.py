# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os

from loguru import logger


from tests.tt_eager.python_api_testing.sweep_tests import (
    comparison_funcs,
    generation_funcs,
)
from tests.tt_eager.python_api_testing.sweep_tests.op_map import op_map as op_map_tt_eager
from tests.ttnn.python_api_testing.sweep_tests.op_map import op_map as op_map_ttnn

from tests.tt_eager.python_api_testing.sweep_tests.common import (
    run_tt_lib_test,
)


def run_single_pytorch_test(
    test_name,
    input_shapes,
    datagen_funcs,
    comparison_func,
    device,
    test_args={},
    env="",
    plot_func=None,
    ttnn_op=False,
):
    op_map = op_map_tt_eager
    if not ttnn_op:
        assert test_name in op_map_tt_eager
    else:
        assert test_name in op_map_ttnn
        op_map = op_map_ttnn

    default_env_dict = {}
    # Get env variables from CLI
    args_env_dict = {}
    if env != "":
        envs = env.split(" ")
        for e in envs:
            if "=" not in e:
                name = e
                value = "1"
            else:
                name, value = e.split("=")
            args_env_dict[name] = value

    # Env variables to use (cli > default)
    if args_env_dict:
        env_dict = args_env_dict
    else:
        env_dict = default_env_dict

    old_env_dict = {}
    assert isinstance(env_dict, dict)
    for key, value in env_dict.items():
        old_env_dict[key] = os.environ.pop(key, None)
        os.environ[key] = value

    if not test_args:
        test_args = generation_funcs.gen_default_dtype_layout_device(input_shapes)[0]
    ################# RUN TEST #################
    logger.info(f"Running with shape: {input_shapes} on device: {device.id()}")
    test_pass, test_output = run_tt_lib_test(
        op_map[test_name]["tt_op"],
        op_map[test_name]["pytorch_op"],
        input_shapes,
        datagen_funcs,
        comparison_func,
        test_args,
        device=device,
        plot_func=plot_func,
    )
    logger.debug(f"Test pass/fail: {test_pass} with {test_output}")
    logger.debug(f"Test args: {test_args}")

    # Unset env variables
    for key, value in old_env_dict.items():
        os.environ.pop(key)
        if value is not None:
            os.environ[key] = value

    assert test_pass, f"{test_name} test failed with input shape {input_shapes}. {test_output}"
    logger.info(f"{test_name} test passed with input shape {input_shapes}.")
