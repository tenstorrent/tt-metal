import os
import sys
import time
import torch
from pathlib import Path
from loguru import logger
from functools import partial

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")

from python_api_testing.sweep_tests import comparison_funcs, generation_funcs
from python_api_testing.sweep_tests.op_map import op_map

from python_api_testing.sweep_tests.common import (
    run_tt_lib_test,
)


def run_single_pytorch_test(
    test_name,
    input_shapes,
    datagen_funcs,
    comparison_func,
    pcie_slot,
    test_args={},
    env="",
):
    assert test_name in op_map.keys()

    default_env_dict = {"TT_PCI_DMA_BUF_SIZE": "1048576"}
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
    logger.info(f"Running with shape: {input_shapes} on device: {pcie_slot}")
    test_pass, test_output = run_tt_lib_test(
        op_map[test_name]["tt_lib_op"],
        op_map[test_name]["pytorch_op"],
        input_shapes,
        datagen_funcs,
        comparison_func,
        pcie_slot,
        test_args,
    )
    logger.debug(f"Test pass/fail: {test_pass} with {test_output}")
    logger.debug(f"Test args: {test_args}")

    # Unset env variables
    for key, value in old_env_dict.items():
        os.environ.pop(key)
        if value is not None:
            os.environ[key] = value

    assert test_pass, f"{test_name} test failed with input shape {input_shapes}."
    logger.info(f"{test_name} test passed with input shape {input_shapes}.")


# Can also run with: pytest tests/python_api_testing/sweep_tests/run_pytorch_ci_tests.py -svvv
# def test_datacopy_test():
#    datagen_func = generation_funcs.gen_func_with_cast(
#        partial(generation_funcs.gen_rand, low=-100, high=100), torch.bfloat16
#    )
#    comparison_func = partial(comparison_funcs.comp_equal)
#    run_single_pytorch_test(
#        "datacopy",
#        [[1, 1, 32, 32]],
#        [datagen_func],
#        comparison_func,
#        0,
#    )

if __name__ == "__main__":
    datagen_func = generation_funcs.gen_func_with_cast(
        partial(generation_funcs.gen_rand, low=-100, high=100), torch.bfloat16
    )
    comparison_func = partial(comparison_funcs.comp_equal)
    run_single_pytorch_test(
        "datacopy",
        [[1, 1, 32, 32]],
        [datagen_func],
        comparison_func,
        0,
    )
