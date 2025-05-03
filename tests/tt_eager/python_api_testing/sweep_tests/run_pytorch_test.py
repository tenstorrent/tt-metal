# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import csv
import sys
import time
import copy
import torch
import argparse
import yaml
import random
from pathlib import Path
from loguru import logger
from functools import partial
import ttnn
from itertools import permutations, product
from tracy import signpost


from tests.tt_eager.python_api_testing.sweep_tests import comparison_funcs, generation_funcs

from tests.tt_eager.python_api_testing.sweep_tests.common import (
    get_test_fieldnames,
    run_test_and_save_results,
    shapes_and_datagen,
)

DTYPES_TT_DICT = {
    "BFLOAT16": ttnn.bfloat16,
    "BFLOAT8_B": ttnn.bfloat8_b,
    "UINT32": ttnn.uint32,
    "UINT16": ttnn.uint16,
    "INT32": ttnn.int32,
    "BFLOAT4_B": ttnn.bfloat4_b,
    "FLOAT32": ttnn.float32,
}

LAYOUTS_TT_DICT = {
    "ROW_MAJOR": ttnn.ROW_MAJOR_LAYOUT,
    "TILE": ttnn.TILE_LAYOUT,
}

MEM_CONFIGS_TT_DICT = {
    "DRAM": ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
    "L1": ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
    "SYSTEM_MEMORY": None,
}


def make_env_combinations(env_dict):
    envs = []

    for key in env_dict:
        if isinstance(env_dict[key], list):
            combinations = []

            for value in env_dict[key]:
                combinations.append((key, value))

            envs.append(combinations)
        else:
            envs.append([(key, env_dict[key])])

    return product(*envs)


def generate_test_sweep_parameters(input_test_config, env=""):
    ################# PARSE TEST CONFIGS #################
    with open(input_test_config, "r") as stream:
        try:
            pytorch_test_configs_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logger.error(exc)
            raise exc

    assert "test-list" in pytorch_test_configs_yaml
    pytorch_test_list = pytorch_test_configs_yaml["test-list"]

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

    # make list
    if isinstance(pytorch_test_list, dict):
        pytorch_test_list = [pytorch_test_list]

    generated_test_sweep_parameters = []
    random.seed(0)

    for i in range(len(pytorch_test_list)):
        for test_name, test_config in pytorch_test_list[i].items():
            # Get env variables from yaml (yaml overrides CLI)
            yaml_env_dict = test_config.get("env", {})

            # Env variables to use (precedence yaml > cli > default)
            if yaml_env_dict:
                env_dict = yaml_env_dict
            elif args_env_dict:
                env_dict = args_env_dict
            else:
                env_dict = {}

            env_dict_combinations = make_env_combinations(env_dict)

            coregrid_dict = test_config.get("coregrid", {})

            for env_dict in env_dict_combinations:
                shape_dict = test_config["shape"]
                datagen_dict = test_config["datagen"]

                comparison_dict = test_config["comparison"]
                comparison_args = comparison_dict.get("args", {})
                comparison_func = partial(getattr(comparison_funcs, comparison_dict["function"]), **comparison_args)

                test_args_gen = getattr(
                    generation_funcs,
                    test_config.get("args-gen", "gen_default_dtype_layout_device"),
                )
                sanitize_args = test_config.get("sanitize-args", True)

                # Optional test args for dtype, etc...
                test_args = test_config.get("args", {})

                if coregrid_dict:
                    test_args["coregrid"] = coregrid_dict

                # Set tests parameters --------------------------
                test_tt_dtypes = []
                test_tt_layouts = []
                test_mem_configs = []
                test_coregrid_configs = []

                if "inputs" in test_args:
                    for input_spec in test_args["inputs"]:
                        test_tt_dtypes.append([])
                        test_tt_layouts.append([])
                        test_mem_configs.append([])

                        assert "data-type" in input_spec, f"For each input you need to specify 'data-type'"
                        assert "data-layout" in input_spec, f"For each input you need to specify 'data-layout'"
                        assert "buffer-type" in input_spec, f"For each input you need to specify 'buffer-type'"

                        for dtype in input_spec["data-type"]:
                            test_tt_dtypes[-1].append(DTYPES_TT_DICT[dtype])

                        for layout in input_spec["data-layout"]:
                            test_tt_layouts[-1].append(LAYOUTS_TT_DICT[layout])

                        for buffer_type in input_spec["buffer-type"]:
                            test_mem_configs[-1].append(MEM_CONFIGS_TT_DICT[buffer_type])
                else:
                    for i in range(shape_dict["num-shapes"]):
                        test_tt_dtypes.append([])
                        test_tt_layouts.append([])
                        test_mem_configs.append([])

                        if "data-layout" in test_args:
                            for layout in test_args["data-layout"]:
                                test_tt_layouts[-1].append(LAYOUTS_TT_DICT[layout])
                        else:
                            test_tt_layouts[-1] = generation_funcs.supported_tt_layouts

                        if "data-type" in test_args:
                            for dtype in test_args["data-type"]:
                                test_tt_dtypes[-1].append(DTYPES_TT_DICT[dtype])
                        else:
                            test_tt_dtypes[-1] = generation_funcs.supported_tt_dtypes

                        if "buffer-type" in test_args:
                            for buffer_type in test_args["buffer-type"]:
                                test_mem_configs[-1].append(MEM_CONFIGS_TT_DICT[buffer_type])
                        else:
                            test_mem_configs[-1] = generation_funcs.supported_mem_configs

                if "coregrid" in test_args:
                    test_coregrid_configs.append(coregrid_dict.get("xmin", {}))
                    test_coregrid_configs.append(coregrid_dict.get("xmax", {}))
                    test_coregrid_configs.append(coregrid_dict.get("ymin", {}))
                    test_coregrid_configs.append(coregrid_dict.get("ymax", {}))

                if "outputs" in test_args:
                    for out_spec in test_args["outputs"]:
                        test_mem_configs.append([])

                        assert "out-buffer-type" in out_spec, f"For output you need to specify 'out-buffer-type'"

                        for buffer_type in out_spec["out-buffer-type"]:
                            test_mem_configs[-1].append(buffer_type)
                else:
                    test_mem_configs.append([])

                    if "out-buffer-type" in test_args:
                        for buffer_type in test_args["out-buffer-type"]:
                            test_mem_configs[-1].append(MEM_CONFIGS_TT_DICT[buffer_type])
                    else:
                        test_mem_configs[-1] = [MEM_CONFIGS_TT_DICT["DRAM"]]
                # Set tests parameters --------------------------

                ################# RUN TEST SWEEP #################
                for input_shapes, datagen_funcs, generated_test_args in shapes_and_datagen(
                    shape_dict,
                    datagen_dict,
                    test_args_gen,
                    test_tt_dtypes,
                    test_tt_layouts,
                    test_mem_configs,
                    sanitize_args=sanitize_args,
                    coregrid=test_coregrid_configs,
                ):
                    data_seed = random.randint(0, 20000000)
                    # input_shapes = input_shapes.copy()

                    parameters = {}
                    parameters["data_seed"] = data_seed
                    parameters["input_shapes"] = input_shapes
                    parameters["env_dict"] = env_dict
                    parameters["test_name"] = test_name
                    parameters["generated_test_args"] = generated_test_args
                    parameters["comparison_func"] = comparison_func
                    parameters["datagen_funcs"] = datagen_funcs

                    generated_test_sweep_parameters.append(parameters)

    return generated_test_sweep_parameters, test_config["output-file"]


def run_sweep_test(parameters, op_map, device, results_csv_writer=None):
    data_seed = parameters["data_seed"]
    input_shapes = parameters["input_shapes"]
    test_name = parameters["test_name"]

    torch.manual_seed(parameters["data_seed"])
    logger.info(f"Running with shape: {input_shapes} and seed: {data_seed}")

    old_env_dict = {}

    for key, value in parameters["env_dict"]:
        old_env_dict[key] = os.environ.pop(key, None)

        if value != "" and value is not None:
            os.environ[key] = value

    test_pass = run_test_and_save_results(
        results_csv_writer,
        test_name,
        input_shapes,
        data_seed,
        parameters["env_dict"],
        parameters["generated_test_args"],
        op_map[test_name]["tt_op"],
        op_map[test_name]["pytorch_op"],
        input_shapes,
        parameters["datagen_funcs"],
        parameters["comparison_func"],
        parameters["generated_test_args"],
        device,
    )

    # Unset env variables
    for key, value in old_env_dict.items():
        os.environ.pop(key, None)
        if value is not None:
            os.environ[key] = value

    return test_pass


def run_sweep_tests(test_sweep_parameters, output_folder, output_file, run_tests_for_ci, op_map, device):
    # Create output folder
    output_folder = Path(output_folder)

    if not output_folder.exists():
        output_folder.mkdir(parents=True)
        write_to_csv = True
        logger.info(f"Starting pytorch tests in: {output_folder}. Writing to csv.")
    else:
        write_to_csv = False
        logger.info(f"Not logging results in {output_folder}. Delete that folder to write csv results.")

    start_time = time.time()
    run_id = 0

    for parameters in test_sweep_parameters:
        results_csv_path = output_folder / output_file

        # Moved this here so that we don't need to maintain a hardcoded list of headers per op
        skip_header = results_csv_path.exists()

        with open(results_csv_path, "a", newline="") as results_csv:
            results_csv_writer = None

            if write_to_csv:
                results_csv_writer = csv.DictWriter(results_csv, fieldnames=get_test_fieldnames(["args"]))

                if not skip_header:
                    results_csv_writer.writeheader()
                    results_csv.flush()

            test_profiling_key = f"test_sweep_separator - {run_id}"
            logger.info(f"Starting profiling test {test_profiling_key}")
            signpost(header=test_profiling_key)

            test_pass = run_sweep_test(parameters, op_map, device, results_csv_writer)

            ttnn.synchronize_device(device)
            signpost(header=f"{test_profiling_key} - end")
            logger.info(f"Stopped profiling test {test_profiling_key}")
            run_id += 1

            results_csv.flush()

            # Check if test passed
            if run_tests_for_ci and not test_pass:
                logger.error(f"{parameters['test_name']} test failed with input shape {parameters['input_shapes']}.")
                sys.exit(1)

    duration = time.time() - start_time
    logger.info(f"Tests run in {duration:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pytorch testing infra")
    parser.add_argument(
        "-i",
        "--input-test-config",
        help="Input pytorch test config",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output-folder-path",
        default="pytorch_test_folder",
        help="Output pytorch test folder",
    )
    parser.add_argument(
        "-s",
        "--device-id",
        default=0,
        type=int,
        help="Id of the device to run on",
    )
    parser.add_argument(
        "-e",
        "--env",
        type=str,
        default="",
        help="Env variables to set",
    )
    parser.add_argument(
        "--run-tests-for-ci",
        action="store_true",
        help="If set, assert on test result after every test.",
    )
    args = parser.parse_args()

    device_id = args.device_id
    device = ttnn.open_device(device_id)
    ttnn.SetDefaultDevice(device)

    logger.info(f"Running on device {device_id} for test.")

    test_sweep_parameters, output_file = generate_test_sweep_parameters(args.input_test_config, args.env)
    run_sweep_tests(test_sweep_parameters, args.output_folder_path, output_file, args.run_tests_for_ci, device)

    ttnn.close_device(device)
