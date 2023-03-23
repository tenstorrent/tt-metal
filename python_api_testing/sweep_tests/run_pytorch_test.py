import csv
import sys
import time
import torch
import argparse
import yaml
from pathlib import Path
from loguru import logger
from functools import partial

from python_api_testing.sweep_tests import comparison_funcs

from python_api_testing.sweep_tests.common import (
    fieldnames,
    run_test_and_save_results,
    shapes_and_datagen,
)

from python_api_testing.sweep_tests.op_map import op_map


def run_pytorch_test(args):
    # Create output folder
    output_folder = Path(args.output_folder_path)
    if output_folder.exists():
        logger.error(
            f"Directory {output_folder} already exists! Remove this folder or provide a different path to start pytorch tests."
        )
        sys.exit(1)
    output_folder.mkdir()
    logger.info(f"Starting pytorch tests in: {output_folder}")

    ################# PARSE ARGS #################
    pcie_slot = args.pcie_slot
    logger.info(f"Running on device {pcie_slot} for test.")
    sys.exit(0)
    ################# PARSE TEST CONFIGS #################
    with open(args.input_test_config, "r") as stream:
        try:
            pytorch_test_configs_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    assert "test-list" in pytorch_test_configs_yaml
    pytorch_test_list = pytorch_test_configs_yaml["test-list"]

    for test_name, test_config in pytorch_test_list.items():
        assert test_name in op_map
        shape_dict = test_config["shape"]
        datagen_dict = test_config["datagen"]
        results_csv_path = output_folder / test_config["output-file"]

        comparison_dict = test_config["comparison"]
        comparison_args = comparison_dict.get("args", {})
        comparison_func = partial(
            getattr(comparison_funcs, comparison_dict["function"]), **comparison_args
        )

        skip_header = False
        if results_csv_path.exists():
            skip_header = True

        ################# RUN TEST SWEEP #################
        with open(results_csv_path, "a", newline="") as results_csv:
            results_csv_writer = csv.DictWriter(results_csv, fieldnames=fieldnames)
            if not skip_header:
                results_csv_writer.writeheader()
                results_csv.flush()

            for input_shapes, datagen_funcs in shapes_and_datagen(
                shape_dict, datagen_dict
            ):
                data_seed = int(time.time())
                torch.manual_seed(data_seed)

                test_pass = run_test_and_save_results(
                    results_csv_writer,
                    test_name,
                    input_shapes,
                    data_seed,
                    op_map[test_name]["ttmetal_op"],
                    op_map[test_name]["pytorch_op"],
                    input_shapes,
                    datagen_funcs,
                    comparison_func,
                    pcie_slot,
                )
                results_csv.flush()


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
        "--pcie-slot",
        default=0,
        type=int,
        help="Virtual PCIE slot of GS device to run on",
    )

    args = parser.parse_args()

    run_pytorch_test(args)
