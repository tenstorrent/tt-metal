import csv
import sys
import time
import torch
from pathlib import Path
from loguru import logger
from functools import partial

from python_api_testing.sweep_tests import (
    ll_buda_ops,
    pytorch_ops,
    generation_funcs,
    comparison_funcs,
)

from python_api_testing.sweep_tests.common import (
    fieldnames,
    run_test_and_save_results,
    get_args_from_argparser,
    generic_shape_sweeps,
)


def run_pytorch_transpose_hc_test(args):
    pcie_slot = args.pcie_slot
    # Create results_csv and write header
    results_csv_path = Path(args.output_csv_file_path)
    if results_csv_path.exists():
        logger.error(
            f"Result csv {results_csv_path} already exists! Remove it or provide a different path."
        )
        sys.exit()

    num_samples = int(args.num_samples) if args.num_samples else None
    logger.info(f"Running {num_samples if num_samples else 'all'} samples for test.")

    with open(results_csv_path, "w", newline="") as results_csv:
        results_csv_writer = csv.DictWriter(results_csv, fieldnames=fieldnames)
        results_csv_writer.writeheader()
        results_csv.flush()

        for input_shapes in generic_shape_sweeps(
            [1, 32, 32, 32], [1, 2048, 2048, 32], num_samples=num_samples
        ):
            data_seed = int(time.time())
            torch.manual_seed(data_seed)

            test_pass = run_test_and_save_results(
                results_csv_writer,
                "transpose_hc",
                input_shapes,
                data_seed,
                ll_buda_ops.transpose_hc,
                partial(pytorch_ops.transpose, dim0=-3, dim1=-2),
                input_shapes,
                [
                    partial(generation_funcs.gen_rand, low=-100, high=100),
                ],
                comparison_funcs.comp_pcc,
                pcie_slot,
            )
            results_csv.flush()


if __name__ == "__main__":
    args = get_args_from_argparser()

    run_pytorch_transpose_hc_test(args)
