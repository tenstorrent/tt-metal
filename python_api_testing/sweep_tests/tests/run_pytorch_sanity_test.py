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
)


def run_pytorch_sanity_test(args):
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

        data_seed = int(time.time())
        torch.manual_seed(data_seed)

        # This does not do actual datacopy, waiting for op implementation
        input_shapes = [[1, 1, 32, 32]]
        test_pass = run_test_and_save_results(
            results_csv_writer,
            "datacopy",
            input_shapes,
            data_seed,
            ll_buda_ops.datacopy,
            pytorch_ops.datacopy,
            input_shapes,
            [partial(generation_funcs.gen_randint, low=0, high=2)],
            comparison_funcs.comp_pcc,
            pcie_slot,
        )
        results_csv.flush()

        data_seed = int(time.time())
        torch.manual_seed(data_seed)

        input_shapes = [[1, 1, 32, 32], [1, 1, 32, 32]]
        test_pass = run_test_and_save_results(
            results_csv_writer,
            "eltwise_add",
            input_shapes,
            data_seed,
            ll_buda_ops.eltwise_add,
            pytorch_ops.add,
            input_shapes,
            [
                partial(generation_funcs.gen_randint, low=0, high=2),
                partial(generation_funcs.gen_randint, low=0, high=2),
            ],
            comparison_funcs.comp_pcc,
            pcie_slot,
        )
        results_csv.flush()


if __name__ == "__main__":
    args = get_args_from_argparser()

    run_pytorch_sanity_test(args)
