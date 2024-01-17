# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import argparse
from importlib.machinery import SourceFileLoader

from loguru import logger
import pandas as pd

import ttnn


from tests.ttnn.python_api_testing.sweep_tests.sweep import reproduce, SWEEP_SOURCES_DIR, SWEEP_RESULTS_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--operation", type=str)
    parser.add_argument("--index", type=int)

    parsed_args = parser.parse_args()
    operation = parsed_args.operation
    index = parsed_args.index

    device = ttnn.open(0)

    file_name = (SWEEP_SOURCES_DIR / operation).with_suffix(".py")
    logger.info(f"Running {file_name}")

    sweep_module = SourceFileLoader("sweep_module", str(file_name)).load_module()

    try:
        passed, message = reproduce(sweep_module.run, sweep_module.parameters, index, device=device)
    except Exception as e:
        passed = False
        message = f"Exception: {e}"
        logger.exception(message)

    ttnn.close(device)

    if passed:
        logger.info(f"Passed")
    else:
        logger.info(f"Failed: {message}")
        exit(-1)


if __name__ == "__main__":
    main()
