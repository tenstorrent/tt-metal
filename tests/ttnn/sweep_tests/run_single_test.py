# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import argparse

from loguru import logger

import ttnn


from tests.ttnn.sweep_tests.sweep import run_single_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-name", type=str)
    parser.add_argument("--index", type=int)

    parsed_args = parser.parse_args()
    test_name = parsed_args.test_name
    index = parsed_args.index

    device = ttnn.open(0)
    status, message = run_single_test(test_name, index, device=device)
    ttnn.close(device)

    if status == "passed":
        logger.info(f"Passed")
    elif status in {"failed", "crashed"}:
        logger.info(f"Error: {message}")
        exit(-1)
    else:
        raise RuntimeError(f"Unknown status {status}")


if __name__ == "__main__":
    main()
