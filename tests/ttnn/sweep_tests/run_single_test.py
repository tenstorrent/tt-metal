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

    device = ttnn.open_device(device_id=0)
    status, message = run_single_test(test_name, index, device=device)
    ttnn.close_device(device)

    if status == "passed":
        logger.info(f"Passed")
    elif status == "is_expected_to_fail":
        logger.info(f'Failed as expected with the following error message: "{message}"')
    elif status in "failed":
        logger.info(f'Failed:"{message}"')
        exit(-1)
    elif status in "crashed":
        logger.info(f'Crashed: "{message}"')
        exit(-1)
    elif status in "skipped":
        logger.info(f'Skipped: "{message}"')
    else:
        raise RuntimeError(f"Unknown status {status}")


if __name__ == "__main__":
    main()
