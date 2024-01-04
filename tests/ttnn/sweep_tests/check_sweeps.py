# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import pandas as pd


from tests.ttnn.sweep_tests.sweep import SWEEP_RESULTS_DIR


def main():
    num_total = 0
    num_total_passed = 0
    for file_name in SWEEP_RESULTS_DIR.glob("*.csv"):
        df = pd.read_csv(file_name)
        num_passed = (df["status"] == "passed").sum()
        logger.info(f"{file_name.name}: {num_passed} passed out of {len(df)}")
        num_total += len(df)
        num_total_passed += num_passed
    logger.info(f"Total: {num_total_passed} passed out of {num_total}")


if __name__ == "__main__":
    main()
