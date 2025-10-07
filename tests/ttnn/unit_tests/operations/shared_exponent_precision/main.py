# SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from loguru import logger

from runner import run_experiments
from postprocessing import analyze_results


def main():
    """This function manages the overall testing process for bfloat8_b precision operations."""

    # Run all experiments
    logger.info("=== Starting bfloat8_b precision experiments ===")
    all_results = run_experiments()

    # Analyze results
    logger.info("=== Analyzing results ===")
    analysis = analyze_results(all_results)

    # Save results
    # save_results(all_results, analysis)

    # # Generate report
    # generate_report(analysis)

    return all_results, analysis


if __name__ == "__main__":
    results, analysis = main()
