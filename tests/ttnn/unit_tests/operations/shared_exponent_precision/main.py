# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from loguru import logger

from runner import run_experiments
from postprocessing import (
    save_results_to_json,
    generate_results_doc,
    worst_cases_analysis,
    pattern_impact_analysis,
    plot_charts,
)


def main():
    """This function manages the overall testing process for bfloat8_b precision operations."""

    # Run all experiments
    logger.info("=== Starting bfloat8_b precision experiments ===")
    all_results = run_experiments()

    # Save all_results to JSON
    logger.info("=== Saving results to JSON ===")
    save_results_to_json(all_results)

    # Generate results document
    logger.info("=== Generating results document ===")
    generate_results_doc(all_results)

    # Find worst cases
    logger.info("=== Analyze worst cases ===")
    worst_cases_analysis(all_results)

    # Analyze pattern impact
    logger.info("=== Analyze pattern impact ===")
    pattern_impact_analysis(all_results)

    # Add visualization
    logger.info("=== Plotting charts ===")
    plot_charts(all_results)


if __name__ == "__main__":
    main()
