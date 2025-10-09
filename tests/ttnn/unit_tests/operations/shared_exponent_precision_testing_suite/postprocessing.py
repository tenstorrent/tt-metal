# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import numpy as np
from pathlib import Path
import json

from constants import Filepaths, ResultKeys, OperationType, MatmulTTConfig


#
# Pattern impact analysis
#
def _analyze_pattern_impact(results: dict) -> dict:
    """
    Analyze how different patterns affect precision across all operations.

    Args:
        results: Nested dictionary with structure [shape_type][pattern][distribution][operation][...]

    Returns:
        Dictionary with pattern analysis results
    """
    pattern_stats = {}

    # Flatten and collect metrics by pattern
    flattened_results = _flatten_results(results)

    for case_info, metrics in flattened_results:
        pattern = case_info["pattern"]

        if pattern not in pattern_stats:
            pattern_stats[pattern] = {
                "pcc_values": [],
                "max_abs_errors": [],
                "mean_abs_errors": [],
                "ulp_max_values": [],
                "ulp_mean_values": [],
                "max_rel_errors": [],
                "operations": {},
                "count": 0,
            }

        # Collect metrics
        pattern_stats[pattern]["pcc_values"].append(metrics.get(ResultKeys.PCC_KEY, 1.0))
        pattern_stats[pattern]["max_abs_errors"].append(metrics.get(ResultKeys.MAX_ABS_ERROR_KEY, 0))
        pattern_stats[pattern]["mean_abs_errors"].append(metrics.get(ResultKeys.MEAN_ABS_ERROR_KEY, 0))
        pattern_stats[pattern]["ulp_max_values"].append(metrics.get(ResultKeys.ULP_MAX_KEY, 0))
        pattern_stats[pattern]["ulp_mean_values"].append(metrics.get(ResultKeys.ULP_MEAN_KEY, 0))
        pattern_stats[pattern]["max_rel_errors"].append(metrics.get(ResultKeys.MAX_REL_ERROR_KEY, 0))
        pattern_stats[pattern]["count"] += 1

        # Track operations per pattern
        operation = case_info["operation"]
        if operation not in pattern_stats[pattern]["operations"]:
            pattern_stats[pattern]["operations"][operation] = 0
        pattern_stats[pattern]["operations"][operation] += 1

    # Calculate statistics for each pattern
    pattern_summary = {}
    for pattern, stats in pattern_stats.items():
        if stats["count"] > 0:
            pattern_summary[pattern] = {
                "avg_pcc": np.mean(stats["pcc_values"]),
                "min_pcc": np.min(stats["pcc_values"]),
                "std_pcc": np.std(stats["pcc_values"]),
                "avg_max_abs_error": np.mean(stats["max_abs_errors"]),
                "max_max_abs_error": np.max(stats["max_abs_errors"]),
                "std_max_abs_error": np.std(stats["max_abs_errors"]),
                "avg_mean_abs_error": np.mean(stats["mean_abs_errors"]),
                "avg_ulp_max": np.mean(stats["ulp_max_values"]),
                "max_ulp_max": np.max(stats["ulp_max_values"]),
                "avg_ulp_mean": np.mean(stats["ulp_mean_values"]),
                "avg_max_rel_error": np.mean(stats["max_rel_errors"]),
                "max_max_rel_error": np.max(stats["max_rel_errors"]),
                "num_tests": stats["count"],
                "operations_breakdown": stats["operations"],
            }

    # Rank patterns by different metrics
    pattern_ranking_by_pcc = sorted(pattern_summary.items(), key=lambda x: x[1]["avg_pcc"])  # Lower PCC is worse
    pattern_ranking_by_abs_error = sorted(
        pattern_summary.items(), key=lambda x: x[1]["avg_max_abs_error"], reverse=True
    )  # Higher error is worse
    pattern_ranking_by_ulp = sorted(
        pattern_summary.items(), key=lambda x: x[1]["avg_ulp_max"], reverse=True
    )  # Higher ULP is worse

    return {
        "pattern_summary": pattern_summary,
        "pattern_ranking_by_pcc": pattern_ranking_by_pcc,
        "pattern_ranking_by_abs_error": pattern_ranking_by_abs_error,
        "pattern_ranking_by_ulp": pattern_ranking_by_ulp,
        "worst_pattern_by_pcc": pattern_ranking_by_pcc[0] if pattern_ranking_by_pcc else None,
    }


def _format_pattern_impact_report(pattern_analysis: dict) -> str:
    """
    Format pattern impact analysis into a markdown report.

    Args:
        pattern_analysis: Dictionary returned by analyze_pattern_impact

    Returns:
        Formatted markdown string
    """
    report_lines = ["# Pattern Impact Analysis\n"]

    # Overview section
    report_lines.append("## Overview\n")
    report_lines.append("This report analyzes how different patterns affect precision metrics across all operations.\n")

    # Summary statistics table
    report_lines.append("## Pattern Performance Summary\n")
    report_lines.append(_create_pattern_summary_table(pattern_analysis["pattern_summary"]))

    # Rankings
    report_lines.append("\n## Pattern Rankings\n")

    # Worst by PCC
    report_lines.append("\n### Ranked by Average PCC (Lower is Worse)\n")
    report_lines.append(_create_pattern_ranking_table(pattern_analysis["pattern_ranking_by_pcc"], "avg_pcc"))

    # Worst by absolute error
    report_lines.append("\n### Ranked by Average Max Absolute Error (Higher is Worse)\n")
    report_lines.append(
        _create_pattern_ranking_table(pattern_analysis["pattern_ranking_by_abs_error"], "avg_max_abs_error")
    )

    # Worst by ULP
    report_lines.append("\n### Ranked by Average Max ULP (Higher is Worse)\n")
    report_lines.append(_create_pattern_ranking_table(pattern_analysis["pattern_ranking_by_ulp"], "avg_ulp_max"))

    # Operations breakdown
    report_lines.append("\n## Operations Breakdown by Pattern\n")
    report_lines.append(_create_operations_breakdown_table(pattern_analysis["pattern_summary"]))

    # Worst pattern details
    if pattern_analysis["worst_pattern_by_pcc"]:
        worst_pattern, worst_stats = pattern_analysis["worst_pattern_by_pcc"]
        report_lines.append(f"\n## Worst Pattern Details: {worst_pattern}\n")
        report_lines.append("### Metrics Summary\n")
        for metric, value in worst_stats.items():
            if metric != "operations_breakdown":
                if isinstance(value, float):
                    if value < 1:
                        report_lines.append(f"- **{metric.replace('_', ' ').title()}**: {value:.6f}")
                    elif value < 100:
                        report_lines.append(f"- **{metric.replace('_', ' ').title()}**: {value:.2f}")
                    else:
                        report_lines.append(f"- **{metric.replace('_', ' ').title()}**: {value:,.0f}")
                else:
                    report_lines.append(f"- **{metric.replace('_', ' ').title()}**: {value}")

    return "\n".join(report_lines)


def _create_pattern_summary_table(pattern_summary: dict) -> str:
    """
    Create a comprehensive markdown summary table for all pattern performance metrics.

    Generates a markdown table showing key precision metrics (PCC, absolute errors, ULP values)
    for each pattern, sorted by average PCC for consistent ordering.

    Args:
        pattern_summary: Dictionary containing pattern statistics with keys as pattern names
                        and values as dictionaries containing metrics like 'avg_pcc', 'min_pcc',
                        'num_tests', 'avg_max_abs_error', etc.

    Returns:
        str: Formatted markdown table string with pattern comparison metrics.
             Returns "No pattern data available." if input is empty.
    """
    if not pattern_summary:
        return "No pattern data available."

    headers = [
        "Pattern",
        "Tests",
        "Avg PCC",
        "Min PCC",
        "Std PCC",
        "Avg Max Abs Err",
        "Max Max Abs Err",
        "Avg ULP Max",
        "Max ULP Max",
    ]

    table_lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]

    # Sort by average PCC for consistent ordering
    sorted_patterns = sorted(pattern_summary.items(), key=lambda x: x[1]["avg_pcc"])

    for pattern, stats in sorted_patterns:
        row = [
            pattern,
            str(stats["num_tests"]),
            f"{stats['avg_pcc']:.6f}",
            f"{stats['min_pcc']:.6f}",
            f"{stats['std_pcc']:.6f}",
            f"{stats['avg_max_abs_error']:.2f}",
            f"{stats['max_max_abs_error']:.2f}",
            f"{stats['avg_ulp_max']:,.0f}",
            f"{stats['max_ulp_max']:,.0f}",
        ]
        table_lines.append("| " + " | ".join(row) + " |")

    return "\n".join(table_lines)


def _create_pattern_ranking_table(pattern_ranking: list, metric_key: str) -> str:
    """
    Create a markdown ranking table for patterns sorted by a specific metric.

    Generates a top-10 ranking table showing patterns ordered by performance
    on a specific precision metric (e.g., PCC, absolute error, ULP).

    Args:
        pattern_ranking: List of (pattern_name, stats_dict) tuples, pre-sorted
                        by the target metric in desired order (best to worst or worst to best)
        metric_key: String key for the metric to display values for (e.g., 'avg_pcc',
                   'avg_max_abs_error', 'avg_ulp_max')

    Returns:
        str: Formatted markdown table with rank, pattern name, metric value, and test count.
             Returns "No ranking data available." if input list is empty.
    """
    if not pattern_ranking:
        return "No ranking data available."

    headers = ["Rank", "Pattern", "Value", "Tests"]

    table_lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]

    for rank, (pattern, stats) in enumerate(pattern_ranking[:10], 1):  # Top 10
        value = stats[metric_key]
        if value < 1:
            value_str = f"{value:.6f}"
        elif value < 100:
            value_str = f"{value:.2f}"
        else:
            value_str = f"{value:,.0f}"

        row = [str(rank), pattern, value_str, str(stats["num_tests"])]
        table_lines.append("| " + " | ".join(row) + " |")

    return "\n".join(table_lines)


def _create_operations_breakdown_table(pattern_summary):
    """
    Create a markdown table showing operation distribution across patterns.

    Generates a table that breaks down how many times each operation type
    was tested for each pattern, providing insight into test coverage
    and pattern-operation combinations.

    Args:
        pattern_summary: Dictionary containing pattern statistics where each pattern
                        has an 'operations_breakdown' key with operation counts and
                        'num_tests' for total test count per pattern

    Returns:
        str: Formatted markdown table showing patterns vs operations with test counts.
             Uses '-' for operations not tested with a given pattern.
             Returns "No pattern data available." if input is empty.
    """
    if not pattern_summary:
        return "No pattern data available."

    # Get all unique operations
    all_operations = set()
    for stats in pattern_summary.values():
        all_operations.update(stats["operations_breakdown"].keys())

    operations = sorted(list(all_operations))
    headers = ["Pattern", "Total Tests"] + operations

    table_lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]

    # Sort by total tests for better readability
    sorted_patterns = sorted(pattern_summary.items(), key=lambda x: x[1]["num_tests"], reverse=True)

    for pattern, stats in sorted_patterns:
        row = [pattern, str(stats["num_tests"])]
        for op in operations:
            count = stats["operations_breakdown"].get(op, 0)
            row.append(str(count) if count > 0 else "-")
        table_lines.append("| " + " | ".join(row) + " |")

    return "\n".join(table_lines)


def pattern_impact_analysis(results: dict, output_dir: str = Filepaths.RESULTS_DIRECTORY) -> dict:
    """
    Analyze pattern impact and save the analysis to a markdown file.

    Args:
        results: Nested dictionary with operation results
        output_file: Path to output markdown file

    Returns:
        Pattern analysis dictionary
    """
    # Create directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Perform pattern analysis
    pattern_analysis = _analyze_pattern_impact(results)

    # Generate markdown report
    markdown_content = _format_pattern_impact_report(pattern_analysis)

    # Save to file
    output_file = Path(output_dir) / Path(Filepaths.PATTERN_IMPACT_ANALYSIS_FILENAME)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    logger.info(f"Pattern impact analysis saved to {output_file}")
    return pattern_analysis


#
# Worst cases analysis
#
def _find_worst_cases(results: dict, top_n: int = 10, metrics_to_track=None):
    """
    Find the worst performing test cases across specified precision metrics.

    Analyzes all operation results to identify the cases with the worst precision
    according to various metrics. For PCC, lower values are worse; for error metrics,
    higher values are worse.

    Args:
        results: Nested dictionary with precision test results structured as:
                [shape_type][pattern][distribution][operation][metrics or axis/config]
                Contains all test cases with their precision metrics
        top_n: Number of worst cases to return for each tracked metric (default: 10)
        metrics_to_track: List of metric keys to analyze. If None, defaults to:
                         ["pcc", "max_abs_error", "ulp_max", "mean_abs_error", "max_rel_error"].
                         Each key should correspond to ResultKeys constants.

    Returns:
        dict: Dictionary mapping each metric name to a list of worst cases, where each
              case is a dict with keys:
              - 'value': The metric value that makes it a worst case
              - 'case': Dict with test case info (shape_type, pattern, distribution, operation, params)
              - 'full_metrics': Complete metrics dict for context
    """
    if metrics_to_track is None:
        metrics_to_track = [
            ResultKeys.PCC_KEY,
            ResultKeys.MAX_ABS_ERROR_KEY,
            ResultKeys.ULP_MAX_KEY,
            ResultKeys.MEAN_ABS_ERROR_KEY,
            ResultKeys.MAX_REL_ERROR_KEY,
        ]

    # Initialize worst cases tracker
    worst_cases = {metric: [] for metric in metrics_to_track}

    # Flatten results for analysis
    flattened_results = _flatten_results(results)

    # Collect all cases with their metrics
    for case_info, metrics in flattened_results:
        for metric_name in metrics_to_track:
            if metric_name in metrics:
                worst_cases[metric_name].append(
                    {"value": metrics[metric_name], "case": case_info, "full_metrics": metrics}
                )

    # Sort and keep top N worst for each metric
    for metric_name in worst_cases:
        if metric_name == ResultKeys.PCC_KEY:
            # For PCC, lower is worse
            worst_cases[metric_name] = sorted(worst_cases[metric_name], key=lambda x: x["value"])[:top_n]
        else:
            # For error metrics, higher is worse
            worst_cases[metric_name] = sorted(worst_cases[metric_name], key=lambda x: x["value"], reverse=True)[:top_n]

    return worst_cases


def _flatten_results(results: dict):
    """
    Flatten the nested results dictionary into a list of (case_info, metrics) tuples.

    Converts the hierarchical test results structure into a flat list for easier analysis.
    Handles different operation types with their specific parameter structures.

    Args:
        results: Nested dictionary with structure:
                [shape_type][pattern][distribution][operation][...metrics or parameters...]

    Returns:
        list: List of (case_info, metrics) tuples where:
              - case_info: Dict with keys 'shape_type', 'pattern', 'distribution',
                          'operation', and 'params' (containing operation-specific parameters)
              - metrics: Dict containing precision metrics for this test case

    Note:
        Handles three operation types:
        1. matmul: Direct metrics at operation level
        2. matmul_tt: Nested structure [tile_w][transpose] -> metrics
        3. others: Structure [axis] -> metrics
    """
    flattened = []

    for shape_type, shape_data in results.items():
        if not isinstance(shape_data, dict):
            continue

        for pattern, pattern_data in shape_data.items():
            if not isinstance(pattern_data, dict):
                continue

            for distribution, dist_data in pattern_data.items():
                if not isinstance(dist_data, dict):
                    continue

                for operation, op_data in dist_data.items():
                    if operation == OperationType.MATMUL_KEY:
                        # Direct metrics for matmul
                        case_info = {
                            "shape_type": shape_type,
                            "pattern": pattern,
                            "distribution": distribution,
                            "operation": operation,
                            "params": {},
                        }
                        flattened.append((case_info, op_data))

                    elif operation == OperationType.MATMUL_TT_KEY:
                        # Handle matmul_tt structure: [tile_w][transpose] -> metrics
                        for tile_w, tile_data in op_data.items():
                            if isinstance(tile_data, dict):
                                for transpose, metrics in tile_data.items():
                                    if isinstance(metrics, dict) and "pcc" in metrics:
                                        case_info = {
                                            "shape_type": shape_type,
                                            "pattern": pattern,
                                            "distribution": distribution,
                                            "operation": operation,
                                            "params": {
                                                MatmulTTConfig.TILE_W_KEY: tile_w,
                                                MatmulTTConfig.TRANSPOSE_KEY: transpose,
                                            },
                                        }
                                        flattened.append((case_info, metrics))

                    else:
                        # Handle other operations with axis
                        for axis, metrics in op_data.items():
                            if isinstance(metrics, dict) and "pcc" in metrics:
                                case_info = {
                                    "shape_type": shape_type,
                                    "pattern": pattern,
                                    "distribution": distribution,
                                    "operation": operation,
                                    "params": {"axis": axis},
                                }
                                flattened.append((case_info, metrics))

    return flattened


def _format_worst_cases_report(worst_cases: dict, top_n: int = 5):
    """
    Format worst cases analysis into a detailed markdown report.

    Creates a comprehensive report showing the worst performing test cases
    for each precision metric, with detailed case information and context metrics.

    Args:
        worst_cases: Dictionary returned by _find_worst_cases containing metric names
                    as keys and lists of worst case data as values. Each case contains
                    'value', 'case' (test info), and 'full_metrics' (all precision data)
        top_n: Number of worst cases to include in detailed report per metric (default: 5)

    Returns:
        str: Formatted markdown report string with sections for each metric showing:
             - Case ranking and problematic values
             - Test case details (shape, pattern, distribution, operation, parameters)
             - Related metrics for context
    """
    report_lines = ["# Worst Cases Analysis\n"]

    metric_descriptions = {
        ResultKeys.PCC_KEY: "Lowest PCC (Pearson Correlation Coefficient)",
        ResultKeys.MAX_ABS_ERROR: "Highest Maximum Absolute Error",
        ResultKeys.MEAN_ABS_ERROR: "Highest Mean Absolute Error",
        ResultKeys.MAX_REL_ERROR: "Highest Maximum Relative Error",
        ResultKeys.ULP_MAX: "Highest Maximum ULP Error",
    }

    for metric_name, cases in worst_cases.items():
        if not cases:
            continue

        report_lines.append(f"\n## {metric_descriptions.get(metric_name, metric_name)}\n")

        for i, case_data in enumerate(cases[:top_n], 1):
            case = case_data["case"]
            value = case_data["value"]
            metrics = case_data["full_metrics"]

            report_lines.append(f"\n### Case {i}")
            report_lines.append(f"- **Value**: {value:.6f}" if value < 100 else f"- **Value**: {value:,.0f}")
            report_lines.append(f"- **Shape Type**: {case['shape_type']}")
            report_lines.append(f"- **Pattern**: {case['pattern']}")
            report_lines.append(f"- **Distribution**: {case['distribution']}")
            report_lines.append(f"- **Operation**: {case['operation']}")

            # Add operation-specific parameters
            if case["params"]:
                for param_name, param_value in case["params"].items():
                    report_lines.append(f"- **{param_name.replace('_', ' ').title()}**: {param_value}")

            # Add other relevant metrics for context
            report_lines.append("\n**Other Metrics:**")
            for key, val in metrics.items():
                if key not in ["ulp_percentiles", "input_stats"] and key != metric_name:
                    if isinstance(val, bool):
                        report_lines.append(f"- {key}: {'True' if val else 'False'}")
                    elif isinstance(val, (int, float)):
                        if val < 100:
                            report_lines.append(f"- {key}: {val:.6f}")
                        else:
                            report_lines.append(f"- {key}: {val:,.0f}")

    return "\n".join(report_lines)


def _create_worst_cases_summary_table(worst_cases: dict, metric="pcc", top_n=10) -> str:
    """
    Create a summary table for a specific metric's worst cases.

    Args:
        worst_cases: Dictionary returned by _find_worst_cases
        metric: Metric to create table for
        top_n: Number of rows to include

    Returns:
        Markdown table string
    """
    if metric not in worst_cases or not worst_cases[metric]:
        return f"No data available for {metric}"

    headers = ["Rank", "Value", "Shape Type", "Pattern", "Distribution", "Operation", "Parameters"]
    table_lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]

    for i, case_data in enumerate(worst_cases[metric][:top_n], 1):
        case = case_data["case"]
        value = case_data["value"]

        # Format parameters
        params_str = ", ".join([f"{k}={v}" for k, v in case["params"].items()]) if case["params"] else "-"

        # Format value based on metric type
        if metric == "pcc" or "rel" in metric:
            value_str = f"{value:.6f}"
        elif value < 100:
            value_str = f"{value:.2f}"
        else:
            value_str = f"{value:,.0f}"

        row = [
            str(i),
            value_str,
            case["shape_type"],
            case["pattern"],
            case["distribution"],
            case["operation"],
            params_str,
        ]
        table_lines.append("| " + " | ".join(row) + " |")

    return "\n".join(table_lines)


def worst_cases_analysis(
    results, output_dir=Filepaths.RESULTS_DIRECTORY, top_n_analysis=10, top_n_per_metric=0, metrics_to_track=None
) -> dict:
    """
    Find worst cases and save complete analysis to markdown file.

    Args:
        results: Nested dictionary with operation results
        output_file: Path to output markdown file
        top_n_analysis: Number of worst cases to find for analysis
        top_n_per_metric: Number of cases to show in detailed report per metric
        metrics_to_track: List of metrics to track
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_file = Path(output_dir) / Path(Filepaths.WORST_CASES_ANALYSIS_FILENAME)

    # Find worst cases
    worst_cases = _find_worst_cases(results, top_n=top_n_analysis, metrics_to_track=metrics_to_track)

    # Build complete markdown content
    markdown_content = []

    # Add detailed worst cases report
    if top_n_per_metric > 0:
        markdown_content.append(_format_worst_cases_report(worst_cases, top_n=top_n_per_metric))

    # Add summary tables for each metric
    markdown_content.append("\n\n# Summary Tables\n")

    metric_descriptions = {
        ResultKeys.PCC_KEY: "PCC (Pearson Correlation Coefficient) - Worst Cases",
        ResultKeys.MAX_ABS_ERROR_KEY: "Maximum Absolute Error - Worst Cases",
        ResultKeys.MEAN_ABS_ERROR_KEY: "Mean Absolute Error - Worst Cases",
        ResultKeys.MAX_REL_ERROR_KEY: "Maximum Relative Error - Worst Cases",
        ResultKeys.ULP_MAX_KEY: "Maximum ULP Error - Worst Cases",
    }

    for metric in worst_cases.keys():
        if worst_cases[metric]:  # Only add table if there are cases
            markdown_content.append(f"\n## {metric_descriptions.get(metric, metric)}\n")
            markdown_content.append(_create_worst_cases_summary_table(worst_cases, metric=metric, top_n=top_n_analysis))

    # Write to file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(markdown_content))

    logger.info(f"Worst cases analysis saved to {output_file}")
    return worst_cases


#
# All case report
#
def save_results_to_json(all_results: dict, output_directory: str = Filepaths.RESULTS_DIRECTORY) -> None:
    """Save the raw results to a JSON file for further analysis if needed"""
    # Create output directory
    Path(output_directory).mkdir(parents=True, exist_ok=True)

    # Save file
    file_path = Path(output_directory) / Path(Filepaths.RAW_RESULTS_JSON_FILENAME)
    with open(file_path, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"Raw results saved to {file_path}")


def _create_matmul_table(results: dict, headers: list) -> str:
    """
    Create a markdown table for matmul operation results.

    Formats matmul operation precision metrics into a markdown table with
    predefined columns for operation name and various precision metrics.

    Args:
        results: List of (operation_name, result_dict) tuples where result_dict
                contains precision metrics like PCC, absolute errors, ULP values, etc.
        headers: List of column headers for the markdown table

    Returns:
        str: Formatted markdown table with operation results
    """

    table_lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]

    for operation, result in results:
        row = [
            operation,
            f"{result[ResultKeys.PCC_KEY]:.6f}",
            f"{result[ResultKeys.MAX_ABS_ERROR_KEY]:.2f}",
            f"{result[ResultKeys.MEAN_ABS_ERROR_KEY]:.2f}",
            f"{result[ResultKeys.MAX_REL_ERROR_KEY]:.6f}",
            f"{result[ResultKeys.MEAN_REL_ERROR_KEY]:.6f}",
            f"{result[ResultKeys.ULP_MEAN_KEY]:.2f}",
            f"{result[ResultKeys.ULP_MAX_KEY]:.0f}",
            "True" if result[ResultKeys.ALLCLOSE_1E_2_KEY] else "False",
            "True" if result[ResultKeys.ALLCLOSE_1E_3_KEY] else "False",
        ]
        table_lines.append("| " + " | ".join(row) + " |")

    return "\n".join(table_lines)


def _create_matmul_tt_table(results: dict, headers: list) -> str:
    """
    Create a markdown table for matmul_tt operation results.

    Formats matmul_tt operation precision metrics into a markdown table including
    operation-specific parameters like tile_w and transpose settings.

    Args:
        results: List of (operation_name, tile_w, transpose, result_dict) tuples
                where result_dict contains precision metrics
        headers: List of column headers for the markdown table including operation,
                tile_w, transpose columns plus precision metrics

    Returns:
        str: Formatted markdown table with matmul_tt operation results and parameters
    """

    table_lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]

    for operation, tile_w, transpose, result in results:
        row = [
            operation,
            str(tile_w),
            str(transpose),
            f"{result[ResultKeys.PCC_KEY]:.6f}",
            f"{result[ResultKeys.MAX_ABS_ERROR_KEY]:.2f}",
            f"{result[ResultKeys.MEAN_ABS_ERROR_KEY]:.2f}",
            f"{result[ResultKeys.MAX_REL_ERROR_KEY]:.6f}",
            f"{result[ResultKeys.MEAN_REL_ERROR_KEY]:.6f}",
            f"{result[ResultKeys.ULP_MEAN_KEY]:.2f}",
            f"{result[ResultKeys.ULP_MAX_KEY]:.0f}",
            "True" if result[ResultKeys.ALLCLOSE_1E_2_KEY] else "False",
            "True" if result[ResultKeys.ALLCLOSE_1E_3_KEY] else "False",
        ]
        table_lines.append("| " + " | ".join(row) + " |")

    return "\n".join(table_lines)


def _create_other_ops_table(results: dict, headers: list) -> str:
    """
    Create a markdown table for other operations (non-matmul operations with axis parameter).

    Formats precision metrics for operations that use an axis parameter (like reductions,
    concatenations, etc.) into a markdown table format.

    Args:
        results: List of (operation_name, axis, result_dict) tuples where result_dict
                contains precision metrics and axis is the operation parameter
        headers: List of column headers for the markdown table including operation,
                axis columns plus precision metrics

    Returns:
        str: Formatted markdown table with operation results and axis parameters
    """

    table_lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]

    for operation, axis, result in results:
        row = [
            operation,
            str(axis),
            f"{result[ResultKeys.PCC_KEY]:.6f}",
            f"{result[ResultKeys.MAX_ABS_ERROR_KEY]:.2f}",
            f"{result[ResultKeys.MEAN_ABS_ERROR_KEY]:.2f}",
            f"{result[ResultKeys.MAX_REL_ERROR_KEY]:.6f}",
            f"{result[ResultKeys.MEAN_REL_ERROR_KEY]:.6f}",
            f"{result[ResultKeys.ULP_MEAN_KEY]:.2f}",
            f"{result[ResultKeys.ULP_MAX_KEY]:.0f}",
            "True" if result[ResultKeys.ALLCLOSE_1E_2_KEY] else "False",
            "True" if result[ResultKeys.ALLCLOSE_1E_3_KEY] else "False",
        ]
        table_lines.append("| " + " | ".join(row) + " |")

    return "\n".join(table_lines)


def _dict_to_markdown(results_dict):
    """
    Convert a nested dictionary of operation results to a comprehensive markdown document.

    Transforms the hierarchical test results into a well-structured markdown report
    organized by shape types, patterns, and distributions, with separate tables
    for different operation types.

    Args:
        results_dict: Nested dictionary with precision test results structured as:
                     [shape_type][pattern][distribution][operation][metrics/parameters]
                     Contains all test results organized hierarchically

    Returns:
        str: Complete formatted markdown document with:
             - Hierarchical sections for shape types, patterns, distributions
             - Separate tables for matmul, matmul_tt, and other operations
             - Formatted precision metrics (PCC, errors, ULP values, allclose results)
    """
    headers = [
        "Operation",
        "Axis",
        "PCC",
        "Max Abs Error",
        "Mean Abs Error",
        "Max Rel Error",
        "Mean Rel Error",
        "ULP Mean",
        "ULP Max",
        "Allclose 1e-2",
        "Allclose 1e-3",
    ]

    markdown_lines = ["# Operation Results Report\n"]

    # Iterate through shape types (main sections)
    for shape_type, shape_data in results_dict.items():
        markdown_lines.append(f"\n## ========== Shape type: {shape_type} ==========\n")

        # Iterate through patterns (subsections)
        for pattern, pattern_data in shape_data.items():
            markdown_lines.append(f"\n### ======= Pattern: {pattern} ======= \n")

            # Iterate through distributions (sub-subsections)
            for distribution, distribution_data in pattern_data.items():
                markdown_lines.append(f"\n#### ===== Distribution: {distribution} =====\n")

                # Group results by operation type
                matmul_results = []
                matmul_tt_results = []
                other_results = []

                for operation, operation_data in distribution_data.items():
                    if operation == OperationType.MATMUL_KEY:
                        matmul_results.append((operation, operation_data))
                    elif operation == OperationType.MATMUL_TT_KEY:
                        # Process matmul_tt nested structure
                        for tile_w, tile_data in operation_data.items():
                            for transpose, result in tile_data.items():
                                matmul_tt_results.append((operation, tile_w, transpose, result))
                    else:
                        # Process other operations with axis
                        for axis, result in operation_data.items():
                            other_results.append((operation, axis, result))

                # Generate tables for each operation type
                if matmul_results:
                    markdown_lines.append("\n##### MatMul Operations\n")
                    markdown_lines.append(_create_matmul_table(matmul_results, headers))

                if matmul_tt_results:
                    markdown_lines.append("\n##### MatMul TT Operations\n")
                    markdown_lines.append(_create_matmul_tt_table(matmul_tt_results, headers))

                if other_results:
                    markdown_lines.append("\n##### Other Operations\n")
                    markdown_lines.append(_create_other_ops_table(other_results, headers))

    return "\n".join(markdown_lines)


def generate_results_doc(all_results: dict, output_directory: str = Filepaths.RESULTS_DIRECTORY) -> None:
    """
    Generate and save a comprehensive markdown documentation of all test results.

    Creates a complete markdown report from the nested results dictionary and saves it
    to a file for easy review and sharing of precision analysis results.

    Args:
        all_results: Complete nested dictionary containing all precision test results
                    with structure [shape_type][pattern][distribution][operation][metrics]
        output_directory: Directory path where the markdown report will be saved
                         (default: Filepaths.RESULTS_DIRECTORY)

    Returns:
        None

    Side Effects:
        - Creates output directory if it doesn't exist
        - Writes markdown report to file specified by Filepaths.RAW_RESULTS_MARKDOWN_FILENAME
        - Logs the file path where report was saved
    """
    markdown_content = _dict_to_markdown(all_results)

    # Create output directory
    Path(output_directory).mkdir(parents=True, exist_ok=True)

    # Save results
    file_path = Path(output_directory) / Path(Filepaths.RAW_RESULTS_MARKDOWN_FILENAME)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    logger.info(f"Report saved to {file_path}")
