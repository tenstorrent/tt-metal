# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import numpy as np


def _analyze_operation_sensitivity(results):
    """Analyze which operations are most sensitive to precision loss"""
    operation_stats = {}

    # Collect metrics by operation
    for shape_type, shape_results in results.items():
        if isinstance(shape_results, dict):
            for pattern, pattern_results in shape_results.items():
                if isinstance(pattern_results, dict):
                    for dist, dist_results in pattern_results.items():
                        if isinstance(dist_results, dict):
                            for op_key, metrics in dist_results.items():
                                # Extract operation name
                                if "_axis" in op_key:
                                    op_name, axis_info = op_key.split("_axis")
                                    axis = int(axis_info)
                                else:
                                    op_name = op_key
                                    axis = None

                                # Initialize stats structure
                                if op_name not in operation_stats:
                                    operation_stats[op_name] = {
                                        "all_metrics": [],
                                        "by_axis": {0: [], 1: [], None: []},
                                        "by_shape_type": {},
                                    }

                                # Store metrics
                                metric_summary = {
                                    "pcc": metrics.get("pcc", 1.0),
                                    "max_abs_error": metrics.get("max_abs_error", 0),
                                    "ulp_max": metrics.get("ulp_max", 0),
                                    "shape_type": shape_type,
                                    "pattern": pattern,
                                    "distribution": dist,
                                    "axis": axis,
                                }

                                operation_stats[op_name]["all_metrics"].append(metric_summary)
                                operation_stats[op_name]["by_axis"][axis].append(metric_summary)

                                if shape_type not in operation_stats[op_name]["by_shape_type"]:
                                    operation_stats[op_name]["by_shape_type"][shape_type] = []
                                operation_stats[op_name]["by_shape_type"][shape_type].append(metric_summary)

    # Compute summary statistics
    operation_summary = {}
    for op_name, stats in operation_stats.items():
        all_pcc = [m["pcc"] for m in stats["all_metrics"]]
        all_errors = [m["max_abs_error"] for m in stats["all_metrics"]]
        all_ulp = [m["ulp_max"] for m in stats["all_metrics"]]

        operation_summary[op_name] = {
            "overall": {
                "avg_pcc": np.mean(all_pcc),
                "min_pcc": np.min(all_pcc),
                "std_pcc": np.std(all_pcc),
                "avg_error": np.mean(all_errors),
                "max_error": np.max(all_errors),
                "avg_ulp": np.mean(all_ulp),
                "max_ulp": np.max(all_ulp),
                "num_tests": len(stats["all_metrics"]),
            },
            "axis_comparison": {},
        }

        # Analyze by axis
        for axis in [0, 1]:
            if stats["by_axis"][axis]:
                axis_pcc = [m["pcc"] for m in stats["by_axis"][axis]]
                operation_summary[op_name]["axis_comparison"][f"axis_{axis}"] = {
                    "avg_pcc": np.mean(axis_pcc),
                    "min_pcc": np.min(axis_pcc),
                }

    # Rank operations by sensitivity
    operation_ranking = sorted(operation_summary.items(), key=lambda x: x[1]["overall"]["avg_pcc"])

    return {
        "operation_summary": operation_summary,
        "operation_ranking": operation_ranking,
        "most_sensitive_operation": operation_ranking[0] if operation_ranking else None,
        "axis_effects": {
            op: summary["axis_comparison"] for op, summary in operation_summary.items() if summary["axis_comparison"]
        },
    }


def _analyze_pattern_impact(results):
    """Analyze how different patterns affect precision"""
    pattern_stats = {}

    # Collect all metrics by pattern
    for shape_type, shape_results in results.items():
        if isinstance(shape_results, dict):
            for pattern, pattern_results in shape_results.items():
                if pattern not in pattern_stats:
                    pattern_stats[pattern] = {"pcc_values": [], "max_abs_errors": [], "ulp_max_values": [], "count": 0}

                # Collect metrics from all distributions and operations
                if isinstance(pattern_results, dict):
                    for dist, dist_results in pattern_results.items():
                        if isinstance(dist_results, dict):
                            for op, metrics in dist_results.items():
                                pattern_stats[pattern]["pcc_values"].append(metrics.get("pcc", 1.0))
                                pattern_stats[pattern]["max_abs_errors"].append(metrics.get("max_abs_error", 0))
                                pattern_stats[pattern]["ulp_max_values"].append(metrics.get("ulp_max", 0))
                                pattern_stats[pattern]["count"] += 1

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
                "avg_ulp_max": np.mean(stats["ulp_max_values"]),
                "max_ulp_max": np.max(stats["ulp_max_values"]),
                "num_tests": stats["count"],
            }

    # Rank patterns by impact (worst first)
    pattern_ranking = sorted(pattern_summary.items(), key=lambda x: x[1]["avg_pcc"])  # Lower PCC is worse

    return {
        "pattern_summary": pattern_summary,
        "pattern_ranking": pattern_ranking,
        "worst_pattern": pattern_ranking[0] if pattern_ranking else None,
    }


def _find_worst_cases(results, top_n=10):
    """Find the worst performing cases across all metrics"""

    worst_cases = {"pcc": [], "max_abs_error": [], "ulp_max": []}

    # Flatten results for analysis
    for shape_type, shape_results in results.items():
        if isinstance(shape_results, dict):
            for pattern, pattern_results in shape_results.items():
                if isinstance(pattern_results, dict):
                    for dist, dist_results in pattern_results.items():
                        if isinstance(dist_results, dict):
                            for op, metrics in dist_results.items():
                                case_info = {
                                    "shape_type": shape_type,
                                    "pattern": pattern,
                                    "distribution": dist,
                                    "operation": op,
                                    "metrics": metrics,
                                }

                                # Track worst cases
                                worst_cases["pcc"].append((metrics.get("pcc", 1.0), case_info))
                                worst_cases["max_abs_error"].append((metrics.get("max_abs_error", 0), case_info))
                                worst_cases["ulp_max"].append((metrics.get("ulp_max", 0), case_info))

    # Sort and keep top N worst
    for metric in worst_cases:
        if metric == "pcc":
            worst_cases[metric] = sorted(worst_cases[metric], key=lambda x: x[0])[:top_n]
        else:
            worst_cases[metric] = sorted(worst_cases[metric], key=lambda x: x[0], reverse=True)[:top_n]

    return worst_cases


def analyze_operation_sensitivity(results):
    """Analyze which operations are most sensitive to precision loss"""
    operation_stats = {}

    # Collect metrics by operation
    for shape_type, shape_results in results.items():
        if isinstance(shape_results, dict):
            for pattern, pattern_results in shape_results.items():
                if isinstance(pattern_results, dict):
                    for dist, dist_results in pattern_results.items():
                        if isinstance(dist_results, dict):
                            for op_key, metrics in dist_results.items():
                                # Extract operation name
                                if "_axis" in op_key:
                                    op_name, axis_info = op_key.split("_axis")
                                    axis = int(axis_info)
                                else:
                                    op_name = op_key
                                    axis = None

                                # Initialize stats structure
                                if op_name not in operation_stats:
                                    operation_stats[op_name] = {
                                        "all_metrics": [],
                                        "by_axis": {0: [], 1: [], None: []},
                                        "by_shape_type": {},
                                    }

                                # Store metrics
                                metric_summary = {
                                    "pcc": metrics.get("pcc", 1.0),
                                    "max_abs_error": metrics.get("max_abs_error", 0),
                                    "ulp_max": metrics.get("ulp_max", 0),
                                    "shape_type": shape_type,
                                    "pattern": pattern,
                                    "distribution": dist,
                                    "axis": axis,
                                }

                                operation_stats[op_name]["all_metrics"].append(metric_summary)
                                operation_stats[op_name]["by_axis"][axis].append(metric_summary)

                                if shape_type not in operation_stats[op_name]["by_shape_type"]:
                                    operation_stats[op_name]["by_shape_type"][shape_type] = []
                                operation_stats[op_name]["by_shape_type"][shape_type].append(metric_summary)

    # Compute summary statistics
    operation_summary = {}
    for op_name, stats in operation_stats.items():
        all_pcc = [m["pcc"] for m in stats["all_metrics"]]
        all_errors = [m["max_abs_error"] for m in stats["all_metrics"]]
        all_ulp = [m["ulp_max"] for m in stats["all_metrics"]]

        operation_summary[op_name] = {
            "overall": {
                "avg_pcc": np.mean(all_pcc),
                "min_pcc": np.min(all_pcc),
                "std_pcc": np.std(all_pcc),
                "avg_error": np.mean(all_errors),
                "max_error": np.max(all_errors),
                "avg_ulp": np.mean(all_ulp),
                "max_ulp": np.max(all_ulp),
                "num_tests": len(stats["all_metrics"]),
            },
            "axis_comparison": {},
        }

        # Analyze by axis
        for axis in [0, 1]:
            if stats["by_axis"][axis]:
                axis_pcc = [m["pcc"] for m in stats["by_axis"][axis]]
                operation_summary[op_name]["axis_comparison"][f"axis_{axis}"] = {
                    "avg_pcc": np.mean(axis_pcc),
                    "min_pcc": np.min(axis_pcc),
                }

    # Rank operations by sensitivity
    operation_ranking = sorted(operation_summary.items(), key=lambda x: x[1]["overall"]["avg_pcc"])

    return {
        "operation_summary": operation_summary,
        "operation_ranking": operation_ranking,
        "most_sensitive_operation": operation_ranking[0] if operation_ranking else None,
        "axis_effects": {
            op: summary["axis_comparison"] for op, summary in operation_summary.items() if summary["axis_comparison"]
        },
    }


def _analyze_shape_effects(results):
    """Analyze how different shapes affect precision"""
    shape_stats = {}

    # Process each shape type
    for shape_type, shape_results in results.items():
        shape_metrics = []

        if isinstance(shape_results, dict):
            # For multi_tile and single_tile
            if shape_type in ["single_tile", "multi_tile"]:
                for pattern, pattern_results in shape_results.items():
                    if isinstance(pattern_results, dict):
                        for dist, dist_results in pattern_results.items():
                            if isinstance(dist_results, dict):
                                for op, metrics in dist_results.items():
                                    shape_metrics.append(
                                        {
                                            "pcc": metrics.get("pcc", 1.0),
                                            "max_abs_error": metrics.get("max_abs_error", 0),
                                            "ulp_max": metrics.get("ulp_max", 0),
                                        }
                                    )

            # For rectangular shapes
            elif shape_type == "rectangular":
                for specific_shape, specific_results in shape_results.items():
                    if isinstance(specific_results, dict):
                        for pattern, pattern_results in specific_results.items():
                            if isinstance(pattern_results, dict):
                                for dist, dist_results in pattern_results.items():
                                    if isinstance(dist_results, dict):
                                        for op, metrics in dist_results.items():
                                            shape_metrics.append(
                                                {
                                                    "pcc": metrics.get("pcc", 1.0),
                                                    "max_abs_error": metrics.get("max_abs_error", 0),
                                                    "ulp_max": metrics.get("ulp_max", 0),
                                                    "specific_shape": specific_shape,
                                                }
                                            )

        if shape_metrics:
            pcc_values = [m["pcc"] for m in shape_metrics]
            error_values = [m["max_abs_error"] for m in shape_metrics]
            ulp_values = [m["ulp_max"] for m in shape_metrics]

            shape_stats[shape_type] = {
                "avg_pcc": np.mean(pcc_values),
                "min_pcc": np.min(pcc_values),
                "std_pcc": np.std(pcc_values),
                "avg_error": np.mean(error_values),
                "max_error": np.max(error_values),
                "avg_ulp": np.mean(ulp_values),
                "max_ulp": np.max(ulp_values),
                "num_tests": len(shape_metrics),
                "percentiles": {
                    "pcc_10th": np.percentile(pcc_values, 10),
                    "pcc_90th": np.percentile(pcc_values, 90),
                    "error_90th": np.percentile(error_values, 90),
                    "error_99th": np.percentile(error_values, 99),
                },
            }

    # Compare single vs multi tile
    comparison = {}
    if "single_tile" in shape_stats and "multi_tile" in shape_stats:
        comparison["tile_scaling"] = {
            "pcc_degradation": shape_stats["single_tile"]["avg_pcc"] - shape_stats["multi_tile"]["avg_pcc"],
            "error_increase_ratio": shape_stats["multi_tile"]["avg_error"]
            / (shape_stats["single_tile"]["avg_error"] + 1e-10),
            "ulp_increase_ratio": shape_stats["multi_tile"]["avg_ulp"]
            / (shape_stats["single_tile"]["avg_ulp"] + 1e-10),
        }

    # Analyze rectangular shapes
    if "rectangular" in shape_stats:
        # Extract specific rectangular shape stats if available
        rect_analysis = {
            "aspect_ratio_effects": "Analysis of how aspect ratio affects precision"
            # Add more detailed analysis here based on specific shapes
        }
        comparison["rectangular_effects"] = rect_analysis

    return {
        "shape_summary": shape_stats,
        "shape_comparison": comparison,
    }


# Main module function
def analyze_results(results: dict) -> dict:
    """Comprehensive analysis of all results"""

    analysis = {
        "worst_cases": _find_worst_cases(results),
        "pattern_impact": _analyze_pattern_impact(results),
        "operation_sensitivity": _analyze_operation_sensitivity(results),
        "shape_effects": _analyze_shape_effects(results),
    }
    print(analysis)
    return analysis
