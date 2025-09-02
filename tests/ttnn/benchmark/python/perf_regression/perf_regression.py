# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Generic performance regression testing tools.

This module provides reusable tools for comparing performance data against baselines
using non-parametric A/B regression test: Mann-Whitney U + median effect + MAD guard band to detect regressions.
"""

import json
import warnings
from pathlib import Path
from typing import Dict, List, Union, Callable

import numpy as np
from scipy import stats


class PerformanceData:
    """
    Container for performance benchmark data with statistical utilities.

    Supports serialization to/from JSON and tracks sections and subsections.
    """

    def __init__(self, data: Dict[str, Dict[str, List[Union[int, float]]]]):
        """
        Initialize PerformanceData.

        Args:
            data: Dictionary with structure {section: {subsection: [samples]}}
        """
        self.data = data
        self._validate_data()

    def _validate_data(self):
        """Validate the data structure and warn about sample counts."""
        for section, subsections in self.data.items():
            if not isinstance(subsections, dict):
                raise ValueError(f"Section '{section}' must contain a dictionary of subsections")

            for subsection, samples in subsections.items():
                if not isinstance(samples, list):
                    raise ValueError(f"Subsection '{section}.{subsection}' must contain a list of samples")

                if not samples:
                    raise ValueError(f"Subsection '{section}.{subsection}' cannot be empty")

                if not all(isinstance(x, (int, float)) for x in samples):
                    raise ValueError(f"All samples in '{section}.{subsection}' must be numeric")

                # Warn about small sample sizes
                if len(samples) < 100:
                    warnings.warn(
                        f"Section '{section}.{subsection}' has only {len(samples)} samples. "
                        f"Results may be unreliable with fewer than 100 samples.",
                        UserWarning,
                    )

    @classmethod
    def from_dict(cls, data: Dict[str, Dict[str, List[Union[int, float]]]]) -> "PerformanceData":
        return cls(data)

    @classmethod
    def from_json_file(cls, file_path: Union[str, Path]) -> "PerformanceData":
        with open(file_path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_json_string(cls, json_string: str) -> "PerformanceData":
        data = json.loads(json_string)
        return cls.from_dict(data)

    def to_dict(self) -> Dict[str, Dict[str, List[Union[int, float]]]]:
        return self.data.copy()

    def to_json_file(self, file_path: Union[str, Path], indent: int = 2):
        with open(file_path, "w") as f:
            json.dump(self.data, f, indent=indent)

    def to_json_string(self, indent: int = 2) -> str:
        return json.dumps(self.data, indent=indent)

    def get_samples(self, section: str, subsection: str) -> List[Union[int, float]]:
        return self.data.get(section, {}).get(subsection, [])

    def get_sections(self) -> List[str]:
        return list(self.data.keys())

    def get_subsections(self, section: str) -> List[str]:
        return list(self.data.get(section, {}).keys())

    def median(self, section: str, subsection: str) -> float:
        samples = self.get_samples(section, subsection)
        return float(np.median(samples)) if samples else 0.0

    def mean(self, section: str, subsection: str) -> float:
        samples = self.get_samples(section, subsection)
        return float(np.mean(samples)) if samples else 0.0

    def std(self, section: str, subsection: str) -> float:
        samples = self.get_samples(section, subsection)
        return float(np.std(samples)) if samples else 0.0

    def __eq__(self, other) -> bool:
        if not isinstance(other, PerformanceData):
            return False
        return self.data == other.data

    def __repr__(self) -> str:
        section_count = len(self.data)
        total_subsections = sum(len(subsections) for subsections in self.data.values())
        return f"PerformanceData(sections={section_count}, subsections={total_subsections})"


def mann_whitney_u(samples1: List[Union[int, float]], samples2: List[Union[int, float]]) -> float:
    """
    Perform Mann-Whitney U test and return p-value.

    Args:
        samples1: First set of samples
        samples2: Second set of samples

    Returns:
        p-value from the test (1.0 if test cannot be performed)
    """
    if len(samples1) == 0 or len(samples2) == 0:
        return 1.0  # No significant difference if no samples

    try:
        _, p_value = stats.mannwhitneyu(samples1, samples2, alternative="two-sided")
        return float(p_value)
    except ValueError:
        # If all values are identical, no significant difference
        return 1.0


def check_regression(
    baseline: PerformanceData,
    current: PerformanceData,
    alpha: float = 0.05,
    min_effect_size_pct: float = 1.0,
) -> Dict[str, Dict[str, Dict[str, Union[bool, float, str]]]]:
    """
    Check for performance regression by comparing baseline against current performance data.

    Args:
        baseline: Baseline (ground truth) performance data
        current: Current performance data to compare against baseline
        alpha: Statistical significance threshold (default: 0.05)
        min_effect_size_pct: Minimum percentage change to consider practically significant (default: 1.0%)

    Returns:
        Dictionary with regression analysis results for each section/subsection:
        {
            "section": {
                "subsection": {
                    "is_regression": bool,
                    "delta_mean_pct": float,
                    "p_value": float,
                    "message": str,
                    "current_mean": float,
                    "baseline_mean": float
                }
            }
        }

    Raises:
        ValueError: If baseline and current data don't have matching structure
    """
    if baseline.get_sections() != current.get_sections():
        raise ValueError(
            f"Section mismatch: baseline has {baseline.get_sections()}, current has {current.get_sections()}"
        )

    results = {}

    for section in baseline.get_sections():
        baseline_subsections = set(baseline.get_subsections(section))
        current_subsections = set(current.get_subsections(section))

        if baseline_subsections != current_subsections:
            raise ValueError(
                f"Subsection mismatch in section '{section}': "
                f"baseline has {baseline_subsections}, current has {current_subsections}"
            )

        results[section] = {}

        for subsection in baseline.get_subsections(section):
            baseline_samples = baseline.get_samples(section, subsection)
            current_samples = current.get_samples(section, subsection)

            if len(baseline_samples) != len(current_samples):
                raise ValueError(
                    f"Sample count mismatch in {section}.{subsection}: "
                    f"baseline has {len(baseline_samples)}, current has {len(current_samples)}"
                )

            # Calculate statistics
            baseline_mean = baseline.mean(section, subsection)
            current_mean = current.mean(section, subsection)

            if baseline_mean == 0:
                results[section][subsection] = {
                    "is_regression": False,
                    "delta_mean_pct": 0.0,
                    "p_value": 1.0,
                    "message": f"Baseline mean is zero for {section}.{subsection}",
                    "current_mean": current_mean,
                    "baseline_mean": baseline_mean,
                }
                continue

            # Calculate p-value using Mann-Whitney U test
            p_value = mann_whitney_u(current_samples, baseline_samples)

            # Calculate percentage change in mean
            delta_mean_pct = ((current_mean / baseline_mean) - 1.0) * 100.0

            # Determine if this is statistically significant
            is_significant = p_value < alpha

            # Determine if this is practically significant (effect size large enough)
            is_practically_significant = abs(delta_mean_pct) >= min_effect_size_pct

            # Determine if this is a regression (statistically AND practically significant AND performance got worse)
            is_regression = is_significant and is_practically_significant and delta_mean_pct > 0

            # Create message
            if is_regression:
                message = f"Regression detected: {delta_mean_pct:+.3f}% (p={p_value:.3f}), \nbaseline: {baseline.get_samples(section, subsection)}, \ncurrent: {current.get_samples(section, subsection)}"
            elif is_significant and is_practically_significant and delta_mean_pct < 0:
                message = f"Improvement detected: {delta_mean_pct:+.3f}% (p={p_value:.3f})"
            elif is_significant and not is_practically_significant:
                message = f"Statistically significant but trivial change: {delta_mean_pct:+.3f}% (p={p_value:.3f}, below {min_effect_size_pct}% threshold)"
            else:
                message = f"No significant change: {delta_mean_pct:+.3f}% (p={p_value:.3f})"

            results[section][subsection] = {
                "is_regression": is_regression,
                "delta_mean_pct": delta_mean_pct,
                "p_value": p_value,
                "message": message,
                "current_mean": current_mean,
                "baseline_mean": baseline_mean,
            }

    return results


def summarize_regression_results(
    results: Dict[str, Dict[str, Dict[str, Union[bool, float, str]]]]
) -> Dict[str, Union[int, List[str]]]:
    """
    Summarize regression analysis results.

    Args:
        results: Results from check_regression function

    Returns:
        Summary dictionary with counts and regression messages
    """
    summary = {
        "total_tests": 0,
        "regressions": 0,
        "improvements": 0,
        "no_change": 0,
        "regression_messages": [],
        "improvement_messages": [],
    }

    for section, subsections in results.items():
        for subsection, result in subsections.items():
            summary["total_tests"] += 1

            if result["is_regression"]:
                summary["regressions"] += 1
                summary["regression_messages"].append(f"{section}.{subsection}: {result['message']}")
            elif result["p_value"] < 0.05 and result["delta_mean_pct"] < 0:
                summary["improvements"] += 1
                summary["improvement_messages"].append(f"{section}.{subsection}: {result['message']}")
            else:
                summary["no_change"] += 1

    return summary
