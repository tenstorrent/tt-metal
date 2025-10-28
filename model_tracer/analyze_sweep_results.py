# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
"""
Sweep Test Results Analyzer

Analyzes sweep test result JSON files to provide comprehensive overview
of test execution including pass/fail statistics, performance metrics,
and detailed breakdowns.
"""

import json
import glob
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import defaultdict
import argparse


# Get the base directory from PYTHONPATH if set, otherwise use current working directory
def get_base_dir():
    """Get the tt-metal base directory from PYTHONPATH or current working directory"""
    pythonpath = os.environ.get("PYTHONPATH", "")
    if pythonpath:
        # PYTHONPATH might contain multiple paths separated by ':'
        paths = pythonpath.split(":")
        for path in paths:
            # Look for tt-metal directory
            if "tt-metal" in path:
                # Extract the tt-metal base directory
                if path.endswith("tt-metal"):
                    return path
                # Handle cases like /home/ubuntu/tt-metal/python_env/lib/python3.X/site-packages
                parts = path.split("tt-metal")
                if parts:
                    return parts[0] + "tt-metal"
    # Fallback: assume we're running from within tt-metal and find it
    current_dir = os.getcwd()
    if "tt-metal" in current_dir:
        parts = current_dir.split("tt-metal")
        return parts[0] + "tt-metal"
    # Last resort: use current directory
    return current_dir


BASE_DIR = get_base_dir()


class SweepResultsAnalyzer:
    """Analyzes sweep test results from exported JSON files"""

    def __init__(self, results_dir: str = None):
        if results_dir is None:
            results_dir = os.path.join(BASE_DIR, "tests/sweep_framework/results_export")
        self.results_dir = results_dir
        self.results_data = []
        self.metadata = {}

    def load_latest_results(self) -> bool:
        """Load the most recent sweep test results"""
        try:
            # Find all result files
            result_files = glob.glob(os.path.join(self.results_dir, "eltwise_*.json"))
            metadata_files = glob.glob(os.path.join(self.results_dir, "oprun_*.json"))

            if not result_files:
                print(f"❌ No result files found in {self.results_dir}")
                return False

            # Get the most recent files
            latest_result_file = max(result_files, key=os.path.getmtime)
            latest_metadata_file = max(metadata_files, key=os.path.getmtime) if metadata_files else None

            print(f"📁 Loading results from: {os.path.basename(latest_result_file)}")
            if latest_metadata_file:
                print(f"📁 Loading metadata from: {os.path.basename(latest_metadata_file)}")

            # Load results data
            with open(latest_result_file, "r") as f:
                self.results_data = json.load(f)

            # Load metadata if available
            if latest_metadata_file:
                with open(latest_metadata_file, "r") as f:
                    self.metadata = json.load(f)

            return True

        except Exception as e:
            print(f"❌ Error loading results: {e}")
            return False

    def load_specific_results(self, result_file: str, metadata_file: Optional[str] = None) -> bool:
        """Load specific result files"""
        try:
            # Load results data
            with open(result_file, "r") as f:
                self.results_data = json.load(f)

            # Load metadata if provided
            if metadata_file and os.path.exists(metadata_file):
                with open(metadata_file, "r") as f:
                    self.metadata = json.load(f)

            print(f"✅ Loaded {len(self.results_data)} test results")
            return True

        except Exception as e:
            print(f"❌ Error loading specific results: {e}")
            return False

    def analyze_overview(self) -> Dict[str, Any]:
        """Generate overall test statistics"""
        if not self.results_data:
            return {}

        stats = {
            "total_tests": len(self.results_data),
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "error_tests": 0,
            "success_rate": 0.0,
            "total_duration": 0.0,
            "avg_duration_per_test": 0.0,
        }

        total_duration = 0.0

        for test in self.results_data:
            # Count test outcomes
            if test.get("skipped", False):
                stats["skipped"] += 1
            elif test.get("success", False):
                stats["passed"] += 1
            else:
                stats["failed"] += 1

            # Count error tests
            if test.get("error_message") and test.get("error_message") != "None":
                stats["error_tests"] += 1

            # Calculate duration
            start_ts = test.get("test_start_ts")
            end_ts = test.get("test_end_ts")
            if start_ts and end_ts:
                try:
                    start = datetime.fromisoformat(start_ts.replace("Z", "+00:00"))
                    end = datetime.fromisoformat(end_ts.replace("Z", "+00:00"))
                    duration = (end - start).total_seconds()
                    total_duration += duration
                except:
                    pass

        stats["total_duration"] = total_duration
        stats["avg_duration_per_test"] = total_duration / stats["total_tests"] if stats["total_tests"] > 0 else 0
        stats["success_rate"] = (stats["passed"] / stats["total_tests"]) * 100 if stats["total_tests"] > 0 else 0

        return stats

    def analyze_by_test_case(self) -> Dict[str, Dict[str, Any]]:
        """Analyze results grouped by test case"""
        test_case_stats = defaultdict(
            lambda: {"total": 0, "passed": 0, "failed": 0, "skipped": 0, "avg_duration": 0.0, "configurations": set()}
        )

        for test in self.results_data:
            test_case = test.get("test_case_name", "unknown")
            test_case_stats[test_case]["total"] += 1

            if test.get("skipped", False):
                test_case_stats[test_case]["skipped"] += 1
            elif test.get("success", False):
                test_case_stats[test_case]["passed"] += 1
            else:
                test_case_stats[test_case]["failed"] += 1

            # Track unique configurations
            if test.get("op_params_set"):
                config_key = str(
                    sorted(
                        [
                            (p["param_name"], str(p.get("param_value_json", p.get("param_value_text", ""))))
                            for p in test["op_params_set"]
                        ]
                    )
                )
                test_case_stats[test_case]["configurations"].add(config_key)

        # Convert sets to counts
        for case_name, stats in test_case_stats.items():
            stats["unique_configurations"] = len(stats["configurations"])
            del stats["configurations"]  # Remove the set, keep only the count
            stats["success_rate"] = (stats["passed"] / stats["total"]) * 100 if stats["total"] > 0 else 0

        return dict(test_case_stats)

    def analyze_failures(self) -> List[Dict[str, Any]]:
        """Analyze failed tests in detail"""
        failures = []

        for test in self.results_data:
            if not test.get("success", False) and not test.get("skipped", False):
                failure_info = {
                    "test_name": test.get("full_test_name", "unknown"),
                    "test_case": test.get("test_case_name", "unknown"),
                    "error_message": test.get("error_message", "No error message"),
                    "exception": test.get("exception", "No exception details"),
                    "status": test.get("status", "unknown"),
                    "op_params": self.extract_key_params(test.get("op_params_set", [])),
                }
                failures.append(failure_info)

        return failures

    def extract_key_params(self, op_params_set: List[Dict]) -> Dict[str, Any]:
        """Extract key parameters from op_params_set"""
        key_params = {}

        for param in op_params_set:
            param_name = param.get("param_name", "")
            if any(key in param_name.lower() for key in ["shape", "dtype", "memory", "layout"]):
                value = (
                    param.get("param_value_json") or param.get("param_value_text") or param.get("param_value_numeric")
                )
                key_params[param_name] = value

        return key_params

    def analyze_performance(self) -> Dict[str, Any]:
        """Analyze performance metrics"""
        performance_data = []

        for test in self.results_data:
            if test.get("success", False):
                # Extract performance metric from message (often contains PCC or timing info)
                message = test.get("message", "")
                try:
                    # Try to extract numeric performance metric
                    if message and message.replace(".", "").replace("e-", "").replace("E-", "").isdigit():
                        perf_value = float(message)
                        performance_data.append(perf_value)
                except:
                    pass

        if performance_data:
            return {
                "total_samples": len(performance_data),
                "min_value": min(performance_data),
                "max_value": max(performance_data),
                "avg_value": sum(performance_data) / len(performance_data),
                "median_value": sorted(performance_data)[len(performance_data) // 2],
            }

        return {"total_samples": 0}

    def print_summary(self):
        """Print comprehensive test results summary"""
        if not self.results_data:
            print("❌ No results data loaded")
            return

        print("🔍 SWEEP TEST RESULTS ANALYSIS")
        print("=" * 60)

        # Metadata summary
        if self.metadata:
            print(f"📋 Test Run Information:")
            print(f"   • Host: {self.metadata.get('host', 'unknown')}")
            print(f"   • Card Type: {self.metadata.get('card_type', 'unknown')}")
            print(f"   • Run Type: {self.metadata.get('run_type', 'unknown')}")
            print(f"   • Git SHA: {self.metadata.get('git_sha', 'unknown')}")
            print(f"   • Run Contents: {self.metadata.get('run_contents', 'unknown')}")

            # Duration from metadata
            if self.metadata.get("run_start_ts") and self.metadata.get("run_end_ts"):
                try:
                    start = datetime.fromisoformat(self.metadata["run_start_ts"].replace("Z", "+00:00"))
                    end = datetime.fromisoformat(self.metadata["run_end_ts"].replace("Z", "+00:00"))
                    total_run_time = (end - start).total_seconds()
                    print(f"   • Total Run Time: {total_run_time:.2f} seconds")
                except:
                    pass
            print()

        # Overall statistics
        overview = self.analyze_overview()
        print(f"📊 Overall Test Statistics:")
        print(f"   • Total Tests: {overview['total_tests']}")
        print(f"   • ✅ Passed: {overview['passed']} ({overview['success_rate']:.1f}%)")
        print(f"   • ❌ Failed: {overview['failed']}")
        print(f"   • ⏭️  Skipped: {overview['skipped']}")
        if overview["error_tests"] > 0:
            print(f"   • 🔥 With Errors: {overview['error_tests']}")
        print(f"   • ⏱️  Avg Duration: {overview['avg_duration_per_test']:.2f}s per test")
        print()

        # Test case breakdown
        test_case_stats = self.analyze_by_test_case()
        print(f"🎯 Test Case Breakdown:")
        for case_name, stats in test_case_stats.items():
            print(f"   • {case_name}:")
            print(f"     - Total: {stats['total']} tests")
            print(f"     - Passed: {stats['passed']} ({stats['success_rate']:.1f}%)")
            print(f"     - Failed: {stats['failed']}")
            if stats["skipped"] > 0:
                print(f"     - Skipped: {stats['skipped']}")
            print(f"     - Unique Configs: {stats['unique_configurations']}")
        print()

        # Performance analysis
        perf_stats = self.analyze_performance()
        if perf_stats.get("total_samples", 0) > 0:
            print(f"🚀 Performance Metrics (from {perf_stats['total_samples']} successful tests):")
            print(f"   • Min Value: {perf_stats['min_value']:.6f}")
            print(f"   • Max Value: {perf_stats['max_value']:.6f}")
            print(f"   • Avg Value: {perf_stats['avg_value']:.6f}")
            print(f"   • Median Value: {perf_stats['median_value']:.6f}")
            print()

        # Failure analysis
        failures = self.analyze_failures()
        if failures:
            print(f"🔥 Failure Analysis ({len(failures)} failures):")
            for i, failure in enumerate(failures[:5]):  # Show first 5 failures
                print(f"   {i+1}. {failure['test_case']}:")
                print(f"      - Error: {failure['error_message']}")
                if failure["op_params"]:
                    print(f"      - Key Params: {failure['op_params']}")
            if len(failures) > 5:
                print(f"   ... and {len(failures) - 5} more failures")
        else:
            print("🎉 No failures detected!")


def main():
    parser = argparse.ArgumentParser(description="Analyze sweep test results")
    parser.add_argument(
        "--results-dir",
        default=os.path.join(BASE_DIR, "tests/sweep_framework/results_export"),
        help="Directory containing result files",
    )
    parser.add_argument("--result-file", help="Specific result file to analyze")
    parser.add_argument("--metadata-file", help="Specific metadata file to analyze")
    parser.add_argument("--latest", action="store_true", default=True, help="Analyze latest results (default)")

    args = parser.parse_args()

    analyzer = SweepResultsAnalyzer(args.results_dir)

    if args.result_file:
        success = analyzer.load_specific_results(args.result_file, args.metadata_file)
    else:
        success = analyzer.load_latest_results()

    if success:
        analyzer.print_summary()
    else:
        print("❌ Failed to load results data")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
