import json
import os
import re
from collections import defaultdict


def parse_test_output(result_file):
    """Parse a single test result file to extract test outcomes"""

    if not os.path.exists(result_file):
        return {"status": "not_run", "tests": []}

    with open(result_file, "r") as f:
        content = f.read()

    results = {
        "status": "unknown",
        "total_tests": 0,
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "oom_failures": [],
        "type_errors": [],
        "other_failures": [],
        "test_details": [],
    }

    # Extract test summary
    summary_match = re.search(r"(\d+) passed.*?(\d+) skipped.*?(\d+) failed", content)
    if summary_match:
        results["passed"] = int(summary_match.group(1))
        results["skipped"] = int(summary_match.group(2))
        results["failed"] = int(summary_match.group(3))
        results["total_tests"] = results["passed"] + results["skipped"] + results["failed"]
        results["status"] = "completed"

    # Alternative summary format
    if results["total_tests"] == 0:
        passed_match = re.search(r"(\d+) passed", content)
        skipped_match = re.search(r"(\d+) skipped", content)
        failed_match = re.search(r"(\d+) failed", content)

        if passed_match:
            results["passed"] = int(passed_match.group(1))
        if skipped_match:
            results["skipped"] = int(skipped_match.group(1))
        if failed_match:
            results["failed"] = int(failed_match.group(1))

        results["total_tests"] = results["passed"] + results["skipped"] + results["failed"]
        if results["total_tests"] > 0:
            results["status"] = "completed"

    # Extract individual test results
    lines = content.split("\\n")
    for i, line in enumerate(lines):
        # Test execution lines
        if "PASSED" in line or "FAILED" in line or "SKIPPED" in line:
            test_name_match = re.search(r"test_.*?\\[(.*?)\\]", line)
            if test_name_match:
                input_shape = test_name_match.group(1)

                test_detail = {"input_shape": input_shape, "status": "unknown", "reason": None}

                if "PASSED" in line:
                    test_detail["status"] = "passed"
                elif "SKIPPED" in line:
                    test_detail["status"] = "skipped"
                    # Look for skip reason in subsequent lines
                    for j in range(i + 1, min(i + 5, len(lines))):
                        if "OOM:" in lines[j]:
                            test_detail["reason"] = "out_of_memory"
                            # Extract memory details
                            memory_match = re.search(r"Out of Memory: Not enough space to allocate (\\d+) B", lines[j])
                            if memory_match:
                                test_detail["memory_required"] = int(memory_match.group(1))
                            results["oom_failures"].append(test_detail)
                            break
                        elif "Type error:" in lines[j]:
                            test_detail["reason"] = "type_error"
                            results["type_errors"].append(test_detail)
                            break
                        elif "SKIPPED" in lines[j]:
                            # Generic skip reason
                            test_detail["reason"] = "skipped"
                            break
                elif "FAILED" in line:
                    test_detail["status"] = "failed"
                    test_detail["reason"] = "other_failure"
                    results["other_failures"].append(test_detail)

                results["test_details"].append(test_detail)

    return results


def analyze_all_results():
    """Analyze all test result files"""

    results_dir = "ssinghal/test_results"

    if not os.path.exists(results_dir):
        print("No test results directory found. Run tests first.")
        return

    # Load test generation summary for context
    with open("ssinghal/test_generation_summary.json", "r") as f:
        generation_summary = json.load(f)

    # Load operator mapping for context
    with open("ssinghal/operator_mapping.json", "r") as f:
        operator_mapping = json.load(f)

    all_results = {}
    summary_stats = {
        "total_operators_tested": 0,
        "total_tests_run": 0,
        "total_passed": 0,
        "total_failed": 0,
        "total_skipped": 0,
        "operators_with_oom": 0,
        "operators_with_type_errors": 0,
        "operators_with_other_failures": 0,
    }

    # Comprehensive failure tracking
    oom_failures_by_operator = defaultdict(list)
    type_errors_by_operator = defaultdict(list)
    other_failures_by_operator = defaultdict(list)

    # Process each result file
    for test_info in generation_summary["generated_tests"]:
        operator = test_info["operator"]
        result_file = f"{results_dir}/{operator.lower()}_results.txt"

        print(f"Analyzing {operator}...")

        result = parse_test_output(result_file)
        all_results[operator] = result

        if result["status"] == "completed":
            summary_stats["total_operators_tested"] += 1
            summary_stats["total_tests_run"] += result["total_tests"]
            summary_stats["total_passed"] += result["passed"]
            summary_stats["total_failed"] += result["failed"]
            summary_stats["total_skipped"] += result["skipped"]

            # Track failures by type
            if result["oom_failures"]:
                summary_stats["operators_with_oom"] += 1
                oom_failures_by_operator[operator] = result["oom_failures"]

            if result["type_errors"]:
                summary_stats["operators_with_type_errors"] += 1
                type_errors_by_operator[operator] = result["type_errors"]

            if result["other_failures"]:
                summary_stats["operators_with_other_failures"] += 1
                other_failures_by_operator[operator] = result["other_failures"]

    # Create comprehensive analysis
    comprehensive_analysis = {
        "summary_statistics": summary_stats,
        "detailed_results": all_results,
        "failure_analysis": {
            "oom_failures": dict(oom_failures_by_operator),
            "type_errors": dict(type_errors_by_operator),
            "other_failures": dict(other_failures_by_operator),
        },
    }

    # Save comprehensive analysis
    with open("ssinghal/comprehensive_test_analysis.json", "w") as f:
        json.dump(comprehensive_analysis, f, indent=2)

    # Create detailed OOM failure report
    create_oom_failure_report(oom_failures_by_operator, operator_mapping)

    # Create summary report
    create_summary_report(comprehensive_analysis)

    return comprehensive_analysis


def create_oom_failure_report(oom_failures, operator_mapping):
    """Create a detailed report of OOM failures"""

    oom_report = {
        "summary": {
            "total_operators_with_oom": len(oom_failures),
            "total_oom_failures": sum(len(failures) for failures in oom_failures.values()),
        },
        "failures_by_operator": {},
        "failures_by_shape": defaultdict(list),
        "memory_analysis": {},
    }

    # Analyze failures by operator
    for operator, failures in oom_failures.items():
        op_info = operator_mapping.get(operator, {})

        oom_report["failures_by_operator"][operator] = {
            "ttnn_function": op_info.get("ttnn_function"),
            "category": op_info.get("category"),
            "total_failures": len(failures),
            "failure_details": failures,
        }

        # Group by shape
        for failure in failures:
            shape_key = failure["input_shape"]
            oom_report["failures_by_shape"][shape_key].append(
                {
                    "operator": operator,
                    "memory_required": failure.get("memory_required"),
                    "ttnn_function": op_info.get("ttnn_function"),
                }
            )

    # Convert defaultdict to regular dict for JSON serialization
    oom_report["failures_by_shape"] = dict(oom_report["failures_by_shape"])

    # Save OOM report
    with open("ssinghal/oom_failure_report.json", "w") as f:
        json.dump(oom_report, f, indent=2)

    # Create CSV for easy analysis
    create_oom_csv(oom_failures, operator_mapping)


def create_oom_csv(oom_failures, operator_mapping):
    """Create CSV file with OOM failure data"""

    import csv

    with open("ssinghal/oom_failures.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Operator", "TTNN_Function", "Category", "Input_Shape", "Memory_Required_B", "Memory_Required_MB"]
        )

        for operator, failures in oom_failures.items():
            op_info = operator_mapping.get(operator, {})
            ttnn_func = op_info.get("ttnn_function", "unknown")
            category = op_info.get("category", "unknown")

            for failure in failures:
                memory_b = failure.get("memory_required", 0)
                memory_mb = memory_b / (1024 * 1024) if memory_b else 0

                writer.writerow([operator, ttnn_func, category, failure["input_shape"], memory_b, f"{memory_mb:.2f}"])


def create_summary_report(analysis):
    """Create a human-readable summary report"""

    stats = analysis["summary_statistics"]

    report = f"""
# Vision Model Operator Testing - Comprehensive Results

## Overall Statistics
- **Operators Tested**: {stats['total_operators_tested']}/42 supported operators
- **Total Test Cases**: {stats['total_tests_run']}
- **Passed**: {stats['total_passed']} ({stats['total_passed']/max(stats['total_tests_run'],1)*100:.1f}%)
- **Failed**: {stats['total_failed']} ({stats['total_failed']/max(stats['total_tests_run'],1)*100:.1f}%)
- **Skipped**: {stats['total_skipped']} ({stats['total_skipped']/max(stats['total_tests_run'],1)*100:.1f}%)

## Failure Analysis
- **Operators with OOM Issues**: {stats['operators_with_oom']}
- **Operators with Type Errors**: {stats['operators_with_type_errors']}
- **Operators with Other Failures**: {stats['operators_with_other_failures']}

## Files Generated
- `comprehensive_test_analysis.json` - Complete analysis data
- `oom_failure_report.json` - Detailed OOM failure analysis
- `oom_failures.csv` - OOM failures in spreadsheet format
- `test_summary_report.md` - This summary report

## Next Steps
1. Review OOM failures in `oom_failures.csv` to identify problematic shapes
2. Check type errors to fix operator implementation issues
3. Investigate other failures for operator-specific problems
4. Consider memory optimization strategies for large tensor operations

"""

    # Add operator-specific results
    report += "\\n## Operator Results\\n\\n"

    for operator, result in analysis["detailed_results"].items():
        if result["status"] == "completed":
            report += f"### {operator}\\n"
            report += f"- Tests: {result['total_tests']} | "
            report += f"Passed: {result['passed']} | "
            report += f"Failed: {result['failed']} | "
            report += f"Skipped: {result['skipped']}\\n"

            if result["oom_failures"]:
                report += f"  - OOM Failures: {len(result['oom_failures'])}\\n"
            if result["type_errors"]:
                report += f"  - Type Errors: {len(result['type_errors'])}\\n"
            if result["other_failures"]:
                report += f"  - Other Failures: {len(result['other_failures'])}\\n"

            report += "\\n"

    with open("ssinghal/test_summary_report.md", "w") as f:
        f.write(report)


def main():
    print("Analyzing comprehensive test results...")

    analysis = analyze_all_results()

    if analysis:
        stats = analysis["summary_statistics"]

        print(f"\\n=== COMPREHENSIVE TEST ANALYSIS COMPLETE ===")
        print(f"Operators tested: {stats['total_operators_tested']}")
        print(f"Total test cases: {stats['total_tests_run']}")
        print(f"Success rate: {stats['total_passed']/max(stats['total_tests_run'],1)*100:.1f}%")
        print(f"OOM issues: {stats['operators_with_oom']} operators")
        print(f"Type errors: {stats['operators_with_type_errors']} operators")

        print(f"\\nReports generated:")
        print(f"  - ssinghal/comprehensive_test_analysis.json")
        print(f"  - ssinghal/oom_failure_report.json")
        print(f"  - ssinghal/oom_failures.csv")
        print(f"  - ssinghal/test_summary_report.md")


if __name__ == "__main__":
    main()
