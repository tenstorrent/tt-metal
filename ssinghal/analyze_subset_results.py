import json
import os
import re
from collections import defaultdict


def parse_test_output(result_file, operator_name):
    """Parse a single test result file to extract test outcomes"""

    if not os.path.exists(result_file):
        return {"status": "not_run", "tests": []}

    with open(result_file, "r") as f:
        content = f.read()

    results = {
        "operator": operator_name,
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

    # Extract test summary from end of file
    summary_lines = content.split("\n")[-10:]  # Look at last 10 lines
    for line in summary_lines:
        if "passed" in line and ("skipped" in line or "failed" in line):
            # Parse format like "84 passed, 5 skipped in 42.64s"
            numbers = re.findall(r"(\d+)", line)
            if len(numbers) >= 2:
                if "failed" in line:
                    results["passed"] = int(numbers[0]) if "passed" in line else 0
                    results["failed"] = int(numbers[1]) if "failed" in line else 0
                    results["skipped"] = int(numbers[2]) if len(numbers) > 2 and "skipped" in line else 0
                else:
                    results["passed"] = int(numbers[0]) if "passed" in line else 0
                    results["skipped"] = int(numbers[1]) if "skipped" in line else 0

                results["total_tests"] = results["passed"] + results["failed"] + results["skipped"]
                results["status"] = "completed"
                break

    # Look for specific test results
    lines = content.split("\n")
    collecting_tests = False

    for i, line in enumerate(lines):
        # Look for test collection start
        if "collected" in line and "items" in line:
            collecting_tests = True
            continue

        if collecting_tests and "::" in line and ("PASSED" in line or "SKIPPED" in line or "FAILED" in line):
            # Extract test name and input shape
            test_match = re.search(r"test_\w+\[([^\]]+)\]", line)
            if test_match:
                input_shape = test_match.group(1)

                test_detail = {"input_shape": input_shape, "status": "unknown", "reason": None}

                if "PASSED" in line:
                    test_detail["status"] = "passed"
                elif "SKIPPED" in line:
                    test_detail["status"] = "skipped"
                    # Look for skip reason in surrounding lines
                    for j in range(max(0, i - 5), min(len(lines), i + 5)):
                        if "OOM:" in lines[j] or "Out of Memory" in lines[j]:
                            test_detail["reason"] = "out_of_memory"
                            # Try to extract memory info
                            memory_match = re.search(r"(\d+)\s*B", lines[j])
                            if memory_match:
                                test_detail["memory_required"] = int(memory_match.group(1))
                            results["oom_failures"].append(test_detail)
                            break
                        elif "Type error:" in lines[j] or "incompatible function arguments" in lines[j]:
                            test_detail["reason"] = "type_error"
                            results["type_errors"].append(test_detail)
                            break

                    if test_detail["reason"] is None:
                        test_detail["reason"] = "generic_skip"

                elif "FAILED" in line:
                    test_detail["status"] = "failed"
                    test_detail["reason"] = "other_failure"
                    results["other_failures"].append(test_detail)

                results["test_details"].append(test_detail)

    return results


def analyze_subset_results():
    """Analyze the subset of test results we have"""

    results_dir = "ssinghal/test_results"

    # List of operators we tested
    tested_operators = ["add", "silu", "relu", "sigmoid", "view", "permute"]

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

    for operator in tested_operators:
        result_file = f"{results_dir}/{operator}_results.txt"

        print(f"Analyzing {operator}...")

        result = parse_test_output(result_file, operator)
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

        print(f"  {operator}: {result['passed']} passed, {result['failed']} failed, {result['skipped']} skipped")
        if result["oom_failures"]:
            print(f"    OOM failures: {len(result['oom_failures'])}")
        if result["type_errors"]:
            print(f"    Type errors: {len(result['type_errors'])}")

    # Create analysis
    analysis = {
        "summary_statistics": summary_stats,
        "detailed_results": all_results,
        "failure_analysis": {
            "oom_failures": dict(oom_failures_by_operator),
            "type_errors": dict(type_errors_by_operator),
            "other_failures": dict(other_failures_by_operator),
        },
    }

    # Save analysis
    with open("ssinghal/subset_test_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)

    # Create OOM CSV
    create_oom_csv(oom_failures_by_operator)

    return analysis


def create_oom_csv(oom_failures):
    """Create CSV file with OOM failure data"""

    import csv

    with open("ssinghal/subset_oom_failures.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Operator", "Input_Shape", "Memory_Required_B", "Memory_Required_MB"])

        for operator, failures in oom_failures.items():
            for failure in failures:
                memory_b = failure.get("memory_required", 0)
                memory_mb = memory_b / (1024 * 1024) if memory_b else 0

                writer.writerow([operator, failure["input_shape"], memory_b, f"{memory_mb:.2f}"])


def main():
    print("Analyzing subset test results...")

    analysis = analyze_subset_results()

    if analysis:
        stats = analysis["summary_statistics"]

        print(f"\n=== SUBSET TEST ANALYSIS COMPLETE ===")
        print(f"Operators tested: {stats['total_operators_tested']}")
        print(f"Total test cases: {stats['total_tests_run']}")
        print(f"Success rate: {stats['total_passed']/max(stats['total_tests_run'],1)*100:.1f}%")
        print(f"OOM issues: {stats['operators_with_oom']} operators")
        print(f"Type errors: {stats['operators_with_type_errors']} operators")

        print(f"\nReports generated:")
        print(f"  - ssinghal/subset_test_analysis.json")
        print(f"  - ssinghal/subset_oom_failures.csv")


if __name__ == "__main__":
    main()
