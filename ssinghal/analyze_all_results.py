import json
import os
import re
from collections import defaultdict
import csv


def get_all_test_files():
    """Get all test result files"""
    results_dir = "ssinghal/test_results"

    if not os.path.exists(results_dir):
        return []

    test_files = []
    for file in os.listdir(results_dir):
        if file.endswith("_results.txt"):
            operator = file.replace("_results.txt", "")
            test_files.append(operator)

    return test_files


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
    summary_lines = content.split("\n")[-20:]  # Look at last 20 lines
    for line in summary_lines:
        if "passed" in line and ("skipped" in line or "failed" in line or line.strip().endswith("passed")):
            # Parse various formats
            numbers = re.findall(r"(\d+)", line)
            if len(numbers) >= 1:
                if "failed" in line and "skipped" in line and "passed" in line:
                    # Format: "X passed, Y failed, Z skipped"
                    results["passed"] = int(numbers[0]) if "passed" in line else 0
                    results["failed"] = int(numbers[1]) if len(numbers) > 1 and "failed" in line else 0
                    results["skipped"] = int(numbers[2]) if len(numbers) > 2 and "skipped" in line else 0
                elif "failed" in line and "passed" in line:
                    # Format: "X passed, Y failed"
                    results["passed"] = int(numbers[0]) if "passed" in line else 0
                    results["failed"] = int(numbers[1]) if len(numbers) > 1 else 0
                elif "skipped" in line and "passed" in line:
                    # Format: "X passed, Y skipped"
                    results["passed"] = int(numbers[0]) if "passed" in line else 0
                    results["skipped"] = int(numbers[1]) if len(numbers) > 1 else 0
                elif line.strip().endswith("passed") and len(numbers) == 1:
                    # Format: "X passed"
                    results["passed"] = int(numbers[0])
                elif "failed" in line and len(numbers) == 1:
                    # Format: "X failed"
                    results["failed"] = int(numbers[0])
                elif "skipped" in line and len(numbers) == 1:
                    # Format: "X skipped"
                    results["skipped"] = int(numbers[0])

                results["total_tests"] = results["passed"] + results["failed"] + results["skipped"]
                if results["total_tests"] > 0:
                    results["status"] = "completed"
                break

    # Extract OOM failures specifically
    oom_pattern = r"SKIPPED.*?OOM: (\[[^\]]+\]).*?allocate (\d+) B.*?store (\d+) B"
    oom_matches = re.findall(oom_pattern, content, re.DOTALL)

    for match in oom_matches:
        input_shape = match[0]
        total_memory = int(match[1])
        per_bank_memory = int(match[2])

        oom_failure = {
            "input_shape": input_shape,
            "total_memory_B": total_memory,
            "total_memory_MB": round(total_memory / (1024 * 1024), 2),
            "per_bank_memory_B": per_bank_memory,
            "per_bank_memory_KB": round(per_bank_memory / 1024, 2),
            "reason": "out_of_memory",
        }

        results["oom_failures"].append(oom_failure)

    return results


def analyze_all_operators():
    """Analyze test results for all operators"""

    operators = get_all_test_files()
    print(f"Found test results for {len(operators)} operators")

    all_results = {}
    summary_stats = {
        "total_operators_tested": 0,
        "total_tests_run": 0,
        "total_passed": 0,
        "total_failed": 0,
        "total_skipped": 0,
        "operators_with_oom": 0,
        "total_oom_failures": 0,
    }

    all_oom_failures = []

    for operator in operators:
        result_file = f"ssinghal/test_results/{operator}_results.txt"

        print(f"Analyzing {operator}...")

        result = parse_test_output(result_file, operator)
        all_results[operator] = result

        if result["status"] == "completed":
            summary_stats["total_operators_tested"] += 1
            summary_stats["total_tests_run"] += result["total_tests"]
            summary_stats["total_passed"] += result["passed"]
            summary_stats["total_failed"] += result["failed"]
            summary_stats["total_skipped"] += result["skipped"]

            if result["oom_failures"]:
                summary_stats["operators_with_oom"] += 1
                summary_stats["total_oom_failures"] += len(result["oom_failures"])

                # Add operator info to each OOM failure
                for oom in result["oom_failures"]:
                    oom["operator"] = operator
                    all_oom_failures.append(oom)

        print(f"  {operator}: {result['passed']} passed, {result['failed']} failed, {result['skipped']} skipped")
        if result["oom_failures"]:
            print(f"    OOM failures: {len(result['oom_failures'])}")

    return all_results, summary_stats, all_oom_failures


def create_comprehensive_reports(all_results, summary_stats, all_oom_failures):
    """Create comprehensive analysis reports"""

    # Main analysis report
    analysis = {
        "summary_statistics": summary_stats,
        "detailed_results": all_results,
        "all_oom_failures": all_oom_failures,
    }

    # Save comprehensive analysis
    with open("ssinghal/final_comprehensive_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)

    # Create detailed OOM CSV
    with open("ssinghal/all_oom_failures.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Operator", "Input_Shape", "Total_Memory_B", "Total_Memory_MB", "Per_Bank_Memory_B", "Per_Bank_Memory_KB"]
        )

        for failure in all_oom_failures:
            writer.writerow(
                [
                    failure["operator"],
                    failure["input_shape"],
                    failure["total_memory_B"],
                    failure["total_memory_MB"],
                    failure["per_bank_memory_B"],
                    failure["per_bank_memory_KB"],
                ]
            )

    # Group OOM failures by operator
    oom_by_operator = defaultdict(list)
    for failure in all_oom_failures:
        oom_by_operator[failure["operator"]].append(failure)

    # Sort by memory requirement
    all_oom_failures.sort(key=lambda x: x["total_memory_MB"], reverse=True)

    return analysis, oom_by_operator


def main():
    print("Analyzing comprehensive test results for all operators...")

    all_results, summary_stats, all_oom_failures = analyze_all_operators()
    analysis, oom_by_operator = create_comprehensive_reports(all_results, summary_stats, all_oom_failures)

    print(f"\n=== FINAL COMPREHENSIVE ANALYSIS ===")
    print(f"Operators tested: {summary_stats['total_operators_tested']}")
    print(f"Total test cases: {summary_stats['total_tests_run']}")
    print(f"Success rate: {summary_stats['total_passed']/max(summary_stats['total_tests_run'],1)*100:.1f}%")
    print(f"Total OOM failures: {summary_stats['total_oom_failures']}")
    print(f"Operators with OOM: {summary_stats['operators_with_oom']}")

    if all_oom_failures:
        print(f"\n=== OOM FAILURES BY OPERATOR ===")
        for operator, failures in oom_by_operator.items():
            print(f"{operator}: {len(failures)} failures")

        print(f"\n=== TOP 10 MOST MEMORY-INTENSIVE FAILURES ===")
        for i, failure in enumerate(all_oom_failures[:10], 1):
            print(f"{i:2d}. {failure['operator']:<12} {failure['input_shape']:<25} → {failure['total_memory_MB']} MB")

    print(f"\nReports saved:")
    print(f"  - ssinghal/final_comprehensive_analysis.json")
    print(f"  - ssinghal/all_oom_failures.csv")

    return all_oom_failures


if __name__ == "__main__":
    main()
