#!/usr/bin/env python3
"""
Comprehensive Analysis of TT-Metal Operator Test Results
Generates detailed report on passed/failed/skipped tests, failure reasons, and OOM analysis
"""

import os
import re
import json
from datetime import datetime
from collections import defaultdict


def calculate_memory_mb(shape, data_type_bytes=2):
    """Calculate estimated memory in MB for a tensor shape (assumes bfloat16 = 2 bytes)"""
    if not shape or len(shape) == 0:
        return 0.0

    elements = 1
    for dim in shape:
        elements *= dim
    return (elements * data_type_bytes) / (1024 * 1024)


def parse_tensor_shape(shape_str):
    """Parse tensor shape from string like '[1, 3, 4320, 7680]'"""
    try:
        # Remove brackets and split by comma
        shape_str = shape_str.strip("[]")
        dims = [int(x.strip()) for x in shape_str.split(",")]
        return dims
    except:
        return None


def analyze_test_result_file(filepath):
    """Analyze a single test result file"""

    operator_name = os.path.basename(filepath).replace("test_", "").replace("_results.txt", "")

    try:
        with open(filepath, "r") as f:
            content = f.read()
    except Exception as e:
        return {
            "operator": operator_name,
            "status": "ERROR",
            "error": f"Could not read file: {e}",
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "oom_failures": [],
            "other_failures": [],
            "failure_reasons": [],
        }

    # Extract test results using pytest output patterns
    passed_count = 0
    failed_count = 0
    skipped_count = 0
    oom_failures = []
    other_failures = []
    failure_reasons = []

    # Count PASSED, FAILED, SKIPPED
    passed_count = len(re.findall(r"PASSED", content))
    failed_count = len(re.findall(r"FAILED", content))
    skipped_count = len(re.findall(r"SKIPPED", content))

    total_tests = passed_count + failed_count + skipped_count

    # Extract OOM failures with shapes
    oom_patterns = [
        r"SKIPPED.*OOM:?\s*\[([^\]]+)\]",
        r"OutOfMemoryError.*\[([^\]]+)\]",
        r"out of memory.*shape.*\[([^\]]+)\]",
    ]

    for pattern in oom_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        for match in matches:
            shape = parse_tensor_shape(match)
            if shape:
                oom_failures.append(
                    {
                        "operator": operator_name,
                        "shape": shape,
                        "memory_mb": calculate_memory_mb(shape),
                        "reason": "Out of Memory",
                    }
                )

    # Extract other failure reasons
    if "ttnn.matmul: The width of the first tensor must be equal to the height" in content:
        failure_reasons.append("Matrix dimension mismatch")

    if "module 'ttnn' has no attribute" in content:
        failure_reasons.append("Missing ttnn operator implementation")

    if "RuntimeError" in content:
        failure_reasons.append("Runtime error")

    if "ImportError" in content or "ModuleNotFoundError" in content:
        failure_reasons.append("Import/Module error")

    # Determine overall status
    if failed_count > 0:
        overall_status = "FAILED"
    elif passed_count > 0:
        overall_status = "PASSED"
    elif skipped_count > 0:
        overall_status = "SKIPPED"
    else:
        overall_status = "UNKNOWN"

    return {
        "operator": operator_name,
        "status": overall_status,
        "total_tests": total_tests if total_tests > 0 else 1,
        "passed": passed_count,
        "failed": failed_count,
        "skipped": skipped_count,
        "oom_failures": oom_failures,
        "other_failures": other_failures,
        "failure_reasons": failure_reasons,
        "pass_rate": (passed_count / total_tests * 100) if total_tests > 0 else 0,
    }


def generate_comprehensive_report():
    """Generate comprehensive operator test analysis report"""

    print("🔍 Analyzing TT-Metal Operator Test Results...")
    print("=" * 60)

    results_dir = "."

    # Find all test result files
    result_files = [f for f in os.listdir(results_dir) if f.startswith("test_") and f.endswith("_results.txt")]

    if not result_files:
        print("❌ No test result files found!")
        return

    print(f"📁 Found {len(result_files)} test result files")

    # Analyze each test file
    all_results = []
    total_operators = len(result_files)
    total_tests = 0
    total_passed = 0
    total_failed = 0
    total_skipped = 0
    total_oom_failures = 0

    operators_by_status = {"PASSED": [], "FAILED": [], "SKIPPED": [], "ERROR": [], "UNKNOWN": []}

    all_oom_failures = []
    failure_reason_counts = defaultdict(int)

    print(f"\n📊 Analyzing operator test results...")

    for i, result_file in enumerate(sorted(result_files)):
        result = analyze_test_result_file(result_file)
        all_results.append(result)

        # Update totals
        total_tests += result["total_tests"]
        total_passed += result["passed"]
        total_failed += result["failed"]
        total_skipped += result["skipped"]
        total_oom_failures += len(result["oom_failures"])

        # Categorize operators
        operators_by_status[result["status"]].append(result["operator"])

        # Collect failures
        all_oom_failures.extend(result["oom_failures"])

        # Count failure reasons
        for reason in result["failure_reasons"]:
            failure_reason_counts[reason] += 1

        print(
            f"  ({i+1:>2}/{total_operators}) {result['operator']:<20} | {result['status']:<8} | Tests: {result['total_tests']:>3} | Pass: {result['passed']:>3} | Fail: {result['failed']:>3} | Skip: {result['skipped']:>3} | OOM: {len(result['oom_failures']):>2}"
        )

    # Generate comprehensive report
    report_filename = f"comprehensive_operator_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

    with open(report_filename, "w") as f:
        f.write("# TT-Metal Operator Test Comprehensive Analysis Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Analysis Scope:** All operator tests in ssinghal/tests/\n\n")

        # Executive Summary
        f.write("## 📊 EXECUTIVE SUMMARY\n\n")
        f.write("| Metric | Count | Percentage |\n")
        f.write("|--------|-------|------------|\n")
        f.write(f"| **Total Operators Tested** | {total_operators} | 100.0% |\n")
        f.write(f"| **Total Individual Tests** | {total_tests} | 100.0% |\n")
        f.write(f"| **Tests Passed** | {total_passed} | {total_passed/total_tests*100:.1f}% |\n")
        f.write(f"| **Tests Failed** | {total_failed} | {total_failed/total_tests*100:.1f}% |\n")
        f.write(f"| **Tests Skipped** | {total_skipped} | {total_skipped/total_tests*100:.1f}% |\n")
        f.write(f"| **OOM Failures** | {total_oom_failures} | {total_oom_failures/total_tests*100:.1f}% |\n\n")

        # Operator Status Breakdown
        f.write("## 🎯 OPERATOR STATUS BREAKDOWN\n\n")

        for status, operators in operators_by_status.items():
            if operators:
                f.write(f"### {status} Operators ({len(operators)})\n\n")
                for op in sorted(operators):
                    op_result = next(r for r in all_results if r["operator"] == op)
                    f.write(
                        f"- **{op}** | Tests: {op_result['total_tests']} | Pass Rate: {op_result['pass_rate']:.1f}%\n"
                    )
                f.write("\n")

        # Detailed Test Results
        f.write("## 📋 DETAILED TEST RESULTS\n\n")
        f.write("| Operator | Status | Total Tests | Passed | Failed | Skipped | Pass Rate | OOM Count |\n")
        f.write("|----------|--------|-------------|--------|--------|---------|-----------|-----------||\n")

        for result in sorted(all_results, key=lambda x: x["operator"]):
            oom_count = len(result["oom_failures"])
            status_emoji = {"PASSED": "✅", "FAILED": "❌", "SKIPPED": "⏭️", "ERROR": "💥"}.get(result["status"], "❓")
            f.write(
                f"| {result['operator']} | {status_emoji} {result['status']} | {result['total_tests']} | {result['passed']} | {result['failed']} | {result['skipped']} | {result['pass_rate']:.1f}% | {oom_count} |\n"
            )

        f.write("\n")

        # OOM Analysis
        if all_oom_failures:
            f.write("## 🧠 OUT OF MEMORY (OOM) ANALYSIS\n\n")
            f.write(f"**Total OOM Failures:** {len(all_oom_failures)}\n\n")

            # Group OOM failures by operator
            oom_by_operator = defaultdict(list)
            for oom in all_oom_failures:
                oom_by_operator[oom["operator"]].append(oom)

            f.write("### OOM Failures by Operator\n\n")
            f.write("| Operator | OOM Count | Largest Shape | Max Memory (MB) | Max Memory (GB) |\n")
            f.write("|----------|-----------|---------------|-----------------|------------------|\n")

            for op_name, oom_list in sorted(oom_by_operator.items()):
                max_memory = max([oom["memory_mb"] for oom in oom_list])
                largest_shape = None
                for oom in oom_list:
                    if oom["memory_mb"] == max_memory:
                        largest_shape = oom["shape"]
                        break

                f.write(
                    f"| {op_name} | {len(oom_list)} | {largest_shape} | {max_memory:.1f} | {max_memory/1024:.2f} |\n"
                )

            f.write("\n")

            # Detailed OOM shapes
            f.write("### Critical OOM Shapes (Top 20)\n\n")
            sorted_oom = sorted(all_oom_failures, key=lambda x: x["memory_mb"], reverse=True)

            f.write("| Operator | Shape | Memory (MB) | Memory (GB) |\n")
            f.write("|----------|-------|-------------|-------------|\n")

            for oom in sorted_oom[:20]:
                f.write(
                    f"| {oom['operator']} | {oom['shape']} | {oom['memory_mb']:.1f} | {oom['memory_mb']/1024:.3f} |\n"
                )

            if len(sorted_oom) > 20:
                f.write(f"\n*... and {len(sorted_oom) - 20} more OOM failures*\n")

            f.write("\n")

        # Failure Reason Analysis
        if failure_reason_counts:
            f.write("## ❌ FAILURE REASON ANALYSIS\n\n")
            f.write("| Failure Reason | Count |\n")
            f.write("|----------------|-------|\n")

            for reason, count in sorted(failure_reason_counts.items(), key=lambda x: x[1], reverse=True):
                f.write(f"| {reason} | {count} |\n")

            f.write("\n")

        # Recommendations
        f.write("## 💡 RECOMMENDATIONS\n\n")
        f.write("### Immediate Actions:\n\n")
        f.write("1. **Memory Optimization Priority:**\n")

        if all_oom_failures:
            oom_by_op = defaultdict(int)
            for oom in all_oom_failures:
                oom_by_op[oom["operator"]] += 1

            top_oom_ops = sorted(oom_by_op.items(), key=lambda x: x[1], reverse=True)[:5]
            for i, (op_name, count) in enumerate(top_oom_ops):
                f.write(f"   {i+1}. **{op_name}** operator ({count} OOM failures)\n")

        f.write("\n2. **Implementation Strategies:**\n")
        f.write("   - **Tensor Chunking:** Split large tensors into smaller manageable pieces\n")
        f.write("   - **Memory Pooling:** Implement efficient memory reuse patterns\n")
        f.write("   - **Streaming Processing:** Process data in streams rather than loading entire tensors\n")
        f.write("   - **Precision Reduction:** Consider using smaller data types where appropriate\n\n")

        f.write("---\n\n")
        f.write("*Report generated by TT-Metal Operator Test Analyzer*\n")

    # Save raw data as JSON
    json_data = {
        "summary": {
            "total_operators": total_operators,
            "total_tests": total_tests,
            "total_passed": total_passed,
            "total_failed": total_failed,
            "total_skipped": total_skipped,
            "total_oom_failures": total_oom_failures,
            "analysis_timestamp": datetime.now().isoformat(),
        },
        "operators_by_status": operators_by_status,
        "detailed_results": all_results,
        "oom_failures": all_oom_failures,
        "failure_reason_counts": dict(failure_reason_counts),
    }

    json_filename = f"operator_test_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(json_filename, "w") as f:
        json.dump(json_data, f, indent=2)

    print(f"\n✅ Analysis complete!")
    print(f"📄 Detailed report saved: {report_filename}")
    print(f"📊 Raw data saved: {json_filename}")

    # Print summary to console
    print(f"\n" + "=" * 60)
    print(f"📋 QUICK SUMMARY")
    print(f"=" * 60)
    print(f"Operators Analyzed:     {total_operators}")
    print(f"Total Individual Tests: {total_tests}")
    print(f"Passed:                 {total_passed} ({total_passed/total_tests*100:.1f}%)")
    print(f"Failed:                 {total_failed} ({total_failed/total_tests*100:.1f}%)")
    print(f"Skipped:                {total_skipped} ({total_skipped/total_tests*100:.1f}%)")
    print(f"OOM Failures:           {total_oom_failures} ({total_oom_failures/total_tests*100:.1f}%)")
    print("")
    print(f"Operators by Status:")
    for status, ops in operators_by_status.items():
        if ops:
            print(f"  {status}: {len(ops)} operators")


if __name__ == "__main__":
    generate_comprehensive_report()
