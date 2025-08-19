#!/usr/bin/env python3
"""
Comprehensive Test Results Analyzer
Analyzes all operator test results and generates detailed reports with:
- Operator name
- Test counts (passed/failed)
- Memory issues
- Input sizes
- Error patterns
"""

import os
import glob
import re
import json
from datetime import datetime
from collections import defaultdict
import math


def calculate_tensor_memory(shape, dtype="bfloat16"):
    """Calculate memory usage for a tensor given its shape and data type"""
    if not shape:
        return 0

    # Bytes per element for different data types
    dtype_sizes = {"bfloat16": 2, "float16": 2, "float32": 4, "int32": 4, "int8": 1, "bool": 1}

    try:
        # Calculate total elements
        total_elements = 1
        for dim in shape:
            total_elements *= dim

        # Calculate bytes
        bytes_per_element = dtype_sizes.get(dtype, 2)  # Default to bfloat16
        total_bytes = total_elements * bytes_per_element

        # Convert to human readable format
        if total_bytes < 1024:
            return f"{total_bytes}B"
        elif total_bytes < 1024**2:
            return f"{total_bytes/1024:.1f}KB"
        elif total_bytes < 1024**3:
            return f"{total_bytes/(1024**2):.1f}MB"
        else:
            return f"{total_bytes/(1024**3):.2f}GB"
    except:
        return "Unknown"


def extract_shapes_from_text(text):
    """Extract tensor shapes from log text"""
    shapes = []

    # Pattern for Shape([...]) format
    shape_pattern = r"Shape\(\[([^\]]+)\]\)"
    matches = re.findall(shape_pattern, text)

    for match in matches:
        try:
            # Parse the shape dimensions
            dims_str = match.strip()
            if dims_str:
                dims = [int(x.strip()) for x in dims_str.split(",")]
                shapes.append(dims)
        except:
            continue

    # Pattern for input_shape parameters
    input_shape_pattern = r"input_shape\d*[=:]?\s*\[([^\]]+)\]"
    matches = re.findall(input_shape_pattern, text)

    for match in matches:
        try:
            dims_str = match.strip()
            if dims_str:
                dims = [int(x.strip()) for x in dims_str.split(",")]
                shapes.append(dims)
        except:
            continue

    return shapes


def analyze_test_file(file_path):
    """Analyze a single test result file"""

    operator_name = os.path.basename(file_path).replace("_results.txt", "").replace("test_", "")

    result = {
        "operator": operator_name,
        "file_path": file_path,
        "total_tests": 0,
        "passed_tests": 0,
        "failed_tests": 0,
        "memory_errors": 0,
        "timeout_errors": 0,
        "collection_errors": 0,
        "other_errors": 0,
        "input_shapes": [],
        "error_types": defaultdict(int),
        "memory_usage_estimates": [],
        "specific_errors": [],
        "test_duration": 0,
    }

    try:
        with open(file_path, "r") as f:
            content = f.read()

        # Extract basic test statistics
        passed_matches = re.findall(r"PASSED", content)
        failed_matches = re.findall(r"FAILED", content)
        error_matches = re.findall(r"ERROR", content)

        result["passed_tests"] = len(passed_matches)
        result["failed_tests"] = len(failed_matches)
        result["total_tests"] = result["passed_tests"] + result["failed_tests"]

        # Extract shapes and calculate memory usage
        shapes = extract_shapes_from_text(content)
        result["input_shapes"] = shapes

        for shape in shapes:
            memory_est = calculate_tensor_memory(shape)
            result["memory_usage_estimates"].append({"shape": shape, "estimated_memory": memory_est})

        # Check for specific error types
        if re.search(r"OutOfMemoryError|OOM|out of memory", content, re.IGNORECASE):
            result["memory_errors"] += 1
            result["error_types"]["memory"] += 1

        if re.search(r"timeout|TimeoutError", content, re.IGNORECASE):
            result["timeout_errors"] += 1
            result["error_types"]["timeout"] += 1

        if re.search(r"error during collection|collection error", content, re.IGNORECASE):
            result["collection_errors"] += 1
            result["error_types"]["collection"] += 1

        # Extract specific error messages
        error_patterns = [
            r"critical.*?\|.*?ttnn\..*?:.*?(?=\n)",
            r"ERROR.*?:.*?(?=\n)",
            r"AttributeError.*?(?=\n)",
            r"TypeError.*?(?=\n)",
            r"AssertionError.*?(?=\n)",
        ]

        for pattern in error_patterns:
            matches = re.findall(pattern, content, re.MULTILINE)
            for match in matches:
                result["specific_errors"].append(match.strip())

        # Extract test duration if available
        duration_match = re.search(r"(\d+\.\d+)s", content)
        if duration_match:
            result["test_duration"] = float(duration_match.group(1))

        # Categorize error types
        if "matmul" in content and "width of the first tensor must be equal to the height" in content:
            result["error_types"]["dimension_mismatch"] += 1
        if "AttributeError" in content:
            result["error_types"]["attribute_error"] += 1
        if "TypeError" in content:
            result["error_types"]["type_error"] += 1
        if "list indices must be integers" in content:
            result["error_types"]["syntax_error"] += 1

    except Exception as e:
        result["analysis_error"] = str(e)

    return result


def generate_comprehensive_report(results):
    """Generate comprehensive analysis report"""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Calculate aggregate statistics
    total_operators = len(results)
    total_tests = sum(r["total_tests"] for r in results)
    total_passed = sum(r["passed_tests"] for r in results)
    total_failed = sum(r["failed_tests"] for r in results)
    total_memory_errors = sum(r["memory_errors"] for r in results)

    success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0

    # Sort results by various criteria
    failed_operators = sorted(
        [r for r in results if r["failed_tests"] > 0], key=lambda x: x["failed_tests"], reverse=True
    )
    memory_problematic = sorted(
        [r for r in results if r["memory_errors"] > 0], key=lambda x: x["memory_errors"], reverse=True
    )

    report = f"""
================================================================================
                        TT-METAL OPERATOR TEST ANALYSIS REPORT
================================================================================
Generated: {timestamp}
Analysis of: {total_operators} operators with {total_tests} total tests
================================================================================

📊 EXECUTIVE SUMMARY
================================================================================
Total Operators Tested:     {total_operators}
Total Tests Executed:       {total_tests}
Tests Passed:               {total_passed} ({total_passed/total_tests*100:.1f}%)
Tests Failed:               {total_failed} ({total_failed/total_tests*100:.1f}%)
Memory-Related Failures:    {total_memory_errors}
Overall Success Rate:       {success_rate:.1f}%

"""

    # Detailed operator analysis
    report += """
================================================================================
📋 DETAILED OPERATOR ANALYSIS
================================================================================
Format: Operator | Total Tests | Passed | Failed | Memory Errors | Status
"""

    for result in sorted(results, key=lambda x: x["operator"]):
        status = "✅ PASS" if result["failed_tests"] == 0 else "❌ FAIL"
        memory_indicator = f"🧠{result['memory_errors']}" if result["memory_errors"] > 0 else ""

        report += f"""
{result['operator']:<20} | {result['total_tests']:>3} | {result['passed_tests']:>3} | {result['failed_tests']:>3} | {result['memory_errors']:>3} | {status} {memory_indicator}"""

    # Memory analysis section
    if memory_problematic:
        report += f"""

================================================================================
🧠 MEMORY ISSUES ANALYSIS
================================================================================
Operators with Memory Problems: {len(memory_problematic)}

"""
        for result in memory_problematic:
            report += f"""
Operator: {result['operator']}
Memory Errors: {result['memory_errors']}
Input Shapes Tested:"""

            if result["memory_usage_estimates"]:
                for est in result["memory_usage_estimates"][:5]:  # Show first 5 shapes
                    report += f"""
  Shape: {est['shape']} → Estimated Memory: {est['estimated_memory']}"""
            else:
                report += "\n  No shape information available"

            if result["specific_errors"]:
                report += f"""
Key Errors:"""
                for error in result["specific_errors"][:3]:  # Show first 3 errors
                    report += f"""
  • {error[:100]}{'...' if len(error) > 100 else ''}"""
            report += "\n"

    # Failed operators analysis
    if failed_operators:
        report += f"""
================================================================================
❌ FAILED OPERATORS ANALYSIS
================================================================================
Top Failed Operators (by failure count):

"""
        for result in failed_operators[:15]:  # Top 15 failed operators
            failure_rate = (result["failed_tests"] / result["total_tests"] * 100) if result["total_tests"] > 0 else 0
            report += f"""
{result['operator']:<20} | Failed: {result['failed_tests']:>2}/{result['total_tests']:>2} ({failure_rate:>5.1f}%) | Memory: {result['memory_errors']}"""

            # Show top error types for this operator
            if result["error_types"]:
                top_errors = sorted(result["error_types"].items(), key=lambda x: x[1], reverse=True)[:3]
                error_summary = ", ".join([f"{err}: {count}" for err, count in top_errors])
                report += f" | Errors: {error_summary}"
            report += "\n"

    # Input size analysis
    report += f"""

================================================================================
📏 INPUT SIZE ANALYSIS
================================================================================
"""

    # Collect all shapes and their frequencies
    all_shapes = defaultdict(int)
    large_shapes = []

    for result in results:
        for shape_info in result["memory_usage_estimates"]:
            shape_tuple = tuple(shape_info["shape"])
            all_shapes[shape_tuple] += 1

            # Consider "large" shapes (>100MB estimated)
            if "GB" in shape_info["estimated_memory"] or (
                "MB" in shape_info["estimated_memory"] and float(shape_info["estimated_memory"].split("MB")[0]) > 100
            ):
                large_shapes.append(
                    {
                        "operator": result["operator"],
                        "shape": shape_info["shape"],
                        "memory": shape_info["estimated_memory"],
                    }
                )

    # Show most common shapes
    common_shapes = sorted(all_shapes.items(), key=lambda x: x[1], reverse=True)[:10]
    report += f"""
Most Common Input Shapes:"""
    for shape, count in common_shapes:
        memory_est = calculate_tensor_memory(list(shape))
        report += f"""
  {str(list(shape)):<30} | Used {count:>2} times | ~{memory_est}"""

    # Show potentially problematic large shapes
    if large_shapes:
        report += f"""

Large Memory Shapes (>100MB):"""
        for item in sorted(large_shapes, key=lambda x: x["memory"], reverse=True)[:10]:
            report += f"""
  {item['operator']:<20} | {str(item['shape']):<30} | ~{item['memory']}"""

    # Error pattern analysis
    report += f"""

================================================================================
🔍 ERROR PATTERN ANALYSIS
================================================================================
"""

    all_error_types = defaultdict(int)
    for result in results:
        for error_type, count in result["error_types"].items():
            all_error_types[error_type] += count

    if all_error_types:
        report += """
Error Type Distribution:"""
        for error_type, count in sorted(all_error_types.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_failed * 100) if total_failed > 0 else 0
            report += f"""
  {error_type:<20} | {count:>3} occurrences ({percentage:>5.1f}% of failures)"""

    # Recommendations
    report += f"""

================================================================================
💡 RECOMMENDATIONS
================================================================================
"""

    recommendations = []

    if total_memory_errors > 0:
        recommendations.append(
            f"🧠 Memory Issues: {total_memory_errors} operators have memory-related failures. Consider optimizing tensor sizes or implementing memory-efficient variants."
        )

    if len(failed_operators) > 0:
        worst_operators = [r["operator"] for r in failed_operators[:5]]
        recommendations.append(
            f"❌ Priority Fixes: Focus on {', '.join(worst_operators)} - these operators have the most failures."
        )

    if "dimension_mismatch" in all_error_types and all_error_types["dimension_mismatch"] > 5:
        recommendations.append(
            "📏 Dimension Mismatches: Many operators have tensor dimension compatibility issues. Review matrix multiplication input requirements."
        )

    if "syntax_error" in all_error_types:
        recommendations.append(
            "🐛 Syntax Errors: Some test files have syntax issues that prevent execution. Review test file structure."
        )

    if not recommendations:
        recommendations.append("✅ Overall system appears healthy with isolated failures.")

    for i, rec in enumerate(recommendations, 1):
        report += f"""
{i}. {rec}"""

    report += f"""

================================================================================
📁 DETAILED RESULTS
================================================================================
Individual test results are available in the ssinghal/test_results/ directory.
For specific operator analysis, examine the corresponding *_results.txt file.

Report generated by TT-Metal Test Analyzer
================================================================================
"""

    return report


def main():
    """Main analysis function"""
    results_dir = "ssinghal/test_results"

    print("🔍 Analyzing TT-Metal operator test results...")
    print(f"📁 Scanning directory: {results_dir}")

    # Find all test result files
    result_files = glob.glob(f"{results_dir}/test_*_results.txt")

    if not result_files:
        print("❌ No test result files found!")
        return

    print(f"📊 Found {len(result_files)} test result files")

    # Analyze each file
    all_results = []

    for i, file_path in enumerate(sorted(result_files), 1):
        operator_name = os.path.basename(file_path).replace("_results.txt", "").replace("test_", "")
        print(f"  ({i:>2}/{len(result_files)}) Analyzing {operator_name}...")

        result = analyze_test_file(file_path)
        all_results.append(result)

    # Generate comprehensive report
    print("📝 Generating comprehensive report...")
    report = generate_comprehensive_report(all_results)

    # Save report
    report_file = f"{results_dir}/comprehensive_analysis_report.txt"
    with open(report_file, "w") as f:
        f.write(report)

    # Save raw data as JSON for further analysis
    json_file = f"{results_dir}/detailed_analysis_data.json"
    with open(json_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"✅ Analysis complete!")
    print(f"📄 Report saved to: {report_file}")
    print(f"📊 Raw data saved to: {json_file}")
    print("\n" + "=" * 80)
    print("📋 QUICK SUMMARY")
    print("=" * 80)

    # Print quick summary
    total_tests = sum(r["total_tests"] for r in all_results)
    total_passed = sum(r["passed_tests"] for r in all_results)
    total_failed = sum(r["failed_tests"] for r in all_results)
    memory_issues = sum(1 for r in all_results if r["memory_errors"] > 0)

    print(f"Operators Analyzed: {len(all_results)}")
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {total_passed} ({total_passed/total_tests*100:.1f}%)")
    print(f"Failed: {total_failed} ({total_failed/total_tests*100:.1f}%)")
    print(f"Operators with Memory Issues: {memory_issues}")
    print()
    print(f"📖 View full report: {report_file}")


if __name__ == "__main__":
    main()
