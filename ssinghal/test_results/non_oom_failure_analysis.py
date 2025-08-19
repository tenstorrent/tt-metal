#!/usr/bin/env python3
"""
Non-OOM Failure Analysis of TT-Metal Operator Test Results
Analyzes all failure causes excluding Out of Memory issues
"""

import os
import re
import json
from datetime import datetime
from collections import defaultdict


def extract_non_oom_failures(filepath):
    """Extract all non-OOM failure information from a test result file"""

    operator_name = os.path.basename(filepath).replace("test_", "").replace("_results.txt", "")
    failures = []

    try:
        with open(filepath, "r") as f:
            content = f.read()
    except Exception as e:
        return []

    # Skip if this is primarily an OOM file
    if content.count("OOM") > content.count("FAILED") and "FAILED" in content:
        # Still check for non-OOM failures even in OOM-heavy files
        pass

    # Extract different types of non-OOM failures
    failure_patterns = {
        "import_error": [r"ImportError: (.+)", r"ModuleNotFoundError: (.+)", r"No module named (.+)"],
        "attribute_error": [r"AttributeError: module \'ttnn\' has no attribute \'(\w+)\'", r"AttributeError: (.+)"],
        "runtime_error": [r"RuntimeError: (.+?)(?:\n|$)", r"ttnn\.([a-zA-Z_]+): (.+?)(?:\n|$)"],
        "value_error": [r"ValueError: (.+?)(?:\n|$)"],
        "assertion_error": [r"AssertionError: (.+?)(?:\n|$)", r"assert (.+?) failed"],
        "dimension_mismatch": [
            r"ttnn\.matmul: The width of the first tensor must be equal to the height of the second tensor",
            r"dimension mismatch: (.+)",
            r"shape mismatch: (.+)",
        ],
        "type_error": [r"TypeError: (.+?)(?:\n|$)"],
        "collection_error": [r"ERROR collecting (.+)", r"ERRORS (.+)"],
        "syntax_error": [r"SyntaxError: (.+)", r"IndentationError: (.+)"],
        "tt_metal_error": [r"TT_THROW @ (.+)", r"TT_FATAL @ (.+)", r"Device assertion failed: (.+)"],
        "memory_layout_error": [
            r"Unsupported memory layout (.+)",
            r"Layout not supported (.+)",
            r"Memory config error (.+)",
        ],
        "device_error": [r"Device not available (.+)", r"Device error (.+)", r"Cannot allocate device (.+)"],
        "timeout_error": [r"TimeoutError: (.+)", r"Test timed out (.+)", r"timeout (.+)"],
    }

    # Check for FAILED tests and extract their reasons
    failed_test_pattern = r"FAILED (.+?) - (.+)"
    failed_matches = re.findall(failed_test_pattern, content, re.MULTILINE)

    for test_name, reason in failed_matches:
        # Skip if this is an OOM failure
        if "OOM" not in reason and "out of memory" not in reason.lower():
            failures.append(
                {
                    "operator": operator_name,
                    "test_name": test_name,
                    "failure_type": "test_failed",
                    "reason": reason.strip(),
                    "severity": "high",
                }
            )

    # Extract specific error patterns
    for error_type, patterns in failure_patterns.items():
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                # Skip OOM-related matches
                if isinstance(match, tuple):
                    match_text = " ".join(match)
                else:
                    match_text = match

                if "OOM" not in match_text and "out of memory" not in match_text.lower():
                    severity = (
                        "critical" if error_type in ["import_error", "collection_error", "syntax_error"] else "high"
                    )

                    failures.append(
                        {
                            "operator": operator_name,
                            "failure_type": error_type,
                            "reason": match_text[:200] + "..." if len(match_text) > 200 else match_text,
                            "severity": severity,
                            "pattern_matched": pattern,
                        }
                    )

    # Look for specific ttnn operator issues
    ttnn_specific_patterns = [
        r"ttnn\.(\w+) is not implemented",
        r"ttnn\.(\w+) does not support (.+)",
        r"ttnn\.(\w+) requires (.+)",
        r"Operation (\w+) not supported on (.+)",
    ]

    for pattern in ttnn_specific_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple):
                match_text = " ".join(match)
            else:
                match_text = match

            failures.append(
                {
                    "operator": operator_name,
                    "failure_type": "ttnn_limitation",
                    "reason": match_text,
                    "severity": "medium",
                    "category": "implementation_gap",
                }
            )

    # Remove duplicates
    unique_failures = []
    seen_reasons = set()
    for failure in failures:
        reason_key = (failure["operator"], failure["failure_type"], failure["reason"][:100])
        if reason_key not in seen_reasons:
            seen_reasons.add(reason_key)
            unique_failures.append(failure)

    return unique_failures


def analyze_non_oom_failures():
    """Analyze non-OOM failures across all test result files"""

    print("❌ Non-OOM Failure Analysis of TT-Metal Operator Tests")
    print("=" * 65)

    results_dir = "."
    result_files = [f for f in os.listdir(results_dir) if f.startswith("test_") and f.endswith("_results.txt")]

    if not result_files:
        print("❌ No test result files found!")
        return

    print(f"📁 Analyzing {len(result_files)} test result files for non-OOM failures...")

    all_failures = []
    failures_by_operator = defaultdict(list)
    failures_by_type = defaultdict(list)
    failures_by_severity = defaultdict(list)

    operators_with_failures = 0
    total_failure_count = 0

    for i, result_file in enumerate(sorted(result_files)):
        operator_name = result_file.replace("test_", "").replace("_results.txt", "")
        failures = extract_non_oom_failures(result_file)

        if failures:
            operators_with_failures += 1
            all_failures.extend(failures)
            failures_by_operator[operator_name] = failures
            total_failure_count += len(failures)

            print(f"  ({i+1:>2}/{len(result_files)}) {operator_name:<25} | Non-OOM Failures: {len(failures):>2}")

            # Categorize failures
            for failure in failures:
                failures_by_type[failure["failure_type"]].append(failure)
                failures_by_severity[failure["severity"]].append(failure)
        else:
            print(f"  ({i+1:>2}/{len(result_files)}) {operator_name:<25} | ✅ No non-OOM failures")

    print(f"\n📊 Non-OOM Failure Summary:")
    print(f"  Total Non-OOM Failures: {total_failure_count}")
    print(
        f"  Operators with Non-OOM Failures: {operators_with_failures}/{len(result_files)} ({operators_with_failures/len(result_files)*100:.1f}%)"
    )
    print(f"  Unique Failure Types: {len(failures_by_type)}")

    # Generate comprehensive report
    generate_non_oom_report(
        all_failures,
        failures_by_operator,
        failures_by_type,
        failures_by_severity,
        total_failure_count,
        operators_with_failures,
        len(result_files),
    )

    return {
        "all_failures": all_failures,
        "failures_by_operator": dict(failures_by_operator),
        "failures_by_type": dict(failures_by_type),
        "failures_by_severity": dict(failures_by_severity),
        "summary": {
            "total_failure_count": total_failure_count,
            "operators_with_failures": operators_with_failures,
            "total_operators": len(result_files),
            "unique_failure_types": len(failures_by_type),
        },
    }


def generate_non_oom_report(
    all_failures,
    failures_by_operator,
    failures_by_type,
    failures_by_severity,
    total_failure_count,
    operators_with_failures,
    total_operators,
):
    """Generate comprehensive non-OOM failure report"""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"non_oom_failure_analysis_{timestamp}.md"

    # Sort operators by failure count
    sorted_operators = sorted(failures_by_operator.items(), key=lambda x: len(x[1]), reverse=True)

    # Sort failure types by frequency
    sorted_failure_types = sorted(failures_by_type.items(), key=lambda x: len(x[1]), reverse=True)

    with open(report_filename, "w") as f:
        f.write("# TT-Metal Operator Non-OOM Failure Analysis Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Focus:** All failure causes excluding Out of Memory (OOM) issues\n")
        f.write(f"**Scope:** Analysis of {total_operators} operators with {total_failure_count} non-OOM failures\n\n")

        # Executive Summary
        f.write("## 🚨 NON-OOM FAILURE EXECUTIVE SUMMARY\n\n")
        f.write("| **Metric** | **Value** | **Impact** |\n")
        f.write("|------------|-----------|------------|\n")
        f.write(f"| **Total Non-OOM Failures** | {total_failure_count:,} | Implementation Issues |\n")
        f.write(
            f"| **Operators Affected** | {operators_with_failures}/{total_operators} | {operators_with_failures/total_operators*100:.1f}% |\n"
        )
        f.write(f"| **Unique Failure Types** | {len(failures_by_type)} | Error Categories |\n")
        f.write(f"| **Critical Failures** | {len(failures_by_severity.get('critical', []))} | Blocking Issues |\n")
        f.write(f"| **High Priority Failures** | {len(failures_by_severity.get('high', []))} | Major Issues |\n")
        f.write(f"| **Medium Priority Failures** | {len(failures_by_severity.get('medium', []))} | Minor Issues |\n\n")

        # Failure Type Breakdown
        f.write("## 📊 FAILURE TYPE BREAKDOWN\n\n")
        f.write("| **Failure Type** | **Count** | **Percentage** | **Severity** | **Description** |\n")
        f.write("|------------------|-----------|----------------|--------------|------------------|\n")

        failure_type_descriptions = {
            "attribute_error": "Missing ttnn operator implementations",
            "runtime_error": "Runtime execution failures",
            "dimension_mismatch": "Tensor dimension compatibility issues",
            "import_error": "Module import/loading failures",
            "value_error": "Invalid parameter values",
            "assertion_error": "Test assertion failures",
            "collection_error": "Test collection/discovery failures",
            "type_error": "Data type incompatibility",
            "tt_metal_error": "TT-Metal framework errors",
            "memory_layout_error": "Memory layout incompatibility",
            "device_error": "Device access/availability issues",
            "timeout_error": "Test execution timeouts",
            "ttnn_limitation": "TTNN framework limitations",
            "test_failed": "General test failures",
        }

        for failure_type, failures in sorted_failure_types:
            count = len(failures)
            percentage = count / total_failure_count * 100

            # Determine overall severity for this type
            severities = [f.get("severity", "medium") for f in failures]
            if "critical" in severities:
                severity = "🚨 Critical"
            elif "high" in severities:
                severity = "⚠️ High"
            else:
                severity = "🟡 Medium"

            description = failure_type_descriptions.get(failure_type, "Other failure type")

            f.write(
                f"| **{failure_type.replace('_', ' ').title()}** | {count} | {percentage:.1f}% | {severity} | {description} |\n"
            )

        f.write("\n")

        # Operator-Specific Failures
        f.write("## 🎯 OPERATOR-SPECIFIC FAILURE ANALYSIS\n\n")
        f.write("| **Operator** | **Failure Count** | **Primary Issue** | **Status** |\n")
        f.write("|--------------|-------------------|-------------------|------------|\n")

        for operator_name, failures in sorted_operators:
            failure_count = len(failures)

            # Find most common failure type for this operator
            type_counts = defaultdict(int)
            for failure in failures:
                type_counts[failure["failure_type"]] += 1

            primary_issue = max(type_counts.items(), key=lambda x: x[1])[0] if type_counts else "unknown"
            primary_issue_display = primary_issue.replace("_", " ").title()

            # Determine status
            has_critical = any(f.get("severity") == "critical" for f in failures)
            has_high = any(f.get("severity") == "high" for f in failures)

            if has_critical:
                status = "🚨 CRITICAL"
            elif has_high:
                status = "⚠️ HIGH"
            else:
                status = "🟡 MEDIUM"

            f.write(f"| **{operator_name}** | {failure_count} | {primary_issue_display} | {status} |\n")

        f.write("\n")

        # Detailed Failure Analysis by Type
        f.write("## 🔍 DETAILED FAILURE ANALYSIS BY TYPE\n\n")

        for failure_type, failures in sorted_failure_types:
            if len(failures) > 0:
                f.write(f"### {failure_type.replace('_', ' ').title()} ({len(failures)} failures)\n\n")

                # Group by operator
                operators_affected = defaultdict(list)
                for failure in failures:
                    operators_affected[failure["operator"]].append(failure)

                f.write(f"**Operators Affected:** {len(operators_affected)}\n\n")

                # Show top examples
                f.write("**Examples:**\n")
                for i, failure in enumerate(failures[:5]):  # Top 5 examples
                    reason = failure["reason"][:100] + "..." if len(failure["reason"]) > 100 else failure["reason"]
                    f.write(f"{i+1}. **{failure['operator']}**: {reason}\n")

                if len(failures) > 5:
                    f.write(f"... and {len(failures) - 5} more similar failures\n")

                f.write("\n")

        # Critical Issues Requiring Immediate Attention
        f.write("## �� CRITICAL ISSUES REQUIRING IMMEDIATE ATTENTION\n\n")

        critical_failures = failures_by_severity.get("critical", [])
        if critical_failures:
            f.write(f"**{len(critical_failures)} Critical Failures Found:**\n\n")

            critical_by_type = defaultdict(list)
            for failure in critical_failures:
                critical_by_type[failure["failure_type"]].append(failure)

            for failure_type, failures in critical_by_type.items():
                f.write(f"### {failure_type.replace('_', ' ').title()}\n")
                f.write(f"**Affected Operators:** {', '.join(set(f['operator'] for f in failures))}\n")
                f.write(f"**Impact:** Blocks basic functionality\n\n")
        else:
            f.write("✅ No critical failures found.\n\n")

        # Implementation Gaps
        f.write("## 🔧 TTNN IMPLEMENTATION GAPS\n\n")

        attribute_errors = failures_by_type.get("attribute_error", [])
        if attribute_errors:
            missing_operators = set()
            for failure in attribute_errors:
                # Extract operator name from attribute error
                match = re.search(r"has no attribute '(\w+)'", failure["reason"])
                if match:
                    missing_operators.add(match.group(1))

            if missing_operators:
                f.write("**Missing TTNN Operators:**\n")
                for op in sorted(missing_operators):
                    affected_tests = [f["operator"] for f in attribute_errors if op in f["reason"]]
                    f.write(f"- **{op}**: Required by {', '.join(set(affected_tests))}\n")
                f.write("\n")

        # Recommendations
        f.write("## 💡 RECOMMENDATIONS FOR NON-OOM FAILURES\n\n")
        f.write("### 🚨 IMMEDIATE ACTIONS (Critical Priority)\n\n")

        if "import_error" in failures_by_type:
            f.write("1. **Fix Import/Module Errors:**\n")
            f.write("   - Resolve missing module dependencies\n")
            f.write("   - Fix import path issues\n")
            f.write("   - Ensure proper environment setup\n\n")

        if "attribute_error" in failures_by_type:
            f.write("2. **Implement Missing TTNN Operators:**\n")
            missing_ops = set()
            for failure in failures_by_type["attribute_error"]:
                match = re.search(r"has no attribute '(\w+)'", failure["reason"])
                if match:
                    missing_ops.add(match.group(1))

            for op in sorted(list(missing_ops)[:5]):  # Top 5
                f.write(f"   - Implement `ttnn.{op}` operator\n")
            f.write("\n")

        if "dimension_mismatch" in failures_by_type:
            f.write("3. **Fix Dimension Compatibility Issues:**\n")
            f.write("   - Review tensor shape requirements for operations\n")
            f.write("   - Add input validation and shape checking\n")
            f.write("   - Implement automatic shape broadcasting where appropriate\n\n")

        f.write("### ⚠️ HIGH PRIORITY ACTIONS\n\n")
        f.write("4. **Runtime Error Resolution:**\n")
        f.write("   - Debug and fix TT-Metal framework errors\n")
        f.write("   - Improve error handling and reporting\n")
        f.write("   - Add graceful fallbacks for unsupported operations\n\n")

        f.write("5. **Memory Layout Compatibility:**\n")
        f.write("   - Ensure consistent memory layout handling\n")
        f.write("   - Add automatic layout conversion where needed\n")
        f.write("   - Document memory layout requirements\n\n")

        f.write("### 🟡 MEDIUM PRIORITY ACTIONS\n\n")
        f.write("6. **Test Infrastructure Improvements:**\n")
        f.write("   - Fix test collection and discovery issues\n")
        f.write("   - Improve test timeout handling\n")
        f.write("   - Add better error reporting and logging\n\n")

        f.write("7. **Device Management:**\n")
        f.write("   - Improve device availability checking\n")
        f.write("   - Add device resource management\n")
        f.write("   - Implement device fallback strategies\n\n")

        # Summary Statistics
        f.write("## 📈 FAILURE STATISTICS SUMMARY\n\n")

        f.write("### By Severity\n")
        for severity in ["critical", "high", "medium"]:
            count = len(failures_by_severity.get(severity, []))
            percentage = count / total_failure_count * 100 if total_failure_count > 0 else 0
            f.write(f"- **{severity.title()}**: {count} failures ({percentage:.1f}%)\n")

        f.write("\n### By Category\n")
        categories = {
            "Implementation Issues": ["attribute_error", "ttnn_limitation"],
            "Runtime Issues": ["runtime_error", "tt_metal_error", "device_error"],
            "Configuration Issues": ["import_error", "memory_layout_error"],
            "Test Issues": ["collection_error", "timeout_error", "test_failed"],
            "Compatibility Issues": ["dimension_mismatch", "type_error", "value_error"],
        }

        for category, types in categories.items():
            count = sum(len(failures_by_type.get(t, [])) for t in types)
            percentage = count / total_failure_count * 100 if total_failure_count > 0 else 0
            f.write(f"- **{category}**: {count} failures ({percentage:.1f}%)\n")

        f.write("\n---\n\n")
        f.write("*Non-OOM Failure Analysis Report generated by TT-Metal Test Analyzer*\n")
        f.write(f"*Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")

    # Save raw data as JSON
    json_filename = f"non_oom_failure_data_{timestamp}.json"
    json_data = {
        "summary": {
            "total_failure_count": total_failure_count,
            "operators_with_failures": operators_with_failures,
            "total_operators": total_operators,
            "unique_failure_types": len(failures_by_type),
            "analysis_timestamp": datetime.now().isoformat(),
        },
        "failures_by_operator": failures_by_operator,
        "failures_by_type": {k: v for k, v in failures_by_type.items()},
        "failures_by_severity": {k: v for k, v in failures_by_severity.items()},
        "all_failures": all_failures,
    }

    with open(json_filename, "w") as f:
        json.dump(json_data, f, indent=2)

    print(f"\n✅ Non-OOM Failure Analysis Complete!")
    print(f"📄 Detailed report: {report_filename}")
    print(f"📊 Raw data: {json_filename}")

    # Summary output
    print(f"\n❌ NON-OOM FAILURE SUMMARY:")
    print(f"{'='*55}")
    print(f"Total Non-OOM Failures:    {total_failure_count:,}")
    print(
        f"Operators with Failures:    {operators_with_failures}/{total_operators} ({operators_with_failures/total_operators*100:.1f}%)"
    )
    print(f"Unique Failure Types:       {len(failures_by_type)}")

    if sorted_failure_types:
        print(f"\nTop 5 Failure Types:")
        for i, (failure_type, failures) in enumerate(sorted_failure_types[:5]):
            percentage = len(failures) / total_failure_count * 100
            print(
                f"  {i+1}. {failure_type.replace('_', ' ').title():<25} | {len(failures):>3} failures ({percentage:.1f}%)"
            )


if __name__ == "__main__":
    analyze_non_oom_failures()
