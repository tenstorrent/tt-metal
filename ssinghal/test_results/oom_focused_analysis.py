#!/usr/bin/env python3
"""
OOM-Focused Analysis of TT-Metal Operator Test Results
Specialized report on Out of Memory failures, shapes, and memory patterns
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


def calculate_memory_gb(memory_mb):
    """Convert MB to GB"""
    return memory_mb / 1024


def parse_tensor_shape(shape_str):
    """Parse tensor shape from string like '[1, 3, 4320, 7680]'"""
    try:
        # Clean up the string and extract numbers
        shape_str = re.sub(r"[^\d,\s]", "", shape_str)
        dims = [int(x.strip()) for x in shape_str.split(",") if x.strip()]
        return dims if len(dims) > 0 else None
    except:
        return None


def extract_oom_failures_from_file(filepath):
    """Extract all OOM failures from a test result file"""

    operator_name = os.path.basename(filepath).replace("test_", "").replace("_results.txt", "")
    oom_failures = []

    try:
        with open(filepath, "r") as f:
            content = f.read()
    except Exception as e:
        return oom_failures

    # Enhanced OOM detection patterns
    oom_patterns = [
        # SKIPPED with OOM and shape
        r"SKIPPED.*?OOM:?\s*\[([^\]]+)\]",
        # SKIPPED lines with shapes that mention OOM
        r"SKIPPED.*?\[([^\]]+)\].*?OOM",
        # Direct OOM error messages with shapes
        r"OutOfMemoryError.*?\[([^\]]+)\]",
        r"out of memory.*?shape.*?\[([^\]]+)\]",
        # TT_THROW messages with OOM
        r"TT_THROW.*?Failed to allocate.*?\[([^\]]+)\]",
        # pytest.skip with OOM
        r"pytest\.skip.*?OOM.*?\[([^\]]+)\]",
    ]

    # Also look for test parameter lines that resulted in skips
    # Pattern: test_name[shape_params] SKIPPED followed by OOM reason
    test_skip_pattern = r"::test_\w+\[([^\]]+)\]\s+SKIPPED"
    skipped_tests = re.findall(test_skip_pattern, content)

    # Check if these skips were due to OOM
    for shape_param in skipped_tests:
        if "OOM" in content:  # If there's any OOM mention in the file
            shape = parse_tensor_shape(shape_param)
            if shape:
                memory_mb = calculate_memory_mb(shape)
                oom_failures.append(
                    {
                        "operator": operator_name,
                        "shape": shape,
                        "memory_mb": memory_mb,
                        "memory_gb": calculate_memory_gb(memory_mb),
                        "source": "parameter_skip",
                    }
                )

    # Extract explicit OOM patterns
    for pattern in oom_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)
        for match in matches:
            shape = parse_tensor_shape(match)
            if shape:
                memory_mb = calculate_memory_mb(shape)
                oom_failures.append(
                    {
                        "operator": operator_name,
                        "shape": shape,
                        "memory_mb": memory_mb,
                        "memory_gb": calculate_memory_gb(memory_mb),
                        "source": "explicit_oom",
                    }
                )

    # Remove duplicates based on shape
    unique_failures = []
    seen_shapes = set()
    for failure in oom_failures:
        shape_tuple = tuple(failure["shape"])
        if shape_tuple not in seen_shapes:
            seen_shapes.add(shape_tuple)
            unique_failures.append(failure)

    return unique_failures


def analyze_all_oom_failures():
    """Analyze OOM failures across all test result files"""

    print("🧠 OOM-Focused Analysis of TT-Metal Operator Tests")
    print("=" * 60)

    results_dir = "."
    result_files = [f for f in os.listdir(results_dir) if f.startswith("test_") and f.endswith("_results.txt")]

    if not result_files:
        print("❌ No test result files found!")
        return

    print(f"📁 Analyzing {len(result_files)} test result files for OOM failures...")

    all_oom_failures = []
    oom_by_operator = defaultdict(list)
    oom_by_shape = defaultdict(list)
    memory_categories = {
        "small": [],  # < 100MB
        "medium": [],  # 100MB - 500MB
        "large": [],  # 500MB - 1GB
        "critical": [],  # > 1GB
    }

    total_oom_count = 0
    operators_with_oom = 0

    for i, result_file in enumerate(sorted(result_files)):
        operator_name = result_file.replace("test_", "").replace("_results.txt", "")
        oom_failures = extract_oom_failures_from_file(result_file)

        if oom_failures:
            operators_with_oom += 1
            all_oom_failures.extend(oom_failures)
            oom_by_operator[operator_name] = oom_failures
            total_oom_count += len(oom_failures)

            print(f"  ({i+1:>2}/{len(result_files)}) {operator_name:<25} | OOM Failures: {len(oom_failures):>3}")

            # Categorize by memory size
            for failure in oom_failures:
                memory_mb = failure["memory_mb"]
                shape_key = tuple(failure["shape"])
                oom_by_shape[shape_key].append(failure)

                if memory_mb < 100:
                    memory_categories["small"].append(failure)
                elif memory_mb < 500:
                    memory_categories["medium"].append(failure)
                elif memory_mb < 1024:
                    memory_categories["large"].append(failure)
                else:
                    memory_categories["critical"].append(failure)
        else:
            print(f"  ({i+1:>2}/{len(result_files)}) {operator_name:<25} | ✅ No OOM issues")

    print(f"\n📊 OOM Analysis Summary:")
    print(f"  Total OOM Failures: {total_oom_count}")
    print(
        f"  Operators with OOM: {operators_with_oom}/{len(result_files)} ({operators_with_oom/len(result_files)*100:.1f}%)"
    )
    print(f"  Unique Problem Shapes: {len(oom_by_shape)}")

    # Generate comprehensive OOM report
    generate_oom_report(
        all_oom_failures,
        oom_by_operator,
        oom_by_shape,
        memory_categories,
        total_oom_count,
        operators_with_oom,
        len(result_files),
    )

    return {
        "all_oom_failures": all_oom_failures,
        "oom_by_operator": dict(oom_by_operator),
        "oom_by_shape": dict(oom_by_shape),
        "memory_categories": memory_categories,
        "summary": {
            "total_oom_count": total_oom_count,
            "operators_with_oom": operators_with_oom,
            "total_operators": len(result_files),
            "unique_problem_shapes": len(oom_by_shape),
        },
    }


def generate_oom_report(
    all_oom_failures,
    oom_by_operator,
    oom_by_shape,
    memory_categories,
    total_oom_count,
    operators_with_oom,
    total_operators,
):
    """Generate comprehensive OOM-focused report"""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"oom_comprehensive_analysis_{timestamp}.md"

    # Sort failures by memory size (largest first)
    sorted_failures = sorted(all_oom_failures, key=lambda x: x["memory_mb"], reverse=True)

    # Sort operators by OOM count
    sorted_operators = sorted(oom_by_operator.items(), key=lambda x: len(x[1]), reverse=True)

    # Sort shapes by frequency and memory impact
    sorted_shapes = sorted(
        oom_by_shape.items(), key=lambda x: (len(x[1]), max(f["memory_mb"] for f in x[1])), reverse=True
    )

    with open(report_filename, "w") as f:
        f.write("# TT-Metal Operator OOM (Out of Memory) Comprehensive Analysis\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Focus:** Out of Memory failures across all operators\n")
        f.write(f"**Scope:** Analysis of {total_operators} operators with {total_oom_count} OOM failures\n\n")

        # Executive Summary
        f.write("## 🚨 OOM EXECUTIVE SUMMARY\n\n")
        f.write("| **Metric** | **Value** | **Impact** |\n")
        f.write("|------------|-----------|------------|\n")
        f.write(f"| **Total OOM Failures** | {total_oom_count:,} | Critical |\n")
        f.write(
            f"| **Operators Affected** | {operators_with_oom}/{total_operators} | {operators_with_oom/total_operators*100:.1f}% |\n"
        )
        f.write(f"| **Unique Problem Shapes** | {len(oom_by_shape)} | Memory Patterns |\n")
        f.write(f"| **Critical Shapes (>1GB)** | {len(memory_categories['critical'])} | Immediate Fix Needed |\n")
        f.write(f"| **Large Shapes (500MB-1GB)** | {len(memory_categories['large'])} | High Priority |\n")
        f.write(f"| **Medium Shapes (100-500MB)** | {len(memory_categories['medium'])} | Medium Priority |\n")
        f.write(f"| **Small Shapes (<100MB)** | {len(memory_categories['small'])} | Low Priority |\n\n")

        # Critical OOM Shapes Analysis
        f.write("## 🔥 CRITICAL OOM SHAPES (>1GB)\n\n")
        if memory_categories["critical"]:
            f.write("| **Shape** | **Memory (GB)** | **Memory (MB)** | **Operators Affected** | **Frequency** |\n")
            f.write("|-----------|-----------------|-----------------|------------------------|---------------|\n")

            critical_shapes = defaultdict(list)
            for failure in memory_categories["critical"]:
                shape_key = tuple(failure["shape"])
                critical_shapes[shape_key].append(failure)

            for shape, failures in sorted(critical_shapes.items(), key=lambda x: x[1][0]["memory_mb"], reverse=True):
                memory_mb = failures[0]["memory_mb"]
                memory_gb = failures[0]["memory_gb"]
                operators = list(set(f["operator"] for f in failures))
                frequency = len(failures)

                f.write(
                    f"| `{list(shape)}` | **{memory_gb:.2f}GB** | {memory_mb:.1f}MB | {', '.join(operators[:3])}{'...' if len(operators) > 3 else ''} | {frequency} |\n"
                )
        else:
            f.write("✅ No critical shapes (>1GB) found.\n")
        f.write("\n")

        # OOM by Operator Analysis
        f.write("## 📊 OOM FAILURES BY OPERATOR\n\n")
        f.write("| **Operator** | **OOM Count** | **Worst Shape** | **Max Memory** | **Status** |\n")
        f.write("|--------------|---------------|-----------------|----------------|------------|\n")

        for operator_name, failures in sorted_operators:
            max_failure = max(failures, key=lambda x: x["memory_mb"])
            worst_shape = max_failure["shape"]
            max_memory_gb = max_failure["memory_gb"]
            oom_count = len(failures)

            # Determine status based on memory severity
            if max_memory_gb > 1.0:
                status = "🚨 CRITICAL"
            elif max_memory_gb > 0.5:
                status = "⚠️ HIGH"
            elif max_memory_gb > 0.1:
                status = "🟡 MEDIUM"
            else:
                status = "🟢 LOW"

            f.write(f"| **{operator_name}** | {oom_count} | `{worst_shape}` | **{max_memory_gb:.2f}GB** | {status} |\n")
        f.write("\n")

        # Detailed Shape Analysis
        f.write("## 🔍 DETAILED SHAPE ANALYSIS\n\n")
        f.write("### Most Problematic Shapes (by operator count)\n\n")
        f.write("| **Shape** | **Memory** | **Operators Affected** | **Total Failures** |\n")
        f.write("|-----------|------------|------------------------|---------------------|\n")

        for shape, failures in sorted_shapes[:20]:  # Top 20 most problematic shapes
            memory_gb = failures[0]["memory_gb"]
            operators = list(set(f["operator"] for f in failures))
            total_failures = len(failures)

            f.write(f"| `{list(shape)}` | **{memory_gb:.2f}GB** | {len(operators)} ops | {total_failures} failures |\n")
        f.write("\n")

        # Memory Distribution Analysis
        f.write("## 📈 MEMORY DISTRIBUTION ANALYSIS\n\n")
        f.write("### OOM Failures by Memory Category\n\n")
        f.write("| **Category** | **Range** | **Count** | **Percentage** | **Operators** |\n")
        f.write("|--------------|-----------|-----------|----------------|---------------|\n")

        for category, failures in memory_categories.items():
            if failures:
                count = len(failures)
                percentage = count / total_oom_count * 100
                operators = len(set(f["operator"] for f in failures))

                if category == "critical":
                    range_str = "> 1GB"
                elif category == "large":
                    range_str = "500MB - 1GB"
                elif category == "medium":
                    range_str = "100MB - 500MB"
                else:
                    range_str = "< 100MB"

                f.write(f"| **{category.upper()}** | {range_str} | {count:,} | {percentage:.1f}% | {operators} |\n")
        f.write("\n")

        # 8K Resolution Impact Analysis
        f.write("## 🎯 8K RESOLUTION IMPACT ANALYSIS\n\n")

        # Identify 8K related shapes
        eight_k_shapes = []
        for failure in all_oom_failures:
            shape = failure["shape"]
            # Check for 8K indicators (4320, 7680, 2160, 3840, etc.)
            if any(dim in [4320, 7680, 2160, 3840, 1080, 1920, 540, 960] for dim in shape):
                eight_k_shapes.append(failure)

        f.write(
            f"**8K-Related OOM Failures:** {len(eight_k_shapes)}/{total_oom_count} ({len(eight_k_shapes)/total_oom_count*100:.1f}%)\n\n"
        )

        if eight_k_shapes:
            f.write("### Top 8K Resolution OOM Issues\n\n")
            f.write("| **Shape** | **Memory** | **Operator** | **8K Context** |\n")
            f.write("|-----------|------------|--------------|----------------|\n")

            # Sort 8K shapes by memory impact
            sorted_8k = sorted(eight_k_shapes, key=lambda x: x["memory_mb"], reverse=True)[:15]

            for failure in sorted_8k:
                shape = failure["shape"]
                memory_gb = failure["memory_gb"]
                operator = failure["operator"]

                # Determine 8K context
                if 4320 in shape and 7680 in shape:
                    context = "Full 8K input"
                elif 2160 in shape and 3840 in shape:
                    context = "Half 8K"
                elif 1080 in shape and 1920 in shape:
                    context = "Quarter 8K"
                else:
                    context = "8K derivative"

                f.write(f"| `{shape}` | **{memory_gb:.2f}GB** | {operator} | {context} |\n")
        f.write("\n")

        # Recommendations
        f.write("## 💡 OOM RESOLUTION RECOMMENDATIONS\n\n")
        f.write("### 🚨 IMMEDIATE CRITICAL ACTIONS\n\n")

        if memory_categories["critical"]:
            f.write("1. **Critical Memory Issues (>1GB) - URGENT:**\n")
            critical_ops = set(f["operator"] for f in memory_categories["critical"])
            for i, op in enumerate(sorted(critical_ops)[:5]):
                op_failures = [f for f in memory_categories["critical"] if f["operator"] == op]
                max_memory = max(f["memory_gb"] for f in op_failures)
                f.write(f"   {i+1}. **{op}** operator - Max: {max_memory:.2f}GB\n")
            f.write("\n")

        f.write("2. **Technical Solutions:**\n")
        f.write("   - **Tensor Chunking:** Implement automatic splitting for tensors >500MB\n")
        f.write("   - **Memory Streaming:** Process large tensors in streaming fashion\n")
        f.write("   - **Precision Optimization:** Use mixed precision (fp16/bf16) strategically\n")
        f.write("   - **Memory Pooling:** Implement efficient memory reuse patterns\n")
        f.write("   - **Progressive Testing:** Start with smaller shapes, scale up gradually\n\n")

        f.write("3. **Device Configuration:**\n")
        f.write("   - **Memory Limits:** Review and potentially increase device memory allocation\n")
        f.write("   - **Fallback Mechanisms:** Implement CPU fallback for oversized tensors\n")
        f.write("   - **Memory Monitoring:** Add real-time memory usage tracking\n\n")

        f.write("### 📋 PRIORITIZED ACTION PLAN\n\n")
        f.write("#### Phase 1: Critical (Immediate - 1-2 weeks)\n")
        f.write("- Fix operators with >1GB tensor failures\n")
        f.write("- Implement tensor chunking for shapes >500MB\n")
        f.write("- Add memory limit checks before operation execution\n\n")

        f.write("#### Phase 2: High Priority (2-4 weeks)\n")
        f.write("- Optimize operators with 500MB-1GB failures\n")
        f.write("- Implement memory streaming for large operations\n")
        f.write("- Add progressive shape testing framework\n\n")

        f.write("#### Phase 3: Medium Priority (1-2 months)\n")
        f.write("- Address 100-500MB memory issues\n")
        f.write("- Optimize memory usage across all operators\n")
        f.write("- Implement comprehensive memory profiling\n\n")

        f.write("---\n\n")
        f.write("*OOM Analysis Report generated by TT-Metal Memory Analyzer*\n")
        f.write(f"*Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")

    # Save raw OOM data as JSON
    json_filename = f"oom_analysis_data_{timestamp}.json"
    json_data = {
        "summary": {
            "total_oom_count": total_oom_count,
            "operators_with_oom": operators_with_oom,
            "total_operators": total_operators,
            "unique_problem_shapes": len(oom_by_shape),
            "analysis_timestamp": datetime.now().isoformat(),
        },
        "oom_by_operator": oom_by_operator,
        "oom_by_shape": {str(k): v for k, v in oom_by_shape.items()},
        "memory_categories": memory_categories,
        "all_failures": all_oom_failures,
    }

    with open(json_filename, "w") as f:
        json.dump(json_data, f, indent=2)

    print(f"\n✅ OOM Analysis Complete!")
    print(f"📄 Detailed OOM report: {report_filename}")
    print(f"📊 Raw OOM data: {json_filename}")

    # Summary output
    print(f"\n🧠 OOM ANALYSIS SUMMARY:")
    print(f"{'='*50}")
    print(f"Total OOM Failures:       {total_oom_count:,}")
    print(
        f"Operators with OOM:       {operators_with_oom}/{total_operators} ({operators_with_oom/total_operators*100:.1f}%)"
    )
    print(f"Critical Shapes (>1GB):   {len(memory_categories['critical'])}")
    print(f"Large Shapes (500MB-1GB): {len(memory_categories['large'])}")
    print(f"Unique Problem Shapes:    {len(oom_by_shape)}")

    if sorted_operators:
        print(f"\nTop 5 OOM-Problematic Operators:")
        for i, (op, failures) in enumerate(sorted_operators[:5]):
            max_mem = max(f["memory_gb"] for f in failures)
            print(f"  {i+1}. {op:<20} | {len(failures):>3} failures | Max: {max_mem:.2f}GB")


if __name__ == "__main__":
    analyze_all_oom_failures()
