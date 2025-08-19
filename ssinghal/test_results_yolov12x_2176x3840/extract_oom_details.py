#!/usr/bin/env python3
"""
OOM (Out of Memory) Error Extractor for TT-Metal Operators
Extracts all OOM errors with specific tensor shapes and calculates memory requirements
"""

import os
import re
import glob
from collections import defaultdict
import json


def calculate_tensor_memory(shape, dtype="bfloat16"):
    """Calculate memory usage for a tensor given its shape and data type"""
    if not shape:
        return 0, "0B"

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
            readable = f"{total_bytes}B"
        elif total_bytes < 1024**2:
            readable = f"{total_bytes/1024:.1f}KB"
        elif total_bytes < 1024**3:
            readable = f"{total_bytes/(1024**2):.1f}MB"
        else:
            readable = f"{total_bytes/(1024**3):.2f}GB"

        return total_bytes, readable
    except:
        return 0, "Unknown"


def extract_oom_errors_from_file(file_path):
    """Extract OOM errors from a single test result file"""

    operator_name = os.path.basename(file_path).replace("_results.txt", "").replace("test_", "")
    oom_errors = []

    try:
        with open(file_path, "r") as f:
            content = f.read()

        # Pattern to match OOM errors with shapes
        # Example: SKIPPED (OOM: [1, 3, 2176, 3840] - TT_THROW @ ...)
        oom_pattern = r"SKIPPED.*?OOM:\s*\[([^\]]+)\].*?TT_THROW"
        matches = re.findall(oom_pattern, content)

        for match in matches:
            try:
                # Parse the shape dimensions
                dims_str = match.strip()
                if dims_str:
                    dims = [int(x.strip()) for x in dims_str.split(",")]

                    # Calculate memory requirements
                    bytes_required, readable_memory = calculate_tensor_memory(dims)

                    oom_errors.append(
                        {
                            "operator": operator_name,
                            "shape": dims,
                            "memory_bytes": bytes_required,
                            "memory_readable": readable_memory,
                            "file_source": file_path,
                        }
                    )
            except Exception as e:
                print(f"Error parsing shape {match} in {operator_name}: {e}")
                continue

    except Exception as e:
        print(f"Error reading file {file_path}: {e}")

    return oom_errors


def analyze_all_oom_errors():
    """Analyze all OOM errors across all test result files"""

    results_dir = "ssinghal/test_results"

    print("🔍 Scanning for OOM errors in test results...")

    # Find all test result files
    result_files = glob.glob(f"{results_dir}/test_*_results.txt")

    if not result_files:
        print("❌ No test result files found!")
        return []

    all_oom_errors = []
    operators_with_oom = set()

    for file_path in sorted(result_files):
        operator_name = os.path.basename(file_path).replace("_results.txt", "").replace("test_", "")

        oom_errors = extract_oom_errors_from_file(file_path)

        if oom_errors:
            operators_with_oom.add(operator_name)
            all_oom_errors.extend(oom_errors)
            print(f"  📊 {operator_name}: {len(oom_errors)} OOM errors found")
        else:
            print(f"  ✅ {operator_name}: No OOM errors")

    print(f"\n📋 Summary:")
    print(f"  Total operators checked: {len(result_files)}")
    print(f"  Operators with OOM errors: {len(operators_with_oom)}")
    print(f"  Total OOM occurrences: {len(all_oom_errors)}")

    return all_oom_errors


def generate_oom_report(oom_errors):
    """Generate comprehensive OOM report"""

    if not oom_errors:
        print("No OOM errors found to report.")
        return

    # Group errors by operator
    errors_by_operator = defaultdict(list)
    for error in oom_errors:
        errors_by_operator[error["operator"]].append(error)

    # Group errors by unique shape (to avoid duplicates)
    unique_shapes_by_operator = defaultdict(set)
    for error in oom_errors:
        shape_tuple = tuple(error["shape"])
        unique_shapes_by_operator[error["operator"]].add(shape_tuple)

    # Generate the report
    timestamp = "2025-08-18 19:40:00"

    report = f"""# TT-Metal Out of Memory (OOM) Analysis Report

**Generated:** {timestamp}
**Analysis Scope:** All operator tests with explicit OOM failures

## 🚨 Executive Summary

| Metric | Value |
|--------|-------|
| **Total Operators with OOM Issues** | {len(errors_by_operator)} |
| **Total OOM Occurrences** | {len(oom_errors)} |
| **Unique Problematic Shapes** | {sum(len(shapes) for shapes in unique_shapes_by_operator.values())} |

---

## 📊 Detailed OOM Analysis by Operator

"""

    # Sort operators by number of OOM errors (most problematic first)
    sorted_operators = sorted(errors_by_operator.items(), key=lambda x: len(x[1]), reverse=True)

    for operator, operator_errors in sorted_operators:
        # Get unique shapes for this operator
        unique_shapes = {}
        for error in operator_errors:
            shape_key = tuple(error["shape"])
            if shape_key not in unique_shapes:
                unique_shapes[shape_key] = error

        # Sort shapes by memory size (largest first)
        sorted_shapes = sorted(unique_shapes.values(), key=lambda x: x["memory_bytes"], reverse=True)

        report += f"""
### 🔴 **{operator.upper()}** Operator
- **Total OOM Occurrences:** {len(operator_errors)}
- **Unique Problematic Shapes:** {len(unique_shapes)}

| Shape | Expected Memory | Memory (Bytes) | Severity |
|-------|----------------|----------------|----------|"""

        for error in sorted_shapes:
            # Determine severity based on memory size
            if error["memory_bytes"] > 500 * 1024 * 1024:  # > 500MB
                severity = "🚨 CRITICAL"
            elif error["memory_bytes"] > 100 * 1024 * 1024:  # > 100MB
                severity = "⚠️ HIGH"
            elif error["memory_bytes"] > 50 * 1024 * 1024:  # > 50MB
                severity = "🟡 MEDIUM"
            else:
                severity = "🟢 LOW"

            shape_str = str(error["shape"]).replace(" ", "")
            report += f"""
| `{shape_str}` | **{error['memory_readable']}** | {error['memory_bytes']:,} | {severity} |"""

        report += "\n"

    # Add memory threshold analysis
    report += f"""

---

## 📏 Memory Size Distribution

### Critical OOM Shapes (>500MB)
"""

    critical_shapes = [e for e in oom_errors if e["memory_bytes"] > 500 * 1024 * 1024]
    high_shapes = [e for e in oom_errors if 100 * 1024 * 1024 < e["memory_bytes"] <= 500 * 1024 * 1024]
    medium_shapes = [e for e in oom_errors if 50 * 1024 * 1024 < e["memory_bytes"] <= 100 * 1024 * 1024]
    low_shapes = [e for e in oom_errors if e["memory_bytes"] <= 50 * 1024 * 1024]

    if critical_shapes:
        report += f"""
| Operator | Shape | Memory Required |
|----------|-------|-----------------|"""

        for error in sorted(critical_shapes, key=lambda x: x["memory_bytes"], reverse=True):
            shape_str = str(error["shape"]).replace(" ", "")
            report += f"""
| **{error['operator']}** | `{shape_str}` | **{error['memory_readable']}** |"""
    else:
        report += "\n*No critical memory shapes found.*"

    report += f"""

### High Memory OOM Shapes (100MB - 500MB)
Count: {len(high_shapes)}

### Medium Memory OOM Shapes (50MB - 100MB)
Count: {len(medium_shapes)}

### Low Memory OOM Shapes (<50MB)
Count: {len(low_shapes)}

---

## 🔧 Recommendations

### Immediate Actions Required:

1. **Memory Optimization Priority:**"""

    # Get top 5 most memory-intensive shapes
    top_memory_shapes = sorted(oom_errors, key=lambda x: x["memory_bytes"], reverse=True)[:5]

    for i, error in enumerate(top_memory_shapes, 1):
        shape_str = str(error["shape"]).replace(" ", "")
        report += f"""
   {i}. **{error['operator']}** operator with shape `{shape_str}` ({error['memory_readable']})"""

    report += f"""

2. **Implementation Strategies:**
   - **Tensor Chunking:** Split large tensors into smaller manageable pieces
   - **Memory Pooling:** Implement efficient memory reuse patterns
   - **Streaming Processing:** Process data in streams rather than loading entire tensors
   - **Precision Reduction:** Consider using smaller data types where appropriate

3. **Hardware Considerations:**
   - Current device appears to have memory limits around 50-100MB per tensor
   - Consider increasing device memory allocation
   - Implement fallback to CPU processing for oversized tensors

---

## 📁 Raw Data

**Detailed Analysis Data:** `ssinghal/test_results/oom_detailed_data.json`
**Individual Test Results:** `ssinghal/test_results/test_*_results.txt`

---

*Report generated by TT-Metal OOM Analyzer*
"""

    return report


def main():
    """Main analysis function"""

    print("🚨 TT-Metal OOM (Out of Memory) Error Analysis")
    print("=" * 60)

    # Extract all OOM errors
    all_oom_errors = analyze_all_oom_errors()

    if not all_oom_errors:
        print("✅ No OOM errors found in any test results!")
        return

    # Generate comprehensive report
    print("\n📝 Generating OOM analysis report...")
    report = generate_oom_report(all_oom_errors)

    # Save report
    report_file = "ssinghal/test_results/oom_analysis_report.md"
    with open(report_file, "w") as f:
        f.write(report)

    # Save raw data
    json_file = "ssinghal/test_results/oom_detailed_data.json"
    with open(json_file, "w") as f:
        json.dump(all_oom_errors, f, indent=2)

    print(f"✅ OOM analysis complete!")
    print(f"📄 Report saved to: {report_file}")
    print(f"📊 Raw data saved to: {json_file}")

    # Print quick summary
    operators_with_oom = set(error["operator"] for error in all_oom_errors)
    total_memory = sum(error["memory_bytes"] for error in all_oom_errors)
    avg_memory = total_memory / len(all_oom_errors) if all_oom_errors else 0

    print(f"\n📋 Quick Summary:")
    print(f"  Operators with OOM: {len(operators_with_oom)}")
    print(f"  Total OOM occurrences: {len(all_oom_errors)}")
    print(f"  Average memory per OOM: {avg_memory/(1024**2):.1f}MB")
    print(f"  Largest tensor: {max(all_oom_errors, key=lambda x: x['memory_bytes'])['memory_readable']}")


if __name__ == "__main__":
    main()
