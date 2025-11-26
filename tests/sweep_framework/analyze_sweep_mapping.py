#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Analyze and compare legacy sweep_tests (YAML-based) with modern sweep_framework/sweeps (Python-based).
Generates a mapping document showing coverage equivalence.
"""

import os
import re
import yaml
import ast
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Any

# Paths
LEGACY_TESTS_DIR = Path(__file__).parent.parent / "ttnn/python_api_testing/sweep_tests/test_configs"
MODERN_SWEEPS_DIR = Path(__file__).parent / "sweeps"
OP_MAP_FILE = Path(__file__).parent.parent / "ttnn/python_api_testing/sweep_tests/op_map.py"


def normalize_op_name(name: str) -> str:
    """Normalize op names for comparison."""
    # Remove common prefixes
    name = name.lower()
    name = re.sub(r"^ttnn[-_]", "", name)
    name = re.sub(r"^eltwise[-_]", "", name)
    name = re.sub(r"[-_]test$", "", name)
    name = re.sub(r"[-_]", "_", name)
    return name


def extract_base_op_name(name: str) -> str:
    """Extract the base operation name."""
    name = normalize_op_name(name)
    # Handle backward ops
    if name.startswith("backward_"):
        return name.replace("backward_", "") + "_bw"
    return name


def parse_yaml_file(yaml_path: Path) -> List[Dict]:
    """Parse a YAML test config and extract test information."""
    tests = []
    try:
        with open(yaml_path, "r") as f:
            content = yaml.safe_load(f)

        if not content or "test-list" not in content:
            return tests

        for test_entry in content["test-list"]:
            if isinstance(test_entry, dict):
                for op_name, config in test_entry.items():
                    test_info = {
                        "op_name": op_name,
                        "normalized_name": normalize_op_name(op_name),
                        "base_name": extract_base_op_name(op_name),
                        "file": yaml_path.name,
                        "dtypes": [],
                        "layouts": [],
                        "buffer_types": [],
                        "out_buffer_types": [],
                    }

                    if isinstance(config, dict):
                        args = config.get("args", {})
                        if isinstance(args, dict):
                            test_info["dtypes"] = args.get("data-type", [])
                            test_info["layouts"] = args.get("data-layout", [])
                            test_info["buffer_types"] = args.get("buffer-type", [])
                            test_info["out_buffer_types"] = args.get("out-buffer-type", [])

                    tests.append(test_info)
    except Exception as e:
        print(f"Warning: Failed to parse {yaml_path}: {e}")

    return tests


def parse_modern_sweep_file(py_path: Path) -> Dict:
    """Parse a Python sweep module and extract parameter information."""
    info = {
        "file": py_path.relative_to(MODERN_SWEEPS_DIR),
        "op_name": py_path.stem,
        "normalized_name": normalize_op_name(py_path.stem),
        "base_name": extract_base_op_name(py_path.stem),
        "category": str(py_path.parent.relative_to(MODERN_SWEEPS_DIR)),
        "dtypes": set(),
        "layouts": set(),
        "memory_configs": set(),
        "has_parameters": False,
        "suites": [],
    }

    try:
        with open(py_path, "r") as f:
            content = f.read()

        # Check if it has a parameters dict
        if "parameters" in content and "parameters = {" in content:
            info["has_parameters"] = True

            # Extract dtype information
            dtype_patterns = [
                r"ttnn\.(bfloat16|bfloat8_b|bfloat4_b|float32|uint16|uint32|int32)",
            ]
            for pattern in dtype_patterns:
                matches = re.findall(pattern, content)
                info["dtypes"].update(matches)

            # Extract layout information
            if "TILE_LAYOUT" in content:
                info["layouts"].add("TILE")
            if "ROW_MAJOR_LAYOUT" in content:
                info["layouts"].add("ROW_MAJOR")

            # Extract memory config information
            if "DRAM_MEMORY_CONFIG" in content:
                info["memory_configs"].add("DRAM")
            if "L1_MEMORY_CONFIG" in content:
                info["memory_configs"].add("L1")
            if "BLOCK_SHARDED" in content or "block_sharded" in content.lower():
                info["memory_configs"].add("BLOCK_SHARDED")
            if "HEIGHT_SHARDED" in content or "height_sharded" in content.lower():
                info["memory_configs"].add("HEIGHT_SHARDED")
            if "WIDTH_SHARDED" in content or "width_sharded" in content.lower():
                info["memory_configs"].add("WIDTH_SHARDED")

            # Extract suite names
            suite_pattern = r'"(\w+)":\s*\{'
            suites = re.findall(suite_pattern, content)
            info["suites"] = [s for s in suites if s not in ["tt_op", "pytorch_op"]]

    except Exception as e:
        print(f"Warning: Failed to parse {py_path}: {e}")

    return info


def collect_legacy_tests() -> Dict[str, List[Dict]]:
    """Collect all legacy YAML test configs."""
    tests_by_op = defaultdict(list)

    for yaml_file in LEGACY_TESTS_DIR.rglob("*.yaml"):
        tests = parse_yaml_file(yaml_file)
        for test in tests:
            tests_by_op[test["base_name"]].append(test)

    return tests_by_op


def collect_modern_sweeps() -> Dict[str, List[Dict]]:
    """Collect all modern Python sweep modules."""
    sweeps_by_op = defaultdict(list)

    for py_file in MODERN_SWEEPS_DIR.rglob("*.py"):
        if py_file.name.startswith("__"):
            continue

        info = parse_modern_sweep_file(py_file)
        if info["has_parameters"]:
            sweeps_by_op[info["base_name"]].append(info)

    return sweeps_by_op


def compare_parameters(legacy: List[Dict], modern: List[Dict]) -> Dict:
    """Compare parameter coverage between legacy and modern tests."""
    legacy_dtypes = set()
    legacy_layouts = set()
    legacy_memory = set()

    for test in legacy:
        for dtype in test.get("dtypes", []):
            legacy_dtypes.add(dtype.upper())
        for layout in test.get("layouts", []):
            legacy_layouts.add(layout.upper())
        for buf in test.get("buffer_types", []) + test.get("out_buffer_types", []):
            legacy_memory.add(buf.upper())

    modern_dtypes = set()
    modern_layouts = set()
    modern_memory = set()

    for sweep in modern:
        modern_dtypes.update(d.upper() for d in sweep.get("dtypes", set()))
        modern_layouts.update(l.upper() for l in sweep.get("layouts", set()))
        modern_memory.update(m.upper() for m in sweep.get("memory_configs", set()))

    # Normalize dtype names
    dtype_mapping = {
        "BFLOAT16": "BFLOAT16",
        "BFLOAT8_B": "BFLOAT8_B",
        "BFLOAT4_B": "BFLOAT4_B",
        "FLOAT32": "FLOAT32",
        "UINT32": "UINT32",
        "UINT16": "UINT16",
        "INT32": "INT32",
    }

    legacy_dtypes_norm = {dtype_mapping.get(d, d) for d in legacy_dtypes}
    modern_dtypes_norm = {dtype_mapping.get(d, d) for d in modern_dtypes}

    return {
        "dtype_match": len(legacy_dtypes_norm & modern_dtypes_norm) / max(len(legacy_dtypes_norm), 1)
        if legacy_dtypes_norm
        else 1.0,
        "layout_match": len(legacy_layouts & modern_layouts) / max(len(legacy_layouts), 1) if legacy_layouts else 1.0,
        "memory_match": len(legacy_memory & modern_memory) / max(len(legacy_memory), 1) if legacy_memory else 1.0,
        "legacy_dtypes": legacy_dtypes_norm,
        "modern_dtypes": modern_dtypes_norm,
        "legacy_layouts": legacy_layouts,
        "modern_layouts": modern_layouts,
        "legacy_memory": legacy_memory,
        "modern_memory": modern_memory,
    }


def calculate_match_status(comparison: Dict) -> str:
    """Calculate overall match status."""
    avg_match = (comparison["dtype_match"] + comparison["layout_match"] + comparison["memory_match"]) / 3

    if avg_match >= 0.8:
        return "Full"
    elif avg_match >= 0.5:
        return "Partial"
    else:
        return "Minimal"


def generate_mapping_document(legacy_tests: Dict, modern_sweeps: Dict) -> str:
    """Generate the mapping document."""
    lines = []

    # Header
    lines.append("# Sweep Tests Mapping: Legacy (YAML) to Modern (Python)")
    lines.append("")
    lines.append("This document maps legacy sweep tests in `tests/ttnn/python_api_testing/sweep_tests/`")
    lines.append("to their equivalents in the modern `tests/sweep_framework/sweeps/` framework.")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")

    # Calculate statistics
    all_legacy_ops = set(legacy_tests.keys())
    all_modern_ops = set(modern_sweeps.keys())
    matched_ops = all_legacy_ops & all_modern_ops
    legacy_only = all_legacy_ops - all_modern_ops
    modern_only = all_modern_ops - all_legacy_ops

    lines.append(f"- **Total Legacy Tests (unique ops)**: {len(all_legacy_ops)}")
    lines.append(f"- **Total Modern Sweeps (unique ops)**: {len(all_modern_ops)}")
    lines.append(
        f"- **Ops with Modern Equivalent**: {len(matched_ops)} ({100*len(matched_ops)/max(len(all_legacy_ops), 1):.1f}%)"
    )
    lines.append(f"- **Legacy-Only Ops (gaps)**: {len(legacy_only)}")
    lines.append(f"- **Modern-Only Ops (new coverage)**: {len(modern_only)}")
    lines.append("")

    # Detailed Mapping Table
    lines.append("## Detailed Op Mapping")
    lines.append("")
    lines.append("### Legend")
    lines.append("- **Match Status**: Full (≥80%), Partial (50-79%), Minimal (<50%)")
    lines.append("- **Dtypes**: Data types covered (BFLOAT16, BFLOAT8_B, etc.)")
    lines.append("- **Layouts**: TILE, ROW_MAJOR")
    lines.append("- **Memory**: DRAM, L1, sharded variants")
    lines.append("")

    lines.append("### Matched Operations")
    lines.append("")
    lines.append("| Legacy Op | Modern Sweep | Match | Dtype Coverage | Layout Coverage | Memory Coverage |")
    lines.append("|-----------|--------------|-------|----------------|-----------------|-----------------|")

    matched_details = []
    for op in sorted(matched_ops):
        comparison = compare_parameters(legacy_tests[op], modern_sweeps[op])
        status = calculate_match_status(comparison)

        legacy_files = ", ".join(set(t["file"] for t in legacy_tests[op]))[:40]
        modern_files = ", ".join(str(s["file"]) for s in modern_sweeps[op])[:40]

        dtype_cov = f"{comparison['dtype_match']*100:.0f}%"
        layout_cov = f"{comparison['layout_match']*100:.0f}%"
        memory_cov = f"{comparison['memory_match']*100:.0f}%"

        lines.append(f"| {op} | {modern_files} | {status} | {dtype_cov} | {layout_cov} | {memory_cov} |")

        matched_details.append(
            {
                "op": op,
                "status": status,
                "comparison": comparison,
                "legacy_files": [t["file"] for t in legacy_tests[op]],
                "modern_files": [str(s["file"]) for s in modern_sweeps[op]],
            }
        )

    lines.append("")

    # Legacy-only ops (gaps)
    lines.append("### Gaps: Legacy Tests Without Modern Equivalent")
    lines.append("")
    lines.append("These operations have sweep tests in the legacy system but no equivalent in the modern framework.")
    lines.append("")
    lines.append("| Op Name | Legacy Files | Dtypes | Layouts | Memory |")
    lines.append("|---------|--------------|--------|---------|--------|")

    for op in sorted(legacy_only):
        tests = legacy_tests[op]
        files = ", ".join(set(t["file"] for t in tests))[:50]
        dtypes = ", ".join(set(d for t in tests for d in t.get("dtypes", [])))[:30]
        layouts = ", ".join(set(l for t in tests for l in t.get("layouts", [])))[:20]
        memory = ", ".join(set(b for t in tests for b in t.get("buffer_types", [])))[:20]
        lines.append(f"| {op} | {files} | {dtypes} | {layouts} | {memory} |")

    lines.append("")

    # Modern-only ops (new coverage)
    lines.append("### New Coverage: Modern Sweeps Without Legacy Equivalent")
    lines.append("")
    lines.append("These operations are tested in the modern framework but have no legacy equivalent.")
    lines.append("")
    lines.append("| Op Name | Category | Files | Suites |")
    lines.append("|---------|----------|-------|--------|")

    for op in sorted(modern_only):
        sweeps = modern_sweeps[op]
        category = sweeps[0].get("category", "N/A")
        files = ", ".join(str(s["file"]) for s in sweeps)[:50]
        suites = ", ".join(set(s for sw in sweeps for s in sw.get("suites", [])))[:30]
        lines.append(f"| {op} | {category} | {files} | {suites} |")

    lines.append("")

    # Detailed parameter comparison for key ops
    lines.append("## Detailed Parameter Comparison (Selected Ops)")
    lines.append("")
    lines.append("This section shows detailed parameter coverage for key operations.")
    lines.append("")

    # Pick some representative ops
    key_ops = ["add", "relu", "matmul", "softmax", "concat", "sigmoid", "gelu", "exp"]
    for op in key_ops:
        if op in matched_ops:
            comparison = compare_parameters(legacy_tests[op], modern_sweeps[op])
            lines.append(f"### {op}")
            lines.append("")
            lines.append("| Parameter | Legacy | Modern | Match |")
            lines.append("|-----------|--------|--------|-------|")

            l_dtypes = ", ".join(sorted(comparison["legacy_dtypes"])) or "None"
            m_dtypes = ", ".join(sorted(comparison["modern_dtypes"])) or "None"
            lines.append(f"| Dtypes | {l_dtypes} | {m_dtypes} | {comparison['dtype_match']*100:.0f}% |")

            l_layouts = ", ".join(sorted(comparison["legacy_layouts"])) or "None"
            m_layouts = ", ".join(sorted(comparison["modern_layouts"])) or "None"
            lines.append(f"| Layouts | {l_layouts} | {m_layouts} | {comparison['layout_match']*100:.0f}% |")

            l_memory = ", ".join(sorted(comparison["legacy_memory"])) or "None"
            m_memory = ", ".join(sorted(comparison["modern_memory"])) or "None"
            lines.append(f"| Memory | {l_memory} | {m_memory} | {comparison['memory_match']*100:.0f}% |")

            lines.append("")

    # Summary table by category
    lines.append("## Coverage by Category")
    lines.append("")

    categories = defaultdict(lambda: {"total": 0, "matched": 0, "full": 0, "partial": 0})
    for op in all_legacy_ops:
        # Determine category from op name
        if "_bw" in op or "backward" in op:
            cat = "backward"
        elif any(x in op for x in ["complex", "polar", "angle", "conj", "imag", "real"]):
            cat = "complex"
        elif any(x in op for x in ["matmul", "linear"]):
            cat = "matmul"
        elif any(x in op for x in ["softmax", "layernorm", "rmsnorm", "groupnorm"]):
            cat = "normalization"
        elif any(x in op for x in ["concat", "permute", "reshape", "transpose", "pad", "repeat"]):
            cat = "data_movement"
        elif any(x in op for x in ["sum", "mean", "max", "min", "var", "std", "argmax", "topk"]):
            cat = "reduction"
        elif any(x in op for x in ["pool", "upsample"]):
            cat = "pooling"
        elif any(x in op for x in ["loss", "l1", "mse"]):
            cat = "loss"
        else:
            cat = "eltwise"

        categories[cat]["total"] += 1
        if op in matched_ops:
            categories[cat]["matched"] += 1
            comparison = compare_parameters(legacy_tests[op], modern_sweeps[op])
            status = calculate_match_status(comparison)
            if status == "Full":
                categories[cat]["full"] += 1
            elif status == "Partial":
                categories[cat]["partial"] += 1

    lines.append("| Category | Total | Matched | Full Match | Partial Match | Coverage % |")
    lines.append("|----------|-------|---------|------------|---------------|------------|")

    for cat in sorted(categories.keys()):
        stats = categories[cat]
        cov = 100 * stats["matched"] / max(stats["total"], 1)
        lines.append(
            f"| {cat} | {stats['total']} | {stats['matched']} | {stats['full']} | {stats['partial']} | {cov:.1f}% |"
        )

    lines.append("")

    # Recommendations
    lines.append("## Recommendations")
    lines.append("")
    lines.append("### High Priority Gaps (Legacy tests to port to modern framework)")
    lines.append("")

    # Find important gaps
    important_gaps = [op for op in legacy_only if not any(x in op for x in ["preprocess", "model"])]
    for op in sorted(important_gaps)[:20]:
        tests = legacy_tests[op]
        files = ", ".join(set(t["file"] for t in tests))
        lines.append(f"- **{op}**: {files}")

    if len(important_gaps) > 20:
        lines.append(f"- ... and {len(important_gaps) - 20} more")

    lines.append("")
    lines.append("### Parameter Coverage Improvements")
    lines.append("")
    lines.append("Operations with partial matches that could benefit from expanded parameter coverage:")
    lines.append("")

    partial_matches = [d for d in matched_details if d["status"] == "Partial"]
    for detail in partial_matches[:10]:
        comp = detail["comparison"]
        issues = []
        if comp["dtype_match"] < 1.0:
            missing = comp["legacy_dtypes"] - comp["modern_dtypes"]
            if missing:
                issues.append(f"missing dtypes: {', '.join(missing)}")
        if comp["layout_match"] < 1.0:
            missing = comp["legacy_layouts"] - comp["modern_layouts"]
            if missing:
                issues.append(f"missing layouts: {', '.join(missing)}")
        if comp["memory_match"] < 1.0:
            missing = comp["legacy_memory"] - comp["modern_memory"]
            if missing:
                issues.append(f"missing memory configs: {', '.join(missing)}")

        if issues:
            lines.append(f"- **{detail['op']}**: {'; '.join(issues)}")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*Generated by analyze_sweep_mapping.py*")

    return "\n".join(lines)


def main():
    print("Collecting legacy tests...")
    legacy_tests = collect_legacy_tests()
    print(f"Found {len(legacy_tests)} unique ops in legacy tests")

    print("Collecting modern sweeps...")
    modern_sweeps = collect_modern_sweeps()
    print(f"Found {len(modern_sweeps)} unique ops in modern sweeps")

    print("Generating mapping document...")
    document = generate_mapping_document(legacy_tests, modern_sweeps)

    # Write to tt-logbook
    output_path = Path(__file__).parent.parent.parent.parent / "tt-logbook" / "sweep_tests_mapping.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write(document)

    print(f"Mapping document written to: {output_path}")

    # Also print to stdout
    print("\n" + "=" * 80)
    print(document)


if __name__ == "__main__":
    main()
