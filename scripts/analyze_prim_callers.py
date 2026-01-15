#!/usr/bin/env python3
"""
Analyze prim callers in the TTNN codebase.

This script uses text-based parsing to:
1. Find all functions that call device_operation::launch (prim launchers)
2. Find all functions that call prim launchers
3. Filter out proxy functions that only forward calls

Usage:
    python scripts/analyze_prim_callers.py [--output FORMAT]
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class FunctionInfo:
    """Information about a function."""

    name: str
    qualified_name: str
    file: str
    line: int
    is_prim_launcher: bool = False
    calls_prim: list = field(default_factory=list)
    all_calls: list = field(default_factory=list)
    has_loops: bool = False
    has_complex_conditionals: bool = False
    is_proxy: bool = False
    proxy_reason: str = ""
    body: str = ""


class PrimCallerAnalyzer:
    """Analyzer for finding prim launchers and their callers."""

    def __init__(self, project_root: str):
        self.project_root = project_root

        # Results
        self.prim_launchers: dict[str, FunctionInfo] = {}
        self.prim_callers: dict[str, FunctionInfo] = {}

    def _extract_functions_from_file(self, file_path: str) -> list[FunctionInfo]:
        """Extract function definitions from a C++ file using regex."""
        functions = []

        try:
            with open(file_path, "r") as f:
                content = f.read()
        except Exception as e:
            print(f"Warning: Failed to read {file_path}: {e}")
            return functions

        # Track namespace context
        current_namespace = []

        # Find namespace declarations
        lines = content.split("\n")

        # Pattern to match namespace declarations
        namespace_pattern = re.compile(r"namespace\s+([\w:]+)\s*\{")

        # Find prim namespace functions - look for the namespace block
        prim_namespace_match = re.search(r"namespace\s+ttnn::prim\s*\{", content)
        if not prim_namespace_match:
            prim_namespace_match = re.search(r"namespace\s+prim\s*\{", content)

        if prim_namespace_match:
            # Find the corresponding closing brace
            start_pos = prim_namespace_match.end()
            brace_count = 1
            end_pos = start_pos

            for i, char in enumerate(content[start_pos:], start_pos):
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        end_pos = i
                        break

            prim_block = content[start_pos:end_pos]
            prim_start_line = content[: prim_namespace_match.start()].count("\n") + 1

            # Find function definitions in the prim namespace block
            # Match function patterns like: ReturnType function_name(params) {
            func_pattern = re.compile(
                r"^([A-Za-z_][\w:<>,\s\*&]*?)\s+"  # return type
                r"([a-z_][a-z0-9_]*)"  # function name (lowercase)
                r"\s*\([^)]*\)\s*\{",  # parameters and opening brace
                re.MULTILINE,
            )

            for match in func_pattern.finditer(prim_block):
                func_name = match.group(2)
                func_start = match.start()

                # Find the function body
                body_start = match.end() - 1  # Position of opening brace
                brace_count = 1
                body_end = body_start + 1

                for i, char in enumerate(prim_block[body_start + 1 :], body_start + 1):
                    if char == "{":
                        brace_count += 1
                    elif char == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            body_end = i + 1
                            break

                func_body = prim_block[body_start:body_end]
                func_line = prim_start_line + prim_block[:func_start].count("\n")

                # Check if it calls device_operation::launch
                is_launcher = "device_operation::launch" in func_body

                qualified_name = f"ttnn::prim::{func_name}"

                info = FunctionInfo(
                    name=func_name,
                    qualified_name=qualified_name,
                    file=file_path,
                    line=func_line,
                    is_prim_launcher=is_launcher,
                    body=func_body,
                )

                functions.append(info)

        return functions

    def _extract_callers_from_file(self, file_path: str, prim_names: set[str]) -> list[FunctionInfo]:
        """Extract functions that call prim functions."""
        functions = []

        try:
            with open(file_path, "r") as f:
                content = f.read()
        except Exception as e:
            return functions

        # Quick check - does this file contain prim:: calls?
        if "prim::" not in content and "ttnn::prim::" not in content:
            return functions

        # Find all function definitions
        # This regex matches common function definition patterns
        func_pattern = re.compile(
            r"(?:^|\n)\s*"  # Start of line
            r"(?:static\s+)?"  # Optional static
            r"(?:inline\s+)?"  # Optional inline
            r"(?:virtual\s+)?"  # Optional virtual
            r"([A-Za-z_][\w:<>,\s\*&]*?)\s+"  # Return type
            r"([\w:]+::)?"  # Optional class/namespace qualifier
            r"([A-Za-z_][\w]*)"  # Function name
            r"\s*\([^;]*?\)\s*"  # Parameters (not ending with ;)
            r"(?:const\s*)?"  # Optional const
            r"(?:override\s*)?"  # Optional override
            r"\{",  # Opening brace
            re.MULTILINE | re.DOTALL,
        )

        lines = content.split("\n")

        for match in func_pattern.finditer(content):
            return_type = match.group(1).strip()
            qualifier = match.group(2) or ""
            func_name = match.group(3)

            # Skip if this is in prim namespace
            if "prim" in qualifier:
                continue

            func_start = match.start()
            body_start = match.end() - 1

            # Find the function body
            brace_count = 1
            body_end = body_start + 1

            for i, char in enumerate(content[body_start + 1 :], body_start + 1):
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        body_end = i + 1
                        break

            func_body = content[body_start:body_end]
            func_line = content[:func_start].count("\n") + 1

            # Check for prim:: calls
            prim_calls = []
            for prim_name in prim_names:
                short_name = prim_name.split("::")[-1]
                if f"prim::{short_name}" in func_body:
                    prim_calls.append(prim_name)

            if not prim_calls:
                # Check for generic prim:: pattern
                prim_call_pattern = re.compile(r"(?:ttnn::)?prim::(\w+)")
                for m in prim_call_pattern.finditer(func_body):
                    call_name = f"ttnn::prim::{m.group(1)}"
                    if call_name not in prim_calls:
                        prim_calls.append(call_name)

            if prim_calls:
                qualified_name = f"{qualifier}{func_name}" if qualifier else func_name

                # Find all function calls in body
                all_calls = re.findall(r"([\w:]+)\s*\(", func_body)

                # Check for loops
                has_loops = bool(re.search(r"\b(for|while|do)\s*\(", func_body))

                # Check for complex conditionals (if with else)
                has_complex = bool(re.search(r"\bif\s*\([^)]*\)\s*\{[^}]*\}\s*else\s*\{", func_body))

                info = FunctionInfo(
                    name=func_name,
                    qualified_name=qualified_name,
                    file=file_path,
                    line=func_line,
                    calls_prim=prim_calls,
                    all_calls=all_calls,
                    has_loops=has_loops,
                    has_complex_conditionals=has_complex,
                    body=func_body,
                )

                functions.append(info)

        return functions

    def _classify_as_proxy(self, func_info: FunctionInfo) -> tuple[bool, str]:
        """
        Determine if a function is a proxy (just forwards to prim).
        Returns (is_proxy, reason).
        """
        # Must call exactly one prim function
        if len(func_info.calls_prim) != 1:
            if len(func_info.calls_prim) == 0:
                return False, "no prim calls"
            return False, f"multiple prim calls ({len(func_info.calls_prim)})"

        # Check for loops
        if func_info.has_loops:
            return False, "contains loops"

        body = func_info.body

        # Check for other ttnn operation calls
        other_ttnn_calls = []
        ttnn_call_pattern = re.compile(r"ttnn::(?!prim::)(\w+(?:::\w+)*)")
        for m in ttnn_call_pattern.finditer(body):
            call = m.group(1)
            # Filter out trivial accessors and configs
            trivial = [
                "device",
                "dtype",
                "layout",
                "shape",
                "logical_shape",
                "padded_shape",
                "storage_type",
                "arch",
                "memory_config",
                "MeshDevice",
                "Tensor",
                "operations::",
                "SmallVector",
            ]
            if not any(t in call for t in trivial):
                other_ttnn_calls.append(f"ttnn::{call}")

        if other_ttnn_calls:
            # Filter out the prim call itself
            other_ttnn_calls = [c for c in other_ttnn_calls if "prim" not in c]
            if other_ttnn_calls:
                return False, f"calls other ttnn ops: {other_ttnn_calls[:3]}"

        # Check for complex conditionals with meaningful logic
        if func_info.has_complex_conditionals:
            return False, "has complex conditional logic"

        # Check for early return patterns that indicate real logic
        early_return_pattern = re.compile(r"if\s*\([^)]*\)\s*\{?\s*return\s+(?!ttnn::prim)")
        if early_return_pattern.search(body):
            # Check if the early return creates a new tensor
            if re.search(r"if\s*\([^)]*\)\s*\{?\s*return\s+ttnn::", body):
                return False, "early exit creates tensor"

        return True, "simple forwarding"

    def find_prim_launchers(self):
        """Find all prim launcher functions."""
        print("Scanning for device_operation.cpp files...")

        # Find all device_operation.cpp files
        ttnn_ops_dir = Path(self.project_root) / "ttnn" / "cpp" / "ttnn" / "operations"
        device_op_files = list(ttnn_ops_dir.rglob("*_device_operation.cpp"))

        print(f"Found {len(device_op_files)} device operation files")

        for f in device_op_files:
            file_path = str(f)
            functions = self._extract_functions_from_file(file_path)

            for func in functions:
                if func.is_prim_launcher:
                    print(f"  Found: {func.qualified_name} in {f.name}:{func.line}")
                    self.prim_launchers[func.qualified_name] = func

        print(f"\nFound {len(self.prim_launchers)} prim launchers")

    def find_prim_callers(self):
        """Find all functions that call prim launchers."""
        print("\nScanning for prim callers...")

        prim_names = set(self.prim_launchers.keys())

        # Scan all cpp files
        ttnn_ops_dir = Path(self.project_root) / "ttnn" / "cpp" / "ttnn" / "operations"
        cpp_files = list(ttnn_ops_dir.rglob("*.cpp"))

        # Filter out device operation files (we already know the prim launchers)
        cpp_files = [f for f in cpp_files if "_device_operation" not in f.name]

        print(f"Scanning {len(cpp_files)} source files for prim callers...")

        for f in cpp_files:
            file_path = str(f)
            functions = self._extract_callers_from_file(file_path, prim_names)

            for func in functions:
                if func.calls_prim:
                    print(f"  Found caller: {func.qualified_name} in {f.name}:{func.line}")
                    self.prim_callers[func.qualified_name] = func

    def classify_callers(self):
        """Classify callers as proxy or non-proxy."""
        print("\nClassifying callers...")

        for name, info in self.prim_callers.items():
            is_proxy, reason = self._classify_as_proxy(info)
            info.is_proxy = is_proxy
            info.proxy_reason = reason

    def print_results(self, show_all: bool = True):
        """Print analysis results."""
        print("\n" + "=" * 70)
        print("=== PRIM LAUNCHERS (Level 0) ===")
        print("=" * 70)

        for name, info in sorted(self.prim_launchers.items()):
            rel_file = os.path.relpath(info.file, self.project_root)
            print(f"{name}")
            print(f"    Location: {rel_file}:{info.line}")

        print(f"\nTotal: {len(self.prim_launchers)} prim launchers")

        print("\n" + "=" * 70)
        print("=== PRIM CALLERS (Level 1) ===")
        print("=" * 70)

        proxies = []
        non_proxies = []

        for name, info in sorted(self.prim_callers.items()):
            if info.is_proxy:
                proxies.append((name, info))
            else:
                non_proxies.append((name, info))

        if show_all:
            print("\n--- PROXY FUNCTIONS (just forward to prim) ---")
            for name, info in proxies:
                rel_file = os.path.relpath(info.file, self.project_root)
                prim_call = info.calls_prim[0] if info.calls_prim else "?"
                print(f"{name}")
                print(f"    -> {prim_call}")
                print(f"    Location: {rel_file}:{info.line}")
                print(f"    [PROXY: {info.proxy_reason}]")

        print("\n--- NON-PROXY FUNCTIONS (do meaningful work) ---")
        for name, info in non_proxies:
            rel_file = os.path.relpath(info.file, self.project_root)
            prim_calls_str = ", ".join(info.calls_prim[:3])
            if len(info.calls_prim) > 3:
                prim_calls_str += f" (+{len(info.calls_prim) - 3} more)"
            print(f"{name}")
            print(f"    -> {prim_calls_str}")
            print(f"    Location: {rel_file}:{info.line}")
            print(f"    [NOT_PROXY: {info.proxy_reason}]")

        print("\n" + "=" * 70)
        print("=== SUMMARY ===")
        print("=" * 70)
        print(f"Prim launchers:     {len(self.prim_launchers)}")
        print(f"Total callers:      {len(self.prim_callers)}")
        print(f"  - Proxies:        {len(proxies)}")
        print(f"  - Non-proxies:    {len(non_proxies)}")

    def export_json(self, output_path: str):
        """Export results to JSON."""
        results = {
            "prim_launchers": [
                {
                    "name": name,
                    "file": os.path.relpath(info.file, self.project_root),
                    "line": info.line,
                }
                for name, info in sorted(self.prim_launchers.items())
            ],
            "callers": [
                {
                    "name": name,
                    "file": os.path.relpath(info.file, self.project_root),
                    "line": info.line,
                    "calls_prim": info.calls_prim,
                    "is_proxy": info.is_proxy,
                    "reason": info.proxy_reason,
                }
                for name, info in sorted(self.prim_callers.items())
            ],
        }

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults exported to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze prim callers in TTNN codebase")
    parser.add_argument("--output", choices=["text", "json", "both"], default="text", help="Output format")
    parser.add_argument(
        "--json-output", default="prim_callers.json", help="JSON output file path (when --output is json or both)"
    )
    parser.add_argument("--show-proxies", action="store_true", help="Show proxy functions in output")

    args = parser.parse_args()

    # Find project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    print(f"Project root: {project_root}")

    analyzer = PrimCallerAnalyzer(str(project_root))

    # Run analysis
    analyzer.find_prim_launchers()
    analyzer.find_prim_callers()
    analyzer.classify_callers()

    # Output results
    if args.output in ["text", "both"]:
        analyzer.print_results(show_all=args.show_proxies)

    if args.output in ["json", "both"]:
        analyzer.export_json(args.json_output)


if __name__ == "__main__":
    main()
