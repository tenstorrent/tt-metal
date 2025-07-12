#!/usr/bin/env python3
"""
Program Factory Discovery Script

This script automatically discovers all program factories in the codebase
to help track the refactoring effort from direct Program creation to
ProgramDescriptor creation.

Focuses on:
- Files within ttnn/operations/*/device/ folders
- Functions that return a Program
- Functions that create Programs or call CreateKernel
"""

import os
import re
from pathlib import Path
from typing import List, NamedTuple


class ProgramFactory(NamedTuple):
    name: str
    file_path: str
    line_number: int
    function_name: str
    has_program_creation: bool
    has_create_kernel: bool
    has_create_circular_buffer: bool


class ProgramFactoryDiscoverer:
    def __init__(self, repo_root: str = "."):
        self.repo_root = Path(repo_root)
        self.operations_dir = self.repo_root / "ttnn/cpp/ttnn/operations"

    def should_scan_file(self, file_path: Path) -> bool:
        """Determine if a file should be scanned for program factories."""
        file_str = str(file_path)

        # Must be in ttnn/operations/ pattern
        if not re.search(r"ttnn/operations/", file_str):
            return False

        # Must be a header or source file
        if not file_path.suffix in [".hpp", ".cpp", ".h", ".cc"]:
            return False

        # Exclude test and example files
        exclude_patterns = [
            r".*test.*",
            r".*example.*",
            r".*old.*",
            r".*deprecated.*",
            r".*benchmark.*",
        ]

        if any(re.search(pattern, file_str) for pattern in exclude_patterns):
            return False

        return True

    def scan_file(self, file_path: Path) -> List[ProgramFactory]:
        """Scan a single file for program factories."""
        factories = []

        try:
            with open(file_path, "r") as f:
                content = f.read()

            # Split content into lines for line number tracking
            lines = content.split("\n")

            # Look for functions that create Program and call CreateKernel
            # Pattern: function definition followed by Program creation and CreateKernel call
            function_pattern = r"(\w+)\s+(\w+)\s*\([^)]*\)\s*\{"

            for match in re.finditer(function_pattern, content):
                start_pos = match.start()
                line_number = content[:start_pos].count("\n") + 1

                # Extract function name
                function_name = match.group(2)

                # Get function content from this point
                function_content = self.extract_function_content(content, start_pos)

                # Check for Program creation and CreateKernel call
                has_program_creation = bool(
                    re.search(r"Program\s+program\s*\{|Program\s+program\s*;|CreateProgram", function_content)
                )
                has_create_kernel = bool(re.search(r"CreateKernel", function_content))
                has_create_circular_buffer = bool(re.search(r"CreateCircularBuffer", function_content))

                # Only include if it creates a Program AND calls CreateKernel
                if has_program_creation and has_create_kernel:
                    # Skip if it's already using ProgramDescriptor
                    if "ProgramDescriptor" in function_content:
                        continue

                    factory = ProgramFactory(
                        name=function_name,
                        file_path=str(file_path),
                        line_number=line_number,
                        function_name=function_name,
                        has_program_creation=has_program_creation,
                        has_create_kernel=has_create_kernel,
                        has_create_circular_buffer=has_create_circular_buffer,
                    )

                    factories.append(factory)

        except Exception as e:
            print(f"Error scanning {file_path}: {e}")

        # Report if multiple factories found in same file
        if len(factories) > 1:
            print(f"  Found {len(factories)} program factories in {file_path.name}:")
            for factory in factories:
                print(f"    - {factory.name} (line {factory.line_number})")
        elif len(factories) == 1:
            print(f"  Found 1 program factory in {file_path.name}: {factories[0].name}")

        return factories

    def extract_function_content(self, content: str, start_pos: int) -> str:
        """Extract the content of a function starting from the given position."""
        if start_pos >= len(content):
            return ""

        # Find the opening brace
        brace_pos = content.find("{", start_pos)
        if brace_pos == -1:
            return ""

        # Count braces to find the end of the function
        brace_count = 1
        pos = brace_pos + 1

        while pos < len(content) and brace_count > 0:
            if content[pos] == "{":
                brace_count += 1
            elif content[pos] == "}":
                brace_count -= 1
            pos += 1

        return content[brace_pos:pos]

    def discover_factories(self) -> List[ProgramFactory]:
        """Discover all program factories in the codebase."""
        all_factories = []

        if not self.operations_dir.exists():
            print(f"Warning: Operations directory {self.operations_dir} does not exist")
            return all_factories

        print(f"Scanning all files in {self.operations_dir}")

        # Scan ALL files in the operations directory recursively
        for file_path in self.operations_dir.rglob("*"):
            if self.should_scan_file(file_path):
                factories = self.scan_file(file_path)
                all_factories.extend(factories)

        return all_factories

    def generate_markdown(
        self, factories: List[ProgramFactory], output_file: str = "PROGRAM_FACTORY_REFACTORING_TRACKER.md"
    ):
        """Generate a markdown tracker for the discovered program factories."""
        # Group factories by file to identify files with multiple factories
        factories_by_file = {}
        for factory in factories:
            if factory.file_path not in factories_by_file:
                factories_by_file[factory.file_path] = []
            factories_by_file[factory.file_path].append(factory)

        # Find files with multiple factories
        files_with_multiple_factories = {
            file: factories_list for file, factories_list in factories_by_file.items() if len(factories_list) > 1
        }

        with open(output_file, "w") as f:
            f.write("# Program Factory Refactoring Tracker\n\n")
            f.write(f"Total program factories found: {len(factories)}\n")
            f.write(f"Total files: {len(factories_by_file)}\n")
            f.write(f"Files with multiple factories: {len(files_with_multiple_factories)}\n\n")

            # List files with multiple factories first
            if files_with_multiple_factories:
                f.write("## Files with Multiple Program Factories\n\n")
                f.write("These files contain multiple program factory functions and may need special attention:\n\n")

                for file_path, factories_list in files_with_multiple_factories.items():
                    f.write(f"### {file_path} ({len(factories_list)} factories)\n\n")
                    f.write("| Factory Name | Line | Status |\n")
                    f.write("|--------------|------|--------|\n")

                    for factory in factories_list:
                        f.write(f"| {factory.name} | {factory.line_number} | TODO |\n")
                    f.write("\n")

            f.write("## All Program Factories List\n\n")
            f.write("| Factory Name | File | Line | Status |\n")
            f.write("|--------------|------|------|--------|\n")

            for factory in factories:
                f.write(f"| {factory.name} | {factory.file_path} | {factory.line_number} | TODO |\n")

        print(f"Generated tracker: {output_file}")

        # Also print summary to console
        print(f"\n=== SUMMARY ===")
        print(f"Total Program Factories: {len(factories)}")
        print(f"Total Files: {len(factories_by_file)}")
        print(f"Files with Multiple Factories: {len(files_with_multiple_factories)}")

        if files_with_multiple_factories:
            print(f"\n=== FILES WITH MULTIPLE FACTORIES ===")
            for file_path, factories_list in files_with_multiple_factories.items():
                print(f"\n{file_path} ({len(factories_list)} factories):")
                for factory in factories_list:
                    print(f"  - {factory.name} (line {factory.line_number})")


def main():
    discoverer = ProgramFactoryDiscoverer()
    factories = discoverer.discover_factories()

    print(f"\nTotal program factories found: {len(factories)}")

    if factories:
        discoverer.generate_markdown(factories)
    else:
        print("No program factories found!")


if __name__ == "__main__":
    main()
