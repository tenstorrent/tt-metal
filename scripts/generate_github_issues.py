#!/usr/bin/env python3
"""
Generate GitHub issue templates for files containing multiple program factories.
Each file with multiple factories will get its own issue to extract each factory into separate files.
"""

import os
import re
from pathlib import Path
from typing import List, NamedTuple
from datetime import datetime


class ProgramFactory(NamedTuple):
    name: str
    file_path: str
    line_number: int
    function_name: str
    has_program_creation: bool
    has_create_kernel: bool
    has_create_circular_buffer: bool


class GitHubIssueGenerator:
    def __init__(self, repo_root: str = "."):
        self.repo_root = Path(repo_root)
        self.operations_dir = self.repo_root / "ttnn/cpp/ttnn/operations"
        self.output_dir = self.repo_root / "github_issues"

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

    def generate_issue_template(self, file_path: str, factories: List[ProgramFactory]) -> str:
        """Generate a GitHub issue template for a file with multiple program factories."""
        file_name = Path(file_path).name
        relative_path = str(Path(file_path).relative_to(self.repo_root))

        # Create the issue content
        issue_content = f"""# Extract Program Factories from {file_name}

## Overview
This file contains {len(factories)} program factory functions that need to be extracted into separate files as part of the ProgramDescriptor refactoring effort.

**File:** `{relative_path}`

## Program Factories to Extract

"""

        for i, factory in enumerate(factories, 1):
            issue_content += f"""### {i}. {factory.name}
- **Function:** `{factory.function_name}`
- **Line:** {factory.line_number}
- **Status:** TODO

"""

        issue_content += f"""## Tasks

- [ ] Analyze dependencies and shared code between the {len(factories)} program factories
- [ ] Create separate files for each program factory
- [ ] Extract each factory function to its own file
- [ ] Update includes and dependencies
- [ ] Ensure no functionality is lost during extraction
- [ ] Update any references to these functions
- [ ] Test that all extracted factories work correctly

## Notes
- This file contains multiple program factories that should be separated for better maintainability
- Each factory should be moved to its own file following the new ProgramDescriptor pattern
- Consider shared utilities or common code that might need to be extracted as well

## Labels
- `ai-split-program-factory`
- `refactoring`
- `program-descriptor`

---
*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

        return issue_content

    def generate_issues(self):
        """Generate GitHub issue files for files with multiple program factories."""
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)

        # Discover all factories
        all_factories = self.discover_factories()

        # Group factories by file
        factories_by_file = {}
        for factory in all_factories:
            if factory.file_path not in factories_by_file:
                factories_by_file[factory.file_path] = []
            factories_by_file[factory.file_path].append(factory)

        # Find files with multiple factories
        files_with_multiple_factories = {
            file: factories_list for file, factories_list in factories_by_file.items() if len(factories_list) > 1
        }

        print(f"Found {len(files_with_multiple_factories)} files with multiple program factories")

        # Generate issue for each file
        for file_path, factories_list in files_with_multiple_factories.items():
            file_name = Path(file_path).name
            safe_file_name = re.sub(r"[^a-zA-Z0-9._-]", "_", file_name)

            issue_content = self.generate_issue_template(file_path, factories_list)

            # Write issue file
            issue_file = self.output_dir / f"extract_program_factories_{safe_file_name}.md"
            with open(issue_file, "w") as f:
                f.write(issue_content)

            print(f"Generated issue: {issue_file}")

        # Generate summary
        summary_file = self.output_dir / "SUMMARY.md"
        with open(summary_file, "w") as f:
            f.write(f"# Program Factory Extraction Issues Summary\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Total files with multiple program factories: {len(files_with_multiple_factories)}\n\n")

            f.write("## Files to Process\n\n")
            for file_path, factories_list in files_with_multiple_factories.items():
                relative_path = str(Path(file_path).relative_to(self.repo_root))
                f.write(f"- `{relative_path}` ({len(factories_list)} factories)\n")

        print(f"\nGenerated {len(files_with_multiple_factories)} GitHub issue templates in {self.output_dir}")
        print(f"Summary: {summary_file}")


def main():
    generator = GitHubIssueGenerator()
    generator.generate_issues()


if __name__ == "__main__":
    main()
