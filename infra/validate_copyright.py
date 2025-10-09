#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Custom copyright validation script for Tenstorrent.

This script validates that copyright headers in source files follow the expected format:
- SPDX-FileCopyrightText: © YEAR Copyright Holder
- SPDX-License-Identifier: Apache-2.0

Simple logic:
- Files with no copyright → Adds Tenstorrent AI ULC copyright
- Files with existing copyright → Allows any copyright holder (individual contributors or Tenstorrent)
- Individual contributors are responsible for adding their own copyright if they want it

The script is designed to complement the Espressif check-copyright tool by validating
SPDX copyright headers while allowing individual contributors to maintain their copyright.
"""

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class ValidationResult:
    """Result of copyright validation for a single file."""

    is_valid: bool
    errors: List[str]
    copyright_line: Optional[str] = None
    license_line: Optional[str] = None
    fixes_applied: List[str] = None

    def __post_init__(self):
        if self.fixes_applied is None:
            self.fixes_applied = []


class CopyrightValidator:
    """Validates copyright headers in source files."""

    # Class constants for better performance and maintainability
    EXPECTED_COMPANY = "Tenstorrent AI ULC"
    EXPECTED_LICENSE = "Apache-2.0"
    CHECK_EXTENSIONS = frozenset({".cpp", ".cc", ".h", ".hpp", ".py", ".ld"})
    MAX_HEADER_LINES = 10  # Only check first 10 lines for headers

    # Pre-compiled regex patterns for better performance
    SPDX_COPYRIGHT_PATTERN = re.compile(
        r"^\s*(?://|#|\*)\s*SPDX-FileCopyrightText:\s*(?:©|\(c\))\s*(\d{4}(?:-\d{4})?)\s+([^/]+?)\.?\s*(?://.*)?$",
        re.IGNORECASE,
    )

    SPDX_LICENSE_PATTERN = re.compile(
        r"^\s*(?://|#|\*)\s*SPDX-License-Identifier:\s*(.+?)\s*$", re.IGNORECASE
    )

    # Excluded directory patterns
    EXCLUDED_PATTERNS = frozenset(
        {
            ".venv",
            "venv",
            ".git",
            "node_modules",
            "__pycache__",
            ".pytest_cache",
            "build",
            ".tox",
            "dist",
            ".egg-info",
        }
    )

    def should_check_file(self, file_path: Path) -> bool:
        """Check if file should be validated based on extension."""
        return file_path.suffix in self.CHECK_EXTENSIONS

    def _is_excluded_path(self, file_path: Path) -> bool:
        """Check if file path contains excluded patterns."""
        path_str = str(file_path)
        return any(pattern in path_str for pattern in self.EXCLUDED_PATTERNS)

    def find_copyright_lines(
        self, lines: List[str]
    ) -> Tuple[List[str], Optional[str], List[str]]:
        """
        Find SPDX copyright and license lines in file content.

        Args:
            lines: List of lines from the file (pre-split for efficiency)

        Returns:
            Tuple of (copyright_lines, license_line, errors)
        """
        copyright_lines = []
        license_line = None
        errors = []
        has_tenstorrent_copyright = False

        # Only check first MAX_HEADER_LINES for performance
        header_lines = lines[: self.MAX_HEADER_LINES]

        for i, line in enumerate(header_lines, 1):
            # Check for copyright line
            copyright_match = self.SPDX_COPYRIGHT_PATTERN.match(line)
            if copyright_match:
                copyright_lines.append(line.strip())

                # Validate company name - check for forbidden "Tenstorrent Inc."
                year_part, company_part = copyright_match.groups()
                company_name = company_part.strip().rstrip(".")

                # Check if this is a Tenstorrent copyright
                if "tenstorrent" in company_name.lower():
                    has_tenstorrent_copyright = True

                    # Check for forbidden "Tenstorrent Inc." (case insensitive)
                    if "tenstorrent inc" in company_name.lower():
                        errors.append(
                            f"Line {i}: Found forbidden 'Tenstorrent Inc.', "
                            f"should be '{self.EXPECTED_COMPANY}'"
                        )
                    # For Tenstorrent copyrights, ensure they use the correct format
                    elif company_name != self.EXPECTED_COMPANY:
                        errors.append(
                            f"Line {i}: Tenstorrent copyright should use '{self.EXPECTED_COMPANY}', "
                            f"found '{company_name}'"
                        )

            # Check for license line
            license_match = self.SPDX_LICENSE_PATTERN.match(line)
            if license_match:
                if license_line is not None:
                    errors.append(
                        f"Line {i}: Multiple SPDX-License-Identifier lines found"
                    )
                    continue

                license_line = line.strip()

                # Validate license
                license_part = license_match.group(1).strip()
                if license_part != self.EXPECTED_LICENSE:
                    errors.append(
                        f"Line {i}: Expected license '{self.EXPECTED_LICENSE}', "
                        f"found '{license_part}'"
                    )

        # If copyright exists, don't enforce Tenstorrent copyright
        # Individual contributors can use their own copyright
        # Only files without any copyright will get Tenstorrent copyright added

        return copyright_lines, license_line, errors

    def _fix_copyright_issues(
        self,
        all_lines: List[str],
        copyright_lines: List[str],
        license_line: Optional[str],
        errors: List[str],
        file_path: Path,
    ) -> Tuple[List[str], List[str]]:
        """
        Fix copyright issues in the file content.

        Args:
            all_lines: All lines of the file
            copyright_lines: Found copyright lines
            license_line: Found license line
            errors: Validation errors found

        Returns:
            Tuple of (fixed_lines, fixes_applied)
        """
        fixed_lines = all_lines.copy()
        fixes_applied = []

        # Detect file comment style
        comment_style = self._detect_comment_style(all_lines, file_path)

        # Fix forbidden "Tenstorrent Inc." to "Tenstorrent AI ULC"
        for i, line in enumerate(fixed_lines):
            if i >= self.MAX_HEADER_LINES:
                break

            copyright_match = self.SPDX_COPYRIGHT_PATTERN.match(line)
            if copyright_match:
                year_part, company_part = copyright_match.groups()
                company_name = company_part.strip().rstrip(".")

                if "tenstorrent inc" in company_name.lower():
                    # Replace the company name using targeted regex substitution
                    new_company = self.EXPECTED_COMPANY

                    # Create a pattern specifically for replacing Tenstorrent Inc
                    tenstorrent_pattern = re.compile(
                        r"(^\s*(?://|#|\*)\s*SPDX-FileCopyrightText:\s*(?:©|\(c\))\s*\d{4}(?:-\d{4})?\s+)(Tenstorrent Inc\.?)(\s*.*$)",
                        re.IGNORECASE,
                    )

                    new_line = tenstorrent_pattern.sub(
                        lambda m: m.group(1) + new_company + m.group(3), line
                    )
                    fixed_lines[i] = new_line
                    fixes_applied.append(
                        f"Line {i+1}: Fixed 'Tenstorrent Inc.' → '{new_company}'"
                    )

        # Add missing copyright header if needed (only if no copyright exists at all)
        if not copyright_lines:
            # Add Tenstorrent copyright for files with no copyright
            header_lines = self._generate_copyright_header(comment_style)
            # Insert at the beginning or after existing comment block
            insert_pos = self._find_header_insert_position(fixed_lines, comment_style)
            for j, header_line in enumerate(header_lines):
                fixed_lines.insert(insert_pos + j, header_line)
            fixes_applied.append("Added missing SPDX-FileCopyrightText header")
            fixes_applied.append("Added missing SPDX-License-Identifier header")

        # Add missing license header if needed (if copyright exists but license doesn't)
        elif license_line is None:
            license_lines = self._generate_license_header(comment_style)
            insert_pos = self._find_license_insert_position(fixed_lines, comment_style)
            for j, license_line_content in enumerate(license_lines):
                fixed_lines.insert(insert_pos + j, license_line_content)
            fixes_applied.append("Added missing SPDX-License-Identifier header")

        return fixed_lines, fixes_applied

    def _detect_comment_style(self, lines: List[str], file_path: Path = None) -> str:
        """Detect the comment style used in the file."""
        # Prioritize file extension for reliability, then check existing comments
        if file_path:
            ext = file_path.suffix.lower()
            if ext in {".ld"}:
                # For .ld files, check if there are existing C-style comments
                for line in lines[:5]:
                    line = line.strip()
                    if line.startswith("/*") or line.startswith(" *"):
                        return "c_block"
                    elif line.startswith("//"):
                        return "cpp"
                # Default to C block for .ld files
                return "c_block"
            elif ext in {".py"}:
                return "python"
            else:  # .h, .hpp, .cpp, .cc - these are C/C++ files
                # Check if there are existing C-style block comments
                for line in lines[:5]:
                    line = line.strip()
                    if line.startswith("/*") or line.startswith(" *"):
                        return "c_block"
                    elif line.startswith("//"):
                        return "cpp"
                # Default to C++ style for C/C++ files
                return "cpp"

        # Fallback to content-based detection if no file extension
        for line in lines[:5]:
            line = line.strip()
            if line.startswith("/*") or line.startswith(" *"):
                return "c_block"
            elif line.startswith("//"):
                return "cpp"
            elif line.startswith("#") and not line.startswith(
                (
                    "#pragma",
                    "#include",
                    "#define",
                    "#ifdef",
                    "#ifndef",
                    "#endif",
                    "#if",
                    "#else",
                    "#elif",
                )
            ):
                return "python"

        # Final fallback
        return "cpp"

    def _generate_copyright_header(self, comment_style: str) -> List[str]:
        """Generate appropriate copyright header based on comment style."""
        year = str(datetime.now().year)

        if comment_style == "c_block":
            return [
                "/*",
                f" * SPDX-FileCopyrightText: (c) {year} {self.EXPECTED_COMPANY}",
                " *",
                f" * SPDX-License-Identifier: {self.EXPECTED_LICENSE}",
                " */",
                "",
            ]
        elif comment_style == "python":
            return [
                f"# SPDX-FileCopyrightText: © {year} {self.EXPECTED_COMPANY}",
                "#",
                f"# SPDX-License-Identifier: {self.EXPECTED_LICENSE}",
                "",
            ]
        else:  # cpp style
            return [
                f"// SPDX-FileCopyrightText: © {year} {self.EXPECTED_COMPANY}",
                "//",
                f"// SPDX-License-Identifier: {self.EXPECTED_LICENSE}",
                "",
            ]

    def _generate_license_header(self, comment_style: str) -> List[str]:
        """Generate just the license header."""
        if comment_style == "c_block":
            return [f" * SPDX-License-Identifier: {self.EXPECTED_LICENSE}"]
        elif comment_style == "python":
            return [f"# SPDX-License-Identifier: {self.EXPECTED_LICENSE}"]
        else:
            return [f"// SPDX-License-Identifier: {self.EXPECTED_LICENSE}"]

    def _find_header_insert_position(self, lines: List[str], comment_style: str) -> int:
        """Find the best position to insert copyright header."""
        # Check if first line is a proper shebang
        if lines and lines[0].startswith("#!/"):
            # Insert after shebang line
            return 1
        # Insert at the very beginning for most cases
        return 0

    def _find_license_insert_position(
        self, lines: List[str], comment_style: str
    ) -> int:
        """Find position to insert license header (after copyright)."""
        for i, line in enumerate(lines[: self.MAX_HEADER_LINES]):
            if self.SPDX_COPYRIGHT_PATTERN.match(line):
                return i + 1
        return 0

    def _generate_tenstorrent_copyright_line(self, comment_style: str) -> str:
        """Generate a single Tenstorrent copyright line based on comment style."""
        year = str(datetime.now().year)

        if comment_style == "c_block":
            return f" * SPDX-FileCopyrightText: (c) {year} {self.EXPECTED_COMPANY}"
        elif comment_style == "python":
            return f"# SPDX-FileCopyrightText: © {year} {self.EXPECTED_COMPANY}"
        else:  # cpp style
            return f"// SPDX-FileCopyrightText: © {year} {self.EXPECTED_COMPANY}"

    def _find_tenstorrent_copyright_insert_position(
        self, lines: List[str], comment_style: str
    ) -> int:
        """Find position to insert Tenstorrent copyright (should be first copyright line)."""
        first_copyright_pos = -1
        for i, line in enumerate(lines[: self.MAX_HEADER_LINES]):
            if self.SPDX_COPYRIGHT_PATTERN.match(line):
                first_copyright_pos = i
                break

        # Insert before the first copyright line, or at the beginning if none found
        return first_copyright_pos if first_copyright_pos >= 0 else 0

    def validate_file(
        self, file_path: Path, fix_errors: bool = False
    ) -> ValidationResult:
        """
        Validate copyright header in a single file.

        Args:
            file_path: Path to the file to validate
            fix_errors: If True, automatically fix copyright issues

        Returns:
            ValidationResult object containing validation status and details
        """
        if not self.should_check_file(file_path):
            return ValidationResult(is_valid=True, errors=[])

        try:
            # Read the entire file when fixing, or just the header when validating
            with open(file_path, "r", encoding="utf-8") as f:
                if fix_errors:
                    # Read the entire content to preserve original ending format
                    original_content = f.read()
                    # Special case: empty files should maintain their format
                    if len(original_content) == 0:
                        original_ends_with_newline = False
                    else:
                        original_ends_with_newline = original_content.endswith(
                            ("\n", "\r\n", "\r")
                        )
                    all_lines = original_content.splitlines()
                    lines = all_lines[: self.MAX_HEADER_LINES]
                else:
                    # Read only first few lines for header check
                    lines = []
                    for i, line in enumerate(f):
                        lines.append(line.rstrip("\n\r"))
                        if i >= self.MAX_HEADER_LINES - 1:
                            break
                    all_lines = None
                    original_ends_with_newline = None

        except UnicodeDecodeError:
            # Skip binary files
            return ValidationResult(is_valid=True, errors=[])
        except Exception as e:
            return ValidationResult(is_valid=False, errors=[f"Error reading file: {e}"])

        copyright_lines, license_line, errors = self.find_copyright_lines(lines)
        fixes_applied = []

        # Simple logic: if copyright exists, don't enforce Tenstorrent copyright
        # Individual contributors are responsible for adding their own copyright if they want it

        # Apply fixes if requested
        if fix_errors and all_lines is not None:
            fixed_lines, file_fixes = self._fix_copyright_issues(
                all_lines, copyright_lines, license_line, errors, file_path
            )
            if file_fixes:
                try:
                    with open(file_path, "w", encoding="utf-8") as f:
                        # Preserve original file ending format
                        content = "\n".join(fixed_lines)
                        if original_ends_with_newline:
                            content += "\n"
                        f.write(content)
                    fixes_applied = file_fixes
                    # Re-validate after fixing
                    copyright_lines, license_line, errors = self.find_copyright_lines(
                        fixed_lines[: self.MAX_HEADER_LINES]
                    )
                except Exception as e:
                    errors.append(f"Error writing fixed file: {e}")

        # Check for missing headers (only if not fixed)
        if not copyright_lines:
            if not any(
                "Added missing SPDX-FileCopyrightText header" in fix
                for fix in fixes_applied
            ):
                errors.append("Missing SPDX-FileCopyrightText header")

        if license_line is None:
            if not any(
                "Added missing SPDX-License-Identifier header" in fix
                for fix in fixes_applied
            ):
                errors.append("Missing SPDX-License-Identifier header")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            copyright_line=copyright_lines[0] if copyright_lines else None,
            license_line=license_line,
            fixes_applied=fixes_applied,
        )

    def validate_files(
        self, file_paths: List[Path], quiet: bool = False, fix_errors: bool = False
    ) -> bool:
        """
        Validate copyright headers in multiple files.

        Args:
            file_paths: List of file paths to validate
            quiet: If True, only show errors, not success messages
            fix_errors: If True, automatically fix copyright issues

        Returns:
            True if all files are valid, False otherwise
        """
        all_valid = True

        for file_path in file_paths:
            result = self.validate_file(file_path, fix_errors=fix_errors)

            if not result.is_valid:
                all_valid = False
                print(f"❌ {file_path}:")
                for error in result.errors:
                    print(f"    {error}")
            elif result.fixes_applied:
                print(f"🔧 {file_path}:")
                for fix in result.fixes_applied:
                    print(f"    {fix}")
            elif not quiet and self.should_check_file(file_path):
                print(f"✅ {file_path}")

        return all_valid

    def discover_files(self, root_path: Path = None) -> List[Path]:
        """
        Discover all relevant files for copyright validation.

        Args:
            root_path: Root directory to search (defaults to current directory)

        Returns:
            List of file paths to validate
        """
        if root_path is None:
            root_path = Path(".")

        discovered_files = []

        # Use more efficient file discovery
        for extension in self.CHECK_EXTENSIONS:
            pattern = f"**/*{extension}"
            for file_path in root_path.glob(pattern):
                if not self._is_excluded_path(file_path):
                    discovered_files.append(file_path)

        return discovered_files

    def get_git_changed_files(self, staged_only: bool = False) -> List[Path]:
        """
        Get files that have changed in git.

        Args:
            staged_only: If True, only return staged files. If False, return all changed files.

        Returns:
            List of changed file paths that should be validated
        """
        try:
            if staged_only:
                # Get only staged files
                cmd = ["git", "diff", "--cached", "--name-only"]
            else:
                # Get all changed files (staged + unstaged)
                cmd = ["git", "diff", "HEAD", "--name-only"]

            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            changed_files = []

            for line in result.stdout.strip().split("\n"):
                if line:  # Skip empty lines
                    file_path = Path(line)
                    if (
                        file_path.exists()
                        and self.should_check_file(file_path)
                        and not self._is_excluded_path(file_path)
                    ):
                        changed_files.append(file_path)

            return changed_files

        except subprocess.CalledProcessError:
            # Not in a git repository or git command failed
            return []
        except FileNotFoundError:
            # Git not installed
            return []


def main() -> int:
    """Main entry point for the copyright validation script."""
    parser = argparse.ArgumentParser(
        description="Validate copyright headers in source files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  {sys.argv[0]} file1.py file2.h         # Check specific files
  {sys.argv[0]} --quiet                  # Check all files, show only errors
  {sys.argv[0]} --fix                    # Fix copyright issues automatically
  {sys.argv[0]} --git-staged --fix       # Fix copyright issues in staged files
  {sys.argv[0]}                          # Check all files with verbose output
        """,
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Files to check (if none provided, checks all relevant files)",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Only output errors, not success messages"
    )
    parser.add_argument(
        "--git-diff",
        action="store_true",
        help="Only check files that are different from HEAD (staged and unstaged changes)",
    )
    parser.add_argument(
        "--git-staged",
        action="store_true",
        help="Only check files that are staged for commit",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Automatically fix copyright issues where possible",
    )

    args = parser.parse_args()

    validator = CopyrightValidator()

    # Get file paths to validate
    if args.files:
        file_paths = [Path(f) for f in args.files if Path(f).exists()]
        # Warn about non-existent files
        missing_files = [f for f in args.files if not Path(f).exists()]
        if missing_files:
            print(f"Warning: Files not found: {', '.join(missing_files)}")
    elif args.git_staged:
        file_paths = validator.get_git_changed_files(staged_only=True)
        if not file_paths:
            print("No staged files to check")
            return 0
    elif args.git_diff:
        file_paths = validator.get_git_changed_files(staged_only=False)
        if not file_paths:
            print("No changed files to check")
            return 0
    else:
        file_paths = validator.discover_files()

    if not file_paths:
        print("No files to check")
        return 0

    # Validate files
    success = validator.validate_files(
        file_paths, quiet=args.quiet, fix_errors=args.fix
    )

    # Print summary
    if success:
        if not args.quiet:
            print(f"\n✅ All {len(file_paths)} files passed copyright validation")
        return 0
    else:
        print(f"\n❌ Copyright validation failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
