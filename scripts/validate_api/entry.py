#!/usr/bin/env python3
"""Entry point for API validation checks."""

import os
import sys

from validate_includes import validate_includes_correct


def find_cpp_sources(directory: str) -> list[str]:
    """Find all C++ source files in the directory."""
    CPP_EXTENSIONS = (".hpp", ".h", ".cpp", ".cc", ".cxx")

    source_files = [
        os.path.join(root, fname)
        for root, _, files in os.walk(directory)
        for fname in files
        if fname.endswith(CPP_EXTENSIONS)
    ]
    return source_files


def main(directory):
    api_source_files = find_cpp_sources('tt_metal/api')
    includes_ok = validate_includes_correct(api_source_files)
    deprecation_ok = True  # Placeholder for deprecation checks

    if not includes_ok:
        print("\nInclude check failed.")
        sys.exit(1)
    else:
        print("All includes are valid and all allowed prefixes are used.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <directory>")
        sys.exit(1)
    main(sys.argv[1])
