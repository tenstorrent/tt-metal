"""Common utilities for API validation scripts."""

import os


def find_cpp_sources(directory: str, skip_files: set[str] = None) -> list[str]:
    """Find all C++ source files in directory, optionally skipping certain files."""
    CPP_EXTENSIONS = (".hpp", ".h", ".cpp", ".cc", ".cxx")
    skip_files = skip_files or set()

    return [
        os.path.join(root, fname)
        for root, _, files in os.walk(directory)
        for fname in files
        if fname.endswith(CPP_EXTENSIONS) and fname not in skip_files
    ]
