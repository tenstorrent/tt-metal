# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Common utilities for API validation scripts."""

import os
from typing import Callable, List, Optional, Tuple, TypeVar

T = TypeVar("T")

# Shared constant for C++ source file extensions
CPP_EXTENSIONS = (".hpp", ".h", ".cpp", ".cc", ".cxx")


def partition(predicate: Callable[[T], bool], items: List[T]) -> Tuple[List[T], List[T]]:
    """Split items into (matching, non-matching) based on predicate.

    Example: partition(lambda x: x > 5, [3, 7, 1, 9]) → ([7, 9], [3, 1])
    """
    matching = [item for item in items if predicate(item)]
    non_matching = [item for item in items if not predicate(item)]
    return matching, non_matching


def is_cpp_source(filepath: str) -> bool:
    """Check if a file path is a C++ source file."""
    return filepath.endswith(CPP_EXTENSIONS)


def find_cpp_sources(directory: str, skip_files: Optional[set[str]] = None) -> list[str]:
    """Find all C++ source files in directory, optionally skipping certain files."""
    skip_files = skip_files or set()

    return [
        os.path.join(root, fname)
        for root, _, files in os.walk(directory)
        for fname in files
        if fname.endswith(CPP_EXTENSIONS) and fname not in skip_files
    ]
