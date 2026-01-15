#!/usr/bin/env python3
"""
Script to eliminate proxy operations by:
1. Replacing the header with a simple using declaration
2. Deleting the .cpp proxy file
3. Updating nanobind to use mod.def directly
4. Removing the .cpp from CMakeLists.txt
"""

import re
import sys
from pathlib import Path


def get_prim_function_from_cpp(cpp_content: str) -> str | None:
    """Extract the prim function name from the .cpp file."""
    match = re.search(r"ttnn::prim::(\w+)\(", cpp_content)
    return match.group(1) if match else None


def get_device_header_path(hpp_path: Path) -> Path:
    """Get the device header path from the hpp path."""
    op_name = hpp_path.stem
    device_dir = hpp_path.parent / "device"
    return device_dir / f"{op_name}_device_operation.hpp"


def update_header(hpp_path: Path, prim_name: str) -> bool:
    """Update the header to use the prim function directly."""
    device_header = get_device_header_path(hpp_path)
    if not device_header.exists():
        print(f"  Warning: device header not found: {device_header}")
        # Try alternative naming
        device_header = hpp_path.parent / "device" / f"{prim_name}_device_operation.hpp"
        if not device_header.exists():
            print(f"  Warning: alternative device header not found: {device_header}")
            return False

    # Get the relative path from the hpp to the device header
    rel_device_header = f"device/{device_header.name}"

    # Read existing header to get copyright
    original = hpp_path.read_text()
    copyright_match = re.search(
        r"(// SPDX-FileCopyrightText:.*?\n// SPDX-License-Identifier:.*?\n)", original, re.DOTALL
    )
    copyright = (
        copyright_match.group(1)
        if copyright_match
        else "// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.\n// SPDX-License-Identifier: Apache-2.0\n"
    )

    new_content = f"""{copyright}
#pragma once

#include "{rel_device_header}"

namespace ttnn {{
using prim::{prim_name};
}}  // namespace ttnn
"""
    hpp_path.write_text(new_content)
    return True


def update_nanobind(nanobind_path: Path, prim_name: str, op_name: str) -> bool:
    """Update the nanobind file to use mod.def directly."""
    if not nanobind_path.exists():
        print(f"  Warning: nanobind file not found: {nanobind_path}")
        return False

    content = nanobind_path.read_text()

    # Check if already converted
    if "mod.def(" in content and "bind_registered_operation" not in content:
        print(f"  Nanobind already converted")
        return True

    # Replace bind_registered_operation with mod.def
    # This is a simplified version - complex cases may need manual review

    # Remove ttnn-nanobind/decorators.hpp include
    content = re.sub(r'#include "ttnn-nanobind/decorators\.hpp"\n', "", content)

    # Replace bind_registered_operation call
    # Pattern: bind_registered_operation(mod, ttnn::name, doc, ttnn::nanobind_arguments_t{...});
    pattern = r"bind_registered_operation\(\s*mod,\s*ttnn::\w+,\s*(\w+),\s*ttnn::nanobind_arguments_t\{([^}]+)\}\);"

    def replace_bind(match):
        doc_var = match.group(1)
        args = match.group(2)
        return f'mod.def(\n        "{op_name}",\n        &ttnn::{op_name},\n        {doc_var},{args});'

    new_content = re.sub(pattern, replace_bind, content, flags=re.DOTALL)

    if new_content == content:
        print(f"  Warning: Could not find bind_registered_operation pattern")
        return False

    nanobind_path.write_text(new_content)
    return True


def update_cmake(cmake_path: Path, cpp_name: str) -> bool:
    """Remove the .cpp file from CMakeLists.txt."""
    if not cmake_path.exists():
        print(f"  Warning: CMakeLists.txt not found: {cmake_path}")
        return False

    content = cmake_path.read_text()

    # Remove the line with the cpp file
    pattern = rf"\n\s+{re.escape(cpp_name)}\.cpp"
    new_content = re.sub(pattern, "", content)

    if new_content == content:
        print(f"  Warning: Could not find {cpp_name}.cpp in CMakeLists.txt")
        return False

    cmake_path.write_text(new_content)
    return True


def eliminate_proxy(hpp_path_str: str) -> bool:
    """Eliminate a proxy operation."""
    hpp_path = Path(hpp_path_str)

    if not hpp_path.exists():
        print(f"Error: Header not found: {hpp_path}")
        return False

    op_name = hpp_path.stem
    cpp_path = hpp_path.with_suffix(".cpp")
    nanobind_path = hpp_path.parent / f"{op_name}_nanobind.cpp"
    cmake_path = hpp_path.parent / "CMakeLists.txt"

    print(f"\nProcessing: {op_name}")
    print(f"  Header: {hpp_path}")

    # Read the cpp file to get the prim function name
    if not cpp_path.exists():
        print(f"  Cpp file not found (may already be deleted): {cpp_path}")
        return False

    cpp_content = cpp_path.read_text()
    prim_name = get_prim_function_from_cpp(cpp_content)

    if not prim_name:
        print(f"  Error: Could not find prim function in {cpp_path}")
        return False

    print(f"  Prim function: {prim_name}")

    # Update header
    if not update_header(hpp_path, prim_name):
        return False
    print(f"  Updated header")

    # Delete cpp file
    cpp_path.unlink()
    print(f"  Deleted {cpp_path.name}")

    # Update nanobind
    if nanobind_path.exists():
        if update_nanobind(nanobind_path, prim_name, op_name):
            print(f"  Updated nanobind")
        else:
            print(f"  Warning: Nanobind update may need manual review")

    # Update CMakeLists.txt
    if update_cmake(cmake_path, op_name):
        print(f"  Updated CMakeLists.txt")

    return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python eliminate_proxy.py <hpp_path> [<hpp_path2> ...]")
        print("       python eliminate_proxy.py --from-todo <todo_file>")
        sys.exit(1)

    if sys.argv[1] == "--from-todo":
        if len(sys.argv) < 3:
            print("Error: --from-todo requires a todo file path")
            sys.exit(1)

        todo_file = Path(sys.argv[2])
        lines = todo_file.read_text().strip().split("\n")

        for line in lines:
            if line.startswith("[ ]"):
                parts = line.split()
                if len(parts) >= 3:
                    hpp_path = parts[2]
                    eliminate_proxy(hpp_path)
    else:
        for hpp_path in sys.argv[1:]:
            eliminate_proxy(hpp_path)


if __name__ == "__main__":
    main()
