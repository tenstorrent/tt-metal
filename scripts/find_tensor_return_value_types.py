#!/usr/bin/env python3
"""
Find all device operations where tensor_return_value_t resolves to Tensor.

This script searches for device operation structures that define:
  using tensor_return_value_t = namespace::tensor_return_value_t;

And checks if the corresponding *_device_operation_types.hpp file defines:
  using tensor_return_value_t = Tensor;

Results are saved to a JSON file.
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

try:
    from clang.cindex import Index, TranslationUnit, CursorKind, TypeKind, Config

    CLANG_AVAILABLE = True
except ImportError:
    CLANG_AVAILABLE = False
    print("Warning: clang.cindex not available, falling back to regex parsing", file=sys.stderr)

# Compile regex patterns once for performance
TENSOR_RETURN_TYPE_ALIAS_PATTERN = re.compile(
    r"using\s+tensor_return_value_t\s*=\s*([^:;]+::)?tensor_return_value_t\s*;"
)
TENSOR_RETURN_TYPE_DIRECT_PATTERN = re.compile(r"using\s+tensor_return_value_t\s*=\s*Tensor\s*;")
TENSOR_RETURN_TYPE_QUALIFIED_PATTERN = re.compile(r"using\s+tensor_return_value_t\s*=\s*(ttnn::)?Tensor\s*;")


def find_compile_commands(root_dir: Path) -> Optional[Path]:
    """Find compile_commands.json in common locations."""
    common_locations = [
        root_dir / "build_Debug" / "compile_commands.json",
        root_dir / "build" / "compile_commands.json",
        root_dir / "compile_commands.json",
    ]

    for location in common_locations:
        if location.exists():
            return location
    return None


def load_compile_commands(compile_commands_path: Path) -> Optional[Dict[str, Any]]:
    """Load compile_commands.json and return as dict indexed by file path."""
    try:
        with open(compile_commands_path, "r") as f:
            commands = json.load(f)

        # Index by absolute file path
        db = {}
        for entry in commands:
            file_path = Path(entry["file"]).resolve()
            db[str(file_path)] = entry
        return db
    except Exception as e:
        print(f"Warning: Failed to load compile_commands.json: {e}", file=sys.stderr)
        return None


def find_device_operation_files(root_dir: Path) -> List[Path]:
    """Find all *_device_operation.hpp files."""
    files = []
    search_dirs = [
        root_dir / "ttnn" / "cpp" / "ttnn" / "operations",
        root_dir / "tt-train",
    ]

    for search_dir in search_dirs:
        if search_dir.exists():
            files.extend(search_dir.rglob("*_device_operation.hpp"))

    return sorted(files)


def find_types_file(device_op_file: Path) -> Optional[Path]:
    """Find corresponding *_device_operation_types.hpp file."""
    # Try same directory first
    types_file = device_op_file.parent / device_op_file.name.replace(
        "_device_operation.hpp", "_device_operation_types.hpp"
    )
    if types_file.exists():
        return types_file

    # Try parent directory
    types_file = device_op_file.parent.parent / device_op_file.name.replace(
        "_device_operation.hpp", "_device_operation_types.hpp"
    )
    if types_file.exists():
        return types_file

    return None


def extract_namespace_regex(file_path: Path) -> Optional[Tuple[str, int]]:
    """Extract namespace from tensor_return_value_t alias using regex."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            lines = content.split("\n")

        for line_num, line in enumerate(lines, 1):
            match = TENSOR_RETURN_TYPE_ALIAS_PATTERN.search(line)
            if match:
                namespace_part = match.group(1)
                if namespace_part:
                    # Remove trailing ::
                    namespace = namespace_part.rstrip(":")
                    return (namespace, line_num)

        return None
    except Exception as e:
        print(f"Error reading {file_path}: {e}", file=sys.stderr)
        return None


def check_resolves_to_tensor_regex(types_file: Path) -> bool:
    """Check if tensor_return_value_t = Tensor using regex."""
    try:
        with open(types_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Check for exact match: using tensor_return_value_t = Tensor;
        # Allow optional ttnn:: prefix
        return bool(TENSOR_RETURN_TYPE_QUALIFIED_PATTERN.search(content))
    except Exception as e:
        print(f"Error reading {types_file}: {e}", file=sys.stderr)
        return False


def resolve_type_with_clang(
    file_path: Path, compile_db: Optional[Dict[str, Any]], root_dir: Path
) -> Optional[Tuple[str, int]]:
    """Use libclang to resolve tensor_return_value_t type alias.

    Note: For extracting namespace from 'using tensor_return_value_t = namespace::tensor_return_value_t;',
    regex is simpler and more reliable. This function is kept for potential future improvements.
    """
    # For now, regex is more straightforward for namespace extraction
    # Clang is better used for checking if the final type resolves to Tensor
    return None


def check_resolves_to_tensor_clang(types_file: Path, compile_db: Optional[Dict[str, Any]]) -> bool:
    """Use libclang to check if tensor_return_value_t = Tensor."""
    if not CLANG_AVAILABLE:
        return False

    try:
        file_key = str(types_file.resolve())
        compile_entry = compile_db.get(file_key) if compile_db else None

        if not compile_entry:
            return False

        import shlex

        args = shlex.split(compile_entry["command"])
        # Remove compiler, input file, and output file arguments
        args = [a for a in args[1:] if a != file_key and not a.startswith("-o")]

        index = Index.create()
        tu = index.parse(str(types_file), args=args, options=TranslationUnit.PARSE_SKIP_FUNCTION_BODIES)

        def find_tensor_return_alias(cursor):
            if cursor.kind == CursorKind.TYPE_ALIAS_DECL:
                if cursor.spelling == "tensor_return_value_t":
                    # Get the underlying type
                    underlying_type = cursor.underlying_typedef_type
                    if underlying_type:
                        type_spelling = underlying_type.spelling
                        # Check if it's Tensor (with or without namespace qualifiers)
                        # Remove leading :: and check base name
                        base_name = type_spelling.split("::")[-1]
                        if base_name == "Tensor":
                            return True
            for child in cursor.get_children():
                if find_tensor_return_alias(child):
                    return True
            return False

        return find_tensor_return_alias(tu.cursor)

    except Exception as e:
        # Fall back to regex
        return False


def process_file(
    device_op_file: Path, compile_db: Optional[Dict[str, Any]], root_dir: Path, use_clang: bool
) -> Optional[Dict[str, Any]]:
    """Process a single device operation file."""
    # Find corresponding types file
    types_file = find_types_file(device_op_file)
    if not types_file:
        return None

    # Extract namespace (regex is simpler and more reliable for this)
    namespace_info = extract_namespace_regex(device_op_file)

    if not namespace_info:
        return None

    namespace, line_num = namespace_info

    # Check if it resolves to Tensor
    if use_clang and CLANG_AVAILABLE:
        resolves_to_tensor = check_resolves_to_tensor_clang(types_file, compile_db)
    else:
        resolves_to_tensor = check_resolves_to_tensor_regex(types_file)

    if not resolves_to_tensor:
        return None

    return {
        "device_operation_file": str(device_op_file.relative_to(root_dir)),
        "types_file": str(types_file.relative_to(root_dir)),
        "namespace": namespace,
        "line_number": line_num,
        "resolves_to": "Tensor",
    }


def main():
    """Main execution."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Find device operations where tensor_return_value_t resolves to Tensor"
    )
    parser.add_argument(
        "--root", type=Path, default=Path.cwd(), help="Root directory of the project (default: current directory)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tensor_return_value_tensor_matches.json"),
        help="Output file path (default: tensor_return_value_tensor_matches.json)",
    )
    parser.add_argument(
        "--use-clang", action="store_true", help="Use libclang for type resolution (requires compile_commands.json)"
    )
    parser.add_argument(
        "--compile-commands", type=Path, help="Path to compile_commands.json (auto-detected if not specified)"
    )

    args = parser.parse_args()

    root_dir = args.root.resolve()

    # Find compile commands
    compile_db = None
    if args.use_clang:
        if args.compile_commands:
            compile_commands_path = args.compile_commands
        else:
            compile_commands_path = find_compile_commands(root_dir)

        if compile_commands_path:
            compile_db = load_compile_commands(compile_commands_path)
            if compile_db:
                print(f"Loaded compile commands from {compile_commands_path}", file=sys.stderr)
            else:
                print("Warning: Failed to load compile commands, falling back to regex", file=sys.stderr)
                args.use_clang = False
        else:
            print("Warning: compile_commands.json not found, falling back to regex", file=sys.stderr)
            args.use_clang = False

    # Find all device operation files
    print("Finding device operation files...", file=sys.stderr)
    device_op_files = find_device_operation_files(root_dir)
    print(f"Found {len(device_op_files)} device operation files", file=sys.stderr)

    # Process each file
    results = []
    for i, device_op_file in enumerate(device_op_files, 1):
        if i % 50 == 0:
            print(f"Processing {i}/{len(device_op_files)}...", file=sys.stderr)

        result = process_file(device_op_file, compile_db, root_dir, args.use_clang)
        if result:
            results.append(result)

    # Save results
    output_path = args.output
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nFound {len(results)} matches", file=sys.stderr)
    print(f"Results saved to {output_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
