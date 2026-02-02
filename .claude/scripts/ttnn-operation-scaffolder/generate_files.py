#!/usr/bin/env python3
"""
TTNN Operation File Generator

Generates all scaffolding files from JSON config using Jinja2 templates.
This script is purely deterministic - no LLM calls.
"""

import sys
import json
import os
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, Template


def validate_config(config: dict) -> list:
    """
    Validate config for common mistakes.

    Returns list of warning messages (empty if no issues).
    """
    warnings = []

    # Check for memory_config in parameters (common mistake - template adds it automatically)
    if any(p.get("name") == "memory_config" for p in config.get("parameters", [])):
        warnings.append(
            "WARNING: 'memory_config' found in parameters array. "
            "This will cause build errors because the template automatically adds memory_config. "
            "Remove 'memory_config' from the parameters array."
        )

    # Check namespace format
    namespace = config.get("namespace", "")
    operation_name = config.get("operation_name", "")
    category = config.get("category", "")
    expected_namespace = f"ttnn::operations::{category}::{operation_name}"
    if namespace and namespace != expected_namespace:
        warnings.append(
            f"NOTE: namespace '{namespace}' differs from expected pattern '{expected_namespace}'. "
            f"This may be intentional, but verify it matches your intended structure."
        )

    # Check output_shape has required fields
    output_shape = config.get("output_shape", {})
    if "cpp_code" not in output_shape:
        warnings.append(
            "WARNING: output_shape.cpp_code is missing. "
            "This field should contain C++ code that declares 'ttnn::Shape output_shape'."
        )
    elif "output_shape" not in output_shape.get("cpp_code", ""):
        warnings.append(
            "WARNING: output_shape.cpp_code should declare a variable named 'output_shape'. "
            f"Got: {output_shape.get('cpp_code', '')[:50]}..."
        )

    return warnings


def generate_files(config_path: str, repo_root: str = None, check_existing: bool = True) -> dict:
    """
    Generate all scaffolding files from config.

    Args:
        config_path: Path to JSON config file
        repo_root: Repository root directory
        check_existing: If True, check for existing files and warn

    Returns:
        dict with 'created' and 'skipped' file lists
    """
    # Load config
    with open(config_path, "r") as f:
        config = json.load(f)

    # Validate config for common mistakes
    validation_warnings = validate_config(config)
    if validation_warnings:
        print("\n" + "=" * 60)
        print("CONFIG VALIDATION WARNINGS")
        print("=" * 60)
        for warning in validation_warnings:
            print(f"\n{warning}")
        print("=" * 60 + "\n")

        # If memory_config is in parameters, this is a critical error that will cause build failures
        if any("memory_config" in w and "build errors" in w for w in validation_warnings):
            print("ERROR: Critical issue detected. Fix the config before proceeding.", file=sys.stderr)
            return {"created": [], "skipped": [], "error": "memory_config in parameters"}

    # Determine repo root
    # Priority: .git directory (most reliable), then build_metal.sh (tt-metal specific)
    # Avoid using just 'ttnn' as marker since ttnn/cpp/ttnn/ exists and causes false matches
    repo_root_explicit = repo_root is not None
    if repo_root is None:
        current = Path(config_path).resolve()
        while current != current.parent:
            # .git is the most reliable marker for repo root
            if (current / ".git").exists():
                repo_root = str(current)
                break
            # build_metal.sh is tt-metal specific and lives at repo root
            if (current / "build_metal.sh").exists():
                repo_root = str(current)
                break
            current = current.parent
        if repo_root is None:
            repo_root = os.getcwd()
            print(f"Warning: Could not find .git or build_metal.sh, using cwd: {repo_root}", file=sys.stderr)
        else:
            print(f"Note: Auto-detected repo_root: {repo_root}")
    else:
        print(f"Using explicit repo_root: {repo_root}")

    repo_root = Path(repo_root)

    # Setup Jinja2 environment
    script_dir = Path(__file__).parent
    template_dir = script_dir / "templates"
    env = Environment(loader=FileSystemLoader(template_dir), trim_blocks=True, lstrip_blocks=True)

    # Determine operation directory
    operation_path = repo_root / config["operation_path"]
    device_path = operation_path / "device"
    test_dev_path = operation_path / "test_dev"

    # Create directories if they don't exist
    operation_path.mkdir(parents=True, exist_ok=True)
    device_path.mkdir(parents=True, exist_ok=True)
    test_dev_path.mkdir(parents=True, exist_ok=True)

    # Define file mappings: (template_name, output_path)
    operation_name = config["operation_name"]
    files_to_generate = [
        # Implementation files (9)
        ("device_operation_types.hpp.j2", device_path / f"{operation_name}_device_operation_types.hpp"),
        ("device_operation.hpp.j2", device_path / f"{operation_name}_device_operation.hpp"),
        ("device_operation.cpp.j2", device_path / f"{operation_name}_device_operation.cpp"),
        ("program_factory.hpp.j2", device_path / f"{operation_name}_program_factory.hpp"),
        ("program_factory.cpp.j2", device_path / f"{operation_name}_program_factory.cpp"),
        ("operation.hpp.j2", operation_path / f"{operation_name}.hpp"),
        ("operation.cpp.j2", operation_path / f"{operation_name}.cpp"),
        ("nanobind.hpp.j2", operation_path / f"{operation_name}_nanobind.hpp"),
        ("nanobind.cpp.j2", operation_path / f"{operation_name}_nanobind.cpp"),
        # Test files (3) - Stage 1-3 verification tests
        ("test_stage1_api_exists.py.j2", test_dev_path / "test_stage1_api_exists.py"),
        ("test_stage2_validation.py.j2", test_dev_path / "test_stage2_validation.py"),
        ("test_stage3_registration.py.j2", test_dev_path / "test_stage3_registration.py"),
    ]

    created_files = []
    skipped_files = []

    # Generate each file
    for template_name, output_path in files_to_generate:
        # Check if file exists
        if check_existing and output_path.exists():
            print(f"Warning: File already exists, skipping: {output_path}")
            skipped_files.append(str(output_path))
            continue

        # Load template
        template = env.get_template(template_name)

        # Render template
        rendered = template.render(**config)

        # Write file
        with open(output_path, "w") as f:
            f.write(rendered)

        print(f"Created: {output_path}")
        created_files.append(str(output_path))

    return {
        "created": created_files,
        "skipped": skipped_files,
        "operation_path": str(operation_path),
        "operation_name": operation_name,
        "category": config["category"],
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: generate_files.py <config.json> [repo_root] [--force]")
        print("Example: generate_files.py my_operation_scaffolding_config.json")
        print("Options:")
        print("  --force: Overwrite existing files")
        sys.exit(1)

    config_path = sys.argv[1]
    repo_root = None
    check_existing = True

    # Parse remaining args
    for arg in sys.argv[2:]:
        if arg == "--force":
            check_existing = False
        elif repo_root is None:
            repo_root = arg

    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    result = generate_files(config_path, repo_root, check_existing)

    print(f"\nGeneration complete!")
    print(f"Created: {len(result['created'])} files")
    if result["skipped"]:
        print(f"Skipped: {len(result['skipped'])} files (already exist)")

    print_summary(result)


def print_summary(result: dict):
    """Print a summary of generated files."""
    print(f"\n{'='*60}")
    print(f"SCAFFOLDING SUMMARY")
    print(f"{'='*60}")
    print(f"Operation: {result['operation_name']}")
    print(f"Category: {result['category']}")
    print(f"Path: {result['operation_path']}")
    print(f"\nImplementation files: 9")
    print(f"Test files (test_dev/): 3")
    print(f"Total created: {len(result['created'])}")
    if result["skipped"]:
        print(f"Skipped (existing): {len(result['skipped'])}")
    print(f"\nNext steps:")
    print(f"  1. Build: ./build_metal.sh -b Debug")
    print(f"  2. Run Stage 1-3 tests:")
    print(f"     pytest {result['operation_path']}/test_dev/test_stage1_api_exists.py -v")
    print(f"     pytest {result['operation_path']}/test_dev/test_stage2_validation.py -v")
    print(f"     pytest {result['operation_path']}/test_dev/test_stage3_registration.py -v")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
