#!/usr/bin/env python3
"""
Migrate compute_kernel_api from tt_metal/include to tt_metal/hw/inc/api/compute.

This script:
1. Moves all files from tt_metal/include/* to tt_metal/hw/inc/api/compute/
2. Updates all #include directives in device code to use new paths
3. Updates any internal relative includes within the moved headers

Usage:
    python3 include_scripts/migrate_compute_kernel_api.py [--dry-run]
"""

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path


def move_files(repo_root: Path, dry_run: bool = False) -> dict:
    """
    Move files from tt_metal/include to tt_metal/hw/inc/api/compute.
    Returns mapping of old_include_path -> new_include_path.

    IMPORTANT: The mapping includes BOTH styles of include paths:
    - Relative style (relies on -I): "compute_kernel_api/x.h" -> "api/compute/x.h"
    - Full path style (relies on -I .): "tt_metal/include/compute_kernel_api/x.h" -> "api/compute/x.h"
    """
    src_dir = repo_root / "tt_metal/include"
    dst_dir = repo_root / "tt_metal/hw/inc/api/compute"

    moves = {}

    if not src_dir.exists():
        print(f"ERROR: Source directory not found: {src_dir}", file=sys.stderr)
        return moves

    # Walk the source directory
    for item in src_dir.rglob('*'):
        if item.is_file():
            rel_path = item.relative_to(src_dir)

            # Remove "compute_kernel_api/" prefix if present
            rel_parts = rel_path.parts
            if rel_parts[0] == "compute_kernel_api":
                # Strip the "compute_kernel_api" prefix
                new_rel_path = Path(*rel_parts[1:]) if len(rel_parts) > 1 else rel_path
            else:
                new_rel_path = rel_path

            new_path = dst_dir / new_rel_path

            # Store the mapping for include updates
            # Add BOTH relative and full-path mappings to handle inconsistent include styles
            old_include = str(rel_path)
            new_include = f"api/compute/{new_rel_path}"
            moves[old_include] = new_include
            # Also add full project-relative path (e.g., "tt_metal/include/compute_kernel_api/x.h")
            full_path_include = f"tt_metal/include/{rel_path}"
            moves[full_path_include] = new_include

            if dry_run:
                print(f"WOULD MOVE: {item} -> {new_path}")
            else:
                # Create parent directory if needed
                new_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(item), str(new_path))
                print(f"Moved: {old_include} -> {new_include}")

    return moves


def update_includes_in_file(filepath: Path, include_mapping: dict, dry_run: bool = False) -> int:
    """
    Update #include directives in a file.
    Returns number of changes made.
    """
    try:
        content = filepath.read_text(encoding='utf-8', errors='replace')
    except Exception as e:
        print(f"Warning: Could not read {filepath}: {e}", file=sys.stderr)
        return 0

    original_content = content
    changes = 0

    # Update includes
    for old_path, new_path in include_mapping.items():
        # Match both angle brackets and quotes
        pattern1 = f'#include\\s*<{re.escape(old_path)}>'
        pattern2 = f'#include\\s*"{re.escape(old_path)}"'

        replacement1 = f'#include <{new_path}>'
        replacement2 = f'#include "{new_path}"'

        new_content = re.sub(pattern1, replacement1, content)
        if new_content != content:
            changes += content.count(old_path)
            content = new_content

        new_content = re.sub(pattern2, replacement2, content)
        if new_content != content:
            changes += content.count(old_path)
            content = new_content

    if content != original_content:
        if dry_run:
            print(f"WOULD UPDATE: {filepath} ({changes} includes)")
        else:
            filepath.write_text(content, encoding='utf-8')
            print(f"Updated: {filepath} ({changes} includes)")
        return changes

    return 0


def update_all_includes(repo_root: Path, include_mapping: dict, dry_run: bool = False):
    """
    Update all #include directives in the repository.
    """
    exclude_dirs = ["build_", ".git", "__pycache__"]
    total_files = 0
    total_changes = 0

    print("\nUpdating #include directives...", file=sys.stderr)

    for dirpath, dirnames, filenames in os.walk(repo_root):
        # Skip excluded directories
        dirnames[:] = [d for d in dirnames if not any(excl in d for excl in exclude_dirs)]

        for filename in filenames:
            if not (filename.endswith(('.cpp', '.cc', '.h', '.hpp'))):
                continue

            filepath = Path(dirpath) / filename
            changes = update_includes_in_file(filepath, include_mapping, dry_run)
            if changes > 0:
                total_files += 1
                total_changes += changes

    print(f"\nSummary: Updated {total_changes} includes in {total_files} files", file=sys.stderr)


def update_path_references(repo_root: Path, dry_run: bool = False):
    """
    Update path references in non-source files (docs, CODEOWNERS, etc.).
    These are plain path strings, not #include directives.
    """
    # Path mappings for non-include references
    path_mappings = [
        # Directory paths
        ("tt_metal/include/compute_kernel_api/", "tt_metal/hw/inc/api/compute/"),
        ("tt_metal/include/compute_kernel_api", "tt_metal/hw/inc/api/compute"),
        # Top-level header
        ("tt_metal/include/compute_kernel_api.h", "tt_metal/hw/inc/api/compute/compute_kernel_api.h"),
    ]

    # Files to update
    files_to_check = [
        repo_root / "docs/Doxyfile",
        repo_root / ".github/CODEOWNERS",
    ]

    # Also find .rst files in docs
    docs_dir = repo_root / "docs"
    if docs_dir.exists():
        files_to_check.extend(docs_dir.rglob("*.rst"))

    total_files = 0
    print("\nUpdating path references in docs/config files...", file=sys.stderr)

    for filepath in files_to_check:
        if not filepath.exists():
            continue

        try:
            content = filepath.read_text(encoding='utf-8', errors='replace')
        except Exception as e:
            print(f"Warning: Could not read {filepath}: {e}", file=sys.stderr)
            continue

        original_content = content

        for old_path, new_path in path_mappings:
            content = content.replace(old_path, new_path)

        if content != original_content:
            if dry_run:
                print(f"WOULD UPDATE: {filepath}")
            else:
                filepath.write_text(content, encoding='utf-8')
                print(f"Updated: {filepath}")
            total_files += 1

    print(f"\nSummary: Updated {total_files} config/doc files", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    parser.add_argument(
        "repo_root",
        nargs="?",
        help="Repository root directory (default: parent of script directory)"
    )
    args = parser.parse_args()

    # Determine repo root
    if args.repo_root:
        repo_root = Path(args.repo_root).resolve()
    else:
        repo_root = Path(__file__).parent.parent.resolve()

    print(f"Repository root: {repo_root}", file=sys.stderr)
    print(f"Dry run: {args.dry_run}", file=sys.stderr)

    # Step 1: Move files and build mapping
    print("\n=== Step 1: Moving files ===", file=sys.stderr)
    include_mapping = move_files(repo_root, args.dry_run)

    if not include_mapping:
        print("ERROR: No files to move. Exiting.", file=sys.stderr)
        sys.exit(1)

    print(f"\nBuilt mapping for {len(include_mapping)} headers", file=sys.stderr)

    # Step 2: Update all includes
    print("\n=== Step 2: Updating includes ===", file=sys.stderr)
    update_all_includes(repo_root, include_mapping, args.dry_run)

    # Step 3: Update path references in docs/config files
    print("\n=== Step 3: Updating docs/config files ===", file=sys.stderr)
    update_path_references(repo_root, args.dry_run)

    if args.dry_run:
        print("\n=== DRY RUN COMPLETE ===", file=sys.stderr)
        print("Run without --dry-run to apply changes", file=sys.stderr)
    else:
        print("\n=== MIGRATION COMPLETE ===", file=sys.stderr)
        print("Next step: Remove 'root_ + \"tt_metal/include\"' from build.cpp", file=sys.stderr)


if __name__ == "__main__":
    main()
