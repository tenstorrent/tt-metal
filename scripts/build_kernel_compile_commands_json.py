#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def check_bear_exists():
    """Check if bear command exists in the system."""
    try:
        subprocess.run(["which", "bear"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except subprocess.CalledProcessError:
        return False


def find_kernel_files(start_dir, kernel_name):
    """Find all files matching <kernel_name>.cpp recursively from start_dir."""
    results = []
    for root, dirs, files in os.walk(start_dir, topdown=True):
        # Remove build directories from the search
        dirs[:] = [d for d in dirs if not d.startswith("build_")]

        for file in files:
            if file == f"{kernel_name}.cpp":
                results.append(os.path.join(root, file))
    return results


def choose_kernel_file(kernel_files, choose_first=False):
    """Let user choose which kernel file to use or pick first if choose_first flag is set."""
    if not kernel_files:
        return None

    if choose_first or len(kernel_files) == 1:
        return kernel_files[0]

    print(f"Multiple kernel files found. Please choose one:")
    for i, path in enumerate(kernel_files):
        print(f"[{i}] {path}")

    choice = input("Enter number: ")
    try:
        index = int(choice)
        if 0 <= index < len(kernel_files):
            return kernel_files[index]
    except ValueError:
        pass

    print("Invalid choice, using first option.")
    return kernel_files[0]


def merge_compile_commands(original, new_entries):
    """Merge two compile_commands.json files."""
    # Create a set of file paths that exist in the original
    original_files = {entry.get("file") for entry in original}

    # Add entries from new_entries that don't exist in original
    for entry in new_entries:
        if entry.get("file") not in original_files:
            original.append(entry)

    return original


def update_command_path(arguments, old_path, new_path):
    """Update a command argument list by replacing old_path with new_path."""
    result = list(arguments)  # Create a copy

    # Find the index of the old path
    for i, arg in enumerate(result):
        if arg == old_path or os.path.basename(arg) == os.path.basename(old_path):
            result[i] = new_path
            return result

    # If we didn't find an exact match, try replacing the last argument
    if result:
        result[-1] = new_path

    return result


def process_compile_commands(json_path, search_dir, choose_first_kernel_file=False):
    """Process and modify the compile_commands.json file."""
    with open(json_path, "r") as f:
        compile_commands = json.load(f)

    # Replace compiler with clang++-17 for all entries
    for entry in compile_commands:
        # For entries with arguments list
        if "arguments" in entry and entry["arguments"]:
            entry["arguments"][0] = "/usr/bin/clang++-17"
        # For entries with command string
        elif "command" in entry:
            cmd = entry["command"]
            # Split by spaces respecting quotes
            parts = []
            current = ""
            in_quotes = False
            quote_char = None

            for char in cmd:
                if char in ['"', "'"]:
                    if not in_quotes:
                        in_quotes = True
                        quote_char = char
                        current += char
                    elif quote_char == char:
                        in_quotes = False
                        quote_char = None
                        current += char
                    else:
                        # Quote character inside other quotes
                        current += char
                elif char.isspace() and not in_quotes:
                    if current:
                        parts.append(current)
                        current = ""
                else:
                    current += char

            if current:
                parts.append(current)

            # Replace first part (compiler) and rebuild command
            if parts:
                parts[0] = "/usr/bin/clang++-17"
                entry["command"] = " ".join(parts)

    # Process each entry for kernel paths...
    for entry in compile_commands:
        directory = entry.get("directory", "")

        # Check if this is a kernel directory
        # Match pattern like .cache/tt-metal-cache/.../kernels/kernel_name/...
        kernel_dir_match = re.search(r"\.cache/tt-metal-cache/[^/]+/\d+/kernels/([^/]+)/", directory)
        if not kernel_dir_match:
            # Not a kernel directory, leave as is
            continue

        kernel_name = kernel_dir_match.group(1)

        # Find kernel files
        kernel_files = find_kernel_files(search_dir, kernel_name)

        if not kernel_files:
            print(f"Warning: No kernel files found for {kernel_name}, keeping original paths.")
            continue

        # Let user choose or select first
        chosen_file = choose_kernel_file(kernel_files, choose_first_kernel_file)

        # Update the file path
        if chosen_file:
            abs_path = os.path.abspath(chosen_file)
            original_file = entry.get("file", "")

            # Update the file field
            entry["file"] = abs_path

            # Update the command arguments
            if "arguments" in entry:
                entry["arguments"] = update_command_path(entry["arguments"], original_file, abs_path)
            elif "command" in entry:
                # Split the command by spaces, respecting quotes
                cmd = entry["command"]
                parts = []
                current = ""
                in_quotes = False
                quote_char = None

                for char in cmd:
                    if char in ['"', "'"]:
                        if not in_quotes:
                            in_quotes = True
                            quote_char = char
                            current += char
                        elif quote_char == char:
                            in_quotes = False
                            quote_char = None
                            current += char
                        else:
                            # Quote character inside other quotes
                            current += char
                    elif char.isspace() and not in_quotes:
                        if current:
                            parts.append(current)
                            current = ""
                    else:
                        current += char

                if current:
                    parts.append(current)

                # Update the command parts and rebuild the command
                updated_parts = update_command_path(
                    parts, original_file, f'"{abs_path}"' if " " in abs_path else abs_path
                )
                entry["command"] = " ".join(updated_parts)

    # Remove duplicate entries for the same source file
    unique_entries = {}
    for entry in compile_commands:
        file_path = entry.get("file", "")
        if file_path:
            # Keep the entry with the most complete information
            if (
                file_path not in unique_entries
                or ("arguments" in entry and "arguments" not in unique_entries[file_path])
                or (
                    "command" in entry
                    and "command" not in unique_entries[file_path]
                    and "arguments" not in unique_entries[file_path]
                )
            ):
                unique_entries[file_path] = entry

    compile_commands = list(unique_entries.values())

    return compile_commands


def main():
    parser = argparse.ArgumentParser(description="Process compile_commands.json for TT Metal kernels")
    parser.add_argument("--input-command", required=True, help="The command to run with a bear")
    parser.add_argument("--output-dir", required=True, help="Directory to store the updated compile_commands.json")
    parser.add_argument(
        "--choose-first-kernel-file", action="store_true", help="Automatically choose the first kernel file found"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing compile_commands.json without asking"
    )
    parser.add_argument("--merge", action="store_true", help="Merge with existing compile_commands.json without asking")
    parser.add_argument(
        "--search-dir", default=os.getcwd(), help="Directory to search for kernel files (default: current directory)"
    )

    args = parser.parse_args()

    # Convert search_dir to absolute path
    search_dir = os.path.abspath(args.search_dir)

    # Check if bear exists
    if not check_bear_exists():
        print("Error: 'bear' command not found. Please install it first.", file=sys.stderr)
        sys.exit(1)

    # Remember the original working directory
    original_dir = os.getcwd()

    # Create temp directory
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Run bear with the provided command
            print(f"Running bear in temporary directory: {temp_dir}")
            os.chdir(temp_dir)

            bear_cmd = f"bear -- {args.input_command}"
            print(f"Executing: {bear_cmd}")
            subprocess.run(bear_cmd, shell=True, check=True)

            # Check if compile_commands.json was generated
            if not os.path.exists("compile_commands.json"):
                print("Error: bear didn't generate compile_commands.json", file=sys.stderr)
                sys.exit(1)

            # Process the compile_commands.json
            updated_commands = process_compile_commands(
                "compile_commands.json", search_dir, args.choose_first_kernel_file
            )

            # Change back to original directory
            os.chdir(original_dir)

            # Prepare output directory
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "compile_commands.json"

            # Check if output file already exists
            if output_path.exists():
                if args.overwrite:
                    action = "overwrite"
                elif args.merge:
                    action = "merge"
                else:
                    response = input(f"{output_path} already exists. Overwrite or merge? [o/m]: ").lower()
                    action = "overwrite" if response.startswith("o") else "merge"

                if action == "merge":
                    # Merge with existing file
                    with open(output_path, "r") as f:
                        existing_commands = json.load(f)

                    merged_commands = merge_compile_commands(existing_commands, updated_commands)
                    updated_commands = merged_commands

            # Write the updated commands
            with open(output_path, "w") as f:
                json.dump(updated_commands, f, indent=2)

            print(f"Updated compile_commands.json saved to {output_path}")

        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        finally:
            # Always make sure we return to the original directory
            os.chdir(original_dir)


if __name__ == "__main__":
    main()
