#!/usr/bin/env python3
import json
import shlex
import subprocess
from pathlib import Path
from datetime import datetime
import sys
import argparse
import os
import concurrent.futures

def load_compile_command(db_path: str, source_files: list[str]):
    """Find compile command for given source file in compile_commands.json"""
    db = json.loads(Path(db_path).read_text())
    source_paths = [Path(f).resolve() for f in source_files]
    results = []
    for entry in db:
        file_path = Path(entry["file"]).resolve()
        if file_path in source_paths:
            cmd_parts = entry.get("arguments")
            if not cmd_parts:
                cmd_parts = shlex.split(entry["command"])
            results.append((file_path, cmd_parts, Path(entry["directory"])))
    return results

def run_preprocessor(cmd_parts, workdir):
    """Run compiler in preprocess+include-report mode and return stderr"""
    # Remove any compile & output options, add -E -H
    filtered = []
    skip_next = False
    skip_next_two = False
    i = 0
    while i < len(cmd_parts):
        part = cmd_parts[i]
        
        if skip_next_two:
            # Skip this argument and the next one
            skip_next_two = False
            skip_next = True
            i += 1
            continue
            
        if skip_next:
            skip_next = False
            i += 1
            continue
            
        if part == "-c":
            i += 1
            continue
        if part == "-o":
            skip_next = True
            i += 1
            continue
        if part.startswith("-o"):
            i += 1
            continue
        # Remove precompiled header related arguments
        if part == "-Xclang" and i + 1 < len(cmd_parts):
            next_part = cmd_parts[i + 1]
            if next_part in ["-include-pch", "-include", "-fno-pch-timestamp"]:
                if next_part == "-include-pch":
                    # Skip "-Xclang -include-pch -Xclang <pch_file>"
                    skip_next_two = True
                elif next_part == "-include":
                    # Skip "-Xclang -include -Xclang <header_file>"  
                    skip_next_two = True
                else:
                    # Skip "-Xclang -fno-pch-timestamp"
                    skip_next = True
                i += 1
                continue
        # Remove -Winvalid-pch which is related to precompiled headers
        if part == "-Winvalid-pch":
            i += 1
            continue
        # Skip .pch files and .hxx files that are precompiled header related
        if part.endswith('.pch') or (part.endswith('.hxx') and 'cmake_pch' in part):
            i += 1
            continue
            
        filtered.append(part)
        i += 1
        
    filtered += ["-E", "-H", "-o", "/dev/null"]

    proc = subprocess.run(
        filtered,
        cwd=workdir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True
    )
    return proc.stderr.splitlines()

def parse_include_lines(lines):
    stack = []
    results = []
    for line in lines:
        if not line.strip():
            continue
        # Ignore possible compiler warnings / extra lines
        if not line.lstrip().startswith('.'):
            continue
        stripped = line.lstrip('.').strip()
        stripped = os.path.abspath(stripped)
        depth = len(line) - len(line.lstrip('.'))
        if depth == 1:
            stack = [stripped]
        else:
            # Ensure stack is correct depth
            while len(stack) >= depth:
                stack.pop()
            stack.append(stripped)
        results.append((depth, stripped, stack[-2] if len(stack) > 1 else None, list(stack[:-1])))
    return results

def parse_include_tree(parsed_lines):
    """Parse the -H output into a nested list structure"""
    stack = []
    root = []
    for depth, filename, parent in parsed_lines:
        node = {"file": filename, "children": []}
        if depth == 1:
            root.append(node)
            stack = [node]
        else:
            # Ensure stack is correct depth
            while len(stack) >= depth:
                stack.pop()
            stack[-1]["children"].append(node)
            stack.append(node)
    return root

def print_tree(tree, indent=0):
    """Pretty-print include hierarchy tree"""
    for node in tree:
        print("  " * indent + node["file"])
        print_tree(node["children"], indent + 1)

def convert_include_file_to_pch_path(include_file: str) -> str:
    if include_file.startswith('/usr/') and 'include/c++' in include_file:
        return os.path.basename(include_file)
    if include_file.startswith('/opt/openmpi'):
        return os.path.basename(include_file)
    if "enchantum/include" in include_file:
        return f"enchantum/{os.path.basename(include_file)}"
    if "include/flatbuffers" in include_file:
        return f"flatbuffers/{os.path.basename(include_file)}"
    if "include/yaml-cpp" in include_file:
        return f"yaml-cpp/{os.path.basename(include_file)}"
    if "include/nlohmann" in include_file:
        return f"nlohmann/{os.path.basename(include_file)}"
    if include_file.startswith("/usr/lib/llvm-17/lib/clang/17/include/"):
        return os.path.basename(include_file)
    if "tt_metal/third_party/tracy/public/" in include_file:
        return include_file.split("tt_metal/third_party/tracy/public/")[1]
    if "third_party/umd/device/api/umd/" in include_file:
        return include_file.split("third_party/umd/device/api/")[1]
    if "/simde/" in include_file:
        return "simde/" + include_file.split("/simde/")[1]
    if "tt_metal/common/" in include_file:
        return "tt_metal/common/" + include_file.split("tt_metal/common/")[1]
    if "tt_metal/distributed/" in include_file:
        return "tt_metal/distributed/" + include_file.split("tt_metal/distributed/")[1]
    if "tt_metal/fabric/" in include_file:
        return "tt_metal/fabric/" + include_file.split("tt_metal/fabric/")[1]
    if "tt_metal/jit_build/" in include_file:
        return "tt_metal/jit_build/" + include_file.split("tt_metal/jit_build/")[1]
    if "tt_metal/hw/" in include_file:
        return "tt_metal/hw/" + include_file.split("tt_metal/hw/")[1]
    if "tt_metal/llrt/" in include_file:
        return "tt_metal/llrt/" + include_file.split("tt_metal/llrt/")[1]
    if "tt_metal/impl/" in include_file:
        return "tt_metal/impl/" + include_file.split("tt_metal/impl/")[1]
    if "tt_metal/lite_fabric/" in include_file:
        return "tt_metal/lite_fabric/" + include_file.split("tt_metal/lite_fabric/")[1]
    if "google/protobuf" in include_file:
        return "google/protobuf/" + include_file.split("google/protobuf/")[1]
    if "include/xtensor/" in include_file:
        return "xtensor/" + include_file.split("include/xtensor/")[1]
    if "include/pybind11" in include_file:
        return "pybind11/" + include_file.split("include/pybind11/")[1]
    if "include/fmt/" in include_file:
        return "fmt/" + include_file.split("include/fmt/")[1]
    if "tt_metal/api/tt-metalium" in include_file:
        return "tt-metalium/" + include_file.split("tt_metal/api/tt-metalium/")[1]
    if "tt_metal/hostdevcommon/api/" in include_file:
        return include_file.split("tt_metal/hostdevcommon/api/")[1]
    if "/usr/include/x86_64-linux-gnu/" in include_file:
        return include_file.split("/usr/include/x86_64-linux-gnu/")[1]
    if "tt_stl/tt_stl/" in include_file:
        return "tt_stl/" + include_file.split("tt_stl/tt_stl/")[1]
    return include_file

def write_pch_file(output, include_files):
    output.write(f"// SPDX-FileCopyrightText: © {datetime.now().year} Tenstorrent Inc.\n")
    output.write("//\n")
    output.write("// SPDX-License-Identifier: Apache-2.0\n\n")
    output.write("#pragma once\n\n")
    for filename in include_files:
        output.write(f"#include <{filename}>\n")

def generate_pch(db_path: str, source_files: list[str], avoid_includes: list[str], pch_file: str | None, verbose: bool = False):
    compile_commands = load_compile_command(db_path, source_files)
    if not compile_commands:
        print(f"Nothing to do. Check list of source files.", file=sys.stderr)
        sys.exit(1)

    everything = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        def process_compile_commands(args):
            file_path, cmd_parts, workdir = args
            stderr_lines = run_preprocessor(cmd_parts, workdir)
            parsed_lines = parse_include_lines(stderr_lines)
            return (file_path, parsed_lines)

        everything = list(executor.map(process_compile_commands, compile_commands))

    def is_avoid_include(path: str) -> bool:
        for avoid in avoid_includes:
            if path.startswith(avoid):
                return True
        for avoid in source_files:
            if path.startswith(avoid):
                return True
        return False

    all_includes = {}
    for _, parsed_lines in everything:
        for depth, filename, parent, include_tree in parsed_lines:
            if filename not in all_includes:
                all_includes[filename] = {"first_level" : 0, "others" : 0, "first_level_parents": [], "other_parents": []}
            if parent and is_avoid_include(parent):
                depth = 1
            if depth == 1:
                all_includes[filename]["first_level"] += 1
                all_includes[filename]["first_level_parents"].append(parent)
            else:
                all_includes[filename]["others"] += 1
                all_includes[filename]["other_parents"].append((parent, include_tree))

        # tree = parse_include_tree(parsed_lines)
        # print("** Include tree for:", file_path)
        # print_tree(tree)

    if verbose:
        print("Used source files:")
        for file_path, _ in everything:
            print(f"  {file_path}")
        print()

        print("Top-level includes:")
        for filename in sorted(all_includes.keys()):
            counts = all_includes[filename]
            if counts['first_level'] > 0:
                print(f"{counts['first_level']:3} {counts['others']:3}  {filename}")
        print()

    potential_pch = set()
    for filename in all_includes.keys():
        counts = all_includes[filename]
        if counts['first_level'] > 1 and not is_avoid_include(filename):
            potential_pch.add(filename)
    for filename in all_includes.keys():
        counts = all_includes[filename]
        if counts['first_level'] == 1 and counts['others'] > 0 and not is_avoid_include(filename):
            for parent, tree in counts['other_parents']:
                add = True
                for ancestor in tree:
                    if ancestor in potential_pch:
                        add = False
                        break
                if add:
                    potential_pch.add(filename)
                    break

    if verbose:
        print("Potential PCH includes:")
        for filename in sorted(potential_pch):
            counts = all_includes[filename]
            print(f"{counts['first_level']:3} {counts['others']:3}  {filename}")
        print()

    filtered_pch = set()
    for filename in potential_pch:
        counts = all_includes[filename]
        add = True
        for parent, tree in counts['other_parents']:
            for ancestor in tree:
                if ancestor in potential_pch:
                    add = False
                    break
            if not add:
                break
        if add:
            filtered_pch.add(filename)

    if verbose:
        print("Filtered PCH includes (removing those included by others):")
        for filename in sorted(filtered_pch):
            counts = all_includes[filename]
            print(f"{counts['first_level']:3} {counts['others']:3}  {filename}")
        print()

    pch_files = sorted(map(convert_include_file_to_pch_path, filtered_pch))
    if verbose:
        print("PCH file:")
        print()
        write_pch_file(sys.stdout, pch_files)
    if pch_file:
        with open(pch_file, "w") as f:
            write_pch_file(f, pch_files)
    else:
        write_pch_file(sys.stdout, pch_files)

    if verbose:
        print("All includes (top-level and others):")
        for filename in sorted(all_includes.keys()):
            counts = all_includes[filename]
            print(f"{counts['first_level']:3} {counts['others']:3}  {filename}")
        print()

def main():
    parser = argparse.ArgumentParser(description="Show C++ include hierarchy from compile_commands.json")
    parser.add_argument("source", help="Path to the .cpp files to inspect comma-separated")
    parser.add_argument("-a", "--avoid-includes", default="", help="Path to the output directories")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("-o", "--output", default="", help="Path to the output file to write PCH includes. Defaults to console output.")
    parser.add_argument("-p", "--compdb", default="compile_commands.json", help="Path to compile_commands.json (default: ./compile_commands.json)")
    args = parser.parse_args()
    generate_pch(
        args.compdb,
        args.source.split(","),
        avoid_includes=args.avoid_includes.split(",") if args.avoid_includes else [],
        pch_file=args.output if args.output else None,
        verbose=args.verbose
    )

if __name__ == "__main__":
    main()
