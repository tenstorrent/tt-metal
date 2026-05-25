"""Probe: parse a single TU with libclang and report diagnostics.

Used to discover which generated headers are missing before we run codegen.
Usage:
    python probe_parse.py <path/to/source.cpp> [--db /workspace/build_Release/compile_commands.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Pin to the system libclang-20 so the parser matches the compiler used in the build.
from clang import cindex

SYSTEM_LIBCLANG = "/usr/lib/x86_64-linux-gnu/libclang-20.so.1"
if Path(SYSTEM_LIBCLANG).exists():
    cindex.Config.set_library_file(SYSTEM_LIBCLANG)


def load_compile_commands(db_path: Path) -> dict[str, dict]:
    raw = json.loads(db_path.read_text())
    out: dict[str, dict] = {}
    for entry in raw:
        out[os.path.normpath(entry["file"])] = entry
    return out


def parse_one(tu_path: Path, db: dict[str, dict]) -> None:
    key = os.path.normpath(str(tu_path))
    if key not in db:
        print(f"FATAL: {tu_path} not in compile_commands.json", file=sys.stderr)
        sys.exit(2)
    entry = db[key]

    # Reconstruct the argv. libclang wants the args without the compiler name and without the source file.
    if "arguments" in entry:
        argv = list(entry["arguments"])
    else:
        # "command" form — split crudely; tt-metal uses ninja which emits "arguments" usually.
        import shlex
        argv = shlex.split(entry["command"])

    # Drop the compiler (argv[0]) and the source file (last occurrence of the tu path).
    if argv and argv[0].endswith(("clang++", "clang++-20", "clang", "gcc", "g++", "g++-12")):
        argv = argv[1:]
    argv = [a for a in argv if os.path.normpath(a) != key]
    # Walk pruning args. We need to drop:
    #   -o <file>      (linker output, libclang doesn't care)
    #   -c             (it's implied)
    #   -Xclang -include-pch -Xclang <path.pch>   (libclang 18 can't read clang-20 PCH)
    #   -Xclang -include -Xclang <path.hxx>       (the corresponding pch header re-include)
    #   -Winvalid-pch                              (only meaningful with PCH)
    pruned: list[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        # -o <out>
        if a == "-o":
            i += 2
            continue
        if a.startswith("-o") and a != "-o":
            i += 1
            continue
        if a == "-c":
            i += 1
            continue
        # PCH triple-Xclang sequences (libclang 18 vs clang-20 mismatch)
        if a == "-Xclang" and i + 1 < len(argv) and argv[i + 1] in ("-include-pch", "-include"):
            # Pattern: -Xclang <flag> -Xclang <path>
            # Strip 4 tokens. Special-case: only strip "-include" if its arg ends in .hxx (cmake_pch header).
            if argv[i + 1] == "-include":
                pch_arg_idx = i + 3
                if pch_arg_idx < len(argv) and argv[pch_arg_idx].endswith((".hxx", ".hpp", ".h")) and "pch" in argv[pch_arg_idx].lower():
                    i += 4
                    continue
                # otherwise let it through
            else:
                i += 4
                continue
        if a == "-Winvalid-pch":
            i += 1
            continue
        pruned.append(a)
        i += 1

    # Run from the entry's `directory` so relative -I flags resolve correctly.
    cwd = entry.get("directory", "/workspace/build_Release")
    saved_cwd = os.getcwd()
    os.chdir(cwd)
    try:
        index = cindex.Index.create()
        # Drop -Werror so the unused-warning escalations don't show up as errors.
        pruned = [a for a in pruned if a not in ("-Werror", "-pedantic-errors")]
        tu = index.parse(
            str(tu_path),
            args=pruned,
            options=cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD,
        )
    finally:
        os.chdir(saved_cwd)

    fatals = 0
    missing_includes: set[str] = set()
    other_errors: list[str] = []
    for d in tu.diagnostics:
        sev = d.severity
        sev_name = {0: "ignored", 1: "note", 2: "warning", 3: "error", 4: "fatal"}.get(sev, str(sev))
        msg = d.spelling
        if sev >= cindex.Diagnostic.Error:
            fatals += 1
            if "file not found" in msg or "no such file" in msg.lower():
                # e.g. "'foo/bar.h' file not found"
                import re
                m = re.search(r"'([^']+)'", msg)
                if m:
                    missing_includes.add(m.group(1))
                else:
                    other_errors.append(msg)
            else:
                other_errors.append(f"[{sev_name}] {msg} @ {d.location}")

    print(f"TU: {tu_path}")
    print(f"argv ({len(pruned)} args):")
    for a in pruned[:8]:
        print(f"  {a}")
    if len(pruned) > 8:
        print(f"  ... and {len(pruned)-8} more")
    print(f"Diagnostics: {fatals} errors/fatals")
    if missing_includes:
        print(f"\nMissing includes ({len(missing_includes)}):")
        for inc in sorted(missing_includes):
            print(f"  - {inc}")
    if other_errors:
        print(f"\nOther errors ({len(other_errors)}):")
        for e in other_errors[:30]:
            print(f"  {e}")
        if len(other_errors) > 30:
            print(f"  ... and {len(other_errors)-30} more")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tu", help="Translation unit path (absolute)")
    ap.add_argument("--db", default="/workspace/build_Release/compile_commands.json")
    args = ap.parse_args()
    db = load_compile_commands(Path(args.db))
    parse_one(Path(args.tu), db)


if __name__ == "__main__":
    main()
