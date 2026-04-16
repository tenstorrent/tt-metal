#!/usr/bin/env python3
"""Detect (and optionally fix) duplicate keys in named_compile_time_args lists.

NamedCompileTimeArgs is a std::vector, so duplicate keys silently pass Python
validation but generate duplicate static constexpr members in named_args_generated.h,
causing kernel build failures.

Usage:
  python scripts/fix_duplicate_ct_args.py [paths...] [--fix] [--check]

  --fix    Remove the second (and later) occurrence of each duplicate key in-place.
  --check  Exit with code 1 if any duplicates are found (useful in CI).

"""

import argparse
import glob
import re
import sys
from pathlib import Path


def find_duplicates(path: str) -> dict:
    """Return {list_name: {dup_key: "conflict"|"same"}} for all lists with duplicates."""
    src = open(path).read()
    issues = {}

    # Collect individual list definitions: foo_named_compile_time_args = [...]
    # Also track values so we can detect conflicting duplicates.
    individual: dict[str, list[str]] = {}
    individual_kv: dict[str, list[tuple[str, str]]] = {}
    for m in re.finditer(r"(\w+_named_compile_time_args)\s*=\s*\[(.*?)\]", src, re.DOTALL):
        name, body = m.group(1), m.group(2)
        # Extract (key, value_expr) pairs — value is everything after the comma up to the closing paren
        pairs = re.findall(r'\(\s*["\'](\w+)["\'],\s*([^)]+?)\s*\)', body)
        individual[name] = [k for k, _ in pairs]
        individual_kv[name] = pairs
        dups = _dup_report(pairs)
        if dups:
            issues[f"{name} (inline)"] = dups

    # Check combined lists: foo_named_compile_time_args_base = (a + b + ...)
    for m in re.finditer(r"(\w+_named_compile_time_args(?:_base)?)\s*=\s*\((.*?)\)", src, re.DOTALL):
        base_name, body = m.group(1), m.group(2)
        sublists = re.findall(r"(\w+_named_compile_time_args)\b", body)
        all_pairs: list[tuple[str, str]] = []
        for sl in sublists:
            all_pairs.extend(individual_kv.get(sl, []))
        dups = _dup_report(all_pairs)
        if dups:
            issues[f"{base_name} (combined)"] = dups

    return issues


def _dup_report(pairs: list[tuple[str, str]]) -> dict[str, str]:
    """Return {key: "same"|"CONFLICT"} for keys appearing more than once."""
    seen: dict[str, str] = {}
    result: dict[str, str] = {}
    for key, val in pairs:
        if key in seen:
            status = "same" if seen[key] == val else "CONFLICT"
            result[key] = status
        else:
            seen[key] = val
    return result


def fix_file(path: str) -> bool:
    """Remove duplicate keys from individual list definitions. Returns True if changed."""
    src = open(path).read()
    original = src

    def dedup_list_body(body: str) -> str:
        seen: set[str] = set()
        lines = body.split("\n")
        out = []
        for line in lines:
            m = re.search(r'\(\s*["\'](\w+)["\']', line)
            if m:
                key = m.group(1)
                if key in seen:
                    continue  # drop duplicate
                seen.add(key)
            out.append(line)
        return "\n".join(out)

    def replacer(m: re.Match) -> str:
        name, body = m.group(1), m.group(2)
        new_body = dedup_list_body(body)
        return f"{name} = [{new_body}]"

    src = re.sub(
        r"(\w+_named_compile_time_args)\s*=\s*\[(.*?)\]",
        replacer,
        src,
        flags=re.DOTALL,
    )

    if src != original:
        open(path, "w").write(src)
        return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="*", help="Files or glob patterns to check")
    parser.add_argument("--fix", action="store_true", help="Remove duplicate entries in-place")
    parser.add_argument("--check", action="store_true", help="Exit 1 if duplicates found")
    args = parser.parse_args()

    if not args.paths:
        parser.error("At least one file or glob pattern is required.")
    files = []
    for p in args.paths:
        files.extend(sorted(glob.glob(p, recursive=True))) if "*" in p else files.append(p)

    found_any = False
    for path in files:
        if not Path(path).is_file():
            continue

        if args.fix:
            issues_before = find_duplicates(path)
            conflicts = {ln: {k: s for k, s in dups.items() if s == "CONFLICT"} for ln, dups in issues_before.items()}
            conflicts = {ln: d for ln, d in conflicts.items() if d}
            if conflicts:
                print(f"\nWARNING: {path} has conflicting duplicates — skipping auto-fix, manual review required:")
                for list_name, dups in conflicts.items():
                    print(f"  {list_name}: {list(dups.keys())}")
                found_any = True
                continue
            changed = fix_file(path)
            if changed:
                print(f"Fixed: {path}")
            issues = find_duplicates(path)
        else:
            issues = find_duplicates(path)

        if issues:
            found_any = True
            print(f"\n{path}:")
            for list_name, dups in issues.items():
                for key, status in dups.items():
                    tag = " [CONFLICT - manual fix required]" if status == "CONFLICT" else ""
                    print(f"  {list_name}: {key!r}{tag}")

    if not found_any:
        print("All clean.")

    return 1 if (found_any and args.check) else 0


if __name__ == "__main__":
    sys.exit(main())
