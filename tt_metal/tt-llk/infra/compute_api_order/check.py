#!/usr/bin/env python3
"""Prototype static checker for compute-kernel API ordering (tt-metal #62681).

The compute API (`tt_metal/hw/inc/api/compute/`) is stateful: most operations only
work after a matching init call has configured the hardware, and some inits are
invalidated by a reconfiguration. Today those rules exist only as prose and as the
runtime LLK Sanitizer, so a mistake compiles and shows up as a hang or silently
wrong results on hardware.

This prototype reads `rules.yaml` and reports, per kernel and per function scope:

  1. an operation with no preceding matching init;
  2. a block init invalidated by a `reconfig_data_format` / `pack_reconfig_data_format`
     between the init and its operation;
  3. an init that is never paired with its required uninit.

Scope and known limits, by design for a first prototype:

  * One function scope at a time. An init issued in a caller or a helper header is
    not yet seen, which is the largest source of false positives on the corpus
    (matmul inits are usually hoisted into a helper).
  * Threads are not separated: the same source is compiled once each for UNPACK,
    MATH and PACK, and this prototype does not model per-thread state.
  * Template arguments are skipped when matching calls, so `binary_tiles_init<...>`
    matches `binary_tiles_init`.
  * Compile-time argument comparison between an init and its operation is not
    implemented yet.

Usage:
    python3 check.py                     # scan the whole repository
    python3 check.py path/to/kernel.cpp  # scan specific files
    python3 check.py --root . --quiet    # counts only

Exit code 0 = no findings, 1 = findings.
"""
import argparse
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

CALL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*(?:<(?:[^<>]|<[^<>]*>)*>)?\s*\(")
STRIP_RE = re.compile(r"//[^\n]*|/\*.*?\*/|\"(?:\\.|[^\"\\])*\"", re.S)
SCOPE_RE = re.compile(
    r"\b(?:ALWI\s+)?(?:void|int|bool|float|std::\w+)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\([^)]*\)\s*\{")


def load_rules(path):
    rules = {"ops": {}, "invalidate": [], "uninit_pairs": {}}
    section = None
    for raw in open(path, encoding="utf-8"):
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if line.rstrip().endswith(":"):
            section = line.strip()[:-1]
            continue
        if section == "ops":
            key, value = line.strip().split(":", 1)
            value = value.strip()
            if value.startswith("["):
                rules["ops"][key.strip()] = [x.strip() for x in value.strip("[]").split(",")]
            else:
                rules["ops"][key.strip()] = [value]
        elif section == "uninit_pairs":
            key, value = line.strip().split(":", 1)
            rules["uninit_pairs"][key.strip()] = value.strip()
        elif section == "invalidate":
            rules["invalidate"].append(line.strip().lstrip("- ").strip())
    return rules


def scopes(source):
    body = STRIP_RE.sub(" ", source)
    out, i = [], 0
    while True:
        match = SCOPE_RE.search(body, i)
        if not match:
            break
        start = match.end() - 1
        depth, j = 0, start
        while j < len(body):
            if body[j] == "{":
                depth += 1
            elif body[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        out.append((match.group(1), body[start:j + 1]))
        i = j + 1
    return out


class Checker:
    def __init__(self, rules):
        self.ops = rules["ops"]
        self.invalidate = set(rules["invalidate"])
        self.uninit_pairs = rules["uninit_pairs"]
        self.inits = {i for v in self.ops.values() for i in v} | set(self.uninit_pairs)
        self.block_inits = {"tilize_init", "tilize_init_short", "untilize_init", "transpose_init"}

    def check(self, path):
        source = open(path, encoding="utf-8", errors="ignore").read()
        hits = []
        for fname, body in scopes(source):
            live, inited = set(), set()
            for match in CALL_RE.finditer(body):
                name, pos = match.group(1), match.start()
                if name in self.inits:
                    live.add(name)
                    inited.add(name)
                elif name in self.invalidate:
                    live -= self.block_inits
                elif name in self.ops:
                    accepted = self.ops[name]
                    if not (live & set(accepted)):
                        hits.append((fname, name, "missing " + "/".join(accepted) + "()",
                                     body[:pos].count("\n") + 1))
            for init, uninit in self.uninit_pairs.items():
                if init in inited and uninit not in body:
                    hits.append((fname, init, "never paired with " + uninit + "()", 0))
        return hits


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", nargs="*", help="kernel sources; default: every */kernels/*.cpp")
    parser.add_argument("--root", default=".", help="repository root for the default scan")
    parser.add_argument("--quiet", action="store_true", help="print counts only")
    args = parser.parse_args()

    files = sorted(args.files) if args.files else sorted(
        glob.glob(os.path.join(args.root, "**", "kernels", "*.cpp"), recursive=True))

    checker = Checker(load_rules(os.path.join(HERE, "rules.yaml")))
    total, bad_files, per_rule = 0, 0, {}
    for path in files:
        hits = checker.check(path)
        if not hits:
            continue
        bad_files += 1
        total += len(hits)
        for fname, op, why, line in hits:
            per_rule[why.split("()")[0]] = per_rule.get(why.split("()")[0], 0) + 1
            if not args.quiet:
                print(f"{os.path.basename(path)} :: {fname}() line~{line}: {op} — {why}")

    print(f"\nscanned {len(files)} kernels, {bad_files} with findings, {total} findings")
    for rule, count in sorted(per_rule.items(), key=lambda x: -x[1]):
        print(f"  {rule}(): {count}")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
