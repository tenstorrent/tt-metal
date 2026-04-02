#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Verify that two reconstructed JSONs have identical configurations.

Compares the 'operations' key only (ignores top-level 'metadata' which
reconstruct-manifest adds but reconstruct does not).

Usage:
    python model_tracer/verify_reconstruct_match.py <file_a> <file_b>

Exit code 0 = exact match, 1 = differences found.
"""

import json
import sys


def load_operations(path):
    with open(path) as f:
        data = json.load(f)
    return data.get("operations", {})


def compare(a_path, b_path):
    print(f"A: {a_path}")
    print(f"B: {b_path}")

    a_ops = load_operations(a_path)
    b_ops = load_operations(b_path)

    print(f"A: {len(a_ops)} ops, B: {len(b_ops)} ops")

    a_keys = set(a_ops)
    b_keys = set(b_ops)

    ok = True

    if a_keys - b_keys:
        print(f"Operations only in A: {sorted(a_keys - b_keys)}")
        ok = False
    if b_keys - a_keys:
        print(f"Operations only in B: {sorted(b_keys - a_keys)}")
        ok = False

    diff_count = 0
    total_configs = 0

    for op_name in sorted(a_keys & b_keys):
        a_cfgs = a_ops[op_name].get("configurations", [])
        b_cfgs = b_ops[op_name].get("configurations", [])
        total_configs += len(a_cfgs)

        if len(a_cfgs) != len(b_cfgs):
            print(f"  {op_name}: config count A={len(a_cfgs)} B={len(b_cfgs)}")
            ok = False
            diff_count += 1
            continue

        for i, (ac, bc) in enumerate(zip(a_cfgs, b_cfgs)):
            if ac == bc:
                continue

            ok = False
            diff_count += 1
            ah = ac.get("config_hash", "?")[:12]
            bh = bc.get("config_hash", "?")[:12]
            diff_keys = [k for k in set(ac) | set(bc) if ac.get(k) != bc.get(k)]
            print(f"  {op_name}[{i}]: hash A={ah} B={bh}, diff keys={diff_keys}")

            if "executions" in diff_keys:
                a_ex = ac.get("executions", [])
                b_ex = bc.get("executions", [])
                print(f"    A: {len(a_ex)} executions, B: {len(b_ex)} executions")
                for j, (ae, be) in enumerate(zip(a_ex, b_ex)):
                    if ae != be:
                        ediff = [k for k in set(ae) | set(be) if ae.get(k) != be.get(k)]
                        print(f"    exec[{j}] diff: {ediff}")
                        for ek in ediff:
                            print(f"      {ek}: A={ae.get(ek)!r}")
                            print(f"      {ek}: B={be.get(ek)!r}")
                        break

            if diff_count >= 20:
                print("  ... stopping after 20 diffs")
                break

        if diff_count >= 20:
            break

    if ok:
        print(f"\nMATCH: {len(a_keys)} operations, {total_configs} configurations identical")
    else:
        print(f"\nFAILED: {diff_count} difference(s) found")

    return ok


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <file_a> <file_b>")
        sys.exit(2)
    match = compare(sys.argv[1], sys.argv[2])
    sys.exit(0 if match else 1)
