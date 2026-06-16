#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Bisect which CFG register(s) a failing kernel implicitly depends on.

Premise: a kernel passes on a pristine device but FAILS when the whole config space is
poisoned at launch (see helpers/cfg_pollution.py) — so it silently relies on some subset
of those words, config its own init never (re)writes. This tool finds the minimal such
subset by delta-debugging: it re-runs the kernel many times, each time poisoning only a
candidate subset (with the same per-word values, which `word_value` makes subset-
independent) and narrows to the words that matter.

Trial isolation is by full device reset, NOT CFG restore. Restoring only CFG is not enough:
empirically the failure also depends on non-CFG device state (dest/SFPU/regfile/L1) that a
prior kernel run leaves behind and that masks the failure. So each trial is `tt-smi -r` ->
poison subset at launch -> run, exactly the regime in which the failure reproduces. The
candidate not in the subset is left at pristine post-reset values. Trials are memoized by
subset so repeated ddmin probes don't re-burn the (slow) reset+run cycle. A hang (exit 5)
counts as a reproduced failure.

Usage:
  python cfg_bisect.py --worktree DIR --arch blackhole \
      --test test_eltwise_unary_datacopy.py \
      --test-id 'test_eltwise_unary_datacopy.py::test_unary_datacopy[...]' \
      [--seed 0xDEADBEEF] [--states 0,1] [--port 5556] [--timeout 600]
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

PASS, FAIL, HANG = "PASS", "FAIL", "HANG"

# Mirror of cfg_pollution.py: CFG_STATE_SIZE (128-bit entries) and boot-owned addr32 to
# exclude from the candidate universe (those break boot, not kernel compute-init).
_CFG_STATE_SIZE = {"blackhole": 56, "wormhole": 47}
_BOOT_OWNED = {"blackhole": set(), "wormhole": {158, 159, 160, 161}}

# cfg_defines.h relative to the tt-llk worktree (tt-metal/tt_metal/tt-llk); hw/inc is one up.
_CFG_DEFINES_REL = {
    "blackhole": "../hw/inc/internal/tt-1xx/blackhole/cfg_defines.h",
    "wormhole": "../hw/inc/internal/tt-1xx/wormhole/wormhole_b0_defines/cfg_defines.h",
}


def _reset():
    print("[bisect]   tt-smi -r ...", file=sys.stderr)
    subprocess.run(["tt-smi", "-r"], capture_output=True, text=True)


def _run_trial(args, env_extra):
    """Invoke run_test.sh for a single variant; return PASS/FAIL/HANG."""
    cmd = [
        "bash",
        os.path.join(args.worktree, ".claude/scripts/run_test.sh"),
        env_extra.pop("_COMMAND", "simulate"),
        "--worktree",
        args.worktree,
        "--arch",
        args.arch,
        "--test",
        args.test,
        "--test-id",
        args.test_id,
        "--maxfail",
        "1",
        "--port",
        str(args.port),
        "--timeout",
        str(args.timeout),
    ]
    proc = subprocess.run(
        cmd, env={**os.environ, **env_extra}, capture_output=True, text=True
    )
    for line in (proc.stdout + proc.stderr).splitlines():
        if "[CFG-POLLUTE]" in line or "RUN_LLK_TESTS_VERDICT" in line:
            print("    " + line.strip(), file=sys.stderr)
    code = proc.returncode
    if code == 0:
        return PASS
    if code == 5:
        return HANG
    if code == 1:
        return FAIL
    raise SystemExit(
        f"run_test.sh exit {code} (compile/env error):\n{proc.stderr[-2000:]}"
    )


def candidate_items(arch, states):
    """All kernel-owned (state, addr32) words — the bisection universe."""
    addr32 = [a for a in range(_CFG_STATE_SIZE[arch] * 4) if a not in _BOOT_OWNED[arch]]
    return [(s, a) for s in states for a in addr32]


def make_test(args, seed, plan_path, memo, preserve):
    """test(items)->bool: reset, poison exactly `items` at launch, run; True iff it reproduces."""

    def test(items):
        key = frozenset((s, a) for s, a in items)
        if key in memo:
            return memo[key]
        plan = {
            "seed": seed,
            "pollute": [[s, a] for s, a in items],
        }  # no snapshot => no restore
        if preserve:
            plan["preserve"] = [
                [a, m] for a, m in preserve.items()
            ]  # mask known deps / over-reach
        with open(plan_path, "w") as f:
            json.dump(plan, f)
        print(f"[bisect] trial: poison {len(items)} word(s)...", file=sys.stderr)
        _reset()
        verdict = _run_trial(
            args, {"_COMMAND": "simulate", "LLK_POLLUTE_PLAN": plan_path}
        )
        reproduced = verdict in (FAIL, HANG)
        print(
            f"[bisect]   -> {verdict} ({'reproduced' if reproduced else 'clean'})",
            file=sys.stderr,
        )
        memo[key] = reproduced
        return reproduced

    return test


def ddmin(items, test):
    """Zeller delta-debugging: shrink `items` to a 1-minimal failing subset.

    Precondition: test(items) is True. Handles interactions (neither half fails alone) by
    raising granularity; degrades to binary search for a single culprit.
    """
    items = list(items)
    n = 2
    while len(items) >= 2:
        chunk = max(1, len(items) // n)
        subsets = [items[i : i + chunk] for i in range(0, len(items), chunk)]
        for s in subsets:  # any subset fail alone?
            if test(s):
                items, n = s, 2
                break
        else:
            for s in subsets:  # any complement fail?
                comp = [x for x in items if x not in s]
                if comp and test(comp):
                    items, n = comp, max(n - 1, 2)
                    break
            else:
                if n >= len(items):
                    break
                n = min(len(items), 2 * n)  # finer granularity
    return items


def name_addr32(worktree, arch):
    """addr32 -> sorted cfg_defines.h field names at that word (best-effort)."""
    path = os.path.normpath(os.path.join(worktree, _CFG_DEFINES_REL.get(arch, "")))
    names = {}
    try:
        with open(path) as f:
            for line in f:
                m = re.match(r"#define\s+(\w+)_ADDR32\s+(\d+)\b", line)
                if m:
                    names.setdefault(int(m.group(2)), []).append(m.group(1))
    except OSError:
        return {}
    return {a: sorted(set(v)) for a, v in names.items()}


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--worktree", required=True)
    p.add_argument("--arch", required=True, choices=["blackhole", "wormhole"])
    p.add_argument("--test", required=True)
    p.add_argument("--test-id", required=True, dest="test_id")
    p.add_argument("--seed", default="0xDEADBEEF", help="poison seed (int, 0x.. ok)")
    p.add_argument(
        "--states", default="0,1", help="shadow states to consider, e.g. 0,1"
    )
    p.add_argument(
        "--preserve",
        default="",
        help="extra addr32:mask bits to leave unpoisoned "
        "(mask already-found deps to surface the next one), e.g. '6:0xffff,3:0x..'",
    )
    p.add_argument("--port", type=int, default=5556)
    p.add_argument("--timeout", type=int, default=600)
    args = p.parse_args()
    seed = int(args.seed, 0)
    states = tuple(int(x) for x in args.states.split(","))
    preserve = {}
    for tok in args.preserve.split(","):
        tok = tok.strip()
        if tok:
            a, _, m = tok.partition(":")
            preserve[int(a, 0)] = int(m, 0) if m else 0xFFFFFFFF

    tmp = tempfile.mkdtemp(prefix="cfg_bisect_")
    plan_path = os.path.join(tmp, "plan.json")
    items = candidate_items(args.arch, states)
    print(
        f"[bisect] universe: {len(items)} (state, addr32) words, seed=0x{seed:08X}"
        + (
            f", preserve={ {a: hex(m) for a, m in preserve.items()} }"
            if preserve
            else ""
        ),
        file=sys.stderr,
    )

    # Build ELFs + confirm control (pristine, no pollution) PASSES — else nothing to bisect against.
    print(
        "[bisect] control: reset + compile + run, no pollution (expect PASS)...",
        file=sys.stderr,
    )
    _reset()
    if _run_trial(args, {"_COMMAND": "run"}) != PASS:
        raise SystemExit(
            "[bisect] pristine control did not PASS — fix the baseline before bisecting."
        )

    memo = {}
    test = make_test(args, seed, plan_path, memo, preserve)

    print("[bisect] sanity: poison everything (expect reproduce)...", file=sys.stderr)
    if not test(items):
        raise SystemExit(
            "[bisect] whole-space poison did NOT reproduce — nothing to bisect for this seed."
        )

    print("[bisect] bisecting...", file=sys.stderr)
    minimal = ddmin(items, test)

    names = name_addr32(args.worktree, args.arch)
    print("\n========== BISECTION RESULT ==========")
    print(
        f"Minimal failing set: {len(minimal)} word(s)  (seed=0x{seed:08X}, {len(memo)} trials)"
    )
    for s, a in sorted(minimal):
        print(
            f"  state {s} addr32 {a:3d}  ->  {', '.join(names.get(a, ['<unknown>']))}"
        )
    print("======================================")
    print(
        "BISECT_JSON "
        + json.dumps(
            {
                "seed": seed,
                "minimal": [[s, a] for s, a in sorted(minimal)],
                "trials": len(memo),
            }
        )
    )


if __name__ == "__main__":
    main()
