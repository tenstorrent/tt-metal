#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Per-LLK implicit-config-dependency catalog builder — FIELD (bit) granular, multi-seed union.

For one kernel K: thrash the config space before K runs (in-kernel trisc.cpp prologue), bisect to
the cfg_defines.h FIELDS K depends on but its init doesn't (re)establish = K's implicit deps
(latent included). Pollution values are RANDOM from a logged seed; run K across several seeds and
UNION the per-seed bisected deps (a real dep manifests under some seed's value; union converges).

GRANULARITY: bit/field level. Universe items are individual cfg_defines.h fields (name, addr32,
shamt, mask). On the shared port (0) each field is poisoned via masked read-modify-write, so other
fields of the same word stay at reset-default — this disambiguates which FIELD a kernel depends on
(e.g. THCON_SEC0_REG1 has ~20 fields; init writes some, the kernel relies on reset-default for
others). The addr-mod port (1, SETC16) stays word-level (SETC16 can't cheaply RMW a sub-field).
Each field is also tagged init_written (its bits intersect the reachable write surface) so the
output separates init-owned fields from reset-implicit ones.

Each trial: tt-smi -r -> write a plan (subset, this seed's values) to L1 -> run K -> verdict.
Requires K compiled WITH the trisc.cpp pollution prologue.

Usage:
  python cfg_catalog.py --worktree DIR --arch blackhole --test test_matmul.py \
      --test-id '...' --seeds 0x1,0x2,0x3 [--out X] [--only-words 70,71,92] [--max-addr32 N]
"""

import argparse
import json
import os
import random
import re
import subprocess
import sys
import tempfile

PASS, FAIL, HANG, ENVERR = "PASS", "FAIL", "HANG", "ENVERR"
_CODE = {0: PASS, 1: FAIL, 5: HANG}

_CFG_STATE_SIZE = {"blackhole": 56, "wormhole": 47}
_ADDR_MOD_ADDR32 = {
    "blackhole": sorted(
        set(range(12, 20))
        | set(range(28, 36))
        | set(range(37, 41))
        | set(range(47, 55))
    ),
}
_BOOT_OWNED = {"blackhole": set(), "wormhole": {158, 159, 160, 161}}
# Firmware-owned over-reach: never a kernel dep; poisoning hangs the cores. Excluded by field name.
_EXCLUDE_FIELD_SUBSTR = ("DISABLE_RISC_BP",)
_CFG_DEFINES = {
    "blackhole": "../hw/inc/internal/tt-1xx/blackhole/cfg_defines.h",
    "wormhole": "../hw/inc/internal/tt-1xx/wormhole/wormhole_b0_defines/cfg_defines.h",
}
# Reachable write surface (written bits per addr32) — mirror of cfg_pollution._LIVE_MASK (BH), used
# only to ANNOTATE each field init_written. Approximate (static); the bisect itself is ground truth.
_WRITE_MASK = {
    "blackhole": {
        0: 0x0000FFFF,
        1: 0xFFFFFFFF,
        2: 0xFFFFFFFF,
        5: 0x0000FFFF,
        7: 0x0000FFFF,
        12: 0xFFFFFFFF,
        13: 0xFFFFFFFF,
        14: 0xFFFFFFFF,
        15: 0xFFFFFFFF,
        16: 0x0000FFFF,
        17: 0xFFFFFFFF,
        18: 0xFFFFFFFF,
        19: 0x0000FFFF,
        20: 0xFFFFFFFF,
        21: 0xFFFFFFFF,
        24: 0xFFFFFFFF,
        25: 0xFFFFFFFF,
        28: 0x0000FFFF,
        29: 0x0000FFFF,
        30: 0x0000FFFF,
        31: 0x0000FFFF,
        32: 0x0000FFFF,
        33: 0x0000FFFF,
        34: 0x0000FFFF,
        35: 0x0000FFFF,
        37: 0x0000FFFF,
        38: 0x0000FFFF,
        39: 0x0000FFFF,
        40: 0x0000FFFF,
        41: 0x0000FFFF,
        47: 0x0000FFFF,
        48: 0x0000FFFF,
        49: 0x0000FFFF,
        50: 0xFFFFFFFF,
        51: 0x0000FFFF,
        52: 0x0000FFFF,
        53: 0x0000FFFF,
        54: 0x0000FFFF,
        55: 0x0000FFFF,
        56: 0xFFFFFFFF,
        57: 0xFFFFFFFF,
        59: 0xFFFFFFFF,
        64: 0xFFFF000F,
        65: 0xFFFFFFFF,
        68: 0xFFFFFFFF,
        69: 0xFFFFFFFF,
        70: 0xFFFFFFFF,
        71: 0xFFC80000,
        72: 0xFFFFFFFF,
        73: 0x00000030,
        76: 0xFFFFFFFF,
        77: 0xFFFFFFFF,
        84: 0xFFFFFFFF,
        86: 0xFFFFFFFF,
        92: 0xFFFFFFFF,
        93: 0xFFFFFFFF,
        112: 0xFFFF000F,
        113: 0xFFFF0000,
        119: 0x00400000,
        120: 0x0000000F,
        124: 0xFFFFFFFF,
        125: 0xFFFFFFFF,
        140: 0xFFFFFFFF,
        141: 0xFFFFFFFF,
        180: 0xFFFFFFFF,
        181: 0xFFFFFFFF,
        182: 0xFFFFFFFF,
        183: 0xFFFFFFFF,
        186: 0xFFFFFFFF,
        209: 0xFFFFFFFF,
        211: 0xFFFFFFFF,
        220: 0x0000000B,
    },
}


def parse_fields(worktree, arch):
    """addr32 -> [(name, shamt, mask)] for every cfg_defines.h field (has a _MASK)."""
    path = os.path.normpath(os.path.join(worktree, _CFG_DEFINES.get(arch, "")))
    A, S, M = {}, {}, {}
    for line in open(path):
        for pat, d, conv in (
            (r"_ADDR32\s+(\d+)", A, int),
            (r"_SHAMT\s+(\d+)", S, int),
            (r"_MASK\s+(0x[0-9A-Fa-f]+|\d+)", M, lambda x: int(x, 0)),
        ):
            m = re.match(r"#define\s+(\w+?)" + pat + r"\b", line)
            if m:
                d[m.group(1)] = conv(m.group(2))
    fields = {}
    for name, a in A.items():
        if name in M:
            fields.setdefault(a, []).append((name, S.get(name, 0), M[name]))
    return fields


# An item is (addr32, port, mask, name). value depends only on (seed, addr32, port, mask).
def value(seed, item):
    return random.Random((seed, item[0], item[1], item[2])).getrandbits(32)


def candidate_items(arch, worktree, only_words=None, max_addr32=None):
    fields = parse_fields(worktree, arch)
    n = _CFG_STATE_SIZE[arch] * 4
    if max_addr32 is not None:
        n = min(n, max_addr32)

    def want(a):
        return (
            a not in _BOOT_OWNED[arch]
            and a < n
            and (only_words is None or a in only_words)
        )

    items = []
    # Shared port (0): one item per FIELD (masked RMW isolates it).
    for a in range(n):
        if not want(a):
            continue
        flds = [
            f
            for f in fields.get(a, [])
            if not any(s in f[0] for s in _EXCLUDE_FIELD_SUBSTR)
        ]
        if flds:
            for name, shamt, mask in flds:
                items.append((a, 0, mask, name))
        else:
            items.append(
                (a, 0, 0xFFFFFFFF, f"word_{a}")
            )  # no named fields -> whole word
    # addr-mod port (1): word-level (SETC16, no cheap sub-field RMW).
    amnames = {}
    for a, flds in fields.items():
        am = sorted(nm for (nm, _, _) in flds if nm.startswith("ADDR_MOD"))
        if am:
            amnames[a] = am
    for a in _ADDR_MOD_ADDR32.get(arch, []):
        if want(a):
            items.append((a, 1, 0xFFFF, "+".join(amnames.get(a, [f"ADDRMOD_{a}"]))))
    return items


def _reset():
    subprocess.run(["tt-smi", "-r"], capture_output=True, text=True)


# addr32 0 bit 0 = CFG_STATE_ID (thread-private, preserved); addr32 2 bits 22-31 = firmware
# DISABLE_RISC_BP (over-reach). Both are skipped by restore so we never write firmware-owned bits.
_RESTORE_MASK_OVERRIDE = {2: 0x003FFFFF}


def build_restore_entries(arch, pristine_path):
    """Restore plan from a captured pristine snapshot (host snapshot_cfg JSON: [[state,addr32,val]..]).

    State-0 cfg-bus words -> port-0 full-word writes (re-establish the shared banked baseline).
    addr-mod words -> port-1 SETC16 zero writes (reset-default). NOTE: snapshot_cfg()'s addr32
    numbering (Config[state][addr32], the shared double-buffered CFG bus) and _ADDR_MOD_ADDR32's
    numbering (ThreadConfig[thread][idx], a separate per-thread-banked array entirely -- see
    BackendConfiguration.md) are DIFFERENT address spaces that happen to share small integers.
    cfg_read()/cfg_write() (ckernel.h) can only reach Config, never ThreadConfig, and RISCV store
    instructions can't write ThreadConfig at all (SETC16 only) -- so there is no capture of the
    real addr-mod value here to replay; 0 is a guess, not a captured value. Verified empirically:
    replaying the Config-space value that happens to share the addr-mod address's number (via
    either cfg_write or SETC16) breaks far more victims than this reset-default-0 guess does.
    """
    with open(pristine_path) as f:
        snap = json.load(f)
    n = _CFG_STATE_SIZE[arch] * 4
    entries = []
    for state, addr32, val in snap:
        if state != 0 or addr32 >= n or addr32 in _BOOT_OWNED[arch]:
            continue
        entries.append([addr32, val, 0, _RESTORE_MASK_OVERRIDE.get(addr32, 0xFFFFFFFF)])
    for a in _ADDR_MOD_ADDR32.get(arch, []):
        entries.append([a, 0, 1, 0xFFFF])  # thread-private addr-mod -> reset-default 0
    return entries


def build_addrmod_restore_entries(addrmod_path):
    """Per-thread addr-mod restore plan from a captured snapshot (host snapshot_addr_mod JSON:
    [[thread, addr32, val], ...]). Groups by addr32 into [addr32, v_thread0, v_thread1, v_thread2]
    quads for write_inkernel_addrmod_restore(); a thread with no captured entry for an address
    defaults to 0 (reset-default), matching build_restore_entries' fallback for the same address.
    """
    with open(addrmod_path) as f:
        snap = json.load(f)
    by_addr = {}
    for thread, addr32, val in snap:
        by_addr.setdefault(addr32, [0, 0, 0])[thread] = val
    return [[addr32, *by_addr[addr32]] for addr32 in sorted(by_addr)]


def _run(args, env_extra):
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
    if "does not exist" in (proc.stdout + proc.stderr):
        return ENVERR
    return _CODE.get(proc.returncode, ENVERR)


def make_test(args, seed, plan_path, memo, restore_path=None):
    def test(items):
        key = frozenset(items)
        if key in memo:
            return memo[key]
        entries = [
            [a, value(seed, (a, port, mask, name)), port, mask]
            for (a, port, mask, name) in items
        ]
        with open(plan_path, "w") as f:
            json.dump({"entries": entries}, f)
        env = {"_COMMAND": "simulate", "LLK_POLLUTE_INKERNEL": plan_path}
        if (
            restore_path
        ):  # restore-mode: replay pristine in-kernel instead of per-trial tt-smi -r
            env["LLK_POLLUTE_INKERNEL_RESTORE"] = restore_path
        else:
            _reset()
        # Run; absorb transient ENVERR (device re-init hiccup) with a reset+retry — never let it
        # masquerade as a (non-repro) PASS, which would silently drop a real dependency.
        verdict = _run(args, env)
        tries = 0
        while verdict == ENVERR and tries < 3:
            tries += 1
            _reset()
            verdict = _run(args, env)
        # Restore-mode caveat: a HANG wedges the TRISC/backend and only tt-smi -r clears it.
        # Reset AFTER recording so the next trial starts from a clean device (restore re-applies
        # config on top). PASS/FAIL leave the device usable (gate-verified), so no reset there.
        if restore_path and verdict == HANG:
            _reset()
        repro = verdict in (FAIL, HANG)
        memo[key] = repro
        tag = verdict + (
            f" (after {tries} ENVERR retr{'y' if tries==1 else 'ies'})" if tries else ""
        )
        print(f"[catalog]   trial {len(items)} field(s) -> {tag}", file=sys.stderr)
        return repro

    return test


def ddmin(items, test):
    items = list(items)
    n = 2
    while len(items) >= 2:
        chunk = max(1, len(items) // n)
        subsets = [items[i : i + chunk] for i in range(0, len(items), chunk)]
        for s in subsets:
            if test(s):
                items, n = s, 2
                break
        else:
            for s in subsets:
                comp = [x for x in items if x not in s]
                if comp and test(comp):
                    items, n = comp, max(n - 1, 2)
                    break
            else:
                if n >= len(items):
                    break
                n = min(len(items), 2 * n)
    return items


def reproduce_check(items, test, retries=3):
    """Does poisoning the whole universe reproduce? Retry to absorb a transient spurious PASS
    (a single device hiccup would otherwise yield a bogus 0-dep result)."""
    for _ in range(retries):
        if test(items):
            return True
    return False


def find_all_deps(items, test):
    remaining, found = list(items), []
    while remaining and reproduce_check(remaining, test):
        minimal = ddmin(remaining, test)
        found.append(minimal)
        drop = set(minimal)
        remaining = [x for x in remaining if x not in drop]
    return found


def _init_written(arch, addr32, mask):
    return bool(mask & _WRITE_MASK.get(arch, {}).get(addr32, 0))


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--worktree", required=True)
    p.add_argument("--arch", required=True, choices=["blackhole", "wormhole"])
    p.add_argument("--test", required=True)
    p.add_argument("--test-id", required=True, dest="test_id")
    p.add_argument("--seeds", default="0x1,0x2,0x3")
    p.add_argument("--out", default=None)
    p.add_argument(
        "--only-words",
        default=None,
        help="restrict universe to these addr32 (comma list)",
    )
    p.add_argument("--max-addr32", type=int, default=None)
    p.add_argument("--port", type=int, default=5556)
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument(
        "--restore",
        default=None,
        help="pristine snapshot JSON ([[state,addr32,val]..]); enables restore-mode "
        "(replay pristine in-kernel each trial instead of tt-smi -r). ~no per-trial reset.",
    )
    args = p.parse_args()
    seeds = [int(s, 0) for s in args.seeds.split(",") if s.strip()]
    only = {int(x, 0) for x in args.only_words.split(",")} if args.only_words else None
    out = args.out or f"/tmp/cat_{os.path.basename(args.test)}.json"

    items = candidate_items(args.arch, args.worktree, only, args.max_addr32)
    print(
        f"[catalog] {args.test} :: universe={len(items)} fields, seeds={[hex(s) for s in seeds]}",
        file=sys.stderr,
    )

    tmp = tempfile.mkdtemp(prefix="cfg_cat_")
    plan_path = os.path.join(tmp, "plan.json")

    # Restore-mode: build the pristine-replay plan once; one initial reset to establish a clean
    # device, then no per-trial reset (the in-kernel replay re-establishes the baseline each trial).
    restore_path = None
    if args.restore:
        restore_entries = build_restore_entries(args.arch, args.restore)
        restore_path = os.path.join(tmp, "restore.json")
        with open(restore_path, "w") as f:
            json.dump({"entries": restore_entries}, f)
        print(
            f"[catalog] restore-mode: {len(restore_entries)} pristine entries (no per-trial reset)",
            file=sys.stderr,
        )

    print("[catalog] control: reset + compile + run (expect PASS)...", file=sys.stderr)
    _reset()
    # Control: plain run in reset-mode; in restore-mode, run WITH restore + empty poison to prove
    # the pristine replay itself reproduces a clean baseline (the first acceptance check).
    if restore_path:
        empty_plan = os.path.join(tmp, "empty.json")
        with open(empty_plan, "w") as f:
            json.dump({"entries": []}, f)
        ctl_env = {
            "_COMMAND": "run",
            "LLK_POLLUTE_INKERNEL": empty_plan,
            "LLK_POLLUTE_INKERNEL_RESTORE": restore_path,
        }
    else:
        ctl_env = {"_COMMAND": "run"}
    if _run(args, ctl_env) != PASS:
        raise SystemExit(
            "[catalog] pristine control did not PASS — fix baseline before cataloging."
        )
    state = {
        "test": args.test,
        "test_id": args.test_id,
        "arch": args.arch,
        "seeds": [hex(s) for s in seeds],
        "per_seed": {},
        "union": [],
    }
    union = set()
    for seed in seeds:
        memo = {}
        test = make_test(args, seed, plan_path, memo, restore_path=restore_path)
        print(
            f"[catalog] seed 0x{seed:08X}: full-universe poison (expect reproduce)...",
            file=sys.stderr,
        )
        deps = find_all_deps(items, test)
        flat = sorted({it for dep in deps for it in dep})
        state["per_seed"][hex(seed)] = [list(it) for it in flat]
        union |= set(flat)
        print(
            f"[catalog] seed 0x{seed:08X}: {len(flat)} dep field(s), {len(memo)} trials",
            file=sys.stderr,
        )
        state["union"] = _render(args.arch, union)
        with open(out, "w") as f:
            json.dump(state, f, indent=2)

    print("\n========== IMPLICIT-DEPENDENCY CATALOG (field-granular) ==========")
    print(f"{args.test}  (union over {len(seeds)} seeds, {len(union)} dep field(s))")
    for e in state["union"]:
        port = "SETC16 " if e["port"] == 1 else "cfgwrite"
        tag = "INIT-WRITTEN" if e["init_written"] else "reset-implicit"
        print(f"  a{e['addr32']:3d}[{port}] mask={e['mask']:<10} {tag:14} {e['field']}")
    print(f"JSON -> {out}")
    print("==================================================================")


def _render(arch, union):
    out = []
    for a, port, mask, name in sorted(union):
        out.append(
            {
                "addr32": a,
                "port": port,
                "mask": hex(mask),
                "field": name,
                "init_written": _init_written(arch, a, mask),
            }
        )
    return out


if __name__ == "__main__":
    main()
