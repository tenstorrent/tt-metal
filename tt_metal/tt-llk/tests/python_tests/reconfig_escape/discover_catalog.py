#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Catalog builder for reconfig testing. This finds reachable config states.
We sample across the testsuite, capture each executed variant's config footprint, and dedup.
The results are then written into a JSON for pair_sweep.py.

Usage:
  python3 discover_catalog.py --worktree DIR --arch blackhole \
      --out-dir /path/to/discovered --manifest /path/to/discovered/manifest.json
"""

import argparse
import datetime
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

PASS, FAIL, ENVERR = "PASS", "FAIL", "ENVERR"

_CFG_STATE_SIZE = {"blackhole": 56}
_ADDR_MOD_ADDR32 = {
    "blackhole": sorted(
        set(range(12, 20))  # ADDR_MOD_AB_SEC0-7
        | set(range(20, 28))  # ADDR_MOD_AB2_SEC0-7
        | set(range(28, 36))  # ADDR_MOD_DST_SEC0-7
        | set(range(37, 41))  # ADDR_MOD_PACK_SEC0-3
        | set(range(47, 55))  # ADDR_MOD_BIAS_SEC0-7
    ),
}
_BOOT_OWNED = {"blackhole": set()}  # TODO: Wormhole

_RESTORE_SPACE_CONFIG = 0
_RESTORE_SPACE_THREADCONFIG = 1
_RESTORE_SPACE_ADC_CH1X = 2

_DISABLE_RISC_BP_SHAMT = (
    22  # cfg_defines.h: DISABLE_RISC_BP_Disable_main_SHAMT, lowest firmware-owned bit
)
_RESTORE_MASK_OVERRIDE = {2: (1 << _DISABLE_RISC_BP_SHAMT) - 1}


def build_restore_entries(arch, pristine_path):
    """Restore config from a captured clean baseline snapshot."""
    with open(pristine_path) as f:
        snap = json.load(f)
    n = _CFG_STATE_SIZE[arch] * 4
    entries = []
    for state, addr32, val in snap:
        if state != 0 or addr32 >= n or addr32 in _BOOT_OWNED[arch]:
            continue
        mask = _RESTORE_MASK_OVERRIDE.get(addr32, 0xFFFFFFFF)
        entries.append([_RESTORE_SPACE_CONFIG, addr32, val, 0, 0, mask])
    for a in _ADDR_MOD_ADDR32.get(arch, []):
        entries.append(
            [_RESTORE_SPACE_THREADCONFIG, a, 0, 0, 0, 0]
        )  # reset-default guess
    return entries


def op_capture_paths(op):
    snap = op["snapshot_path"]
    return (
        snap,
        snap.replace(".snapshot.json", ".addrmod.json"),
        snap.replace(".snapshot.json", ".adc_ch1x.json"),
    )


def config_diff_entries(arch, snapshot_path, pristine):
    """One kernel's Config space footprint."""
    with open(snapshot_path) as f:
        snap = json.load(f)
    boot_owned = _BOOT_OWNED.get(arch, set())
    entries = []
    for state, addr32, val in snap:
        if state != 0 or addr32 in boot_owned or val == pristine.get(addr32):
            continue
        mask = _RESTORE_MASK_OVERRIDE.get(addr32, 0xFFFFFFFF)
        entries.append([_RESTORE_SPACE_CONFIG, addr32, val, 0, 0, mask])
    return entries


def merge_addrmod_chain(chain_ops):
    """ThreadConfig and addr mod write footprint for a chain of kernels."""
    merged = {}
    for op in chain_ops:
        _, addrmod_path, _ = op_capture_paths(op)
        if not os.path.exists(addrmod_path):
            continue
        with open(addrmod_path) as f:
            snap = json.load(f)
        for thread, addr32, val in snap:
            if val == 0:
                continue
            merged.setdefault(addr32, [0, 0, 0])[thread] = val
    return [
        [_RESTORE_SPACE_THREADCONFIG, addr32, *v, 0]
        for addr32, v in sorted(merged.items())
    ]


def last_ch1x_entry(chain_ops):
    for op in reversed(chain_ops):
        _, _, ch1x_path = op_capture_paths(op)
        if os.path.exists(ch1x_path):
            with open(ch1x_path) as f:
                ch1x = json.load(f)
            return [
                [_RESTORE_SPACE_ADC_CH1X, 0, ch1x["unpacker"], ch1x["packer"], 0, 0]
            ]
    return []


def build_chain_entries(arch, pristine_path, pristine, chain_ops):
    """Compose D write footprints into one restore plan. Most recent write wins per field."""
    merged = {}
    for e in build_restore_entries(arch, pristine_path):
        merged[(e[0], e[1])] = e
    for op in chain_ops:
        snap_path, _, _ = op_capture_paths(op)
        for e in config_diff_entries(arch, snap_path, pristine):
            merged[(e[0], e[1])] = e
    for e in merge_addrmod_chain(chain_ops):
        merged[(e[0], e[1])] = e
    for e in last_ch1x_entry(chain_ops):
        merged[(e[0], e[1])] = e
    return list(merged.values())


def build_addrmod_restore_entries(addrmod_path, ch1x=None):
    with open(addrmod_path) as f:
        snap = json.load(f)
    by_addr = {}
    for thread, addr32, val in snap:
        by_addr.setdefault(addr32, [0, 0, 0])[thread] = val
    entries = [
        [_RESTORE_SPACE_THREADCONFIG, addr32, *by_addr[addr32], 0]
        for addr32 in sorted(by_addr)
    ]
    if ch1x is not None:
        entries.append([_RESTORE_SPACE_ADC_CH1X, 0, ch1x[0], ch1x[1], 0, 0])
    return entries


def _reset_card():
    subprocess.run(["tt-smi", "-r"], capture_output=True, text=True)


def capture_pristine(worktree, arch, test_file, test_id, port, timeout, out_path):
    _reset_card()
    cmd = [
        "bash",
        os.path.join(worktree, ".claude/scripts/run_test.sh"),
        "run",
        "--worktree",
        worktree,
        "--arch",
        arch,
        "--test",
        test_file,
        "--test-id",
        test_id,
        "--maxfail",
        "1",
        "--port",
        str(port),
        "--timeout",
        str(timeout),
    ]
    env = {**os.environ, "LLK_CFG_SNAPSHOT": out_path}
    subprocess.run(cmd, env=env, capture_output=True, text=True)
    if not os.path.exists(out_path):
        raise RuntimeError(f"pristine capture failed for {test_id}")


def pytest_env(worktree):
    env = dict(os.environ)
    reconfig_escape_dir = os.path.join(
        worktree, "tests", "python_tests", "reconfig_escape"
    )
    env["PYTHONPATH"] = reconfig_escape_dir + os.pathsep + env.get("PYTHONPATH", "")
    return env


def collect_all(worktree, arch, timeout):
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "--collect-only",
        "-q",
        f"--timeout={timeout}",
        "-k",
        "",
    ]
    proc = subprocess.run(
        cmd,
        cwd=os.path.join(worktree, "tests", "python_tests"),
        env={**pytest_env(worktree), "CHIP_ARCH": arch},
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in proc.stdout.splitlines() if "::" in line]


def sample_per_test(nodeids, n_per_test, rng):
    by_test = {}
    for nid in nodeids:
        t = nid.split("[", 1)[0]
        by_test.setdefault(t, []).append(nid)
    sampled = []
    for ids in by_test.values():
        sampled += ids if len(ids) <= n_per_test else rng.sample(ids, n_per_test)
    return sampled


_ARGV_BATCH = 500  # comfortably under the OS argv limit even for long parametrized ids


def _chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def compile_all(worktree, arch, nodeids, jobs, timeout):
    """Compiles through a driver which takes nodeids through JSON rather than argv."""
    nodeids_path = os.path.join(
        worktree,
        "tests",
        "python_tests",
        "reconfig_escape",
        "_compile_all_nodeids.json",
    )
    with open(nodeids_path, "w") as f:
        json.dump(nodeids, f)
    driver = (
        "import json, sys, pytest\n"
        f"with open({nodeids_path!r}) as f:\n"
        "    nodeids = json.load(f)\n"
        f"sys.exit(pytest.main(nodeids + ['--compile-producer', '-n', '{jobs}', "
        f"'--timeout={timeout}']))\n"
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", driver],
            cwd=os.path.join(worktree, "tests", "python_tests"),
            env={**pytest_env(worktree), "CHIP_ARCH": arch},
            capture_output=True,
            text=True,
        )
    finally:
        os.remove(nodeids_path)
    return proc


def run_discovery_round(
    worktree,
    arch,
    nodeids,
    pristine_restore,
    candidates_path,
    outdir,
    jobs,
    timeout,
    junit_path,
):
    """Batches due to the OS argv limit. Each batch is a separate pytest invocation with
    its own junit file. The batches' testcases are merged into one junit at junit_path.
    """
    merged_cases = []
    last_proc = None
    for i, batch in enumerate(_chunks(nodeids, _ARGV_BATCH)):
        batch_junit = f"{junit_path}.batch{i}"
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "--compile-consumer",
            "-n",
            str(jobs),
            "-p",
            "xdist_capture_plugin",
            f"--llk-pristine-restore={pristine_restore}",
            f"--llk-capture-outdir={outdir}",
            f"--llk-candidates={candidates_path}",
            f"--timeout={timeout}",
            f"--junitxml={batch_junit}",
        ] + batch
        last_proc = subprocess.run(
            cmd,
            cwd=os.path.join(worktree, "tests", "python_tests"),
            env={**pytest_env(worktree), "CHIP_ARCH": arch},
            capture_output=True,
            text=True,
        )
        if os.path.exists(batch_junit):
            merged_cases.extend(ET.parse(batch_junit).getroot().iter("testcase"))
    if merged_cases:
        suite = ET.Element(
            "testsuite", name="reconfig_escape_discovery", tests=str(len(merged_cases))
        )
        suite.extend(merged_cases)
        root = ET.Element("testsuites")
        root.append(suite)
        ET.ElementTree(root).write(junit_path, encoding="utf-8", xml_declaration=True)
    return last_proc


def parse_junit(junit_path):
    tree = ET.parse(junit_path)
    results = {}
    for case in tree.getroot().iter("testcase"):
        nodeid = f"{case.get('classname')}.py::{case.get('name')}"
        failed = case.find("failure") is not None or case.find("error") is not None
        results[nodeid] = FAIL if failed else PASS
    return results


def _sanitize(nodeid):
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", nodeid)
    if len(safe) <= 200:
        return safe
    digest = hashlib.sha256(nodeid.encode()).hexdigest()[:16]
    return safe[:183] + "_" + digest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--worktree", required=True)
    p.add_argument("--arch", required=True, choices=["blackhole"])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument(
        "--sample-per-test",
        type=int,
        default=150,
        help="max candidates to sample per test function; distinct dedup signatures found "
        "grows with this but with diminishing returns at scale\nTODO: measure further",
    )
    p.add_argument(
        "--dedup-cardinality-cutoff",
        type=int,
        default=15,
        help="exclude a config field from the dedup key if it takes on more than this "
        "many distinct observed values across the whole sample",
    )
    p.add_argument(
        "--jobs",
        type=int,
        default=8,
        help="xdist worker count, each takes one Tensix core",
    )
    p.add_argument(
        "--compile-jobs",
        type=int,
        default=8,
        help="xdist worker count for compiling",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for sampling; default derives from the current ISO year+week",
    )
    p.add_argument("--timeout", type=int, default=90)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("discover_catalog: collecting candidate pool...", file=sys.stderr)
    pool = collect_all(args.worktree, args.arch, args.timeout)
    print(f"discover_catalog: pool: {len(pool)} nodeids", file=sys.stderr)
    if not pool:
        raise SystemExit(
            "discover_catalog: collected 0 test items (worktree/arch misconfigured?)"
        )

    seed = (
        args.seed
        if args.seed is not None
        else int(datetime.date.today().strftime("%Y%V"))
    )
    print(
        f"discover_catalog: seed={seed}"
        + (" (explicit)" if args.seed is not None else " (derived from ISO year+week)"),
        file=sys.stderr,
    )
    rng = random.Random(seed)
    sampled = sample_per_test(pool, args.sample_per_test, rng)
    n_files = len(set(n.split("::", 1)[0] for n in sampled))
    n_tests = len(set(n.split("[", 1)[0] for n in sampled))
    print(
        f"discover_catalog: sampled {len(sampled)} candidates ({n_tests} test functions across "
        f"{n_files} files)",
        file=sys.stderr,
    )
    with open(os.path.join(args.out_dir, "sampled_nodeids.txt"), "w") as f:
        f.write("\n".join(sampled))
    candidates_path = os.path.join(args.out_dir, "candidates.txt")
    with open(candidates_path, "w") as f:
        f.write("\n".join(sampled))

    pristine_snap_path = os.path.join(args.out_dir, "pristine.snapshot.json")
    pristine_restore_path = os.path.join(args.out_dir, "pristine.restore.json")
    probe_test_file = sampled[0].split("::", 1)[0]
    capture_pristine(
        args.worktree,
        args.arch,
        probe_test_file,
        sampled[0],
        5556,
        args.timeout,
        pristine_snap_path,
    )
    restore_entries = build_restore_entries(args.arch, pristine_snap_path)
    with open(pristine_restore_path, "w") as f:
        json.dump({"entries": restore_entries}, f)
    print(
        f"discover_catalog: pristine captured: {pristine_restore_path}", file=sys.stderr
    )

    print(
        f"discover_catalog: compiling {len(sampled)} candidates (producer, -n {args.compile_jobs})...",
        file=sys.stderr,
    )
    cproc = compile_all(
        args.worktree, args.arch, sampled, args.compile_jobs, args.timeout
    )
    if cproc.returncode != 0:
        # A broad random sample of the real suite tends to run into a handful of
        # unrelated broken tests. Don't abort the whole discovery pass over that,
        # the discovery round already excludes anything that fails to compile or run.
        print(
            f"discover_catalog: warning: compile-producer had failures (rc={cproc.returncode}); "
            f"affected candidates will show up as non-PASS baseline and be excluded",
            file=sys.stderr,
        )

    capture_dir = os.path.join(args.out_dir, "captures")
    os.makedirs(capture_dir, exist_ok=True)
    junit_path = os.path.join(args.out_dir, "discovery.junit.xml")

    print(
        f"discover_catalog: discovery round: {len(sampled)} candidates across -n {args.jobs}...",
        file=sys.stderr,
    )
    rproc = run_discovery_round(
        args.worktree,
        args.arch,
        sampled,
        pristine_restore_path,
        candidates_path,
        capture_dir,
        args.jobs,
        args.timeout,
        junit_path,
    )
    if not os.path.exists(junit_path):
        print(rproc.stdout[-4000:], file=sys.stderr)
        print(rproc.stderr[-4000:], file=sys.stderr)
        raise SystemExit("discover_catalog: discovery round produced no junit report")
    baseline = parse_junit(junit_path)

    with open(pristine_snap_path) as f:
        pristine = {addr32: val for state, addr32, val in json.load(f) if state == 0}
    addrmod = set(_ADDR_MOD_ADDR32.get(args.arch, []))
    boot_owned = _BOOT_OWNED[args.arch]

    discovered = []
    for nid in sampled:
        verdict = baseline.get(nid, ENVERR)
        snap_path = os.path.join(capture_dir, _sanitize(nid) + ".snapshot.json")
        rec = {
            "test_id": nid,
            "test_file": nid.split("::", 1)[0],
            "baseline": verdict,
            "snapshot_path": None,
            "signature": None,
        }
        if verdict == PASS and os.path.exists(snap_path):
            with open(snap_path) as f:
                snap = json.load(f)
            diff = {
                a: v
                for s, a, v in snap
                if s == 0
                and a not in addrmod
                and a not in boot_owned
                and v != pristine.get(a)
            }
            rec["snapshot_path"] = snap_path
            rec["signature"] = tuple(sorted(diff.items()))
        discovered.append(rec)

    n_pass = sum(1 for r in discovered if r["baseline"] == PASS)
    n_with_footprint = sum(1 for r in discovered if r["signature"])
    print(
        f"[discover] {n_pass}/{len(discovered)} baseline PASS, {n_with_footprint} with a real "
        f"(non-empty) write footprint",
        file=sys.stderr,
    )

    # Some values in the config space make dedup never converge. There's a pretty clean cut
    # between those (often L1 addresses and such) and real configuration, so exclude them.
    per_addr_vals = {}
    for r in discovered:
        if r["signature"]:
            for a, v in r["signature"]:
                per_addr_vals.setdefault(a, set()).add(v)
    high_card = {
        a
        for a, vals in per_addr_vals.items()
        if len(vals) > args.dedup_cardinality_cutoff
    }
    if high_card:
        print(
            f"discover_catalog: excluding {len(high_card)} fields from the dedup key: "
            + ", ".join(
                f"addr32={a}({len(per_addr_vals[a])} values)" for a in sorted(high_card)
            ),
            file=sys.stderr,
        )

    seen_sigs = {}
    representatives = []
    for r in discovered:
        if not r["signature"]:
            continue
        dedup_key = tuple((a, v) for a, v in r["signature"] if a not in high_card)
        if dedup_key not in seen_sigs:
            seen_sigs[dedup_key] = r["test_id"]
            representatives.append(r)
    print(
        f"discover_catalog: found {len(representatives)} distinct write signatures "
        f"(deduped from {n_with_footprint}, cutoff={args.dedup_cardinality_cutoff})",
        file=sys.stderr,
    )

    with open(os.path.join(args.out_dir, "discovery_raw.json"), "w") as f:
        json.dump(
            [
                {k: (list(v) if isinstance(v, tuple) else v) for k, v in r.items()}
                for r in discovered
            ],
            f,
            indent=2,
        )

    for r in representatives:
        entries = build_restore_entries(args.arch, r["snapshot_path"])

        addrmod_path = r["snapshot_path"].replace(".snapshot.json", ".addrmod.json")
        ch1x_path = addrmod_path.replace(".addrmod.json", ".adc_ch1x.json")
        ch1x = None
        if os.path.exists(ch1x_path):
            with open(ch1x_path) as f:
                ch1x_snap = json.load(f)
            ch1x = [ch1x_snap["unpacker"], ch1x_snap["packer"]]
        entries += build_addrmod_restore_entries(addrmod_path, ch1x=ch1x)

        restore_path = r["snapshot_path"].replace(".snapshot.json", ".restore.json")
        with open(restore_path, "w") as f:
            json.dump({"entries": entries}, f)
        r["restore_path"] = restore_path

    manifest = {
        "arch": args.arch,
        "source": "discover_catalog.py",
        "seed": seed,
        "sample_per_test": args.sample_per_test,
        "pristine_snapshot": pristine_snap_path,
        "ops": [],
    }
    for i, r in enumerate(representatives):
        family = re.sub(r"^test_", "", r["test_file"]).replace(".py", "")
        manifest["ops"].append(
            {
                "key": f"{family}__{i:03d}",
                "op_family": family,
                "test_file": r["test_file"],
                "test_id": r["test_id"],
                "baseline": r["baseline"],
                "snapshot_path": r["snapshot_path"],
                "restore_path": r["restore_path"],
            }
        )
    with open(args.manifest, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n========== DISCOVER CATALOG RESULT ==========", file=sys.stderr)
    print(
        f"sampled: {len(sampled)}\ncandidates: {n_with_footprint}\n"
        f"deduped: {len(representatives)}",
        file=sys.stderr,
    )
    print(f"manifest: {args.manifest}", file=sys.stderr)


if __name__ == "__main__":
    main()
