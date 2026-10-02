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

# Duplicated from helpers/cfg_restore.py's _CFG_STATE_SIZE/_ADDR_MOD_ADDR32_BH rather than
# imported: that module is part of the `helpers` package (relative imports, requires ttexalens)
# while this script runs standalone with only its own directory on PYTHONPATH. Keep numerically
# in sync with cfg_restore.py and trisc.cpp's restore_state() by hand.
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
_BOOT_OWNED = {"blackhole": set()} # TODO: Wormhole

_RESTORE_SPACE_CONFIG = 0  # must match cfg_restore.py's RESTORE_SPACE_CONFIG / trisc.cpp
_RESTORE_SPACE_THREADCONFIG = 1  # must match cfg_restore.py's RESTORE_SPACE_THREADCONFIG
_RESTORE_SPACE_ADC_CH1X = 2  # must match cfg_restore.py's RESTORE_SPACE_ADC_CH1X

# addr32 2 bits 22-31 are firmware-owned: DISABLE_RISC_BP_Disable_{main,trisc,ncrisc} and their
# _bmp_clear_* siblings (cfg_defines.h), all packed into that top range. Masked out of restore so
# we never write firmware-owned bits.
#
# addr32 0 here holds only legacy ALU format fields. CFG_STATE_ID is a same-numbered but separate
# ThreadConfig field, not reachable through this Config-space path.
_DISABLE_RISC_BP_SHAMT = (
    22  # cfg_defines.h: DISABLE_RISC_BP_Disable_main_SHAMT, lowest firmware-owned bit
)
_RESTORE_MASK_OVERRIDE = {2: (1 << _DISABLE_RISC_BP_SHAMT) - 1}


def build_restore_entries(arch, pristine_path):
    """Restore plan from a captured pristine snapshot (host snapshot_cfg JSON: [[state,addr32,val]..]).

    State-0 cfg-bus words become Config-space full-word entries, re-establishing the shared banked
    baseline. Addr-mod words become ThreadConfig entries writing 0, the reset default.

    Config[state][addr32] (the shared double-buffered CFG bus) and ThreadConfig[thread][idx]
    (a separate per-thread-banked array, see BackendConfiguration.md) are different address
    spaces. They happen to reuse the same small integers for addr32.

    cfg_read()/cfg_write() can only reach Config. RISCV stores can't reach ThreadConfig at all;
    only SETC16 can. So there is no captured addr-mod value to replay here -- 0 is a guess, not
    a measurement. Replaying whatever Config-space value happens to share that addr32 number
    instead breaks more victims than this guess does. build_addrmod_restore_entries' real captured
    values, appended after these, overwrite this guess when a caller has them.
    """
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
    """One op's Config-space write footprint only: addr32 words where its captured snapshot
    differs from the pristine baseline. Deliberately does NOT exclude addr-mod-numbered Config
    addresses (unlike main()'s dedup signature) -- those are real, independent Config-space
    registers that happen to share a numeric index with a ThreadConfig addr-mod field, not the
    same state, and must be composed like any other Config word.
    """
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
    """ThreadConfig/addr-mod write footprint for a WHOLE chain of ops, merged per (thread, addr32).

    snapshot_addr_mod reads every (thread, addr32) pair unconditionally, so a single op's capture
    is a full snapshot of the whole addr-mod space, not a diff -- there's no real pristine addr-mod
    capture to diff against (ThreadConfig isn't reachable from a Config-space read at all).
    build_restore_entries' zero-guess convention applies here too: a captured 0 means "this op
    didn't write this thread here," so only nonzero values participate, with later chain members
    winning on a shared (thread, addr32). Concatenating each op's full per-op entry instead would
    collapse the whole chain to its last member for every address, since every op's capture covers
    the full range.
    """
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
        [_RESTORE_SPACE_THREADCONFIG, addr32, *v, 0] for addr32, v in sorted(merged.items())
    ]


def last_ch1x_entry(chain_ops):
    """ADC channel1-X isn't addressed (one physical register per unpacker/packer side), so unlike
    addr-mod there's nothing to merge per-field: whoever wrote it most recently is exactly what a
    real chain would leave, so the last chain member with a capture simply wins.
    """
    for op in reversed(chain_ops):
        _, _, ch1x_path = op_capture_paths(op)
        if os.path.exists(ch1x_path):
            with open(ch1x_path) as f:
                ch1x = json.load(f)
            return [[_RESTORE_SPACE_ADC_CH1X, 0, ch1x["unpacker"], ch1x["packer"], 0, 0]]
    return []


def build_chain_entries(arch, pristine_path, pristine, chain_ops):
    """Composes D real ops' own write footprints into one restore plan via most-recent-writer-
    per-field (chain_ops[0] then [1] ... then [-1], each overwriting the ones before it on any
    field they share) -- the state a victim sees at launch after a real D-deep kernel sequence,
    not just the single most-recent op pair_sweep.py's own depth-1 sweep already covers.

    Folds the whole chain down to one entry per (space, addr32) before returning -- the device
    side's last-write-wins apply loop would get the same final state from a raw concatenation of
    every op's entries, but shipping N separate entries per address makes the wire plan scale with
    chain depth for no reason (and overflows the fixed-size L1 scratch region trisc.cpp reads it
    from at depth 3+). Collapsing here instead bounds the plan size at the address space itself --
    every Config word plus every addr-mod address plus ch1x -- the same size regardless of depth.
    """
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
    """Real per-thread addr-mod restore entries from a captured snapshot (host snapshot_addr_mod
    JSON: [[thread, addr32, val], ...]), plus an optional ADC channel1-X entry.

    A thread with no captured entry for an address defaults to 0 (reset-default), matching
    build_restore_entries' fallback for the same address. ch1x, if given, is
    (unpacker0_val, packer_val) -- address_counters' channel1-X residue (see snapshot_adc_ch1x).
    Entries here are meant to be appended after build_restore_entries' output: applied in that
    order, a real captured value here overwrites that plan's reset-default-0 guess for the same
    address.
    """
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


# Embedded as a string, not a real file on disk, because pytest's `-p xdist_capture_plugin`
# needs to resolve a module by that bare name -- so this gets written out to a generated plugin
# dir at runtime instead. pytest_env also puts the real reconfig_escape/ dir on PYTHONPATH (same
# as for `-p xdist_plan_plugin` below), so the plugin body itself can still import discover_catalog
# normally.
_XDIST_CAPTURE_PLUGIN_SRC = '''\
"""pytest plugin: one-shot restore-to-pristine + direct post-exec residue capture, per test item.

snapshot_build.py reads residue by treating a 2nd launch's pre-launch snapshot as launch 1's
post-exec state. That needs two launches landing on the same core, a real risk under xdist since
nothing guarantees the same worker runs two separate pytest invocations for the same nodeid.

Discovery avoids that: restore-to-pristine in pytest_runtest_setup (the same mechanism
cfg_restore.py already uses), then read the CFG state directly via snapshot_cfg/thread_items in
pytest_runtest_teardown. Same process, same core, right after that item's own kernel finished, no
second launch needed. cfg_restore.py's own capture trick already proves residue survives a full
test-to-test boundary (fixture teardown plus the next test's setup); this only needs it to
survive from test-body-end to that same test's own teardown, a strictly smaller window.

--llk-pristine-restore=PATH   restore plan applied before every candidate item launches.
--llk-capture-outdir=DIR      each candidate's post-exec snapshot written to
                              <outdir>/<sanitized-nodeid>.snapshot.json (only for nodeids present
                              in --llk-candidates, one per line).
--llk-candidates=PATH         file of nodeids (one per line) this plugin should act on; any other
                              collected item is left untouched (defensive, in case of accidental
                              overlap with unrelated tests in the same invocation).
"""

import json
import os

from discover_catalog import _sanitize

_RESTORE_VAR = "LLK_CFG_RESTORE"


def pytest_addoption(parser):
    parser.addoption("--llk-pristine-restore", action="store", default=None)
    parser.addoption("--llk-capture-outdir", action="store", default=None)
    parser.addoption("--llk-candidates", action="store", default=None)


def pytest_configure(config):
    config._llk_restore_path = config.getoption("--llk-pristine-restore")
    config._llk_outdir = config.getoption("--llk-capture-outdir")
    cand_path = config.getoption("--llk-candidates")
    config._llk_candidates = set()
    if cand_path:
        with open(cand_path) as f:
            config._llk_candidates = {line.strip() for line in f if line.strip()}


def pytest_runtest_setup(item):
    if item.nodeid in item.config._llk_candidates and item.config._llk_restore_path:
        os.environ[_RESTORE_VAR] = item.config._llk_restore_path
    else:
        os.environ.pop(_RESTORE_VAR, None)


def pytest_runtest_teardown(item, nextitem):
    os.environ.pop(_RESTORE_VAR, None)
    if item.nodeid not in item.config._llk_candidates or not item.config._llk_outdir:
        return
    from helpers.cfg_restore import snapshot_addr_mod, snapshot_adc_ch1x, snapshot_cfg, thread_items
    from helpers.chip_architecture import get_chip_architecture
    from helpers.test_config import TestConfig

    arch = get_chip_architecture()
    items = thread_items(arch)
    snap = snapshot_cfg(TestConfig.TENSIX_LOCATION, items)
    out_path = os.path.join(item.config._llk_outdir, _sanitize(item.nodeid) + ".snapshot.json")
    with open(out_path, "w") as f:
        json.dump([[s, a, v] for (s, a), v in snap.items()], f)

    # Real per-thread addr-mod residue (ThreadConfig, not reachable via snapshot_cfg's Config[]
    # numbering -- see snapshot_addr_mod's docstring). Captured at the same point, so it reflects
    # this same candidate's just-finished kernel.
    addrmod_snap = snapshot_addr_mod(TestConfig.TENSIX_LOCATION)
    addrmod_path = os.path.join(item.config._llk_outdir, _sanitize(item.nodeid) + ".addrmod.json")
    with open(addrmod_path, "w") as f:
        json.dump([[t, a, v] for (t, a), v in addrmod_snap.items()], f)

    # address_counters channel1-X: separate debug-bus ADC state, not reachable via snapshot_cfg
    # or snapshot_addr_mod (both CFG-bus only). Same rationale as the addrmod capture above.
    ch1x_snap = snapshot_adc_ch1x(TestConfig.TENSIX_LOCATION)
    ch1x_path = os.path.join(item.config._llk_outdir, _sanitize(item.nodeid) + ".adc_ch1x.json")
    with open(ch1x_path, "w") as f:
        json.dump(ch1x_snap, f)
'''


def _write_plugins(plugin_dir):
    os.makedirs(plugin_dir, exist_ok=True)
    with open(os.path.join(plugin_dir, "xdist_capture_plugin.py"), "w") as f:
        f.write(_XDIST_CAPTURE_PLUGIN_SRC)


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


def pytest_env(worktree, plugin_dir=None):
    env = dict(os.environ)
    reconfig_escape_dir = os.path.join(
        worktree, "tests", "python_tests", "reconfig_escape"
    )
    path_parts = (
        [plugin_dir, reconfig_escape_dir] if plugin_dir else [reconfig_escape_dir]
    )
    env["PYTHONPATH"] = (
        os.pathsep.join(path_parts) + os.pathsep + env.get("PYTHONPATH", "")
    )
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
    """Group by (file, function) -- nodeid up to its parametrize '[' -- not by file alone: several
    files bundle multiple test functions with independent parametrize spaces, and a flat per-file
    cap skews toward whichever function pytest happens to list first."""
    by_test = {}
    for nid in nodeids:
        t = nid.split("[", 1)[0]
        by_test.setdefault(t, []).append(nid)
    sampled = []
    for ids in by_test.values():
        sampled += ids if len(ids) <= n_per_test else rng.sample(ids, n_per_test)
    return sampled


_ARGV_BATCH = (
    500  # comfortably under the OS argv-length limit even for long parametrized ids
)


def _chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def compile_all(worktree, arch, nodeids, jobs, timeout):
    """Compiles via a short-argv driver: nodeids go through a JSON file, not argv, so full-suite
    sample sizes never risk the OS argument-list limit. Targets exact nodeids rather than whole
    test files: file-level targeting would collect unrelated tests sharing the file, and running
    those alongside heavy deselection trips real fixture-teardown failures. Runs as a single
    pytest session, not batched subprocesses: test_config.py's PRODUCE-mode session start wipes
    the shared build-artifact directory, so multiple sessions would destroy each other's output.
    """
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
    plugin_dir,
):
    """Batches for the same argv-limit reason as compile_all. Each batch is a real, separate
    pytest invocation with its own junit file; the batches' testcases are merged into one junit at
    junit_path so the caller's parse_junit(junit_path) call is unaffected by batching.
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
            env={**pytest_env(worktree, plugin_dir), "CHIP_ARCH": arch},
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


def run_gate_round(
    worktree, arch, nodeids, plan_map_path, jobs, timeout, junit_path, plugin_dir
):
    """Batched for the same argv-limit reason as run_discovery_round -- the representative count
    is normally well under _ARGV_BATCH, but nothing here should silently break if the catalog
    grows past it."""
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
            "xdist_plan_plugin",
            f"--llk-plan-map={plan_map_path}",
            f"--timeout={timeout}",
            f"--junitxml={batch_junit}",
        ] + batch
        last_proc = subprocess.run(
            cmd,
            cwd=os.path.join(worktree, "tests", "python_tests"),
            env={**pytest_env(worktree, plugin_dir), "CHIP_ARCH": arch},
            capture_output=True,
            text=True,
        )
        if os.path.exists(batch_junit):
            merged_cases.extend(ET.parse(batch_junit).getroot().iter("testcase"))
    if merged_cases:
        suite = ET.Element(
            "testsuite", name="reconfig_escape_gate", tests=str(len(merged_cases))
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
        "grows with this but with diminishing returns at scale",
    )
    p.add_argument(
        "--dedup-cardinality-cutoff",
        type=int,
        default=15,
        help="exclude a CFG field from the dedup key if it takes on more than this "
        "many distinct observed values across the whole sample (address/geometry"
        "-encoding fields, not a discrete mode-select)",
    )
    p.add_argument(
        "--jobs",
        type=int,
        default=8,
        help="xdist worker count for the hardware-touching discovery/gate rounds -- "
        "one physical Tensix core per worker, matching the real LLK CI's own -n 8",
    )
    p.add_argument(
        "--compile-jobs",
        type=int,
        default=8,
        help="xdist worker count for compile-producer steps (CPU-only, no hardware)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for sampling. Default derives from the current ISO year+week, "
        "so successive weekly CI runs sample different candidates instead of "
        "redrawing the same ones forever; pass an explicit value for a "
        "reproducible local run",
    )
    p.add_argument("--timeout", type=int, default=90)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    plugin_dir = os.path.join(args.out_dir, "_plugins")
    _write_plugins(plugin_dir)

    print("[discover] collecting candidate pool...", file=sys.stderr)
    pool = collect_all(args.worktree, args.arch, args.timeout)
    print(f"[discover] pool: {len(pool)} nodeids", file=sys.stderr)
    if not pool:
        raise SystemExit("collected 0 test items -- worktree/arch misconfigured?")

    seed = (
        args.seed
        if args.seed is not None
        else int(datetime.date.today().strftime("%Y%V"))
    )
    print(
        f"[discover] seed={seed}"
        + (" (explicit)" if args.seed is not None else " (derived from ISO year+week)"),
        file=sys.stderr,
    )
    rng = random.Random(seed)
    sampled = sample_per_test(pool, args.sample_per_test, rng)
    n_files = len(set(n.split("::", 1)[0] for n in sampled))
    n_tests = len(set(n.split("[", 1)[0] for n in sampled))
    print(
        f"[discover] sampled {len(sampled)} candidates ({n_tests} test functions across "
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
    print(f"[discover] pristine captured -> {pristine_restore_path}", file=sys.stderr)

    print(
        f"[discover] compiling {len(sampled)} candidates (producer, -n {args.compile_jobs})...",
        file=sys.stderr,
    )
    cproc = compile_all(
        args.worktree, args.arch, sampled, args.compile_jobs, args.timeout
    )
    if cproc.returncode != 0:
        # A broad random sample of the real suite will always catch a handful of pre-existing,
        # unrelated broken tests (bad fixtures, CLI-arg mismatches, etc). Don't abort the whole
        # discovery pass over that -- the discovery round's own per-candidate baseline PASS/FAIL
        # already excludes anything that fails to compile or run.
        print(
            f"[discover] WARNING: compile-producer had failures (rc={cproc.returncode}); "
            f"affected candidates will show up as non-PASS baseline and be excluded",
            file=sys.stderr,
        )

    capture_dir = os.path.join(args.out_dir, "captures")
    os.makedirs(capture_dir, exist_ok=True)
    junit_path = os.path.join(args.out_dir, "discovery.junit.xml")

    print(
        f"[discover] discovery round: {len(sampled)} candidates across -n {args.jobs}...",
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
        plugin_dir,
    )
    if not os.path.exists(junit_path):
        print(rproc.stdout[-4000:], file=sys.stderr)
        print(rproc.stderr[-4000:], file=sys.stderr)
        raise SystemExit("discovery round produced no junit report")
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

    # Exact-value dedup never converges. A handful of CFG words are per-tile L1 pointers or
    # tile-descriptor words (addr32 76/77/124/125 = THCON_SEC0/1_REG3_Base_address[_cntx1], the
    # per-tile L1 src addr, unconditionally rewritten by every unpack execute path) whose value
    # tracks wherever a test happened to allocate its tensors, not a mode/format axis.
    #
    # Measured cardinality across the whole sample separates these cleanly from genuine discrete
    # mode-selects, without needing a hand-curated name list (a couple, addr32 69/70, aren't even
    # named fields). A real discrete mode-select stays under ~15 distinct values across thousands
    # of candidates; an address/geometry-encoding field runs into the hundreds.
    #
    # Excluding those from the dedup key only -- the restore payload still replays every field
    # byte-for-byte, and replaying a stale-but-valid address is harmless -- turns a non-converging
    # accumulation curve into one that visibly decelerates.
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
            f"[discover] excluding {len(high_card)} high-cardinality (address/geometry-like) "
            f"fields from the dedup key: "
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
        f"[discover] {len(representatives)} distinct write-footprint signatures "
        f"(deduped from {n_with_footprint}, cardinality cutoff={args.dedup_cardinality_cutoff})",
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

    # --- Gate phase: only on the deduped representatives ---
    gate_nodeids = [r["test_id"] for r in representatives]
    plan_map = {}
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

        plan_map[r["test_id"]] = restore_path
    plan_map_path = os.path.join(args.out_dir, "gate_plan_map.json")
    with open(plan_map_path, "w") as f:
        json.dump(plan_map, f)
    gate_junit_path = os.path.join(args.out_dir, "gate.junit.xml")

    print(
        f"[discover] gate round: {len(gate_nodeids)} representatives across -n {args.jobs}...",
        file=sys.stderr,
    )
    _reset_card()
    gproc = run_gate_round(
        args.worktree,
        args.arch,
        gate_nodeids,
        plan_map_path,
        args.jobs,
        args.timeout,
        gate_junit_path,
        plugin_dir,
    )
    if not os.path.exists(gate_junit_path):
        print(gproc.stdout[-4000:], file=sys.stderr)
        print(gproc.stderr[-4000:], file=sys.stderr)
        raise SystemExit("gate round produced no junit report")
    gate_results = parse_junit(gate_junit_path)

    manifest = {
        "arch": args.arch,
        "source": "discover_catalog.py",
        "seed": seed,
        "sample_per_test": args.sample_per_test,
        "pristine_snapshot": pristine_snap_path,
        "ops": [],
    }
    for i, r in enumerate(representatives):
        gate_v = gate_results.get(r["test_id"], ENVERR)
        family = re.sub(r"^test_", "", r["test_file"]).replace(".py", "")
        manifest["ops"].append(
            {
                "key": f"{family}__{i:03d}",
                "op_family": family,
                "test_file": r["test_file"],
                "test_id": r["test_id"],
                "baseline": r["baseline"],
                "gate": gate_v,
                "usable": gate_v == PASS,
                "snapshot_path": r["snapshot_path"],
                "restore_path": r["restore_path"],
            }
        )
    with open(args.manifest, "w") as f:
        json.dump(manifest, f, indent=2)

    usable = sum(1 for o in manifest["ops"] if o["usable"])
    print(f"\n========== DISCOVER CATALOG RESULT ==========", file=sys.stderr)
    print(
        f"sampled: {len(sampled)}  baseline-PASS-with-footprint: {n_with_footprint}  "
        f"deduped: {len(representatives)}  gate-usable: {usable}",
        file=sys.stderr,
    )
    print(f"manifest -> {args.manifest}", file=sys.stderr)


if __name__ == "__main__":
    main()
