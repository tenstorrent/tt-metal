#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Deterministic multi-device (craq-sim) pytest runner — the multichip analog of
scripts/run_safe_pytest.sh.

run_safe_pytest.sh is single-device: in sim mode it FORCES slow dispatch and does NO
hang detection (it assumes wall-clock timeouts are meaningless at kHz). The proven
craq-sim multichip recipe is the opposite — FAST dispatch, with a sim *clock*-based
watchdog plus a wall-clock backstop. This runner encodes that recipe and drives a
declarative topology matrix so CCL-op verification is reproducible by construction.

Per topology it:
  * sets the sim env triple (TT_METAL_SIMULATOR + mock-cluster + mesh-graph descriptors)
    plus ARCH_NAME and the sim CQ env, and UNSETS TT_METAL_SLOW_DISPATCH_MODE (fast dispatch);
  * arms the sim hang watchdog (TTSIM_HANG_WATCHDOG_CLOCKS) and a wall-clock backstop;
  * runs pytest in its OWN process (craq-sim's dlopen state is process-global: one
    process == one topology), serially by default for determinism;
  * classifies PASS / FAIL / HANG / ERROR and writes a per-topology log;
  * aggregates: any *required* topology not PASS => non-zero exit.

Usage:
  scripts/run_multidevice_sim_pytest.py --op point_to_point  <pytest args...>
  scripts/run_multidevice_sim_pytest.py --topology bh_8xP150_p2p -- -k shape_coords_layout0 -x
  scripts/run_multidevice_sim_pytest.py --list

Selecting topologies:
  --op OPNAME        run the given pytest args under every topology whose applies_to_ops
                     lists OPNAME (the verifier path: it knows the op, not the topology).
  --topology NAME    run under the named topology (repeatable).
  --list             print the resolved matrix and exit.
Anything after `--` (or any unrecognized trailing args) is passed verbatim to pytest.

Exit codes: 0 all required topologies PASS; 1 a required topology FAIL/ERROR;
2 a required topology HANG; 3 configuration/usage error.
"""
from __future__ import annotations

import argparse
import datetime
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.stderr.write("run_multidevice_sim_pytest: PyYAML required (pip install pyyaml)\n")
    sys.exit(3)

REPO_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = Path(__file__).resolve().parent / "multidevice_sim_topologies.yaml"

# Exit codes
EXIT_OK, EXIT_FAIL, EXIT_HANG, EXIT_CONFIG = 0, 1, 2, 3


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------
def _resolve_default(val):
    """A default is either a scalar or {env: VAR, default: PATH} (env-or-fallback)."""
    if isinstance(val, dict) and "default" in val:
        return os.environ.get(val.get("env", ""), val["default"])
    return val


def load_config(path: Path):
    if not path.is_file():
        sys.stderr.write(f"run_multidevice_sim_pytest: config not found: {path}\n")
        sys.exit(EXIT_CONFIG)
    cfg = yaml.safe_load(path.read_text())
    raw_defaults = cfg.get("defaults", {})
    defaults = {k: _resolve_default(v) for k, v in raw_defaults.items()}
    # `{repo}` resolves to the tt-metal checkout root, for descriptors that ship in-tree
    # (e.g. tests/tt_metal/tt_fabric/custom_mesh_descriptors/) rather than in craq-sim's data/.
    defaults.setdefault("repo", str(REPO_DIR))
    topologies = []
    for t in cfg.get("topologies", []):
        t = dict(t)
        # Expand {placeholder} path fields against the resolved defaults.
        for key in ("sim_so", "cluster_desc", "mesh_graph_desc"):
            if key in t and isinstance(t[key], str):
                t[key] = t[key].format(**defaults)
        t.setdefault("required", False)
        t.setdefault("extra_env", {})
        t.setdefault("applies_to_ops", [])
        # Per-topology overrides fall back to the defaults block.
        t["cq_wait_clocks"] = t.get("cq_wait_clocks", defaults.get("cq_wait_clocks", 200))
        t["hang_watchdog_clocks"] = t.get("hang_watchdog_clocks", defaults.get("hang_watchdog_clocks", 2_000_000))
        t["wall_clock_timeout_s"] = t.get("wall_clock_timeout_s", defaults.get("wall_clock_timeout_s", 900))
        topologies.append(t)
    return defaults, topologies


def select_topologies(topologies, op, names):
    if names:
        by_name = {t["name"]: t for t in topologies}
        missing = [n for n in names if n not in by_name]
        if missing:
            sys.stderr.write(f"run_multidevice_sim_pytest: unknown topology(s): {missing}\n")
            sys.stderr.write(f"  available: {[t['name'] for t in topologies]}\n")
            sys.exit(EXIT_CONFIG)
        return [by_name[n] for n in names]
    if op:
        sel = [t for t in topologies if op in t.get("applies_to_ops", [])]
        if not sel:
            sys.stderr.write(f"run_multidevice_sim_pytest: no topology lists op '{op}' in applies_to_ops\n")
            sys.stderr.write(f"  matrix: {[(t['name'], t['applies_to_ops']) for t in topologies]}\n")
            sys.exit(EXIT_CONFIG)
        return sel
    sys.stderr.write("run_multidevice_sim_pytest: must pass --op OPNAME or --topology NAME (or --list)\n")
    sys.exit(EXIT_CONFIG)


# --------------------------------------------------------------------------------------
# Env
# --------------------------------------------------------------------------------------
def build_env(topo) -> dict:
    """The craq-sim multichip recipe env for one topology (FAST dispatch)."""
    env = dict(os.environ)
    # Do NOT inherit run_safe_pytest's slow-dispatch forcing — the recipe is fast dispatch.
    env.pop("TT_METAL_SLOW_DISPATCH_MODE", None)
    env["TT_METAL_SIMULATOR"] = topo["sim_so"]
    env["TT_METAL_MOCK_CLUSTER_DESC_PATH"] = topo["cluster_desc"]
    env["TT_MESH_GRAPH_DESC_PATH"] = topo["mesh_graph_desc"]
    env["ARCH_NAME"] = topo["arch"]
    env["TT_METAL_DRAM_BACKED_CQ"] = "1"
    env["TT_METAL_DISABLE_PRECOMPILED_FW"] = "1"
    env["TT_METAL_SIMULATOR_CQ_WAIT_CLOCKS"] = str(topo["cq_wait_clocks"])
    # Sim clock-based hang watchdog (primary). libttsim aborts the process when it fires,
    # which the wall-clock backstop below would otherwise have to wait out.
    env["TTSIM_HANG_WATCHDOG_CLOCKS"] = str(topo["hang_watchdog_clocks"])
    # Expose this topology's mesh shape + fabric_config so ONE topology-adaptive test can
    # size its mesh_device fixture per topology — this is what lets `--op` fan a single
    # confirmation/golden test out across every topology in the matrix (a test that hardcodes
    # a mesh shape only matches the one topology with that shape; the rest hang fabric init).
    if topo.get("mesh_shape") is not None:
        env["MULTIDEV_SIM_MESH_SHAPE"] = ",".join(str(d) for d in topo["mesh_shape"])
    if topo.get("fabric_config"):
        env["MULTIDEV_SIM_FABRIC_CONFIG"] = str(topo["fabric_config"])
    env["MULTIDEV_SIM_TOPOLOGY"] = topo["name"]
    for k, v in (topo.get("extra_env") or {}).items():
        env[str(k)] = str(v)
    return env


def _rewrite_junit_path(pytest_args, topo_name):
    """When fanning one --junitxml grade out over MULTIPLE topologies, give each topology its
    own junit (PATH.xml -> PATH.<topo>.xml) so they don't overwrite each other. Returns the
    (possibly) rewritten args + the rewritten junit path (or None if no --junitxml present)."""
    out, junit = [], None
    for a in pytest_args:
        if a.startswith("--junitxml="):
            path = a[len("--junitxml=") :]
            stem, dot, ext = path.rpartition(".")
            junit = f"{stem}.{topo_name}.{ext}" if dot else f"{path}.{topo_name}"
            out.append(f"--junitxml={junit}")
        else:
            out.append(a)
    return out, junit


def validate_paths(topo) -> list[str]:
    """Fail-fast: a missing sim/.so or descriptor is a config error, not a silent wrong run."""
    problems = []
    so = Path(topo["sim_so"])
    if not so.is_file():
        problems.append(f"sim .so missing: {so}  (build+stage per SETUP.md, or set the TT_SIM_*_SO env var)")
    else:
        # craq-sim loads soc_descriptor.yaml from beside the .so.
        soc = so.parent / "soc_descriptor.yaml"
        if not soc.is_file():
            problems.append(f"soc_descriptor.yaml missing beside the sim .so: {soc}")
    for key, label in (("cluster_desc", "mock cluster descriptor"), ("mesh_graph_desc", "mesh-graph descriptor")):
        p = Path(topo[key])
        if not p.is_file():
            problems.append(f"{label} missing: {p}")
    return problems


# --------------------------------------------------------------------------------------
# Run one topology
# --------------------------------------------------------------------------------------
def run_one(topo, pytest_args, log_dir) -> tuple[str, str]:
    """Returns (verdict, log_path). verdict in {PASS, FAIL, HANG, ERROR}."""
    name = topo["name"]
    problems = validate_paths(topo)
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{name}_{stamp}.log"

    print(
        f"\n{'=' * 78}\n[multidevice-sim] topology: {name}  (arch={topo['arch']}, required={topo['required']})",
        flush=True,
    )
    if problems:
        with log_path.open("w") as lf:
            lf.write(f"CONFIG ERROR for topology {name}:\n" + "\n".join(problems) + "\n")
        for p in problems:
            print(f"  CONFIG ERROR: {p}", flush=True)
        print(f"MULTIDEV_SIM_RESULT[{name}]: ERROR (config)", flush=True)
        return "ERROR", str(log_path)

    env = build_env(topo)
    cmd = [sys.executable, "-m", "pytest", *pytest_args]
    timeout_s = int(topo["wall_clock_timeout_s"])

    print(f"  sim_so          = {env['TT_METAL_SIMULATOR']}", flush=True)
    print(f"  cluster_desc    = {env['TT_METAL_MOCK_CLUSTER_DESC_PATH']}", flush=True)
    print(f"  mesh_graph_desc = {env['TT_MESH_GRAPH_DESC_PATH']}", flush=True)
    print(f"  watchdog_clocks = {env['TTSIM_HANG_WATCHDOG_CLOCKS']}   wall_clock_timeout = {timeout_s}s", flush=True)
    print(f"  $ {' '.join(cmd)}", flush=True)
    print(f"  log -> {log_path}", flush=True)

    # Own process group so a wall-clock-timeout kill takes pytest + the in-process sim + any orphans.
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    timed_out = {"v": False}

    def _killer():
        timed_out["v"] = True
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass

    timer = threading.Timer(timeout_s, _killer)
    timer.start()
    watchdog_fired = False
    try:
        with log_path.open("w") as lf:
            for line in proc.stdout:  # tee live output to console + log
                sys.stdout.write(line)
                sys.stdout.flush()
                lf.write(line)
                if "hang watchdog fired" in line:
                    watchdog_fired = True
        proc.wait()
    finally:
        timer.cancel()
    rc = proc.returncode

    if timed_out["v"]:
        verdict = "HANG"
        detail = f"wall-clock timeout after {timeout_s}s"
    elif watchdog_fired:
        verdict = "HANG"
        detail = "sim hang watchdog fired"
    elif rc == 0:
        verdict = "PASS"
        detail = "pytest exit 0"
    elif rc == 5:
        verdict = "ERROR"
        detail = "pytest collected no tests (exit 5) — check the test selector"
    elif rc in (2, 3, 4):
        verdict = "ERROR"
        detail = f"pytest internal/usage/collection error (exit {rc})"
    elif rc < 0:
        verdict = "HANG"
        detail = f"killed by signal {-rc} (likely sim abort/deadlock)"
    else:
        verdict = "FAIL"
        detail = f"pytest reported failures (exit {rc})"

    print(f"MULTIDEV_SIM_RESULT[{name}]: {verdict} ({detail})", flush=True)
    return verdict, str(log_path)


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        add_help=True, description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="topology matrix yaml")
    ap.add_argument("--op", help="run under every topology whose applies_to_ops lists this op")
    ap.add_argument("--topology", action="append", default=[], help="run under this named topology (repeatable)")
    ap.add_argument("--list", action="store_true", help="print the resolved matrix and exit")
    ap.add_argument(
        "--log-dir", type=Path, default=REPO_DIR / "generated" / "multidevice_sim", help="where per-topology logs go"
    )
    ap.add_argument("--timeout", type=int, default=None, help="override the per-topology wall-clock backstop (seconds)")
    args, pytest_args = ap.parse_known_args()
    # Allow an explicit `--` separator before pytest args.
    if pytest_args and pytest_args[0] == "--":
        pytest_args = pytest_args[1:]

    defaults, topologies = load_config(args.config)

    if args.list:
        print(f"config: {args.config}")
        print(f"resolved defaults: {defaults}")
        for t in topologies:
            print(f"\n  {t['name']}  (arch={t['arch']}, required={t['required']}, ops={t['applies_to_ops']})")
            print(
                f"    mesh_shape      = {t.get('mesh_shape', '<unset>')}   fabric_config = {t.get('fabric_config', '<unset>')}"
            )
            print(
                f"      ^ the op's acceptance test MUST open a mesh_device of this shape + fabric_config (else fabric init hangs)"
            )
            print(f"    sim_so          = {t['sim_so']}")
            print(f"    cluster_desc    = {t['cluster_desc']}")
            print(f"    mesh_graph_desc = {t['mesh_graph_desc']}")
            missing = validate_paths(t)
            print(f"    paths           = {'OK' if not missing else 'MISSING: ' + '; '.join(missing)}")
        return EXIT_OK

    if not pytest_args:
        sys.stderr.write("run_multidevice_sim_pytest: no pytest args given (pass them after the options or after --)\n")
        return EXIT_CONFIG

    selected = select_topologies(topologies, args.op, args.topology)
    print(f"[multidevice-sim] running {len(selected)} topology(ies) serially: {[t['name'] for t in selected]}")
    multi = len(selected) > 1

    results = []
    for topo in selected:
        if args.timeout is not None:
            topo = {**topo, "wall_clock_timeout_s": args.timeout}
        # Fanning a --junitxml grade over multiple topologies: give each its own junit file.
        topo_args = _rewrite_junit_path(pytest_args, topo["name"])[0] if multi else pytest_args
        verdict, log_path = run_one(topo, topo_args, args.log_dir)
        results.append((topo, verdict, log_path))

    # Summary + aggregate exit.
    print(f"\n{'=' * 78}\n[multidevice-sim] SUMMARY")
    worst = EXIT_OK
    for topo, verdict, log_path in results:
        req = "required" if topo["required"] else "optional"
        print(f"  {verdict:5s}  {topo['name']:28s} [{req}]  {log_path}")
        if topo["required"] and verdict != "PASS":
            worst = max(worst, EXIT_HANG if verdict == "HANG" else EXIT_FAIL)
    print(f"[multidevice-sim] aggregate exit = {worst}")
    return worst


if __name__ == "__main__":
    sys.exit(main())
