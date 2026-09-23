#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Run a multi-device (CCL) pytest against the craq-sim multichip simulator or real
hardware — one process per selected topology.

This is the companion multi-device sim-infra for the tt-metal tree. The op-gen eval
pipeline (tt_ops_code_gen: eval/run_eval.py, prompts/verifier.txt) drives all CCL
verification through it; `scripts/run_safe_pytest.sh` is the WRONG runner for CCL ops
(in sim it forces slow dispatch and has no multichip/hang awareness).

Topologies are declared in scripts/multidevice_sim_topologies.yaml. Each entry pins a
mesh_shape + fabric_config + arch + runtime (sim|hardware) and, for sim, the craq-sim
descriptor triple. A CCL test MUST open exactly the topology's mesh_shape/fabric_config
or fabric init hangs ("Fabric Router Sync: Timeout").

Usage
-----
  run_multidevice_sim_pytest.py --list [--op OP] [--runtime sim|hardware]
  run_multidevice_sim_pytest.py --op OP        [--runtime R] [--timeout S] -- <pytest args>
  run_multidevice_sim_pytest.py --topology NAME              [--timeout S] -- <pytest args>

Everything after `--` is forwarded verbatim to `pytest` (target, --splits/--group,
--junitxml=..., -q/-v). Each selected topology prints one line:
  MULTIDEV_SIM_RESULT[<topology>]: PASS|FAIL|HANG|ERROR
Aggregate exit code: 0 all-PASS / 1 any FAIL / 2 any HANG / 3 config error.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = REPO / "scripts" / "multidevice_sim_topologies.yaml"
ARCH_SUFFIX = {"wormhole": "wh", "blackhole": "bh"}

PASS, FAIL, HANG, ERROR = "PASS", "FAIL", "HANG", "ERROR"


def load_topologies(matrix):
    import yaml
    cfg = yaml.safe_load(Path(matrix).read_text()) or {}
    return cfg.get("topologies", []) or []


def _craq_dir():
    return Path(os.environ.get("CRAQ_SIM_DIR", "/localdev/wransom/craq-sim"))


def _craq_data_dir():
    return Path(os.environ.get("CRAQ_SIM_DATA_DIR", str(_craq_dir() / "data")))


def _arch_suffix(arch):
    return ARCH_SUFFIX.get(arch, arch)


def _resolve_desc(path_str, arch):
    """Descriptor path: absolute (or env-expanded) as-is, else <data>/<wh|bh>/<name>."""
    p = Path(os.path.expandvars(os.path.expanduser(str(path_str))))
    return p if p.is_absolute() else _craq_data_dir() / _arch_suffix(arch) / p


def _sim_lib(arch):
    env = os.environ.get(f"TTSIM_LIB_{_arch_suffix(arch).upper()}") or os.environ.get("TTSIM_LIB")
    if env:
        return Path(env)
    return _craq_dir() / "src" / "_out" / f"release_{_arch_suffix(arch)}" / "libttsim.so"


def select(topos, op, topology, runtime):
    """Topologies matching --topology (exact name) or --op (+ runtime). --topology wins."""
    out = []
    for t in topos:
        if topology:
            if t.get("name") == topology:
                out.append(t)
            continue
        if op and op not in (t.get("applies_to_ops") or []):
            continue
        if runtime and t.get("runtime", "sim") != runtime:
            continue
        out.append(t)
    return out


def sim_env(t):
    """(env, missing_files) for a sim run. missing_files non-empty => config ERROR."""
    arch = t.get("arch", "blackhole")
    env = os.environ.copy()
    env.pop("TT_METAL_SLOW_DISPATCH_MODE", None)  # keep FAST dispatch
    lib = _sim_lib(arch)
    cluster = _resolve_desc(t["cluster_desc"], arch)
    mesh = _resolve_desc(t["mesh_graph_desc"], arch)
    missing = [str(p) for p in (lib, cluster, mesh) if not p.exists()]
    env.update({
        "TT_METAL_SIMULATOR": str(lib),
        "TT_METAL_MOCK_CLUSTER_DESC_PATH": str(cluster),
        "TT_MESH_GRAPH_DESC_PATH": str(mesh),
        "TT_METAL_DRAM_BACKED_CQ": "1",
        "TT_METAL_DISABLE_PRECOMPILED_FW": "1",
        "TT_METAL_SIMULATOR_CQ_WAIT_CLOCKS": str(t.get("cq_wait_clocks", 200)),
        "TTSIM_HANG_WATCHDOG_CLOCKS": str(t.get("hang_watchdog_clocks", 500000000)),
        "ARCH_NAME": arch,
    })
    return env, missing


def hw_env(t):
    env = os.environ.copy()
    env["ARCH_NAME"] = t.get("arch", "wormhole")
    return env


def run_one(t, pytest_args, timeout):
    name = t.get("name", "?")
    is_hw = t.get("runtime", "sim") == "hardware"
    if is_hw:
        env, missing = hw_env(t), []
    else:
        env, missing = sim_env(t)
    if missing:
        print(f"MULTIDEV_SIM_RESULT[{name}]: {ERROR}", flush=True)
        sys.stderr.write(
            f"[{name}] CONFIG ERROR — sim files not found:\n  " + "\n  ".join(missing) + "\n"
            "  Stage the craq-sim descriptors + libttsim for this arch, or set "
            "CRAQ_SIM_DIR / CRAQ_SIM_DATA_DIR / TTSIM_LIB_<ARCH>.\n")
        return ERROR
    cmd = [sys.executable, "-m", "pytest", *pytest_args]
    sys.stderr.write(
        f"[{name}] {'HARDWARE' if is_hw else 'sim ' + t.get('arch', '')} "
        f"mesh_shape={tuple(t.get('mesh_shape') or ())} fabric_config={t.get('fabric_config')}\n"
        f"[{name}] $ {' '.join(cmd)}\n")
    sys.stderr.flush()
    try:
        rc = subprocess.run(cmd, cwd=REPO, env=env, timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        print(f"MULTIDEV_SIM_RESULT[{name}]: {HANG}", flush=True)
        return HANG
    # pytest exit codes: 0 pass, 1 tests failed, 2 interrupted, 5 no tests collected.
    status = PASS if rc == 0 else (ERROR if rc == 5 else (HANG if rc == 2 else FAIL))
    print(f"MULTIDEV_SIM_RESULT[{name}]: {status}", flush=True)
    return status


def do_list(topos):
    if not topos:
        print("(no matching topologies)")
        return
    for t in topos:
        rt = t.get("runtime", "sim")
        print(f"- {t.get('name')}  runtime={rt}  arch={t.get('arch')}  "
              f"mesh_shape={tuple(t.get('mesh_shape') or ())}  "
              f"fabric_config={t.get('fabric_config')}"
              f"{'  [grade_primary]' if t.get('grade_primary') else ''}")
        print(f"    applies_to_ops: {', '.join(t.get('applies_to_ops') or []) or '(none)'}")
        if rt == "sim":
            arch = t.get("arch", "blackhole")
            lib = _sim_lib(arch)
            cluster = _resolve_desc(t.get("cluster_desc", "?"), arch)
            mesh = _resolve_desc(t.get("mesh_graph_desc", "?"), arch)
            def _mark(p):
                return "" if Path(p).exists() else "  (MISSING)"
            print(f"    libttsim       : {lib}{_mark(lib)}")
            print(f"    cluster_desc   : {cluster}{_mark(cluster)}")
            print(f"    mesh_graph_desc: {mesh}{_mark(mesh)}")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # Split off pytest args at the first bare "--" so argparse never sees them.
    if "--" in argv:
        i = argv.index("--")
        pre, pytest_args = argv[:i], argv[i + 1:]
    else:
        pre, pytest_args = argv, []

    ap = argparse.ArgumentParser(
        prog="run_multidevice_sim_pytest.py",
        description="Run a multi-device (CCL) pytest on craq-sim or hardware, one process per topology.")
    ap.add_argument("--op", help="grade every topology whose applies_to_ops lists this op")
    ap.add_argument("--topology", help="grade exactly this topology by name")
    ap.add_argument("--runtime", choices=["sim", "hardware"],
                    default=os.environ.get("EVAL_RUNTIME", "sim"),
                    help="sim (craq-sim, default) or hardware (real silicon)")
    ap.add_argument("--list", action="store_true", help="print matching topologies and exit")
    ap.add_argument("--timeout", type=int, default=3600, help="per-topology wall-clock backstop (s)")
    ap.add_argument("--matrix", default=str(DEFAULT_MATRIX), help="path to the topology matrix yaml")
    a = ap.parse_args(pre)

    try:
        topos = load_topologies(a.matrix)
    except FileNotFoundError:
        sys.stderr.write(f"topology matrix not found: {a.matrix}\n")
        return 3

    if a.list:
        if a.op or a.topology:
            sel = select(topos, a.op, a.topology, a.runtime)
        else:
            sel = [t for t in topos if t.get("runtime", "sim") == a.runtime] if a.runtime else topos
        do_list(sel)
        return 0

    if not (a.op or a.topology):
        ap.error("one of --op or --topology is required (or --list)")

    sel = select(topos, a.op, a.topology, a.runtime)
    if not sel:
        sys.stderr.write(
            f"no topology matches op={a.op!r} topology={a.topology!r} runtime={a.runtime!r} "
            f"(see --list)\n")
        return 3

    results = [run_one(t, pytest_args, a.timeout) for t in sel]
    if HANG in results:
        return 2
    if FAIL in results:
        return 1
    if ERROR in results:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
