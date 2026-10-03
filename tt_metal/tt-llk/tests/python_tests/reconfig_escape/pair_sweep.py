#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Sweep phase for the weekly reconfig CI job.

Consumes the manifest from discover_catalog.py and sweeps every (X, K) pair in the catalog,
and checks whether K still passes after X's residue. If X passed the self-consistency check,
replay X's captured config residue before K's launch. Otherwise, tt-smi -r -> run X -> run K.

If --depth > 1, synthetic reachable hardware states are generated from the manifest.

Usage:
  python3 pair_sweep.py --worktree DIR --arch blackhole --manifest /path/to/manifest.json \
      --out /path/to/findings.jsonl [--self-pairs] [--jobs 8] [--timeout 90] [--port 5556] \
      [--splits N --group G] [--depth 2 --chains 80 --seed S]
"""

import argparse
import datetime
import json
import os
import random
import subprocess
import sys

import discover_catalog

PASS, FAIL, ENVERR = (
    discover_catalog.PASS,
    discover_catalog.FAIL,
    discover_catalog.ENVERR,
)
HANG = "HANG"
_CODE = {0: PASS, 1: FAIL, 5: HANG}
reset = discover_catalog._reset_card
pytest_env = discover_catalog.pytest_env


def _run_test_sh(
    worktree, mode, test_file, test_id, arch, port, timeout, env_extra=None
):
    cmd = [
        "bash",
        os.path.join(worktree, ".claude/scripts/run_test.sh"),
        mode,
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
    env = {**os.environ, **(env_extra or {})}
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    return proc


def simulate(worktree, arch, test_file, test_id, port, timeout, env_extra=None):
    proc = _run_test_sh(
        worktree, "simulate", test_file, test_id, arch, port, timeout, env_extra
    )
    if "does not exist" in (proc.stdout + proc.stderr):
        return ENVERR
    return _CODE.get(proc.returncode, ENVERR)


def compile_all(worktree, arch, nodeids, jobs, timeout):
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "--compile-producer",
        "-n",
        str(jobs),
        f"--timeout={timeout}",
    ] + nodeids
    return subprocess.run(
        cmd,
        cwd=os.path.join(worktree, "tests", "python_tests"),
        env={**pytest_env(worktree), "CHIP_ARCH": arch},
        capture_output=True,
        text=True,
    )


def run_round(worktree, arch, nodeids, plan_map_path, jobs, timeout, junit_path):
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
        f"--junitxml={junit_path}",
    ] + nodeids
    return subprocess.run(
        cmd,
        cwd=os.path.join(worktree, "tests", "python_tests"),
        env={**pytest_env(worktree), "CHIP_ARCH": arch},
        capture_output=True,
        text=True,
    )


parse_junit = discover_catalog.parse_junit


def verify_ground_truth(
    worktree, arch, polluter_nodeids, victim_nodeid, timeout, junit_path
):
    """Reset, then run the polluter(s) then the real victim on the same Tensix core. Returns
    the victim's status. polluter_nodeids is a list: one element at depth 1, D elements for a
    depth-D chain.
    """
    reset()
    cmd = (
        [
            sys.executable,
            "-m",
            "pytest",
            "--compile-consumer",
            f"--timeout={timeout}",
            f"--junitxml={junit_path}",
        ]
        + list(polluter_nodeids)
        + [
            victim_nodeid,
        ]
    )
    proc = subprocess.run(
        cmd,
        cwd=os.path.join(worktree, "tests", "python_tests"),
        env={**pytest_env(worktree), "CHIP_ARCH": arch},
        capture_output=True,
        text=True,
    )
    if not os.path.exists(junit_path):
        return ENVERR, proc
    return parse_junit(junit_path).get(victim_nodeid, ENVERR), proc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--worktree", required=True)
    p.add_argument("--arch", required=True, choices=["blackhole"])
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True, help="JSONL: every trial result")
    p.add_argument("--self-pairs", action="store_true")
    p.add_argument(
        "--jobs",
        type=int,
        default=8,
        help="xdist worker count, one per Tensix core",
    )
    p.add_argument("--port", type=int, default=5556)
    p.add_argument("--timeout", type=int, default=90)
    p.add_argument(
        "--skip-compile",
        action="store_true",
        help="attempt to use precompiled ELFs",
    )
    p.add_argument(
        "--skip-verify",
        action="store_true",
        help="don't check for false positives (only useful for debugging the tool)",
    )
    p.add_argument(
        "--splits",
        type=int,
        default=1,
        help="shard the polluter loop across this many machines (paired with --group)",
    )
    p.add_argument(
        "--group",
        type=int,
        default=1,
        help="what shard to run, in [1, --splits]",
    )
    p.add_argument(
        "--depth",
        type=int,
        default=1,
        help="test with synthetic plausible hardware states",
    )
    p.add_argument(
        "--chains",
        type=int,
        default=80,
        help="random chains to sample when --depth > 1",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for chain sampling (--depth > 1 only)",
    )
    args = p.parse_args()
    if not 1 <= args.group <= args.splits:
        p.error(f"--group must be in [1, {args.splits}] (got {args.group})")

    with open(args.manifest) as f:
        manifest = json.load(f)
    ops = manifest["ops"]
    op_by_key = {o["key"]: o for o in ops}
    victims = [o for o in ops if o.get("baseline") == PASS]
    restore_x = [o for o in victims if o.get("usable")]
    fallback_x = [o for o in victims if not o.get("usable")]
    nodeids = [v["test_id"] for v in victims]
    nodeid_by_key = {v["key"]: v["test_id"] for v in victims}

    if args.depth > 1:
        if args.depth > len(restore_x):
            p.error(
                f"--depth {args.depth} > {len(restore_x)} usable polluters in the manifest"
            )
        seed = (
            args.seed
            if args.seed is not None
            else int(datetime.date.today().strftime("%Y%V"))
        )
        print(
            f"pair_sweep: depth={args.depth} chains={args.chains} seed={seed}"
            + (
                " (explicit)"
                if args.seed is not None
                else " (derived from ISO year+week)"
            ),
            file=sys.stderr,
        )
        rng = random.Random(seed)
        pristine_path = manifest["pristine_snapshot"]
        with open(pristine_path) as f:
            pristine = {a: v for s, a, v in json.load(f) if s == 0}
        chains_dir = os.path.join(
            os.path.dirname(os.path.abspath(args.out)) or ".", "chains"
        )
        os.makedirs(chains_dir, exist_ok=True)

        chain_members = [rng.sample(restore_x, args.depth) for _ in range(args.chains)]
        restore_x = []
        for ci, chain in enumerate(chain_members):
            entries = discover_catalog.build_chain_entries(
                args.arch, pristine_path, pristine, chain
            )
            restore_path = os.path.join(chains_dir, f"chain{ci:03d}.restore.json")
            with open(restore_path, "w") as f:
                json.dump({"entries": entries}, f)
            restore_x.append(
                {
                    "key": "+".join(o["key"] for o in chain),
                    "restore_path": restore_path,
                    "members": [o["key"] for o in chain],
                }
            )
        fallback_x = []

    if args.splits > 1:
        restore_x = restore_x[args.group - 1 :: args.splits]
        fallback_x = fallback_x[args.group - 1 :: args.splits]

    shard_note = f" (shard {args.group}/{args.splits})" if args.splits > 1 else ""
    print(
        f"pair_sweep: victims={len(victims)} restore-mode polluters={len(restore_x)} "
        f"fallback polluters={len(fallback_x)}{shard_note}",
        file=sys.stderr,
    )

    out_f = open(args.out, "w")
    escapes = []

    def record(mode, x_key, k_key, verdict, k_baseline, extra=None):
        rec = {
            "mode": mode,
            "polluter": x_key,
            "victim": k_key,
            "verdict": verdict,
            "victim_baseline": k_baseline,
        }
        if extra:
            rec.update(extra)
        out_f.write(json.dumps(rec) + "\n")
        out_f.flush()
        if verdict != k_baseline:
            escapes.append(rec)
            print(
                f"pair_sweep: ESCAPE {x_key} -> {k_key}: {verdict} (baseline {k_baseline}) [{mode}]",
                file=sys.stderr,
            )

    # --- Restore-mode phase: one xdist round PER POLLUTER, every other victim run in that round
    #     in parallel across -n jobs workers, all pinned to that round's single restore plan. ---
    if restore_x:
        tmp_dir = os.path.dirname(os.path.abspath(args.out)) or "."
        os.makedirs(tmp_dir, exist_ok=True)
        if args.skip_compile:
            print(
                f"pair_sweep: --skip-compile: assuming all {len(nodeids)} victims are already "
                f"compiled",
                file=sys.stderr,
            )
        else:
            print(
                f"pair_sweep: compiling {len(nodeids)} victims once (producer, -n {args.jobs})...",
                file=sys.stderr,
            )
            cproc = compile_all(
                args.worktree, args.arch, nodeids, args.jobs, args.timeout
            )
            if cproc.returncode != 0:
                print(
                    f"pair_sweep: warning: compile step had failures (rc={cproc.returncode}); "
                    f"affected victims will show up as ENVERR in any round",
                    file=sys.stderr,
                )

        for xi, x in enumerate(restore_x):
            x_members = set(x.get("members", [x["key"]]))
            plan_map = {}
            for k, nodeid in zip(victims, nodeids):
                if k["key"] in x_members and not args.self_pairs:
                    continue
                plan_map[nodeid] = x["restore_path"]
            if not plan_map:
                continue
            round_nodeids = list(plan_map.keys())

            print(
                f"pair_sweep: restore-mode round {xi+1}/{len(restore_x)}: polluter {x['key']}, "
                f"{len(round_nodeids)} victims across -n {args.jobs}...",
                file=sys.stderr,
            )
            plan_map_path = os.path.join(tmp_dir, f"pairsweep_round{xi}.map.json")
            with open(plan_map_path, "w") as f:
                json.dump(plan_map, f)
            junit_path = os.path.join(tmp_dir, f"pairsweep_round{xi}.junit.xml")

            reset()
            rproc = run_round(
                args.worktree,
                args.arch,
                round_nodeids,
                plan_map_path,
                args.jobs,
                args.timeout,
                junit_path,
            )
            results = {}
            if not os.path.exists(junit_path):
                print(
                    f"pair_sweep: warning: round {xi+1} for polluter {x['key']} produced no "
                    f"junit report; every victim in it recorded as ENVERR",
                    file=sys.stderr,
                )
                print(rproc.stdout[-2000:], file=sys.stderr)
                print(rproc.stderr[-2000:], file=sys.stderr)
            else:
                results.update(parse_junit(junit_path))

            for k, nodeid in zip(victims, nodeids):
                if nodeid not in plan_map:
                    continue
                v = results.get(nodeid, ENVERR)
                record(
                    "restore",
                    x["key"],
                    k["key"],
                    v,
                    k["baseline"],
                    extra={"members": x["members"]} if args.depth > 1 else None,
                )

    # Fallback phase: full reset per pair, for X's that failed their own restore-gate.
    # TODO: measure in CI.
    for x in fallback_x:
        for k in victims:
            if x["key"] == k["key"] and not args.self_pairs:
                continue
            reset()
            # Compile both variants together in one producer invocation so neither evicts the
            # other's ELF, then run each via simulate.
            proc = subprocess.run(
                [
                    "bash",
                    os.path.join(args.worktree, ".claude/scripts/run_test.sh"),
                    "compile",
                    "--worktree",
                    args.worktree,
                    "--arch",
                    args.arch,
                    "--test",
                    x["test_file"],
                    "--test-id",
                    x["test_id"],
                    "--port",
                    str(args.port),
                    "--timeout",
                    str(args.timeout),
                ],
                capture_output=True,
                text=True,
            )
            subprocess.run(
                [
                    "bash",
                    os.path.join(args.worktree, ".claude/scripts/run_test.sh"),
                    "compile",
                    "--worktree",
                    args.worktree,
                    "--arch",
                    args.arch,
                    "--test",
                    k["test_file"],
                    "--test-id",
                    k["test_id"],
                    "--port",
                    str(args.port),
                    "--timeout",
                    str(args.timeout),
                ],
                capture_output=True,
                text=True,
            )
            vx = simulate(
                args.worktree,
                args.arch,
                x["test_file"],
                x["test_id"],
                args.port,
                args.timeout,
            )
            vk = simulate(
                args.worktree,
                args.arch,
                k["test_file"],
                k["test_id"],
                args.port,
                args.timeout,
            )
            record(
                "fullreset",
                x["key"],
                k["key"],
                vk,
                k["baseline"],
                extra={"polluter_verdict": vx},
            )

    out_f.close()

    # Verify phase: actually run the failure candidates
    candidates_path = os.path.splitext(args.out)[0] + ".candidates.json"
    if args.skip_verify:
        for e in escapes:
            e["verified"] = None
    else:
        verify_dir = os.path.dirname(os.path.abspath(args.out)) or "."
        to_verify = [e for e in escapes if e["mode"] == "restore"]
        print(
            f"pair_sweep: verifying {len(to_verify)} restore-mode escape(s)...",
            file=sys.stderr,
        )
        verified_so_far = 0
        for i, e in enumerate(escapes):
            if e["mode"] != "restore":
                e["verified"] = True  # fullreset already ran this
                continue
            verified_so_far += 1
            member_keys = e.get("members", [e["polluter"]])
            px = [nodeid_by_key[k] for k in member_keys if k in nodeid_by_key]
            vk = nodeid_by_key.get(e["victim"])
            if len(px) != len(member_keys) or not vk:
                e["verified"] = False
                e["verify_result"] = ENVERR
                continue

            # A chain escape only matters if it's not already visible to depth 1.
            # Check whether any single member alone already breaks this victim.
            if len(member_keys) > 1:
                subsumed_by = []
                for mk in member_keys:
                    m = op_by_key[mk]
                    single_plan_path = os.path.join(
                        verify_dir, f"verify_single_{i}_{mk}.map.json"
                    )
                    with open(single_plan_path, "w") as f:
                        json.dump({vk: m["restore_path"]}, f)
                    single_junit = os.path.join(
                        verify_dir, f"verify_single_{i}_{mk}.junit.xml"
                    )
                    reset()
                    run_round(
                        args.worktree,
                        args.arch,
                        [vk],
                        single_plan_path,
                        1,
                        args.timeout,
                        single_junit,
                    )
                    if os.path.exists(single_junit):
                        v = parse_junit(single_junit).get(vk, ENVERR)
                        if v != e["victim_baseline"]:
                            subsumed_by.append(mk)
                e["subsumed_by_depth1"] = subsumed_by

            junit_path = os.path.join(verify_dir, f"verify_{i}.junit.xml")
            result, _ = verify_ground_truth(
                args.worktree, args.arch, px, vk, args.timeout, junit_path
            )
            e["verify_result"] = result
            e["verified"] = result == FAIL
            tag = "CONFIRMED" if e["verified"] else "not reproduced (noise)"
            if e["verified"] and e.get("subsumed_by_depth1"):
                tag += f" but subsumed by depth-1 member(s) {e['subsumed_by_depth1']}"
            print(
                f"    [{verified_so_far}/{len(to_verify)}] {e['polluter']} -> {e['victim']}: {tag}",
                file=sys.stderr,
            )

    # Patch the verify step results back into the escape lines so report.py is aware.
    escape_by_triple = {(e["mode"], e["polluter"], e["victim"]): e for e in escapes}
    if escape_by_triple:
        with open(args.out) as f:
            records = [json.loads(line) for line in f if line.strip()]
        for rec in records:
            match = escape_by_triple.get((rec["mode"], rec["polluter"], rec["victim"]))
            if match:
                for field in ("verified", "verify_result", "subsumed_by_depth1"):
                    if field in match:
                        rec[field] = match[field]
        with open(args.out, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

    with open(candidates_path, "w") as f:
        json.dump(escapes, f, indent=2)

    verified_escapes = [e for e in escapes if e.get("verified")]
    print(f"\n========== PAIR SWEEP RESULT ==========", file=sys.stderr)
    print(f"trials -> {args.out}", file=sys.stderr)
    print(f"all candidate escapes: {candidates_path}", file=sys.stderr)
    if args.skip_verify:
        print(
            f"ESCAPES (unverified, --skip-verify was passed): {len(escapes)}",
            file=sys.stderr,
        )
        report_escapes = escapes
    else:
        noise = len(escapes) - len(verified_escapes)
        print(
            f"ESCAPES (verified): {len(verified_escapes)} "
            f"({noise} candidate(s) did not reproduce and are excluded)",
            file=sys.stderr,
        )
        report_escapes = verified_escapes
    for e in report_escapes:
        print(
            f"  {e['polluter']} -> {e['victim']}: {e['verdict']} (baseline {e['victim_baseline']}) [{e['mode']}]"
            + (
                f" [subsumed by {e['subsumed_by_depth1']}]"
                if e.get("subsumed_by_depth1")
                else ""
            ),
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
