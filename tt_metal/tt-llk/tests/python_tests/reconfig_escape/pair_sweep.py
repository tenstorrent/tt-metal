#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Pair-trial phase for the weekly reconfig-escape CI job.

Consumes the manifest from discover_catalog.py (or snapshot_build.py -- same shape) and sweeps
every (X, K) ordered pair in the catalog, checking whether K still passes after X's residue.
Two modes, chosen per X:

  restore  (X passed its self-consistency gate): replay X's captured post-execution CFG
           snapshot in-kernel immediately before K's launch. No card reset between trials.
  fullreset (X failed the gate): fall back to the historical ground-truth recipe
           (tt-smi -r -> real X -> real K, no reset in between) for every pair in that row.

The restore-mode phase is parallelized across pytest-xdist, one round PER POLLUTER (not per
pair, and not per victim): compile every victim once (--compile-producer), then for each
polluter X run one `--compile-consumer -n jobs` round with every other victim K as a separate
item, each pinned to X's restore plan via xdist_plan_plugin.py. That turns N**2 serial trials
into N rounds of up to N-1 parallel trials.

A physical core accumulates persistent hardware state across every no-reset launch, regardless
of which op runs. This doesn't reduce to a fixed launch count: it's composition-dependent, not
purely count-dependent, so a single reset per polluter round is not enough on its own. The only
launch count confirmed clean in isolation is 12.

This residue survives a fresh pytest subprocess restart, so it isn't a host-side leak, and only
a real reset clears it. `tt-smi -r` resets the whole chip, not one core, so it can only be
inserted at a round/batch boundary where every worker is synced, never mid-batch. Each
polluter's victims are therefore split into small sub-batches sized to stay under that floor,
with a reset before every sub-batch instead of once per (potentially much larger) round.

A pair is an escape when K's baseline is PASS but K after X is FAIL or HANG. A K-side flake
(K itself sometimes flaky at baseline) is out of scope here: only PASS-baseline ops are used
as victims at all, so any post-X divergence is attributable to X, not to K's own instability.

Restore mode replants a *captured snapshot* of X's residue rather than running X for real, so a
restore-mode escape can be a snapshot/replant-fidelity artifact of the harness rather than a real
hardware effect. Every restore-mode escape is therefore re-checked with a plain-pytest
ground-truth reproduction before it is reported: reset, then one serial (no `-n`, single-core)
pytest invocation running X's real test then K's real test back to back, with no restore
machinery and no plan-map.

Only escapes that reproduce this way are reported. Unverified candidates are still written to
the JSONL (`verified: false`) so nothing is silently dropped, but they're excluded from the
final escape count/summary. Fallback-mode escapes already ran real X then real K with no reset
in between (see the fallback phase below), so they're ground truth already and skip
re-verification. This still isn't an absolute guarantee: a single hardware run can still be
flaky. But it is far stronger evidence than an unverified restore-mode hit.

--splits/--group shard the polluter loop across machines; each machine needs its own --out.
Every shard still sweeps against the full victim set, so a shard's own escapes are already
ground truth for that (polluter, victim) pair. No merge step is needed beyond concatenating
each shard's report, same as every other sharded suite in this repo.

--depth > 1 switches from exhaustive single-op pairs to a depth-D CHAIN sweep: D real ops'
write footprints are composed into one restore plan (most-recent-writer-per-field, via
discover_catalog.build_chain_entries), and --chains random chains are swept against every
victim the same way a single op's restore_path is swept at depth 1. This targets what depth-1
structurally can't see: residue an op left several kernels back, untouched by everything since,
still live when a victim launches (B U (A \\ B) for two ops, generalized to D). Every chain
escape is additionally checked for whether any single member alone already reproduces it via
that member's own depth-1 restore_path -- an escape only a chain finds, not subsumed by any
depth-1 pair, is the novel signal depth-D sweeping is for.

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

PASS, FAIL, ENVERR = discover_catalog.PASS, discover_catalog.FAIL, discover_catalog.ENVERR
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
    """Reset, then run the real polluter(s) then the real victim, back to back, in ONE serial
    pytest invocation (no -n, so they all pin to the same physical core as pytest-xdist's own
    "master" worker) with no restore/plan-map machinery at all. Returns the victim's own verdict
    from that real run. polluter_nodeids is a list: one element at depth 1, D elements for a
    depth-D chain.
    """
    reset()
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "--compile-consumer",
        f"--timeout={timeout}",
        f"--junitxml={junit_path}",
    ] + list(polluter_nodeids) + [
        victim_nodeid,
    ]
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
        help="xdist worker count for each restore-mode round -- one physical Tensix "
        "core per worker, matching discover_catalog.py's own -n 8",
    )
    p.add_argument("--port", type=int, default=5556)
    p.add_argument("--timeout", type=int, default=90)
    p.add_argument(
        "--skip-compile",
        action="store_true",
        help="every victim is already compiled (e.g. discover_catalog.py just "
        "compiled this same manifest's candidates in the same invocation) -- "
        "skip the redundant producer recompile",
    )
    p.add_argument(
        "--skip-verify",
        action="store_true",
        help="report every restore-mode escape as-is, without the plain-pytest "
        "ground-truth re-check. Useful for debugging the restore mechanism "
        "itself; the default (verify) is what CI should use.",
    )
    p.add_argument(
        "--splits",
        type=int,
        default=1,
        help="shard the polluter loop across this many machines (paired with "
        "--group). The victim set is never sharded -- every shard tests its "
        "own slice of polluters against all victims in the manifest.",
    )
    p.add_argument(
        "--group",
        type=int,
        default=1,
        help="1-indexed shard to run, in [1, --splits]",
    )
    p.add_argument(
        "--depth",
        type=int,
        default=1,
        help="1 (default): today's exhaustive single-op pairs, unchanged. >1: switch to a "
        "depth-D chain sweep -- sample --chains random D-op combos instead of every pair; "
        "2 is the minimum that can find anything depth-1 structurally can't",
    )
    p.add_argument(
        "--chains",
        type=int,
        default=80,
        help="random chains to sample when --depth > 1. 80 is sized for a ~3-3.5h sweep "
        "per machine at depth 2, -n 8 (measured ~128s/chain against a ~100-victim pool, "
        "plus margin for verify-phase overhead and hardware timing variance)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for chain sampling (--depth > 1 only). Default derives from the "
        "current ISO year+week (same convention as discover_catalog.py), so a weekly CI "
        "run is reproducible within that week and samples different chains the next",
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
            p.error(f"--depth {args.depth} > {len(restore_x)} usable polluters in the manifest")
        seed = (
            args.seed
            if args.seed is not None
            else int(datetime.date.today().strftime("%Y%V"))
        )
        print(
            f"[pair_sweep] depth={args.depth} chains={args.chains} seed={seed}"
            + (" (explicit)" if args.seed is not None else " (derived from ISO year+week)"),
            file=sys.stderr,
        )
        rng = random.Random(seed)
        pristine_path = manifest["pristine_snapshot"]
        with open(pristine_path) as f:
            pristine = {a: v for s, a, v in json.load(f) if s == 0}
        chains_dir = os.path.join(os.path.dirname(os.path.abspath(args.out)) or ".", "chains")
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
        fallback_x = []  # chain members are already gate-passed usable polluters

    if args.splits > 1:
        restore_x = restore_x[args.group - 1 :: args.splits]
        fallback_x = fallback_x[args.group - 1 :: args.splits]

    shard_note = f" (shard {args.group}/{args.splits})" if args.splits > 1 else ""
    print(
        f"[pair_sweep] victims={len(victims)} restore-mode polluters={len(restore_x)} "
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
                f"[pair_sweep] ESCAPE {x_key} -> {k_key}: {verdict} (baseline {k_baseline}) [{mode}]",
                file=sys.stderr,
            )

    # --- Restore-mode phase: one xdist round PER POLLUTER, every other victim run in that round
    #     in parallel across -n jobs workers, all pinned to that round's single restore plan. ---
    if restore_x:
        tmp_dir = os.path.dirname(os.path.abspath(args.out)) or "."
        os.makedirs(tmp_dir, exist_ok=True)
        if args.skip_compile:
            print(
                f"[pair_sweep] --skip-compile: assuming all {len(nodeids)} victims are already "
                f"compiled",
                file=sys.stderr,
            )
        else:
            print(
                f"[pair_sweep] compiling {len(nodeids)} victims once (producer, -n {args.jobs})...",
                file=sys.stderr,
            )
            cproc = compile_all(
                args.worktree, args.arch, nodeids, args.jobs, args.timeout
            )
            if cproc.returncode != 0:
                print(
                    f"[pair_sweep] WARNING: compile-producer had failures (rc={cproc.returncode}); "
                    f"affected victims will show up as ENVERR in any round",
                    file=sys.stderr,
                )

        # See module docstring for the no-reset-accumulation finding. 10, not the confirmed-clean
        # 12, to leave margin since a round's own launch count per worker isn't otherwise bounded.
        SAFE_LAUNCHES_PER_WORKER = 10

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
            batch_size = SAFE_LAUNCHES_PER_WORKER * args.jobs
            batches = list(discover_catalog._chunks(round_nodeids, batch_size))

            print(
                f"[pair_sweep] restore-mode round {xi+1}/{len(restore_x)}: polluter {x['key']}, "
                f"{len(round_nodeids)} victims across -n {args.jobs}, {len(batches)} sub-batch(es) "
                f"of <={batch_size}...",
                file=sys.stderr,
            )
            results = {}
            for bi, batch_nodeids in enumerate(batches):
                batch_plan_map = {n: plan_map[n] for n in batch_nodeids}
                plan_map_path = os.path.join(
                    tmp_dir, f"pairsweep_round{xi}_batch{bi}.map.json"
                )
                with open(plan_map_path, "w") as f:
                    json.dump(batch_plan_map, f)
                junit_path = os.path.join(
                    tmp_dir, f"pairsweep_round{xi}_batch{bi}.junit.xml"
                )

                reset()
                rproc = run_round(
                    args.worktree,
                    args.arch,
                    batch_nodeids,
                    plan_map_path,
                    args.jobs,
                    args.timeout,
                    junit_path,
                )
                if not os.path.exists(junit_path):
                    print(
                        f"[pair_sweep] WARNING: sub-batch {bi+1}/{len(batches)} for polluter "
                        f"{x['key']} produced no junit report; every victim in it recorded as "
                        f"ENVERR",
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

    # --- Fallback phase: full reset per pair, for X's that failed their own restore-gate. ---
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

    # --- Verify phase: re-check every restore-mode escape with a plain-pytest ground-truth
    #     reproduction (reset, real X, real K, same core, no restore machinery) before reporting
    #     it. Fallback-mode escapes already ran that way, so they're trusted as-is. ---
    candidates_path = os.path.splitext(args.out)[0] + ".candidates.json"
    if args.skip_verify:
        for e in escapes:
            e["verified"] = None
    else:
        verify_dir = os.path.dirname(os.path.abspath(args.out)) or "."
        to_verify = [e for e in escapes if e["mode"] == "restore"]
        print(
            f"[pair_sweep] verifying {len(to_verify)} restore-mode escape(s) with plain "
            f"pytest (reset, real polluter, real victim, same core, no restore machinery)...",
            file=sys.stderr,
        )
        verified_so_far = 0
        for i, e in enumerate(escapes):
            if e["mode"] != "restore":
                e["verified"] = (
                    True  # fullreset already ran real X then real K, no reset between
                )
                continue
            verified_so_far += 1
            member_keys = e.get("members", [e["polluter"]])
            px = [nodeid_by_key[k] for k in member_keys if k in nodeid_by_key]
            vk = nodeid_by_key.get(e["victim"])
            if len(px) != len(member_keys) or not vk:
                e["verified"] = False
                e["verify_result"] = ENVERR
                continue

            # A chain escape only matters if it's NOT already visible to a depth-1 sweep: check
            # whether any single member alone, via its own manifest restore_path, already breaks
            # this victim.
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
                        args.worktree, args.arch, [vk], single_plan_path, 1, args.timeout,
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
            tag = "CONFIRMED" if e["verified"] else "NOT reproduced (noise)"
            if e["verified"] and e.get("subsumed_by_depth1"):
                tag += f" but SUBSUMED by depth-1 member(s) {e['subsumed_by_depth1']}"
            print(
                f"    [{verified_so_far}/{len(to_verify)}] {e['polluter']} -> {e['victim']}: {tag}",
                file=sys.stderr,
            )

    with open(candidates_path, "w") as f:
        json.dump(escapes, f, indent=2)

    verified_escapes = [e for e in escapes if e.get("verified")]
    print(f"\n========== PAIR SWEEP RESULT ==========", file=sys.stderr)
    print(f"trials -> {args.out}", file=sys.stderr)
    print(
        f"all candidate escapes (verified + not) -> {candidates_path}", file=sys.stderr
    )
    if args.skip_verify:
        print(
            f"ESCAPES (UNVERIFIED, --skip-verify was passed): {len(escapes)}",
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
