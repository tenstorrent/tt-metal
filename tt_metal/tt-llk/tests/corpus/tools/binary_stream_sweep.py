#!/usr/bin/env python3
"""laneMQ — orchestrator for the sound 2^32 sem-vs-hand TWO-operand streaming sweep.

The laneMK single-operand sweep (fp32_stream_sweep.py) widened by one dimension: drives
the certified binary kernel's persistent-session streamer (the SFPU_STREAM_BINARY hook
in test_sfpu_binary.py) over the joint bf16 x bf16 space [0, 2^32) in resume-safe bands,
one band-leg per pytest invocation, on a flocked device. Per band: run sem + hand,
compare output SHA. Verdict = BIT-EXACT-ALL-INPUTS iff every band's sem_sha == hand_sha
(full contiguous cover, checked); otherwise DIVERGENT (the DIFF bands are the witness
bands to bisect: re-run a DIFF band at a smaller --band-bits to narrow it). Object
identity is enforced by construction (same certified ELF) + the optional .text gate.

Usage:
  binary_stream_sweep.py --op binarypow \
      --sem-node '<pytest impl-1 node>' --hand-node '<pytest impl-3 node>' \
      --farm <tests/python_tests> --venv <python> --llk-home <shim> \
      --runner-temp <RT> --band-bits 26 --chip 0 --out <dir> [--idmap <tsv>]
"""
import argparse
import math
import os
import re
import shlex
import subprocess
import time
from pathlib import Path

import stream_resume
import ulp_admission

TWO32 = 1 << 32
_SHA_RE = re.compile(r"output_sha256=([0-9a-f]{64})")
_RUNS_RE = re.compile(r"runs=(\d+)")


def parse_corr(corr_file):
    """Parse one SFPU_CORRECTNESS record, refusing ambiguous input."""
    p = Path(corr_file)
    if not p.exists():
        return None
    lines = p.read_text().strip().splitlines()
    if not lines:
        return None
    if len(lines) != 1:
        raise RuntimeError("invalid correctness sidecar: expected exactly one line")
    fields = lines[0].split(",")
    if fields[0] != "SFPU_CORRECTNESS":
        return None
    parsed = {}
    for field in fields[1:]:
        if "=" not in field:
            raise RuntimeError(f"invalid correctness sidecar field: {field!r}")
        key, value = field.split("=", 1)
        if not key or key in parsed:
            raise RuntimeError(f"duplicate/empty correctness sidecar key: {key!r}")
        parsed[key] = value
    return parsed


def validate_corr(corr, args, leg, count):
    if corr is None:
        raise RuntimeError("golden run produced no parseable correctness sidecar")
    if corr.get("op") != args.golden or corr.get("leg") != leg:
        raise RuntimeError(
            "golden sidecar identity mismatch: "
            f"expected op={args.golden},leg={leg}; "
            f"got op={corr.get('op')},leg={corr.get('leg')}"
        )
    try:
        joints = int(corr["joints"])
        n_out = int(corr["n_out_of_tol"])
        max_ulp = float(corr["max_bf16_ulp"])
        classes = ulp_admission.parse_class_ulp(corr["class_ulp"])
    except KeyError as error:
        raise RuntimeError(f"invalid golden sidecar: missing {error.args[0]}") from error
    except ValueError as error:
        raise RuntimeError(f"invalid golden sidecar: {error}") from error
    if joints < 0 or n_out < 0 or n_out > joints:
        raise RuntimeError(
            f"invalid golden sidecar counts: joints={joints}, n_out_of_tol={n_out}"
        )
    unknown_classes = set(classes) - ulp_admission.BINARY_CLASS_NAMES
    if unknown_classes:
        raise RuntimeError(
            "invalid golden sidecar class vocabulary: "
            f"unknown={sorted(unknown_classes)}"
        )
    if (
        not math.isfinite(max_ulp)
        or max_ulp < 0
        or max_ulp > 0xFFFF
        or not max_ulp.is_integer()
    ):
        raise RuntimeError(f"invalid golden sidecar max_bf16_ulp: {max_ulp}")
    class_max = max(class_ulp for _, class_ulp in classes.values())
    if max_ulp != class_max:
        raise RuntimeError(
            "golden sidecar max/class mismatch: "
            f"max_bf16_ulp={max_ulp}, class_max={class_max}"
        )
    within = corr.get("within_contract")
    if within not in ("True", "False") or (within == "True") != (n_out == 0):
        raise RuntimeError(
            "invalid golden sidecar within_contract: "
            f"within_contract={within!r}, n_out_of_tol={n_out}"
        )
    if "status" in corr:
        raise RuntimeError(f"golden sidecar is not checked: status={corr['status']}")
    if joints != count:
        raise RuntimeError(
            f"golden sidecar coverage mismatch: joints={joints}, expected={count}"
        )
    class_joints = sum(class_count for class_count, _ in classes.values())
    if class_joints != count:
        raise RuntimeError(
            "golden sidecar class coverage mismatch: "
            f"class_joints={class_joints}, expected={count}"
        )


def run_band_leg(
    args, node, start, count, out_sha_file, log_file, leg=None,
    compiler_options="", golden="", runner_temp=None,
):
    """One pytest invocation: stream joint [start,start+count) for one leg.

    Returns (sha, wall, runs, corr) — corr is the parsed 3-way sidecar dict (or None).
    """
    out_sha_file = Path(out_sha_file)
    corr_file = str(out_sha_file) + ".corr"
    metadata = Path(str(out_sha_file) + ".provenance.json")
    cache_record = stream_resume.cache_record(
        Path(__file__), args, node, start, count, leg or "",
        compiler_options=compiler_options, golden=golden,
        runner_temp=runner_temp,
    )
    if stream_resume.require_matching_cache(out_sha_file, metadata, cache_record):
        txt = out_sha_file.read_text()
        m = _SHA_RE.search(txt)
        corr = parse_corr(corr_file)
        if not m:
            raise RuntimeError(f"provenance-matched cache has no output SHA: {out_sha_file}")
        if golden and corr is None:
            raise RuntimeError(f"golden cache has no correctness sidecar: {corr_file}")
        if golden:
            validate_corr(corr, args, leg, count)
        return m.group(1), 0.0, 0, corr
    env = dict(os.environ)
    env.update(
        CHIP_ARCH="blackhole",
        SHORT_ARCH="bh",
        LLK_HOME=args.llk_home,
        RUNNER_TEMP=runner_temp or args.runner_temp,
        PYTHONUNBUFFERED="1",
        # Map --chip to the physical device (per-chip parallelism: TT_VISIBLE_DEVICES=n +
        # flock /tmp/tt-dev-n.lock lets N orchestrators run concurrently on N chips).
        TT_VISIBLE_DEVICES=str(args.chip),
        SFPU_STREAM_BINARY=f"{start},{count},{out_sha_file}",
        TT_LLK_EXTRA_COMPILER_OPTIONS=compiler_options,
    )
    if golden and leg:
        # Host-side torch.pow TRUE-MATH tolerance leg rides along. Per-class max
        # ULP is used for candidate<=hand admission; no absolute budget is claimed.
        env["SFPU_GOLDEN"] = f"{golden},{leg}"
    inner = (
        # --compile-consumer: use the prebuilt ELFs in RUNNER_TEMP; never invoke the
        # toolchain (galaxy hosts have none). The ELFs must be compiled beforehand
        # (a --compile-producer pass into --runner-temp).
        f"{shlex.quote(args.venv)} -m pytest -o addopts= -q -s --compile-consumer "
        f"{shlex.quote(node)} > {shlex.quote(str(log_file))} 2>&1"
    )
    cmd = ["flock", "-x", f"/tmp/tt-dev-{args.chip}.lock", "-c", inner]
    t0 = time.time()
    # Retry a failed dispatch a few times: a transient slow/hung band (cold device open,
    # a slow first dispatch) usually clears on a fresh attempt, so one bad band should not
    # abandon the whole op's sweep.
    m = None
    txt = ""
    for _ in range(3):
        # A failed pytest may have emitted a partial result.  Never let that file
        # satisfy a later retry which itself produced nothing.
        out_sha_file.unlink(missing_ok=True)
        Path(corr_file).unlink(missing_ok=True)
        run = subprocess.run(cmd, cwd=args.farm, env=env, timeout=args.timeout)
        txt = out_sha_file.read_text() if out_sha_file.exists() else ""
        m = _SHA_RE.search(txt)
        if run.returncode == 0 and m:
            break
        m = None
    dt = time.time() - t0
    r = _RUNS_RE.search(txt)
    if not m:
        raise RuntimeError(
            f"band [{start},{start+count}) leg {node} produced no SHA; see {log_file}"
        )
    corr = parse_corr(corr_file)
    if golden and corr is None:
        raise RuntimeError(f"golden run produced no correctness sidecar: {corr_file}")
    if golden:
        validate_corr(corr, args, leg, count)
    stream_resume.write_cache_record(metadata, cache_record, out_sha_file)
    return m.group(1), dt, int(r.group(1)) if r else 0, corr


def _identity_gate(args, out):
    """Verify every legacy or tri-arm ELF against its staged identity map."""
    if not args.idmap:
        return True
    here = Path(__file__).resolve().parent
    row = None
    for line in open(args.idmap):
        p = line.rstrip("\n").split("\t")
        if p and p[0] == args.op:
            row = p
            break
    expected_fields = 7 if args.tri_mode else 5
    if not row or len(row) != expected_fields:
        (out / f"{args.op}-VERDICT.txt").write_text(
            f"OP={args.op} VERDICT=REFUSED-IDENTITY(no-idmap-row)\n"
        )
        print(f"OP={args.op} REFUSED-IDENTITY no-idmap-row", flush=True)
        return False
    pairs = (
        ((row[1], row[2]), (row[3], row[4]), (row[5], row[6]))
        if args.tri_mode else ((row[1], row[2]), (row[3], row[4]))
    )

    roots = (
        (
            Path(args.selected_runner_temp) / "tt-llk-build/sources" / args.idmap_source,
            Path(args.baseline_sem_runner_temp) / "tt-llk-build/sources" / args.idmap_source,
            Path(args.baseline_hand_runner_temp) / "tt-llk-build/sources" / args.idmap_source,
        )
        if args.tri_mode else
        (Path(args.runner_temp) / "tt-llk-build/sources" / args.idmap_source,) * 2
    )

    def _text(root, v):
        return subprocess.run(
            [args.venv, str(here / "elf_text_sha.py"), str(root / v / "elf/math.elf")],
            capture_output=True,
            text=True,
        ).stdout.strip()

    actual = [_text(root, variant) for root, (variant, _) in zip(roots, pairs)]
    mismatch = any(got != expected for got, (_, expected) in zip(actual, pairs))
    legacy_alias = not args.tri_mode and actual[0] == actual[1]
    if mismatch or legacy_alias:
        reason = "sem==hand" if legacy_alias else "text-mismatch"
        (out / f"{args.op}-VERDICT.txt").write_text(
            f"OP={args.op} VERDICT=REFUSED-IDENTITY({reason})\n"
        )
        print(f"OP={args.op} REFUSED-IDENTITY {reason}", flush=True)
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", required=True)
    ap.add_argument("--sem-node")
    ap.add_argument("--hand-node")
    ap.add_argument("--selected-sem-node")
    ap.add_argument("--baseline-sem-node")
    ap.add_argument("--baseline-hand-node")
    ap.add_argument("--selected-flags")
    ap.add_argument("--baseline-flags")
    ap.add_argument("--farm", required=True)
    ap.add_argument("--venv", required=True)
    ap.add_argument("--llk-home", required=True)
    ap.add_argument("--runner-temp")
    ap.add_argument("--selected-runner-temp")
    ap.add_argument("--baseline-sem-runner-temp")
    ap.add_argument("--baseline-hand-runner-temp")
    ap.add_argument("--band-bits", type=int, default=26)
    ap.add_argument("--chip", default="0")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start-bit", type=lambda s: int(s, 0), default=0)
    ap.add_argument("--total", type=lambda s: int(s, 0), default=TWO32)
    ap.add_argument(
        "--idmap",
        help="optional op->sem/hand .text identity map; gates before streaming",
    )
    ap.add_argument(
        "--idmap-source",
        default="sfpu_binary_test.cpp",
        help="the test .cpp under tt-llk-build/sources whose ELF variants the idmap indexes",
    )
    ap.add_argument(
        "--golden",
        default="",
        help="op key ('binarypow') for the host-side torch.pow 3-way tolerance leg; "
        "requires per-class candidate ULP <= hand; no absolute ULP budget",
    )
    args = ap.parse_args()

    tri_values = (args.selected_sem_node, args.baseline_sem_node, args.baseline_hand_node)
    tri_mode = any(value is not None for value in tri_values)
    if tri_mode:
        if not all(value is not None for value in tri_values):
            ap.error("tri-arm mode requires selected-, baseline-sem-, and baseline-hand-node")
        if args.selected_flags is None or args.baseline_flags is None:
            ap.error("tri-arm mode requires explicit selected and baseline flags")
        if args.selected_sem_node != args.baseline_sem_node:
            ap.error("A/B compiler gate requires the identical semantic node")
        if not all((args.selected_runner_temp, args.baseline_sem_runner_temp,
                    args.baseline_hand_runner_temp)):
            ap.error("tri-arm mode requires three arm-specific runner temps")
        if args.sem_node is not None or args.hand_node is not None:
            ap.error("do not mix legacy --sem/--hand-node with tri-arm nodes")
    elif args.sem_node is None or args.hand_node is None:
        ap.error("provide both legacy --sem-node/--hand-node or all three tri-arm nodes")
    elif args.runner_temp is None:
        ap.error("legacy two-arm mode requires --runner-temp")
    args.tri_mode = tri_mode

    out = Path(args.out).resolve()  # absolute: run_band_leg runs pytest with cwd=farm
    (out / "bands").mkdir(parents=True, exist_ok=True)

    if not _identity_gate(args, out):
        return 2

    band = 1 << args.band_bits
    n_bands = (args.total + band - 1) // band
    ledger = out / f"{args.op}-STREAM-LEDGER.tsv"

    corr_legs = {"sem": _new_leg(), "hand": _new_leg()}

    rows = []
    covered = 0
    t_all = time.time()
    compiler_equal = True
    semantic_equal = True
    compiler_witness = []
    semantic_witness = []
    for k in range(n_bands):
        s = args.start_bit + k * band
        c = min(band, args.start_bit + args.total - s)
        sem_f = out / "bands" / f"b{k:04d}-sem.txt"
        hand_f = out / "bands" / f"b{k:04d}-hand.txt"
        if tri_mode:
            selected_f = out / "bands" / f"b{k:04d}-selected.txt"
            selected_sha, selected_dt, _, _ = run_band_leg(
                args, args.selected_sem_node, s, c, selected_f,
                out / "bands" / f"b{k:04d}-selected.log",
                leg="selected", compiler_options=args.selected_flags,
                runner_temp=args.selected_runner_temp,
            )
            sem_sha, sem_dt, _, sem_corr = run_band_leg(
                args, args.baseline_sem_node, s, c, sem_f,
                out / "bands" / f"b{k:04d}-sem.log", leg="sem",
                compiler_options=args.baseline_flags, golden=args.golden,
                runner_temp=args.baseline_sem_runner_temp,
            )
            hand_sha, hand_dt, _, hand_corr = run_band_leg(
                args, args.baseline_hand_node, s, c, hand_f,
                out / "bands" / f"b{k:04d}-hand.log", leg="hand",
                compiler_options=args.baseline_flags, golden=args.golden,
                runner_temp=args.baseline_hand_runner_temp,
            )
            compiler_eq = selected_sha == sem_sha
            compiler_equal &= compiler_eq
            if not compiler_eq:
                compiler_witness.append((k, s, c))
        else:
            selected_sha = selected_dt = None
            options = os.environ.get("TT_LLK_EXTRA_COMPILER_OPTIONS", "")
            sem_sha, sem_dt, _, sem_corr = run_band_leg(
                args, args.sem_node, s, c, sem_f,
                out / "bands" / f"b{k:04d}-sem.log", leg="sem",
                compiler_options=options, golden=args.golden,
            )
            hand_sha, hand_dt, _, hand_corr = run_band_leg(
                args, args.hand_node, s, c, hand_f,
                out / "bands" / f"b{k:04d}-hand.log", leg="hand",
                compiler_options=options, golden=args.golden,
            )
        if args.golden:
            _fold_leg(corr_legs["sem"], sem_corr)
            _fold_leg(corr_legs["hand"], hand_corr)
        eq = sem_sha == hand_sha
        semantic_equal &= eq
        if not eq:
            semantic_witness.append((k, s, c))
        covered += c
        rows.append((k, s, c, selected_sha or "-", sem_sha, hand_sha,
                     "-" if not tri_mode else ("EQ" if compiler_eq else "DIFF"),
                     "EQ" if eq else "DIFF", f"{selected_dt:.1f}" if tri_mode else "-",
                     f"{sem_dt:.1f}", f"{hand_dt:.1f}"))
        print(
            f"band {k+1}/{n_bands} [{s},{s+c}) sem={sem_sha[:12]} hand={hand_sha[:12]} "
            f"bc={'EQ' if eq else 'DIFF'} ab="
            f"{('-' if not tri_mode else ('EQ' if compiler_eq else 'DIFF'))}",
            flush=True,
        )
        with open(ledger, "w") as fh:
            fh.write(
                f"# {args.op} two-operand stream sweep; band_bits={args.band_bits}; chip={args.chip}\n"
            )
            fh.write(
                "band\tstart\tcount\tselected_sem_sha256\tbaseline_sem_sha256\t"
                "baseline_hand_sha256\tcompiler_verdict\tsemantic_verdict\t"
                "selected_s\tbaseline_sem_s\tbaseline_hand_s\n"
            )
            for row in rows:
                fh.write("\t".join(str(x) for x in row) + "\n")

    wall = time.time() - t_all
    if covered != args.total:
        raise RuntimeError(f"coverage gap: {covered} != {args.total}")
    # A reduced sweep must not certify itself as exhaustive: covered==args.total
    # only proves internal consistency, not that the whole space was swept.
    if not semantic_equal:
        verdict = "DIVERGENT"
    elif covered == TWO32:
        verdict = "BIT-EXACT-ALL-INPUTS"
    else:
        verdict = "BIT-EXACT-PARTIAL-%d-OF-2^32" % covered
    summary = (
        f"OP={args.op} VERDICT={verdict} start={args.start_bit} "
        f"total={args.total} bands={n_bands} covered={covered} "
        f"(full 2^32={covered==TWO32}) wall_s={wall:.1f} witness_bands={semantic_witness}"
    )
    print(summary, flush=True)
    (out / f"{args.op}-VERDICT.txt").write_text(summary + "\n")

    if tri_mode:
        if not compiler_equal:
            compiler_verdict = "DIVERGENT"
        elif covered == TWO32:
            compiler_verdict = "BIT-EXACT-ALL-INPUTS"
        else:
            compiler_verdict = "BIT-EXACT-PARTIAL-%d-OF-2^32" % covered
        compiler_summary = (
            f"OP={args.op} VERDICT={compiler_verdict} start={args.start_bit} "
            f"total={args.total} bands={n_bands} covered={covered} "
            f"(full 2^32={covered==TWO32}) wall_s={wall:.1f} "
            f"witness_bands={compiler_witness}"
        )
        print(f"COMPILER {compiler_summary}", flush=True)
        (out / f"{args.op}-COMPILER-VERDICT.txt").write_text(
            compiler_summary + "\n"
        )

    numeric_ok = True
    if args.golden:
        numeric_ok = write_correctness_ledger(
            out, args.op, verdict, corr_legs, covered
        )
    return 0 if (compiler_equal if tri_mode else semantic_equal) and numeric_ok else 1


def _new_leg():
    return {
        "joints": 0,
        "max_ulp": -1.0,
        "max_ulp_at": "-",
        "n_out": 0,
        "first_witness": None,
        "checked": False,
        "class_ulp": {},
    }


def _fold_leg(acc, corr):
    if not corr:
        return
    acc["checked"] = True
    acc["joints"] += int(corr["joints"])
    mu = float(corr["max_bf16_ulp"])
    if mu > acc["max_ulp"]:
        acc["max_ulp"] = mu
        acc["max_ulp_at"] = corr.get("max_ulp_joint", "-")
    acc["n_out"] += int(corr["n_out_of_tol"])
    ulp_admission.fold_class_ulp(acc["class_ulp"], corr["class_ulp"])
    fw = corr.get("first_witness", "0x00000000")
    try:
        fwi = int(fw, 0)
    except ValueError:
        fwi = 0
    if int(corr["n_out_of_tol"]) > 0 and fwi != 0:
        cand = (
            fwi,
            corr.get("first_witness_class", "-"),
            corr.get("witness_dev", "?"),
            corr.get("witness_golden", "?"),
        )
        if acc["first_witness"] is None or cand[0] < acc["first_witness"][0]:
            acc["first_witness"] = cand


def write_correctness_ledger(out, op, equiv_verdict, corr_legs, covered):
    sem, hand = corr_legs["sem"], corr_legs["hand"]
    equiv = equiv_verdict.startswith("BIT-EXACT")

    def leg_complete(a):
        return a["checked"] and a["joints"] == covered

    def leg_in(a):
        return leg_complete(a) and a["n_out"] == 0

    ulp_ok, ulp_reason = ulp_admission.candidate_not_worse(
        sem["class_ulp"], hand["class_ulp"]
    )

    if not leg_complete(sem) or not leg_complete(hand):
        verdict = (
            "INCOMPLETE-TOLERANCE-COVERAGE"
            f"(sem={sem['joints']}/{covered},hand={hand['joints']}/{covered})"
        )
    else:
        sem_in, hand_in = leg_in(sem), leg_in(hand)
        if sem_in and hand_in:
            verdict = (
                "TOLERANCE-BOTH-PASS"
                if not equiv
                else "TOLERANCE-PASS-AND-EQUAL"
            )
        elif sem_in and not hand_in:
            verdict = "SEM-TOLERANCE-PASS(hand fails tolerance)"
        elif hand_in and not sem_in:
            verdict = "SEM-TOLERANCE-FAIL"
        else:
            verdict = "BOTH-TOLERANCE-FAIL"

    p = out / f"{op}-CORRECTNESS-LEDGER.tsv"
    with open(p, "w") as fh:
        fh.write(
            "# three-way tolerance check (binary): device vs sem/hand AND vs torch.pow "
            "TRUE-MATH golden. Admission requires both tolerance gates and candidate "
            "max_bf16_ulp <= hand for every same-oracle input class; this is relative "
            "non-regression, not an absolute ULP certificate. "
            "covered=%d full_2^32=%s\n"
            % (covered, covered == TWO32)
        )
        fh.write(
            "op\tequiv\tsem_max_bf16_ulp\thand_max_bf16_ulp\tsem_in_contract\t"
            "hand_in_contract\tsem_n_out\thand_n_out\tulp_nonregression\t"
            "ulp_reason\tverdict\tfirst_witness\twitness_class\n"
        )
        w = sem if sem["first_witness"] else hand
        witness = (
            "-" if w["first_witness"] is None else f"0x{w['first_witness'][0]:08x}"
        )
        wclass = "-" if w["first_witness"] is None else w["first_witness"][1]
        fh.write(
            "\t".join(
                str(x)
                for x in (
                    op,
                    "EQUAL" if equiv else "DIVERGENT",
                    ("%.0f" % sem["max_ulp"]) if sem["checked"] else "n/a",
                    ("%.0f" % hand["max_ulp"]) if hand["checked"] else "n/a",
                    leg_in(sem) if sem["checked"] else "n/a",
                    leg_in(hand) if hand["checked"] else "n/a",
                    sem["n_out"] if sem["checked"] else "n/a",
                    hand["n_out"] if hand["checked"] else "n/a",
                    ulp_ok,
                    ulp_reason,
                    verdict,
                    witness,
                    wclass,
                )
            )
            + "\n"
        )
    print(
        f"OP={op} 3WAY_VERDICT={verdict} sem_max_ulp={sem['max_ulp']:.0f} "
        f"hand_max_ulp={hand['max_ulp']:.0f} sem_out={sem['n_out']} hand_out={hand['n_out']}",
        flush=True,
    )
    # Only semantic absolute correctness is composable per slice.  Hand
    # tolerance and candidate<=hand are diagnostics here; the whole-campaign
    # per-class maxima are compared by galaxy_numeric_admission.py.
    sem_absolute_ok = leg_complete(sem) and leg_in(sem)
    hand_oracle_complete = leg_complete(hand)
    gate_ok = sem_absolute_ok and hand_oracle_complete
    (out / f"{op}-CORRECTNESS-VERDICT.txt").write_text(
        f"OP={op} LOCAL_SEM_ABSOLUTE={'PASS' if sem_absolute_ok else 'FAIL'} "
        f"LOCAL_HAND_ORACLE_COMPLETE={'PASS' if hand_oracle_complete else 'FAIL'} "
        f"LOCAL_HAND_ABSOLUTE={'PASS' if leg_in(hand) else 'FAIL'} "
        f"LOCAL_ULP_COMPARISON={'PASS' if ulp_ok else 'FAIL'} ULP_REASON={ulp_reason} "
        "CAMPAIGN_ADMISSION=DEFERRED_GLOBAL "
        "CONTRACT=TOLERANCE-PLUS-SAME-ORACLE-ULP-NONREGRESSION-"
        "NOT-ABSOLUTE-ULP-CERTIFIED "
        f"sem_joints={sem['joints']} hand_joints={hand['joints']} covered={covered} "
        f"verdict={verdict}\n"
    )
    return gate_ok


if __name__ == "__main__":
    raise SystemExit(main())
