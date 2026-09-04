#!/usr/bin/env python3
"""laneMK — orchestrator for the sound 2^32 sem-vs-hand streaming sweep (one op).

Drives the certified corpus kernel's persistent-session streamer (the LANEMK_STREAM
hook in test_sfpu_unary.py) over [0, 2^32) in resume-safe bands, one band-leg per
pytest invocation, on a flocked device. Per band: run sem + hand, compare output SHA.
Verdict = BIT-EXACT-ALL-INPUTS iff every band's sem_sha == hand_sha (full contiguous
cover, checked); otherwise DIVERGENT (bands that differ are the witness bands to bisect).
Object identity is enforced by construction (same certified ELF) + the .text gate.

Usage:
  fp32_stream_sweep.py --op sign --sem-node '<pytest sem node>' --hand-node '<hand node>' \
      --farm <tests/python_tests> --venv <python> --tile-dim 256,256 \
      --band-bits 29 --chip 0 --out <dir>
"""
import argparse
import os
import re
import shlex
import subprocess
import time
from pathlib import Path

TWO32 = 1 << 32
_SHA_RE = re.compile(r"output_sha256=([0-9a-f]{64})")
_RUNS_RE = re.compile(r"runs=(\d+)")


def parse_corr(corr_file):
    """Parse a per-band .corr sidecar (LANEMR_CORRECTNESS,k=v,...) into a dict, or None."""
    p = Path(corr_file)
    if not p.exists():
        return None
    line = p.read_text().strip().splitlines()
    if not line or "LANEMR_CORRECTNESS" not in line[0]:
        return None
    d = {}
    for kv in line[0].split(","):
        if "=" in kv:
            k, v = kv.split("=", 1)
            d[k] = v
    return d


def run_band_leg(args, node, start, count, out_sha_file, log_file, leg=None):
    """One pytest invocation: stream [start,start+count) for one leg.

    Returns (sha, wall, runs, corr) where corr is the parsed 3-way sidecar dict (or None
    when --golden is off / the op is unchecked)."""
    corr_file = str(out_sha_file) + ".corr"
    if Path(out_sha_file).exists():
        txt = Path(out_sha_file).read_text()
        m = _SHA_RE.search(txt)
        if m:
            return m.group(1), 0.0, 0, parse_corr(corr_file)  # resumed
    env = dict(os.environ)
    env.update(
        CHIP_ARCH="blackhole",
        SHORT_ARCH="bh",
        LLK_HOME=args.llk_home,
        RUNNER_TEMP=args.runner_temp,
        PYTHONUNBUFFERED="1",
        LANEMK_TILE_DIM=args.tile_dim,
        # Map --chip to the physical device (per-chip parallelism: TT_VISIBLE_DEVICES=n +
        # flock /tmp/tt-dev-n.lock lets N orchestrators run concurrently on N chips).
        TT_VISIBLE_DEVICES=str(args.chip),
        LANEMK_STREAM=f"{start},{count},{out_sha_file}",
    )
    if args.golden and leg:
        # host-side TRUE-MATH golden + bf16 ULP-contract leg rides along on this pass.
        env["LANEMR_GOLDEN"] = f"{args.golden},{leg}"
    inner = (
        # --compile-consumer: use the prebuilt ELFs in RUNNER_TEMP; never invoke the
        # toolchain (galaxy hosts have none). The ELFs must be compiled beforehand
        # (build_identity_gate.sh / a --compile-producer pass into --runner-temp).
        f"{shlex.quote(args.venv)} -m pytest -o addopts= -q -s --compile-consumer "
        f"{shlex.quote(node)} > {shlex.quote(str(log_file))} 2>&1"
    )
    cmd = ["flock", "-x", f"/tmp/tt-dev-{args.chip}.lock", "-c", inner]
    t0 = time.time()
    # Retry a failed dispatch a few times: a transient slow/hung band (cold device open,
    # a slow first dispatch) usually clears on a fresh attempt, so one bad band should not
    # abandon the whole op's sweep.
    m = None
    for _ in range(3):
        subprocess.run(cmd, cwd=args.farm, env=env, timeout=args.timeout)
        txt = Path(out_sha_file).read_text() if Path(out_sha_file).exists() else ""
        m = _SHA_RE.search(txt)
        if m:
            break
    dt = time.time() - t0
    r = _RUNS_RE.search(txt)
    if not m:
        raise RuntimeError(
            f"band [{start},{start+count}) leg {node} produced no SHA; see {log_file}"
        )
    return m.group(1), dt, int(r.group(1)) if r else 0, parse_corr(corr_file)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", required=True)
    ap.add_argument("--sem-node", required=True)
    ap.add_argument("--hand-node", required=True)
    ap.add_argument("--farm", required=True)
    ap.add_argument("--venv", required=True)
    ap.add_argument("--llk-home", required=True)
    ap.add_argument("--runner-temp", required=True)
    ap.add_argument("--tile-dim", default="256,256")
    ap.add_argument("--band-bits", type=int, default=29)
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
        "--golden",
        default="",
        help="op key for the host-side TRUE-MATH 3-way leg (threeway_golden.REGISTRY); "
        "when set, each band folds device-vs-torch bf16 ULP and writes CORRECTNESS-LEDGER.tsv",
    )
    args = ap.parse_args()

    out = Path(args.out).resolve()  # absolute: run_band_leg runs pytest with cwd=farm
    (out / "bands").mkdir(parents=True, exist_ok=True)

    # OBJECT-IDENTITY GATE (optional but required for a bookable verdict): the two legs must
    # be the certified pin-59 kernels. Verify each ELF's .text against the recorded map and
    # sem != hand; refuse otherwise. .text is farm-path-dependent so the map is in-farm.
    if args.idmap:
        import subprocess as _sp

        here = Path(__file__).resolve().parent
        root = (
            Path(args.runner_temp) / "tt-llk-build/sources/eltwise_unary_sfpu_test.cpp"
        )
        row = None
        for line in open(args.idmap):
            p = line.rstrip("\n").split("\t")
            if p and p[0] == args.op:
                row = p
                break
        if not row or len(row) < 5:
            (out / f"{args.op}-VERDICT.txt").write_text(
                f"OP={args.op} VERDICT=REFUSED-IDENTITY(no-idmap-row)\n"
            )
            print(f"OP={args.op} REFUSED-IDENTITY no-idmap-row", flush=True)
            return
        sv, ss, hv, hs = row[1], row[2], row[3], row[4]

        def _text(v):
            return _sp.run(
                [
                    args.venv,
                    str(here / "elf_text_sha.py"),
                    str(root / v / "elf/math.elf"),
                ],
                capture_output=True,
                text=True,
            ).stdout.strip()

        sa, ha = _text(sv), _text(hv)
        if sa != ss or ha != hs or sa == ha:
            reason = "sem==hand" if sa == ha else "text-mismatch"
            (out / f"{args.op}-VERDICT.txt").write_text(
                f"OP={args.op} VERDICT=REFUSED-IDENTITY({reason})\n"
            )
            print(f"OP={args.op} REFUSED-IDENTITY {reason}", flush=True)
            return
    band = 1 << args.band_bits
    n_bands = (args.total + band - 1) // band
    ledger = out / f"{args.op}-STREAM-LEDGER.tsv"

    # 3-way per-leg combiners (max ULP, out-of-tol total, earliest witness across bands).
    corr_legs = {"sem": _new_leg(), "hand": _new_leg()}

    rows = []
    covered = 0
    t_all = time.time()
    all_equal = True
    witness_bands = []
    for k in range(n_bands):
        s = args.start_bit + k * band
        c = min(band, args.start_bit + args.total - s)
        sem_f = out / "bands" / f"b{k:04d}-sem.txt"
        hand_f = out / "bands" / f"b{k:04d}-hand.txt"
        sem_sha, sem_dt, sem_runs, sem_corr = run_band_leg(
            args,
            args.sem_node,
            s,
            c,
            sem_f,
            out / "bands" / f"b{k:04d}-sem.log",
            leg="sem",
        )
        hand_sha, hand_dt, hand_runs, hand_corr = run_band_leg(
            args,
            args.hand_node,
            s,
            c,
            hand_f,
            out / "bands" / f"b{k:04d}-hand.log",
            leg="hand",
        )
        if args.golden:
            _fold_leg(corr_legs["sem"], sem_corr)
            _fold_leg(corr_legs["hand"], hand_corr)
        eq = sem_sha == hand_sha
        all_equal &= eq
        if not eq:
            witness_bands.append((k, s, c))
        covered += c
        rows.append(
            (
                k,
                s,
                c,
                sem_sha,
                hand_sha,
                "EQ" if eq else "DIFF",
                f"{sem_dt:.1f}",
                f"{hand_dt:.1f}",
            )
        )
        print(
            f"band {k+1}/{n_bands} [{s},{s+c}) sem={sem_sha[:12]} hand={hand_sha[:12]} "
            f"{'EQ' if eq else 'DIFF'} ({sem_dt:.0f}+{hand_dt:.0f}s)",
            flush=True,
        )
        with open(ledger, "w") as fh:
            fh.write(
                f"# {args.op} stream sweep; tile_dim={args.tile_dim}; band_bits={args.band_bits}; chip={args.chip}\n"
            )
            fh.write(
                "band\tstart\tcount\tsem_sha256\thand_sha256\tverdict\tsem_s\thand_s\n"
            )
            for row in rows:
                fh.write("\t".join(str(x) for x in row) + "\n")

    wall = time.time() - t_all
    assert covered == args.total, f"coverage gap: {covered} != {args.total}"
    verdict = "BIT-EXACT-ALL-INPUTS" if all_equal else "DIVERGENT"
    summary = (
        f"OP={args.op} VERDICT={verdict} bands={n_bands} covered={covered} "
        f"(full 2^32={covered==TWO32}) wall_s={wall:.1f} witness_bands={witness_bands}"
    )
    print(summary, flush=True)
    (out / f"{args.op}-VERDICT.txt").write_text(summary + "\n")

    if args.golden:
        write_correctness_ledger(out, args.op, verdict, corr_legs, covered)


def _new_leg():
    return {
        "patterns": 0,
        "max_ulp": -1.0,
        "max_ulp_input": "-",
        "n_out": 0,
        "first_witness": None,  # (u32:int, class, dev, golden)
        "checked": False,
        "unchecked_reason": "",
        "max_ulp_true": -1.0,  # tanhderiv only
    }


def _fold_leg(acc, corr):
    """Fold a per-band .corr dict into the per-leg combiner (max ULP, earliest witness)."""
    if not corr:
        return
    if corr.get("status") == "UNCHECKED":
        acc["unchecked_reason"] = corr.get("reason", "")
        return
    acc["checked"] = True
    acc["patterns"] += int(corr.get("patterns", 0))
    mu = float(corr.get("max_bf16_ulp", 0))
    if mu > acc["max_ulp"]:
        acc["max_ulp"] = mu
        acc["max_ulp_input"] = corr.get("max_ulp_input", "-")
    if "max_ulp_true_sech2" in corr:
        mt = float(corr.get("max_ulp_true_sech2", 0))
        acc["max_ulp_true"] = max(acc["max_ulp_true"], mt)
    acc["n_out"] += int(corr.get("n_out_of_tol", 0))
    fw = corr.get("first_witness", "0x00000000")
    try:
        fwi = int(fw, 0)
    except ValueError:
        fwi = 0
    if int(corr.get("n_out_of_tol", 0)) > 0 and fwi != 0:
        cand = (
            fwi,
            corr.get("first_witness_class", "-"),
            corr.get("witness_dev", "?"),
            corr.get("witness_golden", "?"),
        )
        if acc["first_witness"] is None or cand[0] < acc["first_witness"][0]:
            acc["first_witness"] = cand


def write_correctness_ledger(out, op, equiv_verdict, corr_legs, covered):
    """Emit the per-op 3-way CORRECTNESS-LEDGER: per-leg max ULP + within-contract + verdict."""
    sem, hand = corr_legs["sem"], corr_legs["hand"]
    equiv = equiv_verdict == "BIT-EXACT-ALL-INPUTS"

    def leg_in(a):
        return a["checked"] and a["n_out"] == 0

    if not (sem["checked"] or hand["checked"]):
        reason = sem["unchecked_reason"] or hand["unchecked_reason"] or "no golden"
        verdict = f"UNCHECKED({reason})"
    else:
        sem_in, hand_in = leg_in(sem), leg_in(hand)
        if sem_in and hand_in:
            verdict = "LICENSED-BOTH-CORRECT" if not equiv else "CORRECT-AND-EQUAL"
        elif sem_in and not hand_in:
            verdict = "SEM-MORE-ACCURATE(hand out-of-contract)"
        elif hand_in and not sem_in:
            verdict = "SEM-BUG(sem out-of-contract)"
        else:
            verdict = "BOTH-OUT-OF-CONTRACT(approx/see-note)"

    p = out / f"{op}-CORRECTNESS-LEDGER.tsv"
    with open(p, "w") as fh:
        fh.write(
            "# laneMR three-way: device (certified pin-59 ELF) vs sem/hand equivalence "
            "AND vs torch TRUE-MATH golden (bf16 ULP contract). covered=%d full_2^32=%s\n"
            % (covered, covered == TWO32)
        )
        fh.write(
            "op\tequiv\tsem_max_bf16_ulp\thand_max_bf16_ulp\tsem_in_contract\t"
            "hand_in_contract\tsem_n_out\thand_n_out\tverdict\tfirst_witness\twitness_class\tnote\n"
        )

        def fw(a):
            return (
                "-" if a["first_witness"] is None else f"0x{a['first_witness'][0]:08x}"
            )

        def fwc(a):
            return "-" if a["first_witness"] is None else a["first_witness"][1]

        witness = fw(sem) if sem["first_witness"] else fw(hand)
        wclass = fwc(sem) if sem["first_witness"] else fwc(hand)
        note = ""
        if sem["max_ulp_true"] >= 0 or hand["max_ulp_true"] >= 0:
            note = (
                "tanhderivlut: max bf16 ULP vs the LICENSED LUT contract shown; "
                "vs TRUE sech^2 sem=%.0f hand=%.0f (licensed LUT approximation)"
                % (sem["max_ulp_true"], hand["max_ulp_true"])
            )
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
                    verdict,
                    witness,
                    wclass,
                    note,
                )
            )
            + "\n"
        )
    print(
        f"OP={op} 3WAY_VERDICT={verdict} sem_max_ulp={sem['max_ulp']:.0f} "
        f"hand_max_ulp={hand['max_ulp']:.0f} sem_out={sem['n_out']} hand_out={hand['n_out']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
