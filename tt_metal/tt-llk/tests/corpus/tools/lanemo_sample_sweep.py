#!/usr/bin/env python3
"""laneMO — orchestrator for the stratified-sampling sem-vs-hand sweep (one op).

Drives the certified corpus kernel's sampling streamer (the LANEMO_SAMPLE hook
in test_sfpu_unary.py) over N stratified operand-A sample tiles, one leg per
pytest invocation, on a flocked device. Runs sem + hand over the SAME seeded
sample stream and compares per-leg output SHA-256.

Verdict:
  * SAMPLED-CONSISTENT  — sem_sha == hand_sha over the identical N-sample stream
                          (0 output-byte diffs). A DISTINCT, WEAKER class than
                          the proven-equal ops: it is N samples with 0 diffs, not
                          a proof. Never reported as "verified"/"proven".
  * SAMPLED-DIVERGENT    — sem_sha != hand_sha; the first diverging checkpoint
                          window (from the per-leg checkpoint SHAs) is the
                          witness locus, recorded for follow-up bisection.
  * SAMPLE-REFUSED(reason) — object-identity gate failed, or the op has no
                          usable sampled-input path (recorded by the runner, not
                          faked as a pass).

Object identity is enforced by construction (same certified ELF) + the optional
.text idmap gate (same mechanism as fp32_stream_sweep).

Usage:
  lanemo_sample_sweep.py --op welford --sem-node '<pytest sem node>' \
      --hand-node '<hand node>' --farm <python_tests> --venv <python> \
      --llk-home <llk> --runner-temp <rt> --n-samples 3000000 --seed 0x1 \
      --ckpt 4096 --chip 0 --out <dir>
"""
import argparse
import os
import re
import shlex
import subprocess
import time
from pathlib import Path

_OUT_RE = re.compile(r"output_sha256=([0-9a-f]{64})")
_IN_RE = re.compile(r"input_sha256=([0-9a-f]{64})")
_N_RE = re.compile(r"n_samples=(\d+)")


def _text_sha(venv, rt):
    """Object-identity attestation: sha256 of the built math kernel's .text.
    Each leg builds into its own RUNNER_TEMP, so the single math.elf there is
    that leg's certified kernel. Empty string if not found (build+run only)."""
    elfs = sorted(Path(rt).glob("tt-llk-build/sources/*/*/elf/math.elf"))
    if not elfs:
        return ""
    here = Path(__file__).resolve().parent
    r = subprocess.run(
        [venv, str(here / "elf_text_sha.py"), str(elfs[-1])],
        capture_output=True,
        text=True,
    )
    return r.stdout.strip()


def run_leg(args, node, rt, out_file, log_file):
    """One pytest invocation streaming the whole sample set for one leg into its
    own RUNNER_TEMP `rt`. Returns (out_sha, in_sha, n_samples, ckpt_shas, wall,
    text_sha)."""
    if Path(out_file).exists():
        txt = Path(out_file).read_text()
        m = _OUT_RE.search(txt)
        if m:
            return (
                m.group(1),
                (_IN_RE.search(txt).group(1) if _IN_RE.search(txt) else ""),
                int(_N_RE.search(txt).group(1)) if _N_RE.search(txt) else 0,
                _ckpts(txt),
                0.0,
                _text_sha(args.venv, rt),
            )
    env = dict(os.environ)
    env.update(
        CHIP_ARCH="blackhole",
        SHORT_ARCH="bh",
        LLK_HOME=args.llk_home,
        RUNNER_TEMP=rt,
        PYTHONUNBUFFERED="1",
        TT_VISIBLE_DEVICES=str(args.chip),
        LANEMO_OP=args.op,
        LANEMO_SAMPLE=f"{args.n_samples},{args.seed},{args.ckpt},{out_file}",
    )
    mode = "--compile-consumer " if args.consume else ""
    inner = (
        f"{shlex.quote(args.venv)} -m pytest -o addopts= -q -s {mode}"
        f"{shlex.quote(node)} > {shlex.quote(str(log_file))} 2>&1"
    )
    cmd = ["flock", "-x", f"/tmp/tt-dev-{args.chip}.lock", "-c", inner]
    t0 = time.time()
    m = None
    for _ in range(3):
        subprocess.run(cmd, cwd=args.farm, env=env, timeout=args.timeout)
        txt = Path(out_file).read_text() if Path(out_file).exists() else ""
        m = _OUT_RE.search(txt)
        if m:
            break
    if not m:
        raise RuntimeError(f"leg {node} produced no SHA; see {log_file}")
    dt = time.time() - t0
    im = _IN_RE.search(txt)
    nm = _N_RE.search(txt)
    return (
        m.group(1),
        (im.group(1) if im else ""),
        (int(nm.group(1)) if nm else 0),
        _ckpts(txt),
        dt,
        _text_sha(args.venv, rt),
    )


def _ckpts(txt):
    for line in txt.splitlines():
        if line.startswith("checkpoint_shas\t"):
            return [c for c in line.split("\t", 1)[1].split(",") if c]
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", required=True)
    ap.add_argument("--sem-node", required=True)
    ap.add_argument("--hand-node", required=True)
    ap.add_argument("--farm", required=True)
    ap.add_argument("--venv", required=True)
    ap.add_argument("--llk-home", required=True)
    ap.add_argument(
        "--runner-temp", required=True, help="parent dir for per-leg RUNNER_TEMPs"
    )
    ap.add_argument("--n-samples", type=lambda s: int(s, 0), default=1_000_000)
    ap.add_argument("--seed", default="0x1")
    ap.add_argument("--ckpt", type=lambda s: int(s, 0), default=4096)
    ap.add_argument("--chip", default="0")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--consume",
        action="store_true",
        help="use prebuilt ELFs (--compile-consumer, galaxy); default builds+runs",
    )
    ap.add_argument(
        "--strategy", default="stratified:specials+expgrid+structure+random"
    )
    args = ap.parse_args()

    out = Path(args.out).resolve()
    (out / "legs").mkdir(parents=True, exist_ok=True)
    sem_rt = f"{args.runner_temp}/sem"
    hand_rt = f"{args.runner_temp}/hand"

    sem_sha, sem_in, sem_n, sem_ck, sem_dt, sem_text = run_leg(
        args, args.sem_node, sem_rt, out / "legs" / "sem.txt", out / "legs" / "sem.log"
    )
    hand_sha, hand_in, hand_n, hand_ck, hand_dt, hand_text = run_leg(
        args,
        args.hand_node,
        hand_rt,
        out / "legs" / "hand.txt",
        out / "legs" / "hand.log",
    )

    # The input stream MUST be identical across legs (same seed) — a differential
    # comparison over different inputs is meaningless. Assert it.
    if sem_in and hand_in and sem_in != hand_in:
        verdict = "SAMPLE-REFUSED(input-stream-mismatch)"
        n_diffs = "n/a"
        witness = "-"
    elif sem_n != hand_n:
        verdict = "SAMPLE-REFUSED(sample-count-mismatch)"
        n_diffs = "n/a"
        witness = "-"
    elif sem_text and hand_text and sem_text == hand_text:
        # Object-identity gate: a CONSISTENT verdict is only meaningful if the two
        # legs are genuinely DIFFERENT kernels. Same .text => trivial equality.
        verdict = "SAMPLE-REFUSED(identical-elf-text)"
        n_diffs = "n/a"
        witness = "-"
    elif sem_sha == hand_sha:
        verdict = "SAMPLED-CONSISTENT"
        n_diffs = 0
        witness = "-"
    else:
        verdict = "SAMPLED-DIVERGENT"
        # first diverging checkpoint window = witness locus
        first = next((i for i, (a, b) in enumerate(zip(sem_ck, hand_ck)) if a != b), -1)
        lo = first * args.ckpt
        hi = lo + args.ckpt
        witness = f"first-diverging-checkpoint=[{lo},{hi}) idx={first}"
        n_diffs = f">=1 (checkpoint-window {first})"

    n_used = sem_n
    obj_id = (
        "sem!=hand"
        if (sem_text and hand_text and sem_text != hand_text)
        else ("unknown" if not (sem_text and hand_text) else "sem==hand")
    )
    ledger = out / f"{args.op}-SAMPLE-LEDGER-ROW.tsv"
    with open(ledger, "w") as fh:
        fh.write(
            "op\tn_samples\tper_leg_elems\tstrategy\tn_diffs\tverdict\tsem_out_sha\thand_out_sha\t"
            "sem_text_sha\thand_text_sha\tobject_identity\twitness\twall_s\n"
        )
        fh.write(
            f"{args.op}\t{n_used}\t{args.n_samples}\t{args.strategy}\t{n_diffs}\t"
            f"{verdict}\t{sem_sha[:16]}\t{hand_sha[:16]}\t{sem_text[:16]}\t{hand_text[:16]}\t"
            f"{obj_id}\t{witness}\t{sem_dt+hand_dt:.1f}\n"
        )
    summary = (
        f"OP={args.op} VERDICT={verdict} n_samples={n_used} "
        f"n_diffs={n_diffs} object_identity={obj_id} witness={witness} "
        f"sem={sem_sha[:12]} hand={hand_sha[:12]} wall_s={sem_dt+hand_dt:.1f}"
    )
    print(summary, flush=True)
    (out / f"{args.op}-VERDICT.txt").write_text(summary + "\n")


if __name__ == "__main__":
    main()
