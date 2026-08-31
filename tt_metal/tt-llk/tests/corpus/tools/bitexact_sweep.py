#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""THE EXHAUSTIVE BIT-EXACT CHECK (lane JN, 2026-08-31).

Per unary board row, certify whether the compiled semantic kernel is a
bit-exact drop-in for the hand (production LLK) kernel over ALL inputs the
row's format can deliver through L1.

Method (per row):
  probe     one sim run per leg (sem/hand) on the row's REGISTERED corr node
            and stimuli, dumping the raw L1 input/result bytes
            (LANEJN_DUMP hook in test_sfpu_unary.py + helpers/stimuli_config.py).
  anchor    the same two nodes run on SILICON with the same dump hook —
            the device-golden truth anchors (device use = validation only).
  validate  the executor (pinned craq-sim libttsim, in-process harness) must
            reproduce BOTH legs' device anchors BITWISE on the registered
            stimuli.  A row whose executor cannot reproduce device bits is
            EXECUTOR-UNVALIDATED — its sweep result never counts.
  sweep     enumerate the row's ENTIRE input bit-pattern space (16-bit
            formats: all 65,536 patterns, NaN payloads and denormals
            included), inject the raw patterns directly into L1
            (LANEJN_RAW_A — bypasses the host float pack path, which
            canonicalizes NaN payloads), run BOTH compiled kernels on every
            chunk in the validated simulator, and compare raw result bytes.
  verdict   BIT-EXACT-ALL-INPUTS / DIVERGENT (with a divergence
            certificate) / NOT-EXHAUSTIBLE / EXECUTOR-UNVALIDATED /
            INFEASIBLE-2^32-THIS-EXECUTOR.

Honesty rules (non-negotiable):
  * multi-input rows are NOT exhaustible (2^64/2^96): refused with a
    structured-sampling stub note; never claimed as all-inputs.
  * fp32/int32 unary rows have a 2^32 space.  The chunked sim executor is
    mechanically capable (use --allow-slow) but at the measured rate
    (~4-6 s per 4096-pattern chunk) a single leg costs ~60 days; without
    --allow-slow those rows are recorded INFEASIBLE-2^32-THIS-EXECUTOR,
    never faked.
  * a row is only BIT-EXACT-ALL-INPUTS when (a) both legs' executors were
    validated bitwise against device anchors, (b) every chunk of the full
    enumeration ran, (c) the dumped input bytes equal the enumeration, and
    (d) zero differing result bytes exist.

Layout of the output root (resume-safe; every stage skips existing goods):
  rows/<row>/probe-{sem,hand}.npz      sim runs, registered stimuli
  rows/<row>/anchor-{sem,hand}.npz     device runs, registered stimuli
  rows/<row>/chunks/c<k>-{sem,hand}.npz
  rows/<row>/validate.json  rows/<row>/verdict.json
  rows/<row>/CERTIFICATE.tsv           (divergent rows: first N diffs)
  BIT-EXACT-LEDGER.tsv

Usage:
  bitexact_sweep.py --out DIR --rows tanh-fresh,log-fresh --stage all
  bitexact_sweep.py --out DIR --all-unary --stage probe --jobs 12
  bitexact_sweep.py --out DIR --all-unary --stage anchor          # device
  bitexact_sweep.py --out DIR --all-unary --stage sweep --jobs 12
  bitexact_sweep.py --out DIR --all-unary --stage verdict         # + ledger
"""

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parent
CORPUS = TOOLS.parent
TESTS = CORPUS.parent  # tt_metal/tt-llk/tests
LLK = TESTS.parent  # tt_metal/tt-llk
PYDIR = TESTS / "python_tests"

DEVICE_LOCK = "/tmp/tt-device.lock"
SILICON_LOCK = "/tmp/tt-llk-sfpu-silicon.lock"

BOARD_ROWS_TSV = TOOLS / "bitexact_board_rows.tsv"
OPS_TSV = CORPUS / "sweep_2x2_ops.tsv"

# Element width by harness DataFormat name.
FMT_BYTES = {
    "Float16_b": 2,
    "Float16": 2,
    "Int16": 2,
    "UInt16": 2,
    "Float32": 4,
    "Int32": 4,
    "UInt32": 4,
}
FMT_SPACE = {2: 1 << 16, 4: 1 << 32}

# Arity / harness classification by the sem corr node's test file.  Only
# test_sfpu_unary.py rows are instrumented (the LANEJN_* hooks); everything
# else is refused with a per-class reason, never silently dropped.
UNARY_HOOKED_FILES = {"test_sfpu_unary.py"}
FILE_CLASS = {
    "test_sfpu_unary.py": ("unary", None),
    "test_eltwise_unary_typecast.py": (
        "unary-unhooked",
        "unary typecast row, separate harness (test_eltwise_unary_typecast.py) "
        "not instrumented with the LANEJN hooks this run",
    ),
    "test_sfpu_binop_scalar.py": (
        "unary-unhooked",
        "tensor x registered-scalar row; separate harness not instrumented "
        "with the LANEJN hooks this run",
    ),
    "test_sfpu_coverage.py": (
        "unary-unhooked",
        "coverage-census row; separate harness not instrumented with the "
        "LANEJN hooks this run",
    ),
    "test_sfpu_binary.py": (
        "binary",
        "two independent tensor operands: input space 2^64 — NOT exhaustible; "
        "structured-sampling stub: stratify per exponent-pair (256x256 "
        "exponent cells x mantissa corners + dense diagonal)",
    ),
    "test_sfpu_ternary.py": (
        "ternary",
        "three independent tensor operands: input space 2^96 — NOT "
        "exhaustible; structured-sampling stub as for binary rows",
    ),
}
STRUCTURAL_REASON = (
    "structural/multi-operand choreography row (cross-lane / cross-tile "
    "state, reductions, attention, topk, prefix scans): the row's semantics "
    "are not a per-element unary map, so 'all inputs' is not a 2^16/2^32 "
    "pattern space — NOT exhaustible by element enumeration"
)


def read_ops():
    ops = {}
    for line in OPS_TSV.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        f = line.split("\t")
        if f[0] == "op":
            continue
        ops[f[0]] = {"kind": f[2], "sem": f[5], "hand": f[7]}
    return ops


def read_board_rows():
    rows = []
    for line in BOARD_ROWS_TSV.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        rows.append(line.split("\t")[0])
    return rows


def classify(row, ops):
    if row not in ops:
        return ("missing", "row absent from sweep_2x2_ops.tsv")
    kind = ops[row]["kind"]
    if kind != "full2x2":
        return ("no-hand-pair", f"kind={kind}: no distinct hand kernel leg")
    tf = ops[row]["sem"].split("::")[0]
    if tf in FILE_CLASS:
        return FILE_CLASS[tf]
    return ("structural", STRUCTURAL_REASON)


def sha12(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


class Ctx:
    def __init__(self, args):
        self.out = Path(args.out)
        self.python = args.python
        self.sim = args.sim
        self.flags = args.flags
        self.jobs = args.jobs
        self.timeout = args.timeout
        self.allow_slow = args.allow_slow
        self.first_n = args.first_n
        self.sim_id = f"craqsim-{sha12(self.sim)}-harness-inprocess"
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "rows").mkdir(exist_ok=True)

    def rowdir(self, row):
        d = self.out / "rows" / row
        d.mkdir(parents=True, exist_ok=True)
        return d


def base_env(ctx, runner_temp):
    env = dict(os.environ)
    env.update(
        CHIP_ARCH="blackhole",
        SHORT_ARCH="bh",
        LLK_HOME=str(LLK),
        RUNNER_TEMP=str(runner_temp),
        TT_LLK_EXTRA_COMPILER_OPTIONS=ctx.flags,
        PYTHONUNBUFFERED="1",
    )
    return env


def run_node_sim(ctx, node, runner_temp, dump, raw_a=None, skip_assert=False, log=None):
    """One in-process craq-sim pytest run of `node`. Returns (ok, seconds)."""
    Path(runner_temp).mkdir(parents=True, exist_ok=True)
    env = base_env(ctx, runner_temp)
    env["TT_METAL_SIMULATOR"] = str(ctx.sim)
    env["LANEJN_DUMP"] = str(dump)
    if raw_a is not None:
        env["LANEJN_RAW_A"] = str(raw_a)
    if skip_assert:
        env["LANEJN_SKIP_ASSERT"] = "1"
    cmd = [
        ctx.python,
        "-m",
        "pytest",
        "-o",
        "addopts=",
        "-q",
        "--run-simulator",
        "--bit-exact-runs",
        "2",
        node,
    ]
    t0 = time.time()
    r = subprocess.run(
        cmd,
        cwd=PYDIR,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=ctx.timeout,
    )
    dt = time.time() - t0
    if log:
        Path(log).write_bytes(r.stdout)
    ok = Path(dump).exists() and (b"1 passed" in r.stdout or skip_assert)
    # With skip_assert the test still must PASS (assert skipped, run intact).
    if skip_assert:
        ok = Path(dump).exists() and b"1 passed" in r.stdout
    return ok, dt


def run_node_device(ctx, node, runner_temp, dump, log):
    """One silicon pytest run under both device flocks (laneJL discipline)."""
    Path(runner_temp).mkdir(parents=True, exist_ok=True)
    env = base_env(ctx, runner_temp)
    env.pop("TT_METAL_SIMULATOR", None)
    env["LANEJN_DUMP"] = str(dump)
    inner = (
        f"rm -rf {shlex.quote(str(LLK / 'perf_data'))} && "
        f"cd {shlex.quote(str(PYDIR))} && "
        f"timeout {ctx.timeout} {shlex.quote(ctx.python)} -m pytest -o addopts= -q "
        f"--bit-exact-runs 2 {shlex.quote(node)} > {shlex.quote(str(log))} 2>&1"
    )
    cmd = [
        "flock",
        "-x",
        DEVICE_LOCK,
        "-c",
        f"flock -x {SILICON_LOCK} -c {shlex.quote(inner)}",
    ]
    t0 = time.time()
    subprocess.run(cmd, env=env, timeout=ctx.timeout + 600)
    dt = time.time() - t0
    logtxt = Path(log).read_bytes() if Path(log).exists() else b""
    return Path(dump).exists() and b"1 passed" in logtxt, dt


def load_dump(path):
    d = np.load(path)
    meta = [str(x) for x in d["meta"]]
    return {
        "src": d["src_raw"].tobytes(),
        "res": d["res_raw"].tobytes(),
        "mathop": meta[0],
        "impl": meta[1],
        "in_fmt": meta[2],
        "out_fmt": meta[3],
        "dims": meta[4],
        "tiles_a": int(meta[5]),
        "tiles_res": int(meta[6]),
    }


# ---------------------------------------------------------------- stages


def stage_probe(ctx, row, ops):
    rd = ctx.rowdir(row)
    res = {}
    for leg in ("sem", "hand"):
        dump = rd / f"probe-{leg}.npz"
        if dump.exists():
            res[leg] = True
            continue
        ok, dt = run_node_sim(
            ctx,
            ops[row][leg],
            rd / f"rt-probe-{leg}",
            dump,
            log=rd / f"probe-{leg}.log",
        )
        res[leg] = ok
        print(f"[probe] {row} {leg}: {'OK' if ok else 'FAIL'} ({dt:.0f}s)", flush=True)
    return all(res.values())


def stage_anchor(ctx, row, ops):
    rd = ctx.rowdir(row)
    for leg in ("sem", "hand"):
        dump = rd / f"anchor-{leg}.npz"
        if dump.exists():
            continue
        ok, dt = run_node_device(
            ctx,
            ops[row][leg],
            rd / f"rt-anchor-{leg}",
            dump,
            rd / f"anchor-{leg}.log",
        )
        print(f"[anchor] {row} {leg}: {'OK' if ok else 'FAIL'} ({dt:.0f}s)", flush=True)


def stage_validate(ctx, row):
    """Executor validation gate: sim must reproduce device bits, both legs."""
    rd = ctx.rowdir(row)
    out = {"validated": False, "detail": {}}
    for leg in ("sem", "hand"):
        p, a = rd / f"probe-{leg}.npz", rd / f"anchor-{leg}.npz"
        if not p.exists() or not a.exists():
            out["detail"][leg] = "missing probe or anchor"
            break
        dp, da = load_dump(p), load_dump(a)
        same_src = dp["src"] == da["src"]
        same_res = dp["res"] == da["res"]
        out["detail"][leg] = {
            "stimuli_bitwise_identical": same_src,
            "result_bitwise_identical": same_res,
            "bytes": len(dp["res"]),
        }
        if not (same_src and same_res):
            break
    else:
        out["validated"] = True
    (rd / "validate.json").write_text(json.dumps(out, indent=1))
    print(
        f"[validate] {row}: {'VALIDATED' if out['validated'] else 'FAILED'}", flush=True
    )
    return out["validated"]


def enum_chunks(space_bits, elem_bytes, chunk_elems):
    """Yield (idx, bytes) covering the FULL pattern space, padded w/ 0."""
    total = 1 << space_bits
    dtype = np.uint16 if elem_bytes == 2 else np.uint32
    n_chunks = (total + chunk_elems - 1) // chunk_elems
    for k in range(n_chunks):
        lo = k * chunk_elems
        hi = min(lo + chunk_elems, total)
        arr = np.arange(lo, hi, dtype=np.uint64).astype(dtype)
        if hi - lo < chunk_elems:
            arr = np.concatenate([arr, np.zeros(chunk_elems - (hi - lo), dtype=dtype)])
        yield k, arr.tobytes()


def stage_sweep(ctx, row, ops):
    rd = ctx.rowdir(row)
    probe = load_dump(rd / "probe-sem.npz")
    eb = FMT_BYTES.get(probe["in_fmt"])
    if eb is None:
        return f"unknown input format {probe['in_fmt']}"
    if eb == 4 and not ctx.allow_slow:
        return "2^32 space: refused without --allow-slow"
    chunk_elems = len(probe["src"]) // eb
    space_bits = 16 if eb == 2 else 32
    cd = rd / "chunks"
    cd.mkdir(exist_ok=True)
    t0 = time.time()
    for k, payload in enum_chunks(space_bits, eb, chunk_elems):
        raw = cd / f"c{k}.bin"
        if not raw.exists() or raw.stat().st_size != len(payload):
            raw.write_bytes(payload)
        for leg in ("sem", "hand"):
            dump = cd / f"c{k}-{leg}.npz"
            if dump.exists():
                continue
            ok, dt = run_node_sim(
                ctx,
                ops[row][leg],
                rd / f"rt-sweep-{leg}",
                dump,
                raw_a=raw,
                skip_assert=True,
                log=cd / f"c{k}-{leg}.log",
            )
            if not ok:
                return f"chunk {k} {leg} FAILED (see {cd}/c{k}-{leg}.log)"
    (rd / "sweep-runtime.txt").write_text(f"{time.time() - t0:.1f}\n")
    print(f"[sweep] {row}: complete ({time.time() - t0:.0f}s)", flush=True)
    return None


def _f32(bits16=None, bits32=None):
    if bits16 is not None:
        return (bits16.astype(np.uint32) << 16).view(np.float32)
    return bits32.view(np.float32)


def stage_verdict(ctx, row, ops):
    rd = ctx.rowdir(row)
    probe = load_dump(rd / "probe-sem.npz")
    in_eb = FMT_BYTES[probe["in_fmt"]]
    out_eb = FMT_BYTES.get(probe["out_fmt"], in_eb)
    idt = np.uint16 if in_eb == 2 else np.uint32
    odt = np.uint16 if out_eb == 2 else np.uint32
    space_bits = 16 if in_eb == 2 else 32
    total = 1 << space_bits
    chunk_elems = len(probe["src"]) // in_eb
    n_chunks = (total + chunk_elems - 1) // chunk_elems
    cd = rd / "chunks"

    seen = 0
    diffs = []  # (input_bits, sem_bits, hand_bits)
    ndiff = 0
    max_abs = 0.0
    max_ulp = 0
    nan_only = 0
    region_hist = {}
    for k in range(n_chunks):
        ds = cd / f"c{k}-sem.npz"
        dh = cd / f"c{k}-hand.npz"
        if not ds.exists() or not dh.exists():
            return None  # sweep incomplete
        s, h = load_dump(ds), load_dump(dh)
        exp_lo = k * chunk_elems
        exp = np.arange(
            exp_lo, min(exp_lo + chunk_elems, total), dtype=np.uint64
        ).astype(idt)
        pad = chunk_elems - exp.size
        if pad:
            exp = np.concatenate([exp, np.zeros(pad, dtype=idt)])
        src_bits = np.frombuffer(s["src"], dtype=idt)
        if not np.array_equal(src_bits, exp) or s["src"] != h["src"]:
            return f"chunk {k}: injected input bytes != enumeration"
        valid = chunk_elems - pad
        sb = np.frombuffer(s["res"], dtype=odt)[:valid]
        hb = np.frombuffer(h["res"], dtype=odt)[:valid]
        seen += valid
        neq = np.nonzero(sb != hb)[0]
        if neq.size:
            ndiff += int(neq.size)
            iv = src_bits[:valid][neq]
            sv, hv = sb[neq], hb[neq]
            if out_eb == 2:
                sf, hf = _f32(bits16=sv), _f32(bits16=hv)
            else:
                sf, hf = _f32(bits32=sv), _f32(bits32=hv)
            both_nan = np.isnan(sf) & np.isnan(hf)
            nan_only += int(both_nan.sum())
            fin = np.isfinite(sf) & np.isfinite(hf)
            if fin.any():
                max_abs = max(max_abs, float(np.abs(sf[fin] - hf[fin]).max()))
                ulp = np.abs(sv[fin].astype(np.int64) - hv[fin].astype(np.int64))
                max_ulp = max(max_ulp, int(ulp.max()))
            # region histogram by input exponent field
            if in_eb == 2:
                expf = (iv >> 7) & 0xFF
            else:
                expf = (iv >> 23) & 0xFF
            for e in np.unique(expf):
                region_hist[int(e)] = region_hist.get(int(e), 0) + int(
                    (expf == e).sum()
                )
            for j in range(min(neq.size, max(0, ctx.first_n - len(diffs)))):
                diffs.append((int(iv[j]), int(sv[j]), int(hv[j])))

    vjson = {
        "row": row,
        "input_format": probe["in_fmt"],
        "output_format": probe["out_fmt"],
        "space": total,
        "patterns_compared": seen,
        "diverging_inputs": ndiff,
        "both_nan_diffs": nan_only,
        "max_abs_delta_finite": max_abs,
        "max_ulp_delta_finite": max_ulp,
        "diff_region_hist_by_input_exponent": region_hist,
        "verdict": "BIT-EXACT-ALL-INPUTS" if ndiff == 0 else "DIVERGENT",
    }
    (rd / "verdict.json").write_text(json.dumps(vjson, indent=1))
    if ndiff:
        with open(rd / "CERTIFICATE.tsv", "w") as f:
            f.write(
                f"# divergence certificate {row}: {ndiff} diverging inputs of "
                f"{seen}; first {len(diffs)} below\n"
            )
            f.write("input_bits\tsem_bits\thand_bits\n")
            for i, sv, hv in diffs:
                f.write(
                    f"0x{i:0{in_eb*2}x}\t0x{sv:0{out_eb*2}x}\t0x{hv:0{out_eb*2}x}\n"
                )
    print(f"[verdict] {row}: {vjson['verdict']} ({ndiff} diffs)", flush=True)
    return vjson


# ---------------------------------------------------------------- ledger


def emit_ledger(ctx, roster, ops):
    lp = ctx.out / "BIT-EXACT-LEDGER.tsv"
    lines = [
        "# THE EXHAUSTIVE BIT-EXACT CHECK — lane JN ledger "
        f"(executor {ctx.sim_id}; flags: pinned ON set; generated {time.strftime('%Y-%m-%d %H:%M:%S')})",
        "row\tformat\tinput-space\tverdict\tdiverging-inputs\tmax-delta\texecutor\tvalidation\truntime-s\tnote",
    ]
    tally = {}
    for row, (cls, reason) in roster:
        fmt = space = "-"
        verdict = note = ""
        ndiff = maxd = "-"
        executor = "-"
        validation = "-"
        runtime = "-"
        rd = ctx.out / "rows" / row
        if cls != "unary":
            verdict = "NOT-EXHAUSTIBLE"
            if cls == "unary-unhooked":
                verdict = "NOT-RUN-UNHOOKED-HARNESS"
            note = reason
        else:
            probe = rd / "probe-sem.npz"
            if probe.exists():
                d = load_dump(probe)
                fmt = f"{d['in_fmt']}->{d['out_fmt']}"
                eb = FMT_BYTES.get(d["in_fmt"], 0)
                space = f"2^{16 if eb == 2 else 32}" if eb else "?"
            vj = rd / "validate.json"
            validated = False
            if vj.exists():
                validated = json.loads(vj.read_text()).get("validated", False)
                validation = (
                    "device-anchor-bitwise-BOTH-LEGS"
                    if validated
                    else "FAILED-device-anchor"
                )
            else:
                validation = "NO-DEVICE-ANCHOR"
            executor = ctx.sim_id
            vfile = rd / "verdict.json"
            if not probe.exists():
                verdict = "EXECUTOR-UNVALIDATED"
                note = "probe run failed/missing"
            elif space == "2^32" and not vfile.exists():
                verdict = "INFEASIBLE-2^32-THIS-EXECUTOR"
                note = (
                    "sound chunked-sim executor would need ~2^20 runs "
                    "(~60 days/leg at measured rate); vectorized host "
                    "executor not built/validated — honestly NOT swept"
                )
            elif vfile.exists():
                v = json.loads(vfile.read_text())
                ndiff = str(v["diverging_inputs"])
                maxd = (
                    f"abs {v['max_abs_delta_finite']:.6g} / ulp {v['max_ulp_delta_finite']}"
                    if v["diverging_inputs"]
                    else "0"
                )
                verdict = v["verdict"]
                if not validated:
                    verdict = "EXECUTOR-UNVALIDATED"
                    note = (
                        f"sweep computed ({v['verdict']}, {ndiff} diffs) but the "
                        "executor did not reproduce device bits on the registered "
                        "stimuli — result does NOT count"
                    )
                rt = rd / "sweep-runtime.txt"
                if rt.exists():
                    runtime = rt.read_text().strip()
            else:
                verdict = "SWEEP-INCOMPLETE"
                note = "probe ok; sweep chunks incomplete"
        tally[verdict] = tally.get(verdict, 0) + 1
        lines.append(
            f"{row}\t{fmt}\t{space}\t{verdict}\t{ndiff}\t{maxd}\t{executor}\t{validation}\t{runtime}\t{note}"
        )
    lines.append("# tally: " + ", ".join(f"{k}={v}" for k, v in sorted(tally.items())))
    lp.write_text("\n".join(lines) + "\n")
    print(f"ledger -> {lp}")
    for k, v in sorted(tally.items()):
        print(f"  {k}: {v}")


# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", default="")
    ap.add_argument("--all-unary", action="store_true")
    ap.add_argument(
        "--stage",
        default="all",
        choices=["probe", "anchor", "validate", "sweep", "verdict", "all", "ledger"],
    )
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--allow-slow", action="store_true")
    ap.add_argument("--first-n", type=int, default=256)
    ap.add_argument(
        "--python",
        default=str(
            (TESTS / ".venv/bin/python")
            if (TESTS / ".venv/bin/python").exists()
            else "/home/ttuser/sfpi-uplift/tt-metal/tt_metal/tt-llk/tests/.venv/bin/python"
        ),
    )
    ap.add_argument(
        "--sim",
        default="/home/ttuser/sfpi-uplift/laneJN-simstage/libttsim.so",
    )
    ap.add_argument("--flags", default=None)
    args = ap.parse_args()

    if args.flags is None:
        sys.path.insert(0, str(CORPUS))
        import sweep_2x2  # noqa: E402

        args.flags = sweep_2x2.ON_FLAGS

    ops = read_ops()
    board = read_board_rows()
    roster = [(r, classify(r, ops)) for r in board]

    if args.all_unary:
        targets = [r for r, (c, _) in roster if c == "unary"]
    else:
        targets = [r for r in args.rows.split(",") if r]
        for r in targets:
            if r not in ops:
                sys.exit(f"unknown row {r}")

    ctx = Ctx(args)
    print(
        f"bitexact_sweep: {len(targets)} target rows, stage={args.stage}, "
        f"executor={ctx.sim_id}, out={ctx.out}"
    )

    def per_row(row, stage):
        try:
            if stage in ("probe", "all"):
                stage_probe(ctx, row, ops)
            if stage in ("sweep", "all"):
                err = stage_sweep(ctx, row, ops)
                if err:
                    print(f"[sweep] {row}: {err}", flush=True)
            if stage in ("verdict", "all"):
                if (ctx.out / "rows" / row / "probe-sem.npz").exists():
                    stage_verdict(ctx, row, ops)
        except Exception as e:  # noqa: BLE001 — record, never crash the fleet
            print(f"[ERROR] {row}: {e!r}", flush=True)

    if args.stage == "anchor":
        for row in targets:  # device: strictly serial under the flocks
            stage_anchor(ctx, row, ops)
    elif args.stage == "validate":
        for row in targets:
            stage_validate(ctx, row)
    elif args.stage == "ledger":
        emit_ledger(ctx, roster, ops)
    else:
        with cf.ThreadPoolExecutor(max_workers=args.jobs) as ex:
            list(ex.map(lambda r: per_row(r, args.stage), targets))
        if args.stage in ("verdict", "all"):
            for row in targets:
                if (ctx.out / "rows" / row / "anchor-sem.npz").exists():
                    stage_validate(ctx, row)
            emit_ledger(ctx, roster, ops)


if __name__ == "__main__":
    main()
