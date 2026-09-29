#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# selftest for prove_all.py — guards ROUTING + CENSUS MATH and exercises the
# engines end-to-end on 3 known ops:
#     zeropad-fresh   (formal_equiv, known EQUAL     -> SMT-PROVEN-ALL-INPUTS)
#     smoothstep-fresh(formal_equiv, known DIVERGENT -> DIVERGENCE-CERTIFIED)
#     binary-bcast    (classify,    known SCOPE      -> NOT-EXHAUSTIBLE)
#
# Part A is a fast pure-unit check of the precedence join + census (no sim) and
# a hardware-free exercise of the silicon_stream engine against the REAL
# galaxy_shard.sh (stub farm, exactly as selftest_galaxy_shard.py does it).
# Part B runs the real driver over the 3 ops and checks the emitted ledger.
#
#   python3 selftest_prove_all.py            # Part A + Part B
#   python3 selftest_prove_all.py --no-live  # Part A only (no pinned instruments)
#
# Exit 0 = PASS.

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
import prove_all as PA  # noqa: E402
import selftest_galaxy_shard as SGS  # noqa: E402  (stub farm + stub streamer)


def part_a_unit():
    """Precedence + census must obey the strict order and machine-certified rule."""
    # silicon supersedes a divergent engine verdict
    r = PA.join_op(
        "x",
        "WIN",
        {"engine": "bitexact", "class": "DIVERGENCE-CERTIFIED", "verdict": "DIVERGENT"},
        {
            "x": {
                "silicon_class": "SILICON-EXHAUSTIVE",
                "silicon_verdict": "BIT-EXACT-ALL-INPUTS",
            }
        },
        {},
    )
    assert r["provability_class"] == "SILICON-EXHAUSTIVE", r
    assert r["machine_certified_equal"] == "YES", r

    # domain overlay upgrades a base DIVERGENT to SMT-PROVEN-DOMAIN
    r = PA.join_op(
        "y",
        "WIN",
        {
            "engine": "formal_equiv",
            "class": "DIVERGENCE-CERTIFIED",
            "verdict": "DIVERGENT",
        },
        {},
        {
            "y": {
                "domain_class": "SMT-PROVEN-DOMAIN",
                "jo_verdict": "PROVEN-EQUIV-ON-DOCUMENTED-DOMAIN",
            }
        },
    )
    assert r["provability_class"] == "SMT-PROVEN-DOMAIN", r
    assert r["machine_certified_equal"] == "no", r  # domain != full machine-certified

    # a pure SMT-all-inputs stays certified-equal
    r = PA.join_op(
        "z",
        "WIN",
        {
            "engine": "formal_equiv",
            "class": "SMT-PROVEN-ALL-INPUTS",
            "verdict": "PROVEN-EQUIV-ALL-INPUTS",
        },
        {},
        {},
    )
    assert (
        r["provability_class"] == "SMT-PROVEN-ALL-INPUTS"
        and r["machine_certified_equal"] == "YES"
    ), r

    # classify not-exhaustible is unreached
    r = PA.join_op(
        "w",
        "PARITY",
        {
            "engine": "classify",
            "class": "NOT-EXHAUSTIBLE",
            "verdict": "NOT-EXHAUSTIBLE",
        },
        {},
        {},
    )
    assert (
        r["provability_class"] == "NOT-EXHAUSTIBLE"
        and r["machine_certified_equal"] == "no"
    ), r

    # census counts what the join produced
    rows = [
        {
            "op": "a",
            "provability_class": "SILICON-EXHAUSTIVE",
            "machine_certified_equal": "YES",
        },
        {
            "op": "b",
            "provability_class": "SMT-PROVEN-ALL-INPUTS",
            "machine_certified_equal": "YES",
        },
        {
            "op": "c",
            "provability_class": "DIVERGENCE-CERTIFIED",
            "machine_certified_equal": "no",
        },
    ]
    cen = PA.census(rows)
    assert cen["SILICON-EXHAUSTIVE"] == 1 and cen["SMT-PROVEN-ALL-INPUTS"] == 1
    mc = sum(1 for r in rows if r["machine_certified_equal"] == "YES")
    assert mc == 2
    print("PART A (unit precedence + census): PASS")


def _run_formal_fake(rc, payload):
    """Drive PA.run_formal with both sim legs and the formal engine FAKED, so
    only the exit-code/verdict plumbing under test executes (no sim, no z3).
    `payload` is the verdict JSON the fake engine writes (dict), a raw string
    to write verbatim, or None for "the prover produced no output file"."""
    tmp = Path(tempfile.mkdtemp(prefix="prove_all_formal_"))
    real_leg, real_run = PA._run_leg, PA.subprocess.run

    def fake_leg(op, leg, node, out_dir, flags, timeout):
        t = Path(out_dir) / f"trace-{op}-{leg}.log"
        t.write_text("SFPUJO I\n")
        # distinct .text hashes so the REFUSED-IDENTITY short-circuit is not hit
        return t, {"path": str(t), "text_sha256": ("a" if leg == "sem" else "b") * 64}, None

    def fake_run(cmd, **kw):
        out = Path(cmd[cmd.index("--out") + 1])
        row = cmd[cmd.index("--row") + 1]
        if payload is not None:
            body = payload if isinstance(payload, str) else json.dumps(payload)
            (out / f"{row}-verdict.json").write_text(body)
        return subprocess.CompletedProcess(cmd, rc, stdout="", stderr="engine stderr")

    PA._run_leg, PA.subprocess.run = fake_leg, fake_run
    try:
        return PA.run_formal(
            "op",
            {"sem_node": "s.py::a", "hand_node": "h.py::b", "reason": "-"},
            tmp,
            "-flags",
            60,
        )
    finally:
        PA._run_leg, PA.subprocess.run = real_leg, real_run


def part_a_formal_routing():
    """formal_equiv encodes its verdict in the EXIT CODE:
        rc=0  PROVEN-EQUIV-ALL-INPUTS
        rc=2  PROVEN-EQUIV-ON-DOCUMENTED-DOMAIN / DIVERGENT / UNDECIDED
        rc=1  SEMANTICS-UNVALIDATED / NO-Z3 / SCOPE-REFUSED (early returns)
    A non-zero rc that still wrote a well-formed verdict JSON is a REACHED
    VERDICT and must be classified; only a crash / missing / unparsable
    verdict file is a genuine prover failure (UNSWEPT)."""
    reached = [
        (0, "PROVEN-EQUIV-ALL-INPUTS", "SMT-PROVEN-ALL-INPUTS"),
        (2, "PROVEN-EQUIV-ON-DOCUMENTED-DOMAIN", "SMT-PROVEN-DOMAIN"),
        (2, "DIVERGENT", "DIVERGENCE-CERTIFIED"),
        (2, "UNDECIDED", "UNDECIDED-Z3-TIMEOUT"),
        (1, "SEMANTICS-UNVALIDATED", "SCOPE-REFUSED"),
        (1, "SCOPE-REFUSED", "SCOPE-REFUSED"),
    ]
    for rc, verdict, want in reached:
        got = _run_formal_fake(rc, {"row": "op", "verdict": verdict, "details": {}})
        assert got["verdict"] == verdict, (rc, verdict, got)
        assert got["class"] == want, (rc, verdict, got["class"], want)
    # NO-Z3 reached the emitter but is not a proof verdict: stays UNSWEPT
    got = _run_formal_fake(1, {"row": "op", "verdict": "NO-Z3"})
    assert got["class"] == "UNSWEPT" and got["verdict"] == "NO-Z3", got
    # genuine prover failures keep the UNSWEPT / PROVER-FAILED path
    for rc, payload in ((0, None), (1, None), (2, None), (2, "{ not json"),
                        (2, {"row": "op"})):
        got = _run_formal_fake(rc, payload)
        assert got["class"] == "UNSWEPT" and got["verdict"] == "PROVER-FAILED", (
            rc, payload, got
        )
    print("PART A (formal_equiv exit-code/verdict routing): PASS")


def part_a_fast_census():
    """The fast-set 'certified' count must count only the certified classes
    (the MACHINE_CERTIFIED pair plus SMT-PROVEN-DOMAIN, i.e. the top of
    CLASS_ORDER) — never infeasible / scope-refused / unswept ops."""
    fast = ["c1", "c2", "d1", "g1", "g2", "g3"]
    rows = [
        {"op": "c1", "provability_class": "SMT-PROVEN-ALL-INPUTS",
         "machine_certified_equal": "YES"},
        {"op": "c2", "provability_class": "SMT-PROVEN-DOMAIN",
         "machine_certified_equal": "no"},
        {"op": "d1", "provability_class": "DIVERGENCE-CERTIFIED",
         "machine_certified_equal": "no"},
        {"op": "g1", "provability_class": "INFEASIBLE-2^32",
         "machine_certified_equal": "no"},
        {"op": "g2", "provability_class": "SCOPE-REFUSED",
         "machine_certified_equal": "no"},
        {"op": "g3", "provability_class": "UNSWEPT",
         "machine_certified_equal": "no"},
    ]
    tmp = Path(tempfile.mkdtemp(prefix="prove_all_census_"))
    txt = PA.write_summary(tmp, rows, {"shas": {}, "timestamp": "t"}, "0s", fast)
    hit = [ln for ln in txt.splitlines() if "certified-or-domain in fast set" in ln]
    assert hit, txt
    n = int(hit[0].rsplit("=", 1)[1])
    assert n == 2, ("refused/infeasible/unswept ops counted as certified", n, txt)
    assert "36 fast ops" not in txt, (
        "stale hardcoded paper reconciliation ('36 fast ops -> 25/11') still "
        "printed; prove_all_fast_ops.tsv has %d rows" % len(PA.read_tsv(PA.FAST_OPS))
    )
    print("PART A (fast-set certified census): PASS")


def part_a_manifest():
    """Routing sanity: the 3 selftest ops route to the expected engines."""
    board = PA.load_board()
    man = PA.load_manifest(board)
    exp = {
        "zeropad-fresh": "formal_equiv",
        "smoothstep-fresh": "formal_equiv",
        "binary-bcast": "classify",
    }
    for op, eng in exp.items():
        assert man[op]["engine"] == eng, (op, man[op]["engine"], eng)
    print("PART A (manifest routing of 3 selftest ops): PASS")


# ---------------------------------------------------------------------------
# Part A (silicon_stream): the driver must DRIVE the existing 32-chip streamer.
#
# Hardware-free by the same trick selftest_galaxy_shard.py uses: a throwaway
# FARM_ROOT whose streamer and elf_text_sha are stubs.  Everything between
# prove_all and those stubs is the real thing — the real galaxy_shard.sh, its
# real identity gate, the real galaxy_combine.py.
# ---------------------------------------------------------------------------
SILICON_MAN_ROW = {
    "op": "myop",
    "board_class": "WIN",
    "arity_space": "single-2^20",  # small so 8 stub slices finish instantly
    "engine": "silicon_stream",
    "sem_node": "sem-node",
    "hand_node": "hand-node",
    "expected_class_ref": "SILICON-EXHAUSTIVE",
    "reason": "selftest stub farm",
}


def _stub_farm(work, *, idmap_sem_body=b"sem-text\n"):
    """A stub FARM_ROOT + the IDMAP galaxy_shard.sh gates against."""
    farm = SGS.Farm(work, "fp32")  # single-operand space -> fp32 streamer
    idmap = work / "idmap.tsv"
    idmap.write_text(
        "myop\tAAA\t%s\tBBB\t%s\n"
        % (
            hashlib.sha256(idmap_sem_body).hexdigest(),
            hashlib.sha256(b"hand-text\n").hexdigest(),
        )
    )
    return farm, idmap


def _run_silicon_stub(work, *, dead="", band_bits=10, npar=8, sem_body=b"sem-text\n"):
    """Invoke prove_all's silicon_stream engine over the stub farm."""
    work.mkdir(parents=True, exist_ok=True)
    farm, idmap = _stub_farm(work, idmap_sem_body=sem_body)
    record = work / "calls.jsonl"
    record.write_text("")
    args = argparse.Namespace(
        silicon_farm_root=str(farm.farm_root),
        silicon_venv=str(farm.venv),
        silicon_idmap=str(idmap),
    )
    saved = dict(os.environ)
    os.environ.update(
        SHARD_RECORD=str(record),
        DEAD_CHIPS=dead,
        STAGGER="0",
        NPAR=str(npar),
        BAND_BITS=str(band_bits),
    )
    try:
        rec = PA.run_silicon_stream("myop", SILICON_MAN_ROW, work / "ev", args, 600)
    finally:
        os.environ.clear()
        os.environ.update(saved)
    calls = [json.loads(l) for l in record.read_text().splitlines() if l.strip()]
    return rec, calls, record


def part_a_silicon_stream():
    """silicon_stream shells out to galaxy_shard.sh and translates its verdict."""
    # 1. the translation table adds no class of its own
    assert set(PA.SILICON_CLASS_MAP) <= {"BIT-EXACT-ALL-INPUTS", "DIVERGENT"}
    for cls in PA.SILICON_CLASS_MAP.values():
        assert cls in PA.RANK, cls
    assert PA.SILICON_CLASS_MAP["BIT-EXACT-ALL-INPUTS"] == "SILICON-EXHAUSTIVE"
    assert "SILICON-EXHAUSTIVE" in PA.MACHINE_CERTIFIED

    with tempfile.TemporaryDirectory(prefix="prove_all_silicon_") as tmp:
        root = Path(tmp)

        # 2. a clean 8-chip shard of 2^20 certifies, and the record is joinable
        rec, calls, record = _run_silicon_stub(root / "ok")
        assert rec["engine"] == "silicon_stream", rec
        assert rec["class"] == "SILICON-EXHAUSTIVE", rec
        assert rec["verdict"] == "BIT-EXACT-ALL-INPUTS", rec
        assert rec["covered"] == str(1 << 20), rec
        assert rec["geometry"]["NPAR"] == "8", rec
        assert len(calls) == 8, calls
        assert sorted(int(c["chip"]) for c in calls) == list(range(8)), calls
        assert {int(c["total"], 0) for c in calls} == {(1 << 20) // 8}, calls
        joined = PA.join_op("myop", "WIN", rec, {}, {})
        assert joined["provability_class"] == "SILICON-EXHAUSTIVE", joined
        assert joined["machine_certified_equal"] == "YES", joined
        assert not PA.operational_failures({"myop": rec}), rec

        # 3. re-entering does NOT short-circuit: prove_all defers resume to the
        #    streamer, so the shard is driven again (here with a new geometry
        #    that prove_all's own cache key would not have noticed).
        before = len(record.read_text().splitlines())
        rec2, calls2, _ = _run_silicon_stub(root / "ok2", band_bits=9)
        assert rec2["class"] == "SILICON-EXHAUSTIVE", rec2
        assert calls2, "re-entry launched no slices"
        assert not PA.cache_eligible("silicon_stream")
        assert PA.cache_eligible("formal_equiv") and PA.cache_eligible("bitexact")
        assert before == 8

        # 4. a dead chip must never be certified — the combiner says INCOMPLETE
        #    and the record stays an auditable operational failure.
        rec3, _, _ = _run_silicon_stub(root / "dead", dead="3")
        assert rec3["class"] == "UNSWEPT", rec3
        assert rec3["verdict"] == "INCOMPLETE", rec3
        assert PA.operational_failures({"myop": rec3}), rec3

        # 5. the shard's own identity gate is honoured, not re-implemented
        rec4, calls4, _ = _run_silicon_stub(root / "refused", sem_body=b"wrong\n")
        assert rec4["class"] == "UNSWEPT", rec4
        assert rec4["verdict"].startswith("REFUSED-IDENTITY"), rec4
        assert not calls4, "a refused identity gate still streamed slices"

        # 6. missing farm config is a named refusal, not a silent pass
        rec5 = PA.run_silicon_stream(
            "myop",
            SILICON_MAN_ROW,
            root / "nofarm",
            argparse.Namespace(
                silicon_farm_root=None, silicon_venv=None, silicon_idmap=None
            ),
            600,
        )
        assert rec5["class"] == "UNSWEPT" and rec5["verdict"] == "NO-FARM", rec5
    print("PART A (silicon_stream drives galaxy_shard.sh, stub farm): PASS")


def part_b_live():
    """Run the driver end-to-end on the 3 ops; check the ledger classes."""
    tmp = Path(tempfile.mkdtemp(prefix="prove_all_selftest_"))
    ops = "zeropad-fresh,smoothstep-fresh,binary-bcast"
    print(f"PART B: running prove_all --only {ops} (out={tmp}) ...")
    r = subprocess.run(
        [
            sys.executable,
            str(HERE / "prove_all.py"),
            "--only",
            ops,
            "--out",
            str(tmp),
            "--timeout",
            "600",
        ],
        capture_output=True,
        text=True,
    )
    sys.stdout.write(r.stdout[-1500:])
    if r.returncode != 0:
        sys.stderr.write(r.stderr[-1500:])
        raise SystemExit("driver exited non-zero")
    ledger = tmp / "MASTER-COVERAGE-LEDGER.tsv"
    got = {}
    for row in PA.read_tsv(ledger):
        got[row["op"]] = row["provability_class"]
    expect = {
        "zeropad-fresh": "SMT-PROVEN-ALL-INPUTS",
        "smoothstep-fresh": "DIVERGENCE-CERTIFIED",
        "binary-bcast": "NOT-EXHAUSTIBLE",
    }
    ok = True
    for op, cls in expect.items():
        actual = got.get(op)
        flag = "OK" if actual == cls else "MISMATCH"
        if actual != cls:
            ok = False
        print(f"  {op:20s} expect={cls:24s} got={actual} [{flag}]")
    # census of the 3-op run must be internally consistent
    rows = PA.read_tsv(ledger)
    assert len(rows) == 3, rows
    if not ok:
        raise SystemExit("PART B: ledger class mismatch")
    print("PART B (live 3-op end-to-end): PASS")


if __name__ == "__main__":
    part_a_unit()
    part_a_formal_routing()
    part_a_fast_census()
    part_a_manifest()
    part_a_silicon_stream()
    if "--no-live" in sys.argv[1:]:
        print("PART B skipped (--no-live): needs the pin-59 instruments")
    else:
        part_b_live()
    print("\nSELFTEST: ALL PASS")
