#!/usr/bin/env python3
"""Self-test for the three W3 harness measurement-gap fixes (laneDO).

Drives the REAL sweep code (imported, not re-implemented) with filesystem
fixtures and stubbed device/classify workers — no toolchain, no simulator,
no device:

  1. KNOB-LEG MODES (harness gap 1): knob_legs() produces EXACT flag lists —
     solo knobs keep the historical OFF vs OFF+flag shape; drop-one knobs
     (replay-exec-record, planner-residency, init-hoist) get reviewed-ON
     minus exactly their flag token(s) vs the full reviewed-ON set; on-plus
     knobs (replay-loop-unroll, int-abs, lut-select-leaf-ext, repr-prop —
     the default-off booking flags) get plain reviewed-ON vs reviewed-ON
     plus exactly their outside-ON token(s), in-ON tokens deduped; a
     drop-one flag outside the ON set refuses loudly, and an on-plus flag
     entirely inside it refuses loudly (mirror images); attribute_knobs and
     knob_silicon build their legs from knob_legs (no site left on the old
     hardcoded solo shape); init-hoist HAS a knob row (it had none) and is
     drop-one (laneCJ census-timing fact: solo init-hoist ALWAYS refuses).
  2. EQZ-CLASS SEM LEG (harness gap 2): a fresh-body row (fresh_cpp sem
     nodes) whose sem OFF/ON pair is byte-identical MEASURES one physical
     sem leg that fills both sem cells (the hand OFF==ON rule verbatim) and
     produces a real vs_hand number; a NON-fresh row with the same
     classification keeps the recorded refusal byte-for-byte (cells,
     note, zero sem device jobs, verdict strings) — unaffected rows are
     verdict-byte-stable.  _silicon_jobs enumerates the identical fold.
  3. EVIDENCE-ROOT PIN GUARD, sweep level (DD shipped the wrapper half):
     check_evidence_root_pin refuses a foreign-pin root (PIN_STAMP or
     existing preflight.json), passes same-pin resumes and fresh roots,
     and the SWEEP-LEVEL flow refuses END-TO-END — a real
     `python3 sweep_2x2.py` subprocess against a foreign-pin root exits
     nonzero at the guard BEFORE any compiler work, while a same-pin root
     proceeds past the guard (and dies later at the missing compiler,
     proving the guard's placement, not its absence).

Run standalone or from the sweep wrappers; exits nonzero on any failure so
a broken gate can never bless a sweep.
"""

import importlib.util
import json
import pathlib
import subprocess
import sys
import tempfile
import types

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("sweep_2x2", HERE / "sweep_2x2.py")
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)

FAILS = []


def check(name, cond, detail=""):
    if cond:
        print(f"SELFTEST PASS: {name}")
    else:
        print(f"SELFTEST FAIL: {name} {detail}")
        FAILS.append(name)


# ---------------- 1. knob-leg modes ----------------

DROP_ONE_EXPECTED = ("replay-exec-record", "planner-residency", "init-hoist")

check(
    "knob modes: the three dependent/service knobs are drop-one",
    all(sweep.knob_mode(k) == "drop-one" for k in DROP_ONE_EXPECTED),
    {k: sweep.knob_mode(k) for k in DROP_ONE_EXPECTED},
)
check(
    "knob modes: init-hoist HAS a knob row (gap: it had none)",
    "init-hoist" in sweep.KNOBS
    and sweep.KNOBS["init-hoist"] == "-mtt-tensix-optimize-init-hoist",
    sweep.KNOBS.get("init-hoist"),
)
check(
    "knob modes: every KNOB_MODES key is a KNOBS key, values legal",
    all(k in sweep.KNOBS for k in sweep.KNOB_MODES)
    and all(v in ("solo", "drop-one", "on-plus") for v in sweep.KNOB_MODES.values()),
    sweep.KNOB_MODES,
)

ON_TOKENS = sweep.ON_FLAGS.split()
OFF_TOKENS = sweep.OFF_FLAGS.split()
for knob in DROP_ONE_EXPECTED:
    flag_tokens = sweep.KNOBS[knob].split()
    legs = sweep.knob_legs(knob)
    off_leg, knob_leg = dict(legs)["off"], dict(legs)["knob"]
    expected_off = [t for t in ON_TOKENS if t not in flag_tokens]
    check(
        f"drop-one {knob}: knob leg is the FULL reviewed-ON set, exactly",
        knob_leg == sweep.ON_FLAGS,
        knob_leg,
    )
    check(
        f"drop-one {knob}: off leg is reviewed-ON minus exactly its flag "
        "token(s), order preserved (exact list)",
        off_leg.split() == expected_off and all(t in ON_TOKENS for t in flag_tokens),
        off_leg,
    )
    check(
        f"drop-one {knob}: leg names stay ('off','knob') — downstream "
        "delta/sample naming unchanged",
        [n for n, _ in legs] == ["off", "knob"],
        legs,
    )

# solo knobs keep the historical shape, exactly
for knob in ("ccmask", "replay-hoist", "invariant-loadi"):
    legs = sweep.knob_legs(knob)
    check(
        f"solo {knob}: OFF vs OFF+flag, exact flag lists",
        dict(legs)["off"] == sweep.OFF_FLAGS
        and dict(legs)["knob"] == f"{sweep.OFF_FLAGS} {sweep.KNOBS[knob]}"
        and dict(legs)["knob"].split() == OFF_TOKENS + sweep.KNOBS[knob].split(),
        legs,
    )

# a drop-one flag outside the reviewed ON set refuses loudly
try:
    sweep.drop_one_flags("-mtt-tensix-optimize-not-in-on-set")
    check("drop-one: flag outside the ON set refuses (SystemExit)", False)
except SystemExit as e:
    check(
        "drop-one: flag outside the ON set refuses (SystemExit)",
        "NOT in the reviewed ON set" in str(e),
        e,
    )

# ---- on-plus mode (default-off booking flags; coordinator extension) ----
ON_PLUS_EXPECTED = ("replay-loop-unroll", "int-abs", "lut-select-leaf-ext", "repr-prop")
check(
    "knob modes: the four default-off booking knobs are on-plus",
    all(sweep.knob_mode(k) == "on-plus" for k in ON_PLUS_EXPECTED),
    {k: sweep.knob_mode(k) for k in ON_PLUS_EXPECTED},
)
for knob in ON_PLUS_EXPECTED:
    legs = sweep.knob_legs(knob)
    off_leg, knob_leg = dict(legs)["off"], dict(legs)["knob"]
    new_tokens = [t for t in sweep.KNOBS[knob].split() if t not in ON_TOKENS]
    check(
        f"on-plus {knob}: off leg is the PLAIN reviewed-ON set, exactly",
        off_leg == sweep.ON_FLAGS,
        off_leg,
    )
    check(
        f"on-plus {knob}: knob leg is reviewed-ON plus exactly its "
        "outside-ON token(s), order preserved (exact list)",
        knob_leg.split() == ON_TOKENS + new_tokens and len(new_tokens) >= 1,
        knob_leg,
    )
    check(
        f"on-plus {knob}: leg names stay ('off','knob') — the knob leg "
        "contains the flag, delta sign convention unchanged",
        [n for n, _ in legs] == ["off", "knob"],
        legs,
    )

# The replay completion profitability guard is intentionally a DEFAULT-OFF
# booking knob.  Its proof acts on shapes produced by replay-hoist in the
# reviewed ON pipeline, so the only meaningful A/B is ON vs ON+guard.
_completion_knob = "replay-hoist-completion-guard"
_completion_flag = "-mtt-tensix-replay-hoist-completion-guard"
_completion_legs = sweep.knob_legs(_completion_knob)
check(
    "replay completion guard: exact flag and on-plus mode",
    sweep.KNOBS.get(_completion_knob) == _completion_flag
    and sweep.knob_mode(_completion_knob) == "on-plus",
    {
        "flag": sweep.KNOBS.get(_completion_knob),
        "mode": sweep.knob_mode(_completion_knob),
    },
)
check(
    "replay completion guard: legs are exactly ON vs ON+guard",
    dict(_completion_legs)["off"] == sweep.ON_FLAGS
    and dict(_completion_legs)["knob"].split() == ON_TOKENS + [_completion_flag]
    and [name for name, _ in _completion_legs] == ["off", "knob"],
    _completion_legs,
)
check(
    "replay completion guard: not promoted into reviewed ON",
    _completion_flag not in ON_TOKENS,
    sweep.ON_FLAGS,
)

# lut-select-leaf-ext's parent lut-select token is already IN the ON set:
# on-plus must DEDUPE it (append only leaf-ext + license), never double it.
_lleg = dict(sweep.knob_legs("lut-select-leaf-ext"))["knob"].split()
check(
    "on-plus lut-select-leaf-ext: in-ON parent token deduped (appears once), "
    "leaf-ext + finite-math license appended",
    _lleg.count("-mtt-tensix-optimize-lut-select") == 1
    and _lleg[-2:]
    == ["-mtt-tensix-optimize-lut-select-leaf-ext", "-ffinite-math-only"],
    _lleg[-4:],
)

# an on-plus flag ENTIRELY inside the reviewed ON set refuses loudly (an
# A/A of the full ON set — the mirror image of drop-one's check)
try:
    sweep.on_plus_flags("-mtt-tensix-optimize-ccmask")
    check("on-plus: flag entirely inside the ON set refuses (SystemExit)", False)
except SystemExit as e:
    check(
        "on-plus: flag entirely inside the ON set refuses (SystemExit)",
        "ENTIRELY inside the reviewed ON" in str(e),
        e,
    )


# attribute_knobs and knob_silicon build their legs from knob_legs: capture
# the legs each site passes to classify() for one solo and one drop-one knob.
def mk_row(op, nodes=None, kind="full2x2"):
    return {
        "op": op,
        "corpus_id": f"fix__{op}",
        "kind": kind,
        "marker": "TILE_LOOP",
        "metric": "MATH_ISOLATE",
        "per_tile": True,
        "issue_slot_lb": None,
        "pin_flags": "",
        "extra_env": {},
        "sel_extra_env": {"sem": {}, "hand": {}},
        "craq_archs": "bh",
        "nodes": {
            sel: (nodes or {}).get(sel, "")
            for sel in ("sem-corr", "sem-perf", "hand-corr", "hand-perf")
        },
    }


def mk_sweep(ev):
    sw = object.__new__(sweep.Sweep)
    sw.a = types.SimpleNamespace(
        force=False,
        dry_run=False,
        knob_silicon_rows=None,
        knob_attribution=True,
        skip_craq_gate=True,
        classify_workers=1,
        priority_ops=None,
        prev_run=None,
        baseline=None,
        allow_win_to_parity=False,
        max_drift_pct=5.0,
        max_abs_drift_pct=10.0,
        red_loss_growth_pct=5.0,
    )
    sw.ev = pathlib.Path(ev)
    sw.ev.mkdir(parents=True, exist_ok=True)
    sw.reds = []
    sw.reused = []
    return sw


with tempfile.TemporaryDirectory() as td:
    td = pathlib.Path(td)
    sw = mk_sweep(td / "ev")
    seen_legs = {}

    def fake_classify(row, sel, legs=None, tag="classify"):
        seen_legs[tag] = legs
        return {"status": "OK", "all": "IDENTICAL"}

    sw.classify = fake_classify
    row = mk_row("kop", {"sem-perf": "perf.py::t[mathop:K]"})
    (sw.ev / "kop").mkdir()  # classify() creates this in the real flow
    sw.attribute_knobs(row, {"sem-perf": {"status": "OK", "all": "CHANGED"}})
    check(
        "attribute_knobs: solo knob leg spec comes from knob_legs (ccmask)",
        seen_legs.get("knobs/ccmask") == sweep.knob_legs("ccmask"),
        seen_legs.get("knobs/ccmask"),
    )
    check(
        "attribute_knobs: drop-one knob leg spec comes from knob_legs "
        "(init-hoist: ON-minus-flag vs full ON)",
        seen_legs.get("knobs/init-hoist") == sweep.knob_legs("init-hoist")
        and dict(seen_legs["knobs/init-hoist"])["knob"] == sweep.ON_FLAGS,
        seen_legs.get("knobs/init-hoist"),
    )
    check(
        "attribute_knobs: every KNOBS knob got exactly its knob_legs spec",
        all(seen_legs.get(f"knobs/{k}") == sweep.knob_legs(k) for k in sweep.KNOBS),
        sorted(seen_legs),
    )

    # FY-F1 INJECTION (lane GF): the knob-only-row blindness.  A row whose
    # ONLY effect rides a default-off booking flag classifies byte-identical
    # on the main sem OFF-vs-ON pair (unaryshift-fresh / castfp32tofp16a /
    # unarybitwise-fresh at ON-28: lane FY measured their unroll knob legs
    # MANUALLY because the pregate returned SKIP_NOT_CHANGED).  A row
    # REGISTERED for knob silicon with a CLEAN byte-identical main verdict
    # must still get its knob legs — the on-plus knob's (ON vs ON+flag)
    # span is never measured by the main pair.
    sw_fy = mk_sweep(td / "ev-fy")
    sw_fy.a.knob_silicon_rows = ["unaryshift-fresh"]

    def fake_classify_fy(row, sel, legs=None, tag="classify"):
        # the unroll knob fires on this row; every other knob refuses
        if tag == "knobs/replay-loop-unroll":
            return {"status": "OK", "all": "CHANGED"}
        return {"status": "OK", "all": "IDENTICAL"}

    sw_fy.classify = fake_classify_fy
    row_fy = mk_row("unaryshift-fresh", {"sem-perf": "perf.py::t[mathop:LeftShift]"})
    (sw_fy.ev / "unaryshift-fresh").mkdir()
    att = sw_fy.attribute_knobs(
        row_fy, {"sem-perf": {"status": "OK", "all": "IDENTICAL"}}
    )
    check(
        "FY-F1: REGISTERED knob-silicon row with byte-identical main legs "
        "still gets knob attribution (on-plus knob fire found, not "
        "SKIP_NOT_CHANGED)",
        att.get("status") == "OK"
        and att.get("firing_knobs") == ["replay-loop-unroll"]
        and att.get("single_knob_attribution") == "replay-loop-unroll",
        att,
    )
    att2 = sw_fy.attribute_knobs(
        mk_row("otherop", {"sem-perf": "perf.py::t[mathop:O]"}),
        {"sem-perf": {"status": "OK", "all": "IDENTICAL"}},
    )
    check(
        "FY-F1: UNREGISTERED row keeps the historical cost pregate "
        "(SKIP_NOT_CHANGED on byte-identical main)",
        att2.get("status") == "SKIP_NOT_CHANGED",
        att2,
    )
    att3 = sw_fy.attribute_knobs(
        row_fy, {"sem-perf": {"status": "COMPILE_FAIL", "all": None}}
    )
    check(
        "FY-F1: only a CLEAN byte-identical main verdict opens the "
        "registered-row path (COMPILE_FAIL still skips)",
        att3.get("status") == "SKIP_NOT_CHANGED",
        att3,
    )

    # knob_silicon: classification IDENTICAL short-circuits before any device
    # work, so a classify stub is enough to capture the legs it passes.
    sw2 = mk_sweep(td / "ev2")
    seen2 = {}

    def fake_classify2(row, sel, legs=None, tag="classify"):
        seen2[tag] = legs
        return {"status": "OK", "all": "IDENTICAL"}

    sw2.classify = fake_classify2
    row2 = mk_row(
        "kop2",
        {"sem-perf": "perf.py::t[mathop:K2]", "sem-corr": "corr.py::t[mathop:K2]"},
    )
    (sw2.ev / "kop2").mkdir()  # classify() creates this in the real flow
    sw2.knob_silicon(
        row2,
        {
            "op": "kop2",
            "selector": "sem-perf",
            "status": "OK",
            "firing_knobs": ["init-hoist", "ccmask", "replay-loop-unroll"],
        },
    )
    out = json.loads((sw2.ev / "kop2" / "knob-silicon.json").read_text())
    check(
        "knob_silicon: drop-one knob's legs/flags come from knob_legs "
        "(entry records mode + both flag sets)",
        seen2.get("knobs/init-hoist") == sweep.knob_legs("init-hoist")
        and out["init-hoist"]["flags"] == sweep.ON_FLAGS
        and out["init-hoist"]["off_flags"] == dict(sweep.knob_legs("init-hoist"))["off"]
        and out["init-hoist"]["mode"] == "drop-one",
        out.get("init-hoist"),
    )
    check(
        "knob_silicon: solo knob entry keeps the historical flag shape",
        out["ccmask"]["flags"] == f"{sweep.OFF_FLAGS} {sweep.KNOBS['ccmask']}"
        and out["ccmask"]["mode"] == "solo",
        out.get("ccmask"),
    )
    check(
        "knob_silicon: on-plus knob's legs/flags come from knob_legs "
        "(off = plain ON, knob = ON + flag, mode recorded)",
        seen2.get("knobs/replay-loop-unroll") == sweep.knob_legs("replay-loop-unroll")
        and out["replay-loop-unroll"]["off_flags"] == sweep.ON_FLAGS
        and out["replay-loop-unroll"]["flags"]
        == f"{sweep.ON_FLAGS} {sweep.KNOBS['replay-loop-unroll']}"
        and out["replay-loop-unroll"]["mode"] == "on-plus",
        out.get("replay-loop-unroll"),
    )
    check(
        "knob_silicon: byte-identical knob pair still refuses (no device run)",
        out["init-hoist"]["status"] == "REFUSAL_BYTE_IDENTICAL"
        and out["ccmask"]["status"] == "REFUSAL_BYTE_IDENTICAL"
        and out["replay-loop-unroll"]["status"] == "REFUSAL_BYTE_IDENTICAL",
        out,
    )

# ---------------- 2. eqz-class sem leg ----------------

FRESH_NODES = {
    "sem-corr": "test_u.py::test_eqz_fresh_cpp[edges-fresh_cpp]",
    "sem-perf": "perf_u.py::t[mathop:EqualZero-fresh_cpp_impl:1]",
    "hand-corr": "test_u.py::test_eqz_fresh_cpp[edges-production]",
    "hand-perf": "perf_u.py::t[mathop:EqualZero-fresh_cpp_impl:0]",
}
PLAIN_NODES = {
    "sem-corr": "test_u.py::test_op[functional]",
    "sem-perf": "perf_u.py::t[mathop:Op-impl:sem]",
    "hand-corr": "test_u.py::test_op[functional]",
    "hand-perf": "perf_u.py::t[mathop:Op-impl:hand]",
}
fresh_row = mk_row("eqz-fresh", FRESH_NODES)
plain_row = mk_row("plainop", PLAIN_NODES)

check(
    "fresh_body_row: fresh_cpp sem nodes detected, plain row and pinpair not",
    sweep.fresh_body_row(fresh_row)
    and not sweep.fresh_body_row(plain_row)
    and not sweep.fresh_body_row(mk_row("pp", FRESH_NODES, kind="pinpair")),
)

CLS_SEM_IDENT = {
    "sem-corr": {"status": "OK", "all": "IDENTICAL"},
    "sem-perf": {"status": "OK", "all": "IDENTICAL"},
    "hand-corr": {"status": "OK", "all": "IDENTICAL"},
    "hand-perf": {"status": "OK", "all": "CHANGED"},
}


def mk_silicon_sweep(ev, perf_values):
    """A Sweep whose device/classify plumbing is stubbed; silicon() itself,
    the leg folding, and the ratio computation are the REAL code."""
    sw = mk_sweep(ev)
    sw.a.dry_run = False
    sw.device_jobs = []

    def fake_device_job(
        row, sel, label, leg, flags, tag="silicon", expected_texts=None
    ):
        sw.device_jobs.append((sel, label, leg, flags))
        return 0

    sw._device_job = fake_device_job
    # marker/per_tile: the lane-ET dual-metric read (KERNEL cell) goes
    # through the same stub — kernel == diag values keeps every ratio
    # assertion unchanged.
    sw._perf_value = lambda row, sel, label, leg, tag="silicon", marker=None, per_tile=None: perf_values[
        sel
    ]
    sw._classify_texts = lambda row, sel, leg, tag="classify": {"math.elf": "x"}
    sw._macro_lb_gate = lambda row, cls, result: None
    sw._issue_slot_check = lambda row, result: None
    sw._measured_flags_gate = lambda row, cls, result: None
    return sw


with tempfile.TemporaryDirectory() as td:
    td = pathlib.Path(td)
    # fresh-body row, sem pair byte-identical: ONE physical sem leg fills
    # both sem cells; vs_hand computes normally (sem 90 vs hand 100 = -10%).
    sw = mk_silicon_sweep(td / "ev", {"sem-perf": 90.0, "hand-perf": 100.0})
    res = sw.silicon(fresh_row, CLS_SEM_IDENT)
    sem_jobs = [j for j in sw.device_jobs if j[0] == "sem-perf"]
    check(
        "eqz rule: fresh-body sem OFF==ON runs exactly one physical leg "
        "(3 perf procs, leg 'off', OFF flags)",
        [j[2] for j in sem_jobs] == ["off", "off", "off"]
        and all(j[3] == sweep.OFF_FLAGS for j in sem_jobs),
        sem_jobs,
    )
    check(
        "eqz rule: both sem cells filled with the measured value (no refusal)",
        res["cells"]["sem_off"] == 90.0 and res["cells"]["sem_on"] == 90.0,
        res["cells"],
    )
    check(
        "eqz rule: vs_hand_pct is a REAL number computed normally",
        isinstance(res.get("vs_hand_pct"), float)
        and abs(res["vs_hand_pct"] - (-10.0)) < 1e-9,
        res.get("vs_hand_pct"),
    )
    check(
        "eqz rule: the row's note names the fresh-body one-leg fold",
        any("fresh-body row" in n and "eqz-class rule" in n for n in res["notes"]),
        res["notes"],
    )
    check(
        "eqz rule: ROW-VERDICT class is measured (WIN here), not REFUSAL",
        sweep.Sweep._row_class(res) == "WIN",
        sweep.Sweep._row_class(res),
    )
    verdict = sw._row_verdict(res, {}, {}, {})
    check(
        "eqz rule: _row_verdict books a real vs-hand verdict (no refusal branch)",
        verdict["rag"] == "GREEN"
        and not any("refusal" in v.lower() for v in verdict["verdicts"]),
        verdict,
    )

    # NON-fresh row, same classification: the refusal path byte-stable.
    sw2 = mk_silicon_sweep(td / "ev2", {"sem-perf": 90.0, "hand-perf": 100.0})
    res2 = sw2.silicon(plain_row, CLS_SEM_IDENT)
    check(
        "unaffected rows: non-fresh sem OFF==ON keeps the recorded refusal "
        "(cells, zero sem device jobs)",
        res2["cells"]["sem_off"] == "REFUSAL_BYTE_IDENTICAL"
        and res2["cells"]["sem_on"] == "REFUSAL_BYTE_IDENTICAL"
        and not [j for j in sw2.device_jobs if j[0] == "sem-perf"],
        (res2["cells"], sw2.device_jobs),
    )
    check(
        "unaffected rows: refusal note byte-stable",
        "sem-perf OFF/ON byte-identical: recorded refusal, no device run"
        in res2["notes"],
        res2["notes"],
    )
    verdict2 = sw2._row_verdict(res2, {}, {}, {})
    check(
        "unaffected rows: refusal verdict strings byte-stable",
        verdict2["verdicts"] == ["refusal byte-identical (no baseline history): GREEN"]
        and verdict2["rag"] == "GREEN",
        verdict2,
    )

    # _silicon_jobs enumerates the identical fold (batched executor path).
    sw3 = mk_silicon_sweep(td / "ev3", {"sem-perf": 90.0, "hand-perf": 100.0})
    jobs_fresh = sw3._silicon_jobs(fresh_row, CLS_SEM_IDENT)
    jobs_plain = sw3._silicon_jobs(plain_row, CLS_SEM_IDENT)
    sem_perf_fresh = [
        j for j in jobs_fresh if j["sel"] == "sem-perf" and j["kind"] == "perf"
    ]
    sem_perf_plain = [
        j for j in jobs_plain if j["sel"] == "sem-perf" and j["kind"] == "perf"
    ]
    check(
        "_silicon_jobs mirrors the eqz rule: fresh row enumerates 3 sem-perf "
        "off-leg jobs, plain row enumerates zero",
        len(sem_perf_fresh) == 3
        and all(
            j["leg"] == "off" and j["flags"] == sweep.OFF_FLAGS for j in sem_perf_fresh
        )
        and not sem_perf_plain,
        (len(sem_perf_fresh), len(sem_perf_plain)),
    )

# ---------------- 3. evidence-root pin guard (sweep level) ----------------

PIN_A = "aa" * 32
PIN_B = "bb" * 32

with tempfile.TemporaryDirectory() as td:
    td = pathlib.Path(td)

    # fresh/empty root: ok; stamp is idempotent and never overwrites
    ev = td / "ev"
    ev.mkdir()
    ok, _ = sweep.check_evidence_root_pin(ev, PIN_A)
    check("pin guard: fresh root passes", ok)
    sweep.stamp_evidence_root_pin(ev, PIN_A)
    sweep.stamp_evidence_root_pin(ev, PIN_B)  # must NOT overwrite
    check(
        "pin guard: stamp written once, never overwritten",
        (ev / "PIN_STAMP").read_text().strip() == PIN_A,
    )
    ok, _ = sweep.check_evidence_root_pin(ev, PIN_A)
    check("pin guard: same-pin root resumes", ok)
    ok, detail = sweep.check_evidence_root_pin(ev, PIN_B)
    check(
        "pin guard: foreign PIN_STAMP refuses, message names both pins",
        not ok
        and "EVIDENCE-ROOT PIN COLLISION" in detail
        and PIN_A in detail
        and PIN_B in detail,
        detail,
    )

    # no stamp, but an existing preflight.json from another pin: refuse
    ev2 = td / "ev2"
    ev2.mkdir()
    (ev2 / "preflight.json").write_text(json.dumps({"cc1plus_sha256": PIN_A}))
    ok, detail = sweep.check_evidence_root_pin(ev2, PIN_B)
    check(
        "pin guard: foreign preflight.json (no stamp) refuses",
        not ok and "preflight.json" in detail,
        detail,
    )
    ok, _ = sweep.check_evidence_root_pin(ev2, PIN_A)
    check("pin guard: matching preflight.json passes", ok)

    # END-TO-END: the real sweep_2x2.py CLI refuses a foreign-pin root at
    # the guard, BEFORE any compiler work (no toolchain exists here).
    fake_venv = td / "venv"
    (fake_venv / "bin").mkdir(parents=True)
    (fake_venv / "bin" / "python").write_text("#!/bin/sh\nexit 0\n")
    ev3 = td / "ev3"
    ev3.mkdir()
    (ev3 / "PIN_STAMP").write_text(PIN_A + "\n")

    def run_sweep(root):
        return subprocess.run(
            [
                sys.executable,
                str(HERE / "sweep_2x2.py"),
                "--evidence-root",
                str(root),
                "--cc1plus-sha",
                PIN_B,
                "--venv",
                str(fake_venv),
                "--compiler",
                str(td / "no-such-compiler"),
                "--ops",
                "eqz-fresh",
            ],
            capture_output=True,
            text=True,
        )

    p = run_sweep(ev3)
    out = p.stdout + p.stderr
    check(
        "END-TO-END: sweep_2x2.py refuses a foreign-pin root (rc!=0, "
        "collision named) BEFORE any compiler work",
        p.returncode != 0
        and "EVIDENCE-ROOT PIN COLLISION" in out
        and "missing compiler" not in out,
        (p.returncode, out[-400:]),
    )
    ev4 = td / "ev4"
    ev4.mkdir()
    (ev4 / "PIN_STAMP").write_text(PIN_B + "\n")
    p = run_sweep(ev4)
    out = p.stdout + p.stderr
    check(
        "END-TO-END: a same-pin root passes the guard (run proceeds to the "
        "next preflight check — the deliberately missing compiler)",
        p.returncode != 0
        and "EVIDENCE-ROOT PIN COLLISION" not in out
        and "missing compiler" in out,
        (p.returncode, out[-400:]),
    )

print()
if FAILS:
    print(f"SELFTEST: {len(FAILS)} FAILURE(S): {FAILS}")
    sys.exit(1)
print("SELFTEST: all green")
