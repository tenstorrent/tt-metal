#!/usr/bin/env python3
"""Self-test for the sweep enforcement-layer gates (ledger item 10).

Drives the REAL gate logic in sweep_2x2.py (imported, not re-implemented),
with filesystem fixtures only — no toolchain, no simulator, no device:

  REVIEW_RECORD gate (check_review_record):
    1. missing record            -> refuse
    2. wrong-pin record          -> refuse (pin-match: full sha quoted)
    3. malformed record          -> refuse (reviewer/sections required)
    4. valid record              -> pass
  Macro-launch classification (scan_disasm_text / classify_macro_launch):
    5. SFPLOADMACRO in ON leg           -> MACRO_LAUNCH
    6. pure ttreplay launches ON > OFF  -> REPLAY_LAUNCH
    7. record-form replays (…,1,1 / …,0,1) and balanced launches -> None
    8. single-leg (pinpair) scan uses only the SFPLOADMACRO criterion
  issue_slot_lb requirement (macro_lb_red):
    9.  macro-launch row + empty lb -> RED message naming the row and §1
    10. macro-launch row + lb set   -> no message
    11. non-macro row + empty lb    -> no message

Run by both sweep wrappers before anything; exit 0 all-green, 1 otherwise.
"""
import importlib.util
import pathlib
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("sweep_2x2", HERE / "sweep_2x2.py")
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)

PIN = "7e" * 32  # any full 64-hex value
FAILS = []


def check(name, cond, detail=""):
    if cond:
        print(f"SELFTEST PASS: {name}")
    else:
        print(f"SELFTEST FAIL: {name} {detail}")
        FAILS.append(name)


VALID_RECORD = f"""# REVIEW_RECORD fixture
Pin: cc1plus sha256 `{PIN}`
Reviewer: fixture reviewer (independent: no)

## Reviewed commits/branches
- repo abc123 — fixture

## Gates checked
- [x] fixture gate
"""

with tempfile.TemporaryDirectory() as td:
    td = pathlib.Path(td)

    # 1. missing record
    ok, detail = sweep.check_review_record(td / "REVIEW_RECORD-missing.md", PIN)
    check("missing review record refuses", not ok and "missing" in detail)

    # 2. wrong pin (full sha not quoted)
    p = td / "REVIEW_RECORD-wrongpin.md"
    p.write_text(VALID_RECORD.replace(PIN, "ab" * 32))
    ok, detail = sweep.check_review_record(p, PIN)
    check("wrong-pin record refuses (pin-match)", not ok and "pin-match" in detail)

    # 3. malformed: no reviewer / no sections
    p = td / "REVIEW_RECORD-malformed.md"
    p.write_text(f"pin {PIN}\nnothing else\n")
    ok, detail = sweep.check_review_record(p, PIN)
    check(
        "malformed record refuses (reviewer+sections required)",
        not ok and "Reviewer" in detail and "Gates" in detail,
    )

    # 4. valid record passes
    p = td / "REVIEW_RECORD-ok.md"
    p.write_text(VALID_RECORD)
    ok, detail = sweep.check_review_record(p, PIN)
    check("valid record passes", ok, detail)

    # THE CHECKED-IN record for the CURRENT pin must satisfy the real gate
    # against the real reviewed pin value from sweep_2x2.conf.
    conf = (HERE / "sweep_2x2.conf").read_text()
    import re as _re

    m = _re.search(r"^_REVIEWED_CC1PLUS_SHA256=([0-9a-f]{64})$", conf, _re.M)
    check("conf carries a full cc1plus pin", bool(m))
    if m:
        cur = m.group(1)
        rec = HERE / "review_records" / f"REVIEW_RECORD-{cur[:12]}.md"
        ok, detail = sweep.check_review_record(rec, cur)
        check(f"checked-in record for pin {cur[:12]} passes the real gate", ok, detail)

# 5-7. macro-launch classification on fixture disassembly text
ON_MACRO = "  b324:\t4c038002\tsfploadmacro\t0,L0,0,0,7\n  b364:\t3c000002\tsfpnop\n"
ON_REPLAY = (
    "  b89c:\t10000100\tttreplay\t0,4,0,0\n" * 3
    + "  b87c:\t10000104\tttreplay\t0,4,0,1\n"
)
OFF_PLAIN = (
    "  b2e0:\t1000010c\tttreplay\t0,4,1,1\n  b2e8:\tc0038001\tsfpload\tL0,0,0,7\n"
)

on = sweep.scan_disasm_text(ON_MACRO)
check(
    "SFPLOADMACRO detected -> MACRO_LAUNCH",
    sweep.classify_macro_launch(on, sweep.scan_disasm_text(OFF_PLAIN))
    == "MACRO_LAUNCH",
)
on = sweep.scan_disasm_text(ON_REPLAY)
off = sweep.scan_disasm_text(OFF_PLAIN)
check(
    "ON-only pure replay launches -> REPLAY_LAUNCH",
    on["replay_launch"] == 3
    and sweep.classify_macro_launch(on, off) == "REPLAY_LAUNCH",
)
check(
    "record-form replays and balanced legs -> None",
    sweep.classify_macro_launch(off, off) is None
    and off["replay_launch"] == 0,  # …,1,1 is the record path, not a launch
)
check(
    "single-leg scan: replay launches alone never classify (SFPLOADMACRO only)",
    sweep.classify_macro_launch(on, None) is None
    and sweep.classify_macro_launch(sweep.scan_disasm_text(ON_MACRO), None)
    == "MACRO_LAUNCH",
)

# 9-11. issue_slot_lb requirement
scan = {"classification": "MACRO_LAUNCH", "sfploadmacro_on": 64, "replay_launch_on": 0}
msg = sweep.macro_lb_red("fixture-op", "TILE_LOOP", None, scan)
check(
    "macro-launch row with EMPTY lb is RED, names the row and the §1 caveat",
    bool(msg) and "fixture-op" in msg and "EMPTY issue_slot_lb" in msg and "§1" in msg,
    repr(msg),
)
check(
    "macro-launch row with lb set passes",
    sweep.macro_lb_red("fixture-op", "TILE_LOOP", 10.0, scan) is None,
)
check(
    "non-macro row with empty lb passes",
    sweep.macro_lb_red("fixture-op", "TILE_LOOP", None, {"classification": None})
    is None
    and sweep.macro_lb_red("fixture-op", "TILE_LOOP", None, None) is None,
)

# The RED must actually flow into a sweep: drive Sweep._macro_lb_gate on a
# minimal instance (no argparse / preflight needed for this method).
inst = object.__new__(sweep.Sweep)
inst.reds = []
row = {"op": "fixture-op", "marker": "TILE_LOOP", "issue_slot_lb": None}
cls = {"sem-perf": {"macro_scan": scan}}
result = {"notes": []}
inst._macro_lb_gate(row, cls, result)
check(
    "Sweep._macro_lb_gate appends the RED and the report note",
    len(inst.reds) == 1 and len(result["notes"]) == 1,
)

if FAILS:
    print(f"enforcement-gates self-test: FAILED ({len(FAILS)}: {', '.join(FAILS)})")
    sys.exit(1)
print(
    "enforcement-gates self-test: ALL GREEN (review-record missing/wrong-pin/"
    "malformed refuse + valid/checked-in pass; macro/replay-launch classify; "
    "empty-lb RED wired through Sweep)"
)
