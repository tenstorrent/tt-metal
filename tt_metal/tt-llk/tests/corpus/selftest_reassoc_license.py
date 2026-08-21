#!/usr/bin/env python3
"""Self-test for the LICENSED reassociation knob leg (lane EJ).

Drives the REAL sweep code (imported, not re-implemented) — no
toolchain, no simulator, no device:

  1. KNOB WIRING: the "reassoc" knob exists, is on-plus, and its flag
     string carries the FULL effective license — -fassociative-math,
     the -fno-signed-zeros -fno-trapping-math pair GCC requires for the
     flag to take effect (toplev.cc:1623 clears it otherwise), and
     -mtt-tensix-optimize-reassoc — each exactly once, all OUTSIDE the
     reviewed ON set, so the on-plus knob leg is ON + exactly these
     tokens (the delta reads as the license's own effect and every
     device jobkey differs from any unlicensed cell's by construction).
  2. LICENSED BOOKKEEPING: LICENSED_KNOBS names reassoc (with the owner
     ratification date 2026-08-21 in the citation);
     licensed_craq_disposition implements the licensed CRAQ gate:
     unlicensed knobs keep the historical all-PASS rule BYTE-IDENTICALLY
     (any non-PASS closes the gate, no note); the licensed knob's OFF
     leg must PASS (broken baseline still withholds); a licensed
     knob-leg mismatch keeps the gate OPEN and returns a non-silent
     LICENSED-EXPECTED note naming the device-golden tolerance
     authority; missing/empty legs never open any gate.
  3. FAIL-CLOSED SCHEMA: LICENSED_KNOBS keys must be KNOBS keys (loud
     import error otherwise — checked by construction here since the
     module imported), and the licensed note text names both the
     expectation (bit-exact fail = license working) and the authority
     (device-golden at the row's documented tolerance).

Run standalone or from the sweep wrappers; exits nonzero on any failure.
"""

import importlib.util
import pathlib
import sys

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


# ---- 1. knob wiring ----
check("reassoc knob exists", "reassoc" in sweep.KNOBS)
tokens = sweep.KNOBS["reassoc"].split()
expected = [
    "-fassociative-math",
    "-fno-signed-zeros",
    "-fno-trapping-math",
    "-mtt-tensix-optimize-reassoc",
]
check(
    "reassoc knob carries the full effective license, each token once",
    tokens == expected,
    f"got {tokens}",
)
check("reassoc mode is on-plus", sweep.knob_mode("reassoc") == "on-plus")
on_tokens = sweep.ON_FLAGS.split()
check(
    "license tokens are all OUTSIDE the reviewed ON set",
    all(t not in on_tokens for t in tokens),
)
legs = dict(sweep.knob_legs("reassoc"))
check("reassoc off leg is plain reviewed-ON", legs["off"] == sweep.ON_FLAGS)
check(
    "reassoc knob leg is ON + exactly the license tokens",
    legs["knob"] == sweep.ON_FLAGS + " " + " ".join(expected),
    f"got {legs['knob']}",
)

# ---- 2. licensed bookkeeping ----
check("LICENSED_KNOBS names reassoc", "reassoc" in sweep.LICENSED_KNOBS)
check(
    "licensed citation carries the owner ratification date",
    "2026-08-21" in sweep.LICENSED_KNOBS["reassoc"],
)
check(
    "every licensed knob is a real knob",
    all(k in sweep.KNOBS for k in sweep.LICENSED_KNOBS),
)

d = sweep.licensed_craq_disposition

# Historical rule byte-identical for unlicensed knobs.
check(
    "unlicensed all-PASS opens",
    d("int-abs", {"off": "PASS", "knob": "PASS"}) == (True, None),
)
check(
    "unlicensed knob-leg FAIL closes with no note",
    d("int-abs", {"off": "PASS", "knob": "FAIL(rc=1)"}) == (False, None),
)
check(
    "unlicensed off-leg FAIL closes",
    d("int-abs", {"off": "FAIL(rc=1)", "knob": "PASS"}) == (False, None),
)
check("missing legs never open (unlicensed)", d("int-abs", None) == (False, None))
check("empty legs never open", d("int-abs", {}) == (False, None))

# Licensed knob.
check(
    "licensed all-PASS opens with no note",
    d("reassoc", {"off": "PASS", "knob": "PASS"}) == (True, None),
)
gate, note = d("reassoc", {"off": "PASS", "knob": "FAIL(rc=1)"})
check("licensed knob-leg mismatch keeps the gate OPEN", gate is True)
check("licensed mismatch is never silent", bool(note))
check(
    "licensed note is LICENSED-EXPECTED and names the authority",
    note is not None
    and note.startswith("LICENSED-EXPECTED")
    and "device-golden" in note
    and "tolerance" in note,
    f"got {note!r}",
)
check(
    "licensed OFF-leg failure still closes the gate (baseline authority)",
    d("reassoc", {"off": "FAIL(rc=1)", "knob": "PASS"}) == (False, None),
)
check("missing legs never open (licensed)", d("reassoc", None) == (False, None))

# UNSUPPORTED/SKIPPED on the licensed knob leg is also non-PASS: gate
# stays open with the note (recorded, adjudicated at device-golden).
gate, note = d("reassoc", {"off": "PASS", "knob": "UNSUPPORTED"})
check("licensed knob-leg UNSUPPORTED: open + noted", gate is True and bool(note))

# ---- 3. sign convention / leg naming untouched ----
check(
    "leg names stay off/knob (downstream cells unchanged)",
    [n for n, _ in sweep.knob_legs("reassoc")] == ["off", "knob"],
)

print()
if FAILS:
    print(f"SELFTEST: {len(FAILS)} FAILURE(S): {FAILS}")
    sys.exit(1)
print("SELFTEST: all reassoc-license checks passed")
