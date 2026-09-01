#!/usr/bin/env python3
"""Self-test for witness_preflight.py (conf R9's compile half).

Drives the REAL gate logic (imported, not re-implemented) with filesystem
fixtures only — no toolchain, no compile:

  1. table parsing: a well-formed table parses; malformed rows (field
     count, non -mtt-tensix- flag, non-rvtt dump flag, exact duplicates)
     are named errors;
  2. ON-set coherence: a witnessed flag outside sweep_2x2.py's reviewed ON
     set is reported stale (config error, never a silent skip);
  3. PRESENT: a fixture dump tree containing the required line -> found;
  4. MISSING: the line absent -> not found, and the RED message names the
     flag and says 'fire witness stale on the union' (the pin-11 lesson);
  5. STALE regex: the required line present but only OUTSIDE the entry's
     own dump pass files -> not found (dump-flag scoping is real);
  6. grouping: entries sharing a pytest node compile once;
  7. the CHECKED-IN conf's active table parses clean, names only ON-set
     flags, and remains within the reviewed fourteen-compile budget.

NOTE: the shipping table deliberately carries a prgm-const row that is RED
on the pin-11 union (lane BT owns the fix) — that RED is the system
working, so THIS selftest proves the mechanism on SYNTHETIC witnesses only
and never compiles anything.

Run by the nightly wrapper with the other gate self-tests; exit 0 green.
"""
import importlib.util
import pathlib
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "witness_preflight", HERE / "witness_preflight.py"
)
wp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wp)

sweep_spec = importlib.util.spec_from_file_location("sweep_2x2", HERE / "sweep_2x2.py")
sweep = importlib.util.module_from_spec(sweep_spec)
sweep_spec.loader.exec_module(sweep)

FAILS = []


def check(name, cond, detail=""):
    if cond:
        print(f"SELFTEST PASS: {name}")
    else:
        print(f"SELFTEST FAIL: {name} {detail}")
        FAILS.append(name)


GOOD_TABLE = """# fixture conf
_REVIEWED_FIRE_WITNESSES="
-mtt-tensix-optimize-ccmask|perf_x.py::node[a]|-fdump-tree-rvtt_ccmask|synthetic: fixture fired
-mtt-tensix-optimize-drain-schedule|perf_x.py::node[a]|-fdump-rtl-rvtt_macro_planner|synthetic drain fired
-mtt-tensix-macro-planner|perf_y.py::node[b]|-fdump-rtl-rvtt_macro_planner|synthetic formed
"
"""

# 1. parsing: well-formed
rows, errors = wp.parse_witness_table(GOOD_TABLE)
check(
    "well-formed table parses without errors", rows is not None and not errors, errors
)
check("well-formed table yields 3 rows", len(rows) == 3)

# malformed variants
for name, bad in (
    ("3-field row", "-mtt-tensix-optimize-ccmask|n|-fdump-tree-rvtt_ccmask\n"),
    ("empty field", "-mtt-tensix-optimize-ccmask|n||line\n"),
    ("non-tensix flag", "-march=rv32i|n|-fdump-tree-rvtt_ccmask|line\n"),
    (
        "non-rvtt dump flag",
        "-mtt-tensix-optimize-ccmask|n|-fdump-tree-optimized|line\n",
    ),
):
    text = '_REVIEWED_FIRE_WITNESSES="\n' + bad + '"\n'
    _, errs = wp.parse_witness_table(text)
    check(f"malformed table refused: {name}", bool(errs), errs)

dup = (
    '_REVIEWED_FIRE_WITNESSES="\n'
    "-mtt-tensix-optimize-ccmask|n|-fdump-tree-rvtt_ccmask|line\n"
    "-mtt-tensix-optimize-ccmask|n|-fdump-tree-rvtt_ccmask|line\n"
    '"\n'
)
_, errs = wp.parse_witness_table(dup)
check("exact duplicate row refused", any("duplicate" in e for e in errs), errs)

_, errs = wp.parse_witness_table("# no table at all\n")
check("missing table refused", bool(errs), errs)

# 2. ON-set coherence
rows, _ = wp.parse_witness_table(GOOD_TABLE)
stale = wp.check_on_set(rows, sweep.ON_FLAGS)
check("fixture flags (real ON-set members) are coherent", stale == [], stale)
rows_bad, _ = wp.parse_witness_table(
    '_REVIEWED_FIRE_WITNESSES="\n'
    "-mtt-tensix-optimize-not-a-real-flag|n|-fdump-tree-rvtt_ccmask|line\n"
    '"\n'
)
stale = wp.check_on_set(rows_bad, sweep.ON_FLAGS)
check(
    "non-ON-set witnessed flag reported stale",
    stale == ["-mtt-tensix-optimize-not-a-real-flag"],
    stale,
)

# 3-5. dump scanning on fixture trees
with tempfile.TemporaryDirectory() as td:
    td = pathlib.Path(td)
    build = td / "tt-llk-build/sources/x.cpp/aa/elf"
    build.mkdir(parents=True)
    (build / "math.elf--.100t.rvtt_ccmask").write_text(
        "noise\nsynthetic: fixture fired here\nmore\n"
    )
    (build / "math.elf--.200r.rvtt_macro_planner").write_text(
        "planner noise\nno drain line here\n"
    )
    # 5. the ccmask line ALSO present in the WRONG pass's dump — must not
    # satisfy the macro_planner-scoped entry.
    (build / "math.elf--.100t.rvtt_ccmask").write_text(
        "synthetic: fixture fired here\nsynthetic drain fired\n"
    )
    verdicts = wp.scan_dumps(td / "tt-llk-build", rows)
    by_flag = {v["flag"]: v for v in verdicts}
    check(
        "PRESENT: required line found in its pass's dump",
        by_flag["-mtt-tensix-optimize-ccmask"]["found"],
    )
    check(
        "MISSING: absent line not found (drain entry)",
        not by_flag["-mtt-tensix-optimize-drain-schedule"]["found"],
    )
    check(
        "dump-flag scoping: a line in another pass's dump does not satisfy",
        not by_flag["-mtt-tensix-optimize-drain-schedule"]["found"]
        and by_flag["-mtt-tensix-optimize-drain-schedule"]["dump_files_scanned"] == 1,
    )
    msg = wp.stale_message(by_flag["-mtt-tensix-optimize-drain-schedule"])
    check(
        "RED message names the flag and 'fire witness stale on the union'",
        "fire witness stale on the union: -mtt-tensix-optimize-drain-schedule" in msg
        and "MISSING" in msg,
        msg,
    )

# 6. grouping
groups = wp.group_rows(rows)
check(
    "entries sharing a node compile once",
    len(groups) == 2
    and sorted(len(g["rows"]) for g in groups) == [1, 2]
    and sorted(groups[0]["dump_flags"] + groups[1]["dump_flags"]).count(
        "-fdump-rtl-rvtt_macro_planner"
    )
    == 2,
    groups,
)

# 7. the CHECKED-IN table must satisfy its own structural gate
conf_rows, conf_errors = wp.parse_witness_table((HERE / "sweep_2x2.conf").read_text())
check(
    "checked-in conf table parses clean",
    conf_rows is not None and not conf_errors,
    conf_errors,
)
if conf_rows:
    stale = wp.check_on_set(conf_rows, sweep.ON_FLAGS)
    check("checked-in table names only reviewed ON-set flags", stale == [], stale)
    n_groups = len(wp.group_rows(conf_rows))
    # Budget refreshed 5 -> 8 at the ON-28 promotion (tt-metal 461ed6c796,
    # 2026-08-23): +window-pairing +replay-record-hoist +lreg-alloc each
    # added its own R9 union fire witness (one compile group apiece).  The
    # stale 5 made this selftest RED at every canon tip since the
    # promotion (lane GE noted it pre-existing; lane GF refreshed).  Keep
    # Keep the ratified GF budget.  A future promotion that adds a witness
    # group must update this budget in the same reviewed commit.
    # Budget refreshed 8 -> 14 at the knob-promotion-round-2 ceremony
    # (lane HE, 2026-08-26): +window-pairing-stride +crossrow-pairing
    # +record-hoist-peel +lut-select-fp16 +native-compare +pressure-park
    # each added its own R9 union fire witness (one compile group apiece;
    # all 18 rows / 14 groups verified ALL GREEN on the installed pin-29
    # binary in the same ceremony).
    # Budget refreshed 14 -> 16 at the knob-promotion-round-3 ceremony
    # (lane HQ, 2026-08-26): +park-ordering (softplus node) and
    # +store-source-tier (fill node) each added its own R9 union fire
    # witness compile group; full table verified ALL GREEN on the
    # installed pin-32 binary in the same ceremony.
    # Budget refreshed 16 -> 17 at the knob-promotion-round-4 ceremony
    # (lane KM, 2026-09-01).  The 17th compile group PREDATES round 4:
    # the lane-IZ dst-ownership witness (erfinv fp32 corr node) landed
    # without this bump, so the selftest was RED at every canon tip
    # since (pre-existing at pin 51, found and fixed here).  Round 4's
    # own +priced-placement row SHARES that erfinv node (one compile,
    # two dump flags), so the group count stays 17.
    check(
        f"checked-in table stays within the 17-compile budget ({n_groups})",
        1 <= n_groups <= 17,
    )

if FAILS:
    print(f"witness-preflight self-test: FAILED ({len(FAILS)}: {', '.join(FAILS)})")
    sys.exit(1)
print(
    "witness-preflight self-test: ALL GREEN (parse/refuse, ON-set coherence, "
    "present/missing/scoped-stale on synthetic witnesses, node grouping, "
    "checked-in table structurally clean)"
)
