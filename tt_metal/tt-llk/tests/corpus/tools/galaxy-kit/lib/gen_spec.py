#!/usr/bin/env python3
"""galaxy-kit spec generator: derive the benchmark row/arm specs from the
sweep ops TSV + FINAL-BOARD, and the compile-producer groups.

Writes into --work:
  ROWS.tsv   one line per (op, leg): marker/metric/per_tile + board class
  ARMS.tsv   one line per (op, leg, arm): flagskey, env, corr node, perf node
  flags/<key>.txt          exact TT_LLK_EXTRA_COMPILER_OPTIONS per key
  producer-groups.tsv      (group, flagskey, env, node count)
  producer-nodes/<group>.txt

Default scope = every FINAL-BOARD WIN/PARITY/LOSS op + every extra blaze
row with a raced hand arm + the trigonometry-fresh booked licensed legs.
--ops restricts to a comma list / @file of op names (legs kept).
"""
import argparse
import ast
import csv
import pathlib
import re

ap = argparse.ArgumentParser()
ap.add_argument("--farm", required=True, help="tt-llk dir (pin checkout)")
ap.add_argument("--board", required=True, help="FINAL-BOARD.tsv path")
ap.add_argument("--work", required=True, help="output workdir")
ap.add_argument("--ops", default="", help="comma list or @file filter")
ap.add_argument("--no-trig-special", action="store_true")
a = ap.parse_args()

FARM = pathlib.Path(a.farm)
WORK = pathlib.Path(a.work)
WORK.mkdir(parents=True, exist_ok=True)
OPS = FARM / "tests/corpus/sweep_2x2_ops.tsv"
SWEEP = FARM / "tests/corpus/sweep_2x2.py"

# ---- ON flag set parsed from the sweep itself (never hand-copied) ----
ON = None
for node in ast.parse(SWEEP.read_text()).body:
    if isinstance(node, ast.Assign) and any(
        getattr(t, "id", None) == "ON_FLAGS" for t in node.targets
    ):
        ON = " ".join(ast.literal_eval(node.value).split())
assert ON, "ON_FLAGS parse failed"
print(f"ON set: {len(ON.split())} flags")

# booked trig legs (pin55-trig-remeasure-20260902; sem licensed vs hand ctl)
TRIG_COMMON = (
    " -mtt-tensix-optimize-loop-prgm-reclaim"
    " -mtt-tensix-optimize-stochrnd-store-fold"
    " -mtt-tensix-optimize-cyclic-region-schedule"
    " -mtt-tensix-optimize-rename-temporal"
)
TRIG_SEM = (
    ON
    + TRIG_COMMON
    + (
        " -fassociative-math -fno-signed-zeros -fno-trapping-math"
        " -mtt-tensix-optimize-reassoc-mad-restructure"
    )
)
TRIG_HAND = ON + TRIG_COMMON

rows = list(
    csv.DictReader((l for l in OPS.open() if not l.startswith("#")), delimiter="\t")
)
ops_by_name = {r["op"]: r for r in rows}

board = {}
with pathlib.Path(a.board).open() as f:
    next(f)  # leading comment line
    for r in csv.DictReader(f, delimiter="\t"):
        board[r["op"]] = r["class"]

only = None
if a.ops:
    src = a.ops
    if src.startswith("@"):
        src = pathlib.Path(src[1:]).read_text()
    only = {t.strip() for t in re.split(r"[\s,]+", src) if t.strip()}

FLAGS = {"on": ON, "trigsem": TRIG_SEM, "trighand": TRIG_HAND}


def flags_key_for(row):
    pf = (row.get("pin_flags") or "").strip()
    if pf:
        key = "pin-" + re.sub(r"[^a-z0-9]+", "", row["op"])[:16]
        FLAGS[key] = pf
        return key
    return "on"


def env_of(row, arm):
    parts = []
    for col in ("extra_env", f"{arm}_extra_env"):
        v = (row.get(col) or "").strip()
        if v:
            parts.append(v)
    return ";".join(parts)


targets, seen = [], set()
for r in rows:
    if r["kind"] == "skip":
        continue
    cls = board.get(r["op"], "NOT-ON-BOARD")
    if cls in ("WIN", "PARITY", "LOSS"):
        fk = flags_key_for(r)
        targets.append((r["op"], "plain", "board", r, fk, fk, cls))
        seen.add(r["op"])
for r in rows:
    if r["kind"] == "skip" or r["op"] in seen:
        continue
    if "blaze" in r["op"] and r["sem_perf"] and r["hand_perf"]:
        fk = flags_key_for(r)
        targets.append(
            (r["op"], "plain", "blaze", r, fk, fk, board.get(r["op"], "NOT-ON-BOARD"))
        )
if not a.no_trig_special and "trigonometry-fresh" in ops_by_name:
    tr = ops_by_name["trigonometry-fresh"]
    targets.append(
        ("trigonometry-fresh", "booked", "special", tr, "trigsem", "trighand", "WIN")
    )
if only is not None:
    targets = [t for t in targets if t[0] in only]
    missing = only - {t[0] for t in targets}
    if missing:
        print(f"WARNING: --ops names with no target row: {sorted(missing)}")

(WORK / "flags").mkdir(exist_ok=True)
for k, v in FLAGS.items():
    (WORK / "flags" / f"{k}.txt").write_text(v + "\n")

with (WORK / "ROWS.tsv").open("w") as w:
    w.write(
        "op\tleg\tset\tboard_class\tkind\tmarker\tmetric\tper_tile\t"
        "sem_flagskey\thand_flagskey\tsem_env\thand_env\n"
    )
    for op, leg, s, r, fks, fkh, cls in targets:
        w.write(
            f"{op}\t{leg}\t{s}\t{cls}\t{r['kind']}\t{r['marker']}\t"
            f"{r['metric']}\t{r['per_tile']}\t{fks}\t{fkh}\t"
            f"{env_of(r,'sem')}\t{env_of(r,'hand')}\n"
        )

groups = {}
with (WORK / "ARMS.tsv").open("w") as w:
    w.write("op\tleg\tarm\tflagskey\tenv\tcorr_node\tperf_node\n")
    for op, leg, s, r, fks, fkh, cls in targets:
        for arm, fk in (("sem", fks), ("hand", fkh)):
            corr, perf = r[f"{arm}_corr"], r[f"{arm}_perf"]
            if not perf:
                continue
            env = env_of(r, arm)
            w.write(f"{op}\t{leg}\t{arm}\t{fk}\t{env}\t{corr}\t{perf}\n")
            g = groups.setdefault((fk, env), set())
            if corr:
                g.add(corr)
            g.add(perf)

(WORK / "producer-nodes").mkdir(exist_ok=True)
with (WORK / "producer-groups.tsv").open("w") as w:
    w.write("group\tflagskey\tenv\tnodes\n")
    for (fk, env), nodes in sorted(groups.items()):
        gname = fk + ("" if not env else "+" + re.sub(r"[^A-Za-z0-9]+", "_", env)[:30])
        (WORK / "producer-nodes" / f"{gname}.txt").write_text(
            "\n".join(sorted(nodes)) + "\n"
        )
        w.write(f"{gname}\t{fk}\t{env}\t{len(nodes)}\n")

print(
    f"targets: {len(targets)} rows "
    f"(board {sum(1 for t in targets if t[2]=='board')}, "
    f"blaze {sum(1 for t in targets if t[2]=='blaze')}, "
    f"special {sum(1 for t in targets if t[2]=='special')}); "
    f"{len(groups)} producer groups"
)
