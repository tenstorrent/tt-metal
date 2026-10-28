#!/bin/bash
# BH: does a board reset between tests change the measured numbers? (Stefan's SoL observation: after a reset
# a test always landed in the fast state.) Tree = today's main + stability stack + stage-2 harness.
# Usage: bh_reset.sh <tag> <sol true|false> <nodes per module>
# Arms: N1/N2 = one serialized session (no reset), twice; R1/R2 = one process per test with tt-smi -r before each, twice.
set -uo pipefail
TAG=$1 SOL=$2 PER=${3:-40}
W=${REPLAY_ROOT:-/tmp/replay}; REPO=$W/repo; OUT=$W/out/$TAG; mkdir -p "$OUT"
exec > >(tee -a "$OUT/log.txt") 2>&1
echo "running" > "$OUT/status"
[[ -d $REPO ]] || git clone -q --filter=blob:none --no-checkout https://github.com/tenstorrent/tt-metal "$REPO"
T=$W/trees/91077836f3d8
if [[ ! -f $T/.ready ]]; then
  git -C "$REPO" fetch -q origin "perf-trial/replay/91077836f3d8:refs/heads/replay-91077836f3d8"
  git -C "$REPO" worktree add -q --no-checkout "$T" replay-91077836f3d8
  git -C "$T" sparse-checkout set --no-cone /tt_metal/tt-llk/ /tt_metal/hw/ /tt_metal/hostdevcommon/ /ttnn/cpp/ttnn/operations/experimental/ /tt_metal/sfpi-info.sh /tt_metal/sfpi-version
  git -C "$T" checkout -q && (cd "$T/tt_metal/tt-llk/tests" && ./setup_testing_env.sh > /dev/null) && touch "$T/.ready"
fi
L=$T/tt_metal/tt-llk
cd $L/tests/python_tests
export TT_LLK_DISABLE_ASSERTS=1 RUNNER_TEMP=$W/tmp-$TAG LLK_PERF_RUN_TYPES=L1_TO_L1
SOLARG=(); [[ $SOL == true ]] && SOLARG=(--speed-of-light)
M=(perf_math_matmul.py perf_eltwise_binary.py perf_eltwise_unary_sfpu.py)
B=(-q --override-ini=log_cli=false -m "perf and not accuracy" --timeout=120 "${SOLARG[@]}")
pytest "${B[@]}" --collect-only -n 0 "${M[@]}" 2>/dev/null | grep "::" > $OUT/all.txt
python3 - $OUT $PER <<'PY'
import sys, collections
out, per = sys.argv[1], int(sys.argv[2])
nodes = [l.strip() for l in open(f"{out}/all.txt")]
by = collections.defaultdict(list)
for n in nodes: by[n.split("::")[0]].append(n)
pick = []
for m, ns in by.items():
    step = max(1, len(ns) // per); pick += ns[::step][:per]
open(f"{out}/nodes.txt", "w").write("\n".join(pick) + "\n")
print(len(pick), "nodes")
PY
mapfile -t NODES < $OUT/nodes.txt
pytest "${B[@]}" --compile-producer -n 10 "${NODES[@]}" > $OUT/compile.log 2>&1; echo "compiled $(tail -n 1 $OUT/compile.log)"
for arm in N1 N2; do
  PERF_RUN_TAG=bhr-$arm pytest "${B[@]}" --compile-consumer -n 0 "${NODES[@]}" > $OUT/$arm.log 2>&1
  cp -r $L/perf_data/runs/bhr-$arm $OUT/; echo "$arm $(date +%T) $(tail -n 1 $OUT/$arm.log)"
done
for arm in R1 R2; do
  mkdir -p $OUT/bhr-$arm; i=0
  for n in "${NODES[@]}"; do
    i=$((i+1)); tt-smi -r > $OUT/reset.log 2>&1 || echo "reset failed at $i"
    PERF_RUN_TAG=bhr-$arm-$i pytest "${B[@]}" --compile-consumer -n 0 "$n" >> $OUT/$arm.log 2>&1
    cp -r $L/perf_data/runs/bhr-$arm-$i $OUT/bhr-$arm/ 2>/dev/null; rm -rf $L/perf_data/runs/bhr-$arm-$i
  done
  echo "$arm $(date +%T) $i tests"
done
python3 - $OUT <<'PY'
import glob, sys, pandas as pd
O = sys.argv[1]
def load(d):
    fs = [f for f in glob.glob(f"{O}/{d}/**/*.csv", recursive=True) if not f.endswith(".post.csv")]
    fr = pd.concat([pd.read_csv(f, low_memory=False, dtype=str) for f in fs])
    fr = fr[fr["marker"] == "TILE_LOOP"]
    col = [c for c in fr.columns if c.startswith("mean(")][0]
    keys = [c for c in fr.columns if c != "marker" and not c.startswith(("mean(", "std(", "min(", "max(", "median("))]
    fr = fr.drop_duplicates(keys)
    return fr.set_index(keys)[col].astype(float)
arms = {a: load(f"bhr-{a}") for a in ("N1", "N2", "R1", "R2")}
for a, b in (("N1", "N2"), ("R1", "R2"), ("N1", "R1"), ("N2", "R2")):
    x, y = arms[a].align(arms[b], join="inner")
    d = (y - x) / x * 100
    print(f"{a} vs {b}: {len(d)} points, {(d != 0).sum()} not bit-exact, max {d.abs().max():.2f} %, "
          f"{b} faster in {(d < 0).sum()}, slower in {(d > 0).sum()}, median {d.median():+.3f} %")
PY
echo done > "$OUT/status"
