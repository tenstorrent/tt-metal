#!/bin/bash
# Perf counters on vs off, same code (main + stability stack, 91077836f3d8). Full non-SoL suite, L1_TO_L1, 15 workers.
# Two builds (with and without --dump-perf-counters, i.e. -DPERF_COUNTERS_COMPILED), then 4 measuring passes:
# F1 (off), C1 (on), F2 (off), C2 (on). F1 vs F2 is the noise floor; off vs on is the answer.
# Usage: counters.sh <tag> <wormhole|blackhole>
set -uo pipefail
TAG=$1 ARCH=$2
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
export TT_LLK_DISABLE_ASSERTS=1 LLK_PERF_RUN_TYPES=L1_TO_L1 CHIP_ARCH=$ARCH
COMMON=(-q --override-ini=log_cli=false -m "perf and not accuracy" --timeout=60 --perf-splits 1 --perf-group 1)
for b in off on; do
  F=(); [[ $b == on ]] && F=(--dump-perf-counters)
  RUNNER_TEMP=$W/tmp-$b pytest "${COMMON[@]}" "${F[@]}" --compile-producer -n 10 . > $OUT/compile-$b.log 2>&1
  echo "compiled $b $(date +%T) $(grep -E 'passed|failed' $OUT/compile-$b.log | tail -n 1)"
done
for arm in F1 C1 F2 C2; do
  b=off; F=(); [[ $arm == C* ]] && b=on && F=(--dump-perf-counters)
  RUNNER_TEMP=$W/tmp-$b PERF_RUN_TAG=cnt-$arm pytest "${COMMON[@]}" "${F[@]}" --compile-consumer --dist loadgroup -n 15 . > $OUT/$arm.log 2>&1
  echo "$arm $(date +%T) $(grep -E 'passed|failed' $OUT/$arm.log | tail -n 1)"
  mkdir -p $OUT/$arm; cp -r $L/perf_data/runs/cnt-$arm/. $OUT/$arm/ 2>/dev/null || cp -r $L/tests/python_tests/perf_data/runs/cnt-$arm/. $OUT/$arm/
done
python3 - $OUT <<'PY'
import glob, sys, pandas as pd
O = sys.argv[1]
def load(d):
    fs = [f for f in glob.glob(f"{O}/{d}/**/*.csv", recursive=True) if not f.endswith(".post.csv") and "counter" not in f.lower()]
    return pd.concat([pd.read_csv(f, low_memory=False, dtype=str) for f in fs])
raw = {a: load(a) for a in ("F1", "C1", "F2", "C2")}
col = "mean(L1_TO_L1)"
skip = ("mean(", "std(", "min(", "max(", "median(")
keys = sorted(set.intersection(*[{c for c in f.columns if not c.startswith(skip) and c not in ("run_id", "timestamp")} for f in raw.values()]))
arms = {}
for a, f in raw.items():
    f[col] = pd.to_numeric(f[col], errors="coerce")
    arms[a] = f.groupby(keys, dropna=False)[col].median()
print("points per arm:", {a: len(v) for a, v in arms.items()})
mi = keys.index("marker") if "marker" in keys else None
for a, b in (("F1", "F2"), ("C1", "C2"), ("F1", "C1"), ("F2", "C2")):
    x, y = arms[a].align(arms[b], join="inner")
    d = y - x; p = d / x * 100
    gate = ((p.abs() > 2) & (d.abs() > 30)).sum()
    tl = [i for i in x.index if mi is not None and i[mi] == "TILE_LOOP"]
    pt = p.loc[tl]
    print(f"{a} vs {b}: {len(x)} points, {int((d != 0).sum())} not bit-exact, max {p.abs().max():.3f} %, median {p.median():+.4f} %, "
          f"over the gate rule {int(gate)}; TILE_LOOP {len(tl)}: {int((pt != 0).sum())} not bit-exact, max {pt.abs().max():.3f} %, median {pt.median():+.4f} %")
    for k, v in p.abs().sort_values(ascending=False).head(3).items():
        if v > 0: print(f"    {v:.3f} %  {[s for s in k if isinstance(s, str) and s.startswith('perf_')]} {k[mi] if mi is not None else ''}")
PY
echo done > "$OUT/status"
