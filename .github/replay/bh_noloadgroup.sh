#!/bin/bash
# BH: is the full suite repeatable without --dist loadgroup (#57015)? Tree = main + stability stack (91077836f3d8).
# One compile, then 4 measuring passes of the full non-SoL suite, L1_TO_L1, 15 workers like CI, interleaved:
# N1 (no loadgroup), L1 (loadgroup), N2, L2. Same code in every pass.
# Usage: bh_noloadgroup.sh <tag>
set -uo pipefail
TAG=$1
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
cd $L/tests
sed 's/ --dist loadgroup / /' run_llk_perf_blackhole.sh > run_llk_perf_blackhole_nolg.sh; chmod +x run_llk_perf_blackhole_nolg.sh
grep -c "loadgroup" run_llk_perf_blackhole.sh run_llk_perf_blackhole_nolg.sh
export SPEED_OF_LIGHT=false TT_LLK_DISABLE_ASSERTS=1 RUNNER_TEMP=$W/tmp-$TAG LLK_PERF_RUN_TYPES=L1_TO_L1 LLK_PERF_EXITFIRST=false
LLK_PERF_PHASE=compile ./run_llk_perf_blackhole.sh 1 1 > $OUT/compile.log 2>&1; echo "compiled $(date +%T) $(tail -n 1 $OUT/compile.log)"
for arm in N1 L1 N2 L2; do
  S=./run_llk_perf_blackhole.sh; [[ $arm == N* ]] && S=./run_llk_perf_blackhole_nolg.sh
  PERF_RUN_TAG=bhlg-$arm LLK_PERF_PHASE=run $S 1 1 > $OUT/$arm.log 2>&1
  echo "$arm $(date +%T) $(grep -E 'passed|failed' $OUT/$arm.log | tail -n 1)"
  mkdir -p $OUT/$arm; cp -r $L/tests/python_tests/perf_data/runs/bhlg-$arm/. $OUT/$arm/ 2>/dev/null || cp -r $L/perf_data/runs/bhlg-$arm/. $OUT/$arm/
done
python3 - $OUT <<'PY'
import glob, sys, pandas as pd
O = sys.argv[1]
def load(d):
    fs = [f for f in glob.glob(f"{O}/{d}/**/*.csv", recursive=True) if not f.endswith(".post.csv")]
    fr = pd.concat([pd.read_csv(f, low_memory=False, dtype=str) for f in fs])
    col = "mean(L1_TO_L1)"
    keys = [c for c in fr.columns if not c.startswith(("mean(", "std(", "min(", "max(", "median(")) and c not in ("run_id", "timestamp")]
    fr[col] = pd.to_numeric(fr[col], errors="coerce")
    return fr.groupby(keys, dropna=False)[col].median()
arms = {a: load(a) for a in ("N1", "L1", "N2", "L2")}
print("points per arm:", {a: len(v) for a, v in arms.items()})
for a, b in (("N1", "N2"), ("L1", "L2"), ("N1", "L1"), ("N2", "L2")):
    x, y = arms[a].align(arms[b], join="inner")
    d = (y - x); p = d / x * 100
    gate = ((p.abs() > 2) & (d.abs() > 30)).sum()
    tl = [i for i in x.index if "TILE_LOOP" in i]
    ptl = p.loc[tl]
    print(f"{a} vs {b}: {len(x)} points, {int((d != 0).sum())} not bit-exact, max {p.abs().max():.2f} %, "
          f"over the gate rule (2 % and 30 cycles) {int(gate)}; TILE_LOOP {len(tl)} points, {int((ptl != 0).sum())} not bit-exact, max {ptl.abs().max():.2f} %")
    top = p.abs().sort_values(ascending=False).head(5)
    for k, v in top.items():
        if v > 0: print(f"    {v:.2f} %  {[s for s in k if isinstance(s, str) and ('perf_' in s or s in ('TILE_LOOP','INIT','UNINIT'))]}")
PY
echo done > "$OUT/status"
