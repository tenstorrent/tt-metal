#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Perf: cycles of the original and the generated LLKs with the LLK perf test, LLK asserts off.
# The HAL perf kernel is a copy of the perf test source with every '#include "<header>"' of a header the
# agent wrote (tests/hal_llk/) pointed to "../hal_llk/<header>", the same change the agent makes in its
# HAL test source. Original and HAL alternate REPS times, each pytest run after tt-smi -r.
#
# Usage: perf.sh <ws copy> <out dir> <reps> <perf python test> <-k expression>
#   <ws copy> = hard-link copy of the run's workspace (made by hal-llk-gen; never the run's own ws)
# Output: <out>/{orig,hal}_r<i>/{pytest.txt,perf_data/}, <out>/perf_summary.{json,md}, <out>/SOURCE.txt
set -uo pipefail
CW=${1:?ws copy}
OUT=${2:?out dir}
REPS=${3:?reps}
PERF_PY=${4:?perf python test, e.g. perf_eltwise_binary}
PERF_K=${5-}
TOOL_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CT=$CW/tt_metal/tt-llk/tests
mkdir -p "$OUT"

src=$(grep -oE 'sources/[A-Za-z0-9_]+\.cpp' "$CT/python_tests/$PERF_PY.py" 2>/dev/null | sort -u)
[[ $(echo "$src" | grep -c .) == 1 && -f $CT/$src ]] || { echo "perf: no single perf source in python_tests/$PERF_PY.py"; exit 2; }
stem=$(basename "$src" .cpp)
rm -f "$CT/sources/${stem}_hal.cpp" "$CT/python_tests/${PERF_PY}_hal.py"   # hard links: remove, never write in place
args=()
for h in $(ls "$CT/hal_llk"); do args+=(-e "s#^\#include \"$h\"#\#include \"../hal_llk/$h\"#"); done
(( ${#args[@]} )) || { echo "perf: $CT/hal_llk is empty"; exit 2; }
sed "${args[@]}" "$CT/$src" > "$CT/sources/${stem}_hal.cpp"
sed "s#sources/$stem.cpp#sources/${stem}_hal.cpp#g" "$CT/python_tests/$PERF_PY.py" > "$CT/python_tests/${PERF_PY}_hal.py"
nhal=$(grep -c '^#include "../hal_llk/' "$CT/sources/${stem}_hal.cpp" || true)
norig=$(grep '^#include "llk_' "$CT/sources/${stem}_hal.cpp" | grep -vc '"llk_defs.h"' || true)
echo "perf kernel sources/${stem}_hal.cpp: $nhal includes point to hal_llk/, $norig original llk_*.h includes left (llk_defs.h not counted)" \
  | tee "$OUT/SOURCE.txt"
[[ $norig == 0 && $nhal -gt 0 ]] || { echo "perf: the HAL perf kernel still includes original LLK headers"; exit 2; }

rc_all=0
for i in $(seq 1 "$REPS"); do
  for side in orig hal; do
    lab=${side}_r$i; py=$PERF_PY; [[ $side == hal ]] && py=${PERF_PY}_hal
    mkdir -p "$OUT/$lab" "$OUT/tmp/$lab"
    tt-smi -r > "$OUT/$lab/reset.txt" 2>&1
    kargs=(); [[ -n "$PERF_K" ]] && kargs=(-k "$PERF_K")
    (cd "$CT" && TT_LLK_DISABLE_ASSERTS=1 PERF_RUN_TAG=$lab PERF_KEEP_RUNS=0 TMPDIR=$OUT/tmp/$lab \
       timeout 90m pytest "python_tests/$py.py" ${kargs[@]+"${kargs[@]}"} -q > "$OUT/$lab/pytest.txt" 2>&1)
    rc=$?; (( rc == 0 )) || rc_all=1
    echo "$lab: exit $rc, $(tail -1 "$OUT/$lab/pytest.txt")" | tee -a "$OUT/progress.txt"
    mkdir -p "$OUT/$lab/perf_data"
    cp -r "$CW/tt_metal/tt-llk/perf_data/runs/$lab/." "$OUT/$lab/perf_data/" 2>/dev/null
    rm -rf "$OUT/tmp/$lab"
  done
done
rmdir "$OUT/tmp" 2>/dev/null
python3 "$TOOL_DIR/python/perf_summary.py" "$OUT" --reps "$REPS" --json "$OUT/perf_summary.json" --md "$OUT/perf_summary.md" \
  || { echo "perf: perf_summary.py failed"; exit 1; }
exit $rc_all
