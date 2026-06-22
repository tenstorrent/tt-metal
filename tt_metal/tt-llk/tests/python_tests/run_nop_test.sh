#!/usr/bin/env bash
# Usage:   ./run_nop_test.sh [OUTDIR]        # long, so run detached if you wish:
#          nohup ./run_nop_test.sh /path/out >/path/out/run.log 2>&1 &
# Env:     N     variants sampled per test file   (default 10)
#          MAXV  variants campaigned per kernel   (default 5)
#          TTNOP path to the ttnop binary         (default below)
#
# Reset device on TENSIX TIMED OUT.
set -uo pipefail

PT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"  # the python_tests dir
cd "$PT"

OUT="${1:-$PT/nop_results}"
N="${N:-10}"
MAXV="${MAXV:-5}"
export TTNOP="${TTNOP:-/localdev/iklikovac/ttnop/ttnop}"  # inherited by pytest
mkdir -p "$OUT"

echo ">> building nodeid list ($N variants/file)..."
python build_nodeids.py "$N" \
    | grep -v test_device_print > "$OUT/nodeids.txt"
count=$(wc -l < "$OUT/nodeids.txt")
echo ">> $count nodeids"
[ "$count" -gt 0 ] || { echo "!! no nodeids collected; aborting"; exit 1; }

# Run one mode. Trailing `|| true` because baseline-fail variants make pytest exit non-zero.
run_mode() {   # $1 = label (-> out_<label>/, <label>.log); rest = mode env assignments
    local label="$1"; shift
    echo ">> mode: $label"
    LLK_NODEIDS="$OUT/nodeids.txt" LLK_PT="$PT" LLK_NOP=1 \
        LLK_MAX_VARIANTS="$MAXV" "$@" LLK_OUTDIR="$OUT/out_$label" \
        python runner.py > "$OUT/$label.log" 2>&1 || true
}

run_mode single    LLK_PLAN_MODE=single    LLK_NOPS=2000 LLK_CLASSES=store,load
run_mode magnitude LLK_PLAN_MODE=magnitude LLK_LOOP=1 LLK_NSET=200,2000,20000 \
                   LLK_CLASSES=store,load,op,other
run_mode cross     LLK_PLAN_MODE=cross     LLK_LOOP=1 LLK_NSET=4,8,64 LLK_CLASSES=store,load
run_mode skew      LLK_PLAN_MODE=skew      LLK_LOOP=1 LLK_CLASSES=store,load \
                   LLK_SKEW="500:0,0:500,500:50,50:500,2000:200"

echo
echo "===== RESULTS ====="
for d in "$OUT"/out_*; do
    [ -d "$d" ] || continue
    echo "== $(basename "$d") =="
    grep -rhE "FAIL-MISMATCH|FAIL-TIMEOUT" "$d"/*.txt 2>/dev/null
    grep -rh "# SUMMARY" "$d"/*.txt 2>/dev/null | awk '{
        for (i=1;i<=NF;i++){
            if ($i~/^pass=/)     p+=substr($i,6)
            if ($i~/mismatch=/)  m+=substr($i,10)
            if ($i~/timeout=/)   t+=substr($i,9)
            if ($i~/err=/)       e+=substr($i,5)
        }
    } END { print "  pass="p" mismatch="m" timeout="t" err="e }'
done
