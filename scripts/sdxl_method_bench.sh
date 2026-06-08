#!/bin/bash
# E2E test-method benchmark on a single SDXL model layer (resnetblock2d, one case), controlling for ccache.
# Question: how does test wall-time depend on METHOD (cold vs warm JIT reuse) and ccache state?
#
# Cells (each = one full `pytest` invocation wall, plain — no tracy; wall includes python+ttnn import+device open):
#   cold_ccoff   : empty JIT cache, ccache DISABLED (TT_METAL_CCACHE_KERNEL_SUPPORT unset) -> true full serial compile
#   cold_ccempty : empty JIT cache, ccache ENABLED but a FRESH private CCACHE_DIR (wiped each pass) -> full compile + populate
#   cold_ccwarm  : empty JIT cache, ccache ENABLED + a PRE-WARMED private CCACHE_DIR -> compile served by ccache
#   warm         : PRE-POPULATED JIT cache (ship-the-cache steady state) -> zero compile (ccache irrelevant)
#
# warmup+warm (in-job two-pass) is DERIVED = cold(ccstate) + warm  (for SDXL the hardware-free parallel meta-collect
# is blind to the C++-migrated ops, so the only "warmup" is an on-device compile pass == a cold run).
#
# ccache is isolated to private CCACHE_DIRs under /tmp so the shared ~/.cache/ccache (15GB) is never touched.
# Passes are interleaved with a per-pass shuffled cell order to control for host/thermal drift.
set -uo pipefail
WT=/localdev/mstaletovic/2026_05_28/0104_mstaletovic_agent_eval/wt_origin_main
cd "$WT"; export PYTHONPATH="$WT"
TEST="models/demos/stable_diffusion_xl_base/tests/pcc/test_module_tt_resnetblock2d.py"
SEL="image_resolution0-"
N="${1:-4}"                      # passes per cell
RES=/tmp/sdxl_method_results.txt
JIT_COLD=/tmp/jit_cold; JIT_WARM=/tmp/jit_warm
CC_WARM=/tmp/bench_cc_warm; CC_EMPTY=/tmp/bench_cc_empty
RUNLOG=/tmp/sdxl_method_run.log
: > /tmp/sdxl_method_data.tsv
echo "SDXL method bench  $(date)  host=$(hostname)  N=$N passes/cell" | tee "$RES"
echo "test=$TEST -k $SEL" | tee -a "$RES"

run_case() {  # $1=cell label ; sets env, times one pytest, appends "<cell>\t<wall_ms>\t<hit>\t<result>"
  local cell="$1" jit cc_env=() label="$1" t0 t1 wall hit res
  case "$cell" in
    cold_ccoff)   jit="$JIT_COLD"; rm -rf "$jit"; cc_env=(env -u TT_METAL_CCACHE_KERNEL_SUPPORT) ;;
    cold_ccempty) jit="$JIT_COLD"; rm -rf "$jit" "$CC_EMPTY"; cc_env=(env TT_METAL_CCACHE_KERNEL_SUPPORT=1 CCACHE_DIR="$CC_EMPTY") ;;
    cold_ccwarm)  jit="$JIT_COLD"; rm -rf "$jit";            cc_env=(env TT_METAL_CCACHE_KERNEL_SUPPORT=1 CCACHE_DIR="$CC_WARM") ;;
    warm)         jit="$JIT_WARM";                            cc_env=(env TT_METAL_CCACHE_KERNEL_SUPPORT=1) ;;  # JIT pre-populated, no compile
  esac
  t0=$(date +%s%N)
  "${cc_env[@]}" TT_METAL_CACHE="$jit" timeout 400 python3 -m pytest "$TEST" -k "$SEL" -p no:cacheprovider >"$RUNLOG" 2>&1
  t1=$(date +%s%N); wall=$(( (t1-t0)/1000000 ))
  hit=$(grep -aoE "JIT cache stats: [0-9/]+ hits \([0-9.]+%\)" "$RUNLOG" | tail -1 | grep -oE "\([0-9.]+%\)" | tr -d '()')
  res=$(grep -aoE "[0-9]+ (passed|failed)" "$RUNLOG" | tail -1)
  printf "%s\t%s\t%s\t%s\n" "$cell" "$wall" "${hit:-?}" "${res:-NORESULT}" >> /tmp/sdxl_method_data.tsv
  printf "  %-13s wall=%6dms  jit_hit=%-7s %s\n" "$cell" "$wall" "${hit:-?}" "${res:-NORESULT}" | tee -a "$RES"
}

echo "" | tee -a "$RES"; echo "=== PRE-WARM (untimed): warm ccache dir + warm JIT dir ===" | tee -a "$RES"
rm -rf "$CC_WARM" "$JIT_WARM" "$JIT_COLD"
TT_METAL_CCACHE_KERNEL_SUPPORT=1 CCACHE_DIR="$CC_WARM" TT_METAL_CACHE="$JIT_WARM" \
  python3 -m pytest "$TEST" -k "$SEL" -p no:cacheprovider >"$RUNLOG" 2>&1 \
  && echo "  pre-warm ok (JIT_WARM + CC_WARM populated)" | tee -a "$RES" || echo "  PRE-WARM FAILED" | tee -a "$RES"

CELLS=(cold_ccoff cold_ccempty cold_ccwarm warm)
for p in $(seq 1 "$N"); do
  echo "" | tee -a "$RES"; echo "=== PASS $p/$N ($(date '+%T')) — shuffled order ===" | tee -a "$RES"
  for cell in $(printf "%s\n" "${CELLS[@]}" | shuf); do run_case "$cell"; done
done

echo "" | tee -a "$RES"; echo "================= MEDIANS (ms) =================" | tee -a "$RES"
python3 - <<'PY' | tee -a "$RES"
import statistics, collections
d=collections.defaultdict(list)
for ln in open("/tmp/sdxl_method_data.tsv"):
    p=ln.rstrip("\n").split("\t")
    if len(p)>=2 and p[1].isdigit(): d[p[0]].append(int(p[1]))
med={}
for c in ["cold_ccoff","cold_ccempty","cold_ccwarm","warm"]:
    if d[c]:
        med[c]=statistics.median(d[c])
        print(f"{c:13} median={med[c]:6d}  min={min(d[c]):6d} max={max(d[c]):6d}  n={len(d[c])}  all={d[c]}")
print()
w=med.get("warm")
if w:
    print("DERIVED warmup+warm (in-job two-pass = cold + warm):")
    for c in ["cold_ccoff","cold_ccempty","cold_ccwarm"]:
        if c in med:
            tp=med[c]+w
            print(f"  {c:13}: cold={med[c]:6d}  vs  warmup+warm={tp:6d}  ({'COLD wins' if med[c]<tp else 'WARMUP+WARM wins'} in-job)")
    print()
    print("SHIP-THE-CACHE (warm reuse only, warmup amortized elsewhere):")
    for c in ["cold_ccoff","cold_ccempty","cold_ccwarm"]:
        if c in med:
            print(f"  vs {c:13}: {med[c]/w:.2f}x faster  (cold {med[c]}ms -> warm {w}ms)")
PY
echo "DONE $(date)" | tee -a "$RES"
touch /tmp/sdxl_method_DONE
