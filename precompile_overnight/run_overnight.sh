#!/bin/bash
# Overnight A/B benchmark, FULL groupnorm golden suite, cold cache each:
#   A = "normal" run   : cold, NO precompile (today's behavior) -> compile inline, sequential
#   B = "new" run      : cold, parallel precompile (warmup, workers=8 greedy) -> warm execute
#   C = fallback        : only if B fails (e.g. OOM) -> new run at workers=4
#
# Each run is a separate run_safe_pytest invocation (device lock + reset between).
# A cgroup memory sampler runs throughout (mem.log) to capture peak usage / OOM.
# Everything is logged here; SUMMARY.txt is the at-a-glance result.
set +e

REPO=/localdev/mstaletovic/2026_05_28/0104_mstaletovic_agent_eval/clones/groupnorm_sc_N_1_HW_C_full_run1/tt-metal
RES=$REPO/precompile_overnight
SUITE=.claude/eval/golden_tests/groupnorm_sc_N_1_HW_C/test_golden.py
SUMMARY=$RES/SUMMARY.txt
CG=/sys/fs/cgroup/memory
LIM=$(cat $CG/memory.limit_in_bytes 2>/dev/null || echo 0)

cd "$REPO" || exit 1
source python_env/bin/activate 2>/dev/null

: > "$SUMMARY"
log(){ echo "$@" | tee -a "$SUMMARY"; }
sess_time(){ grep -oE "in [0-9]+\.[0-9]+s" "$1" | tail -1; }   # pytest session wall
counts(){ grep -oE "[0-9]+ (passed|failed|skipped|xfailed|error)" "$1" | tr '\n' ' '; }

log "==================================================================="
log "Overnight precompile A/B — FULL groupnorm golden suite"
log "started : $(date)"
log "host    : $(hostname)"
log "cgroup  : limit $(awk -v l=$LIM 'BEGIN{printf "%.0f GB", l/1073741824}')"
log "suite   : $SUITE (all RUN cases, no size cap)"
log "==================================================================="

# --- memory sampler (epoch usage_bytes total_rss total_cache failcnt) ---
( echo "epoch usage_bytes total_rss total_cache failcnt" > "$RES/mem.log"
  while true; do
    echo "$(date +%s) $(cat $CG/memory.usage_in_bytes 2>/dev/null) $(awk '/^total_rss /{print $2}' $CG/memory.stat 2>/dev/null) $(awk '/^total_cache /{print $2}' $CG/memory.stat 2>/dev/null) $(cat $CG/memory.failcnt 2>/dev/null)" >> "$RES/mem.log"
    sleep 5
  done ) &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

# ------------------------------------------------------------------ RUN A
log ""; log ">>> RUN A  normal (cold, no precompile)  start $(date)"
rm -rf /tmp/ovn_cache_A
t0=$(date +%s)
TT_METAL_CACHE=/tmp/ovn_cache_A scripts/run_safe_pytest.sh --run-all "$SUITE" > "$RES/A_normal.log" 2>&1
EA=$?; WA=$(( $(date +%s) - t0 ))
log "RUN A  normal       : exit=$EA  total_wall=${WA}s ($((WA/60))m)  session=$(sess_time "$RES/A_normal.log")"
log "RUN A  test counts  : $(counts "$RES/A_normal.log")"
log "RUN A  cache size   : $(du -sh /tmp/ovn_cache_A 2>/dev/null | cut -f1)"
rm -rf /tmp/ovn_cache_A

# ------------------------------------------------------------------ RUN B
log ""; log ">>> RUN B  new workers=8 greedy (cold -> precompile -> warm)  start $(date)"
rm -rf /tmp/ovn_cache_B8
t0=$(date +%s)
TT_METAL_CACHE=/tmp/ovn_cache_B8 EVAL_PRECOMPILE=1 EVAL_PRECOMPILE_WORKERS=8 \
  scripts/run_safe_pytest.sh --run-all "$SUITE" > "$RES/B_new_w8.log" 2>&1
EB=$?; WB=$(( $(date +%s) - t0 ))
log "RUN B  new(w8)      : exit=$EB  total_wall=${WB}s ($((WB/60))m)  session=$(sess_time "$RES/B_new_w8.log")"
log "RUN B  precompile   : $(grep -oE 'EVAL_PRECOMPILE: [0-9]+ unique programs compiled in [0-9.]+s.*' "$RES/B_new_w8.log" | tail -1)"
log "RUN B  test counts  : $(counts "$RES/B_new_w8.log")"
log "RUN B  OOM/killed?  : $(grep -cE 'Killed|build failed|trisc.*failure' "$RES/B_new_w8.log") hits"
rm -rf /tmp/ovn_cache_B8

# ------------------------------------------------------------------ RUN C (fallback)
EC=0; WC=0
if [ $EB -ne 0 ]; then
  log ""; log ">>> RUN C  new workers=4 (fallback, B8 exit=$EB)  start $(date)"
  rm -rf /tmp/ovn_cache_B4
  t0=$(date +%s)
  TT_METAL_CACHE=/tmp/ovn_cache_B4 EVAL_PRECOMPILE=1 EVAL_PRECOMPILE_WORKERS=4 \
    scripts/run_safe_pytest.sh --run-all "$SUITE" > "$RES/C_new_w4.log" 2>&1
  EC=$?; WC=$(( $(date +%s) - t0 ))
  log "RUN C  new(w4)      : exit=$EC  total_wall=${WC}s ($((WC/60))m)  session=$(sess_time "$RES/C_new_w4.log")"
  log "RUN C  precompile   : $(grep -oE 'EVAL_PRECOMPILE: [0-9]+ unique programs compiled in [0-9.]+s.*' "$RES/C_new_w4.log" | tail -1)"
  log "RUN C  test counts  : $(counts "$RES/C_new_w4.log")"
  log "RUN C  OOM/killed?  : $(grep -cE 'Killed|build failed|trisc.*failure' "$RES/C_new_w4.log") hits"
  rm -rf /tmp/ovn_cache_B4
fi

# ------------------------------------------------------------------ summary
log ""
log "=== memory peak during the night ==="
awk -v lim=$LIM 'NR>1 && $2>m{m=$2;r=$3;c=$4} END{printf "peak usage = %.1f GB (rss %.1f, cache %.1f)  /  limit %.0f GB\n", m/1073741824, r/1073741824, c/1073741824, lim/1073741824}' "$RES/mem.log" | tee -a "$SUMMARY"
fc=$(tail -1 "$RES/mem.log" | awk '{print $5}')
log "memory.failcnt (end) = ${fc:-?}"

log ""
log "=== A/B verdict ==="
log "A normal (cold, sequential)         : ${WA}s ($((WA/60))m)  exit=$EA"
if [ $EB -eq 0 ]; then
  log "B new    (cold, precompile w8)       : ${WB}s ($((WB/60))m)  exit=$EB"
  if [ $WB -gt 0 ]; then log "SPEEDUP (A/B)                        : $(awk -v a=$WA -v b=$WB 'BEGIN{printf "%.2fx", a/b}')"; fi
else
  log "B new    (cold, precompile w8)       : FAILED exit=$EB (see B_new_w8.log; likely OOM)"
  if [ $EC -eq 0 ] && [ $WC -gt 0 ]; then
    log "C new    (cold, precompile w4)       : ${WC}s ($((WC/60))m)  exit=$EC"
    log "SPEEDUP (A/C)                        : $(awk -v a=$WA -v c=$WC 'BEGIN{printf "%.2fx", a/c}')"
  fi
fi
log ""
log "finished: $(date)"
log "logs: A_normal.log  B_new_w8.log  $( [ $EB -ne 0 ] && echo C_new_w4.log )  mem.log"
echo ""
echo "ALL DONE — results in $SUMMARY"
