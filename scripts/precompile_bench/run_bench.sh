#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# run_bench.sh — comprehensive local benchmark of the precompile system on a 75-test
# layernorm suite. Measures the e2e cost of the two ways to run a suite:
#
#   COLD       : the standard run — kernels JIT-compile inline & serial during the run.
#   PRECOMPILE : the new path — WARMUP (phase 1 = collect + precompile, hardware-free,
#                parallel) then a WARM run that reuses the on-disk JIT cache.
#
# Two orthogonal cache axes:
#   * JIT binary cache (TT_METAL_CACHE): reset before every COLD run and before every
#     WARMUP, so COLD is truly cold and WARMUP truly fills it; the WARM run reuses it.
#   * Compiler cache (ccache): kernel-JIT uses ccache only when TT_METAL_CCACHE_KERNEL_SUPPORT
#     is set (build.cpp:115). Three ccache conditions:
#       off     — flag unset (default/CI behaviour; ccache not used by kernel JIT)
#       deleted — flag set, isolated CCACHE_DIR emptied first (cold compiler cache)
#       warm    — flag set, CCACHE_DIR reused from the preceding 'deleted' run of same method
#
# Per-phase instrumentation: wall/user/sys/%CPU/peak-RSS (run_and_time.py -> getrusage),
# CPU-over-time + compiler-vs-python split (cpu_sampler.py, segmented by marks.csv), and
# JIT telemetry (hits/total, cached, jitted) parsed from logs. WARMUP is sub-split into
# collect vs compile via the plugin's "compiled N programs in Xs" line.
#
# North-star metric: COLD wall  vs  (probe + WARMUP + WARM) wall, per ccache condition.
#
# Output: $BENCH_OUT (default /tmp/lnbench). Then: summarize.py $BENCH_OUT
# Holds the device flock for the whole run. Env knobs: BENCH_REPEATS, BENCH_OUT,
# PRECOMPILE_WORKERS, SAFE_PYTEST_DISPATCH_TIMEOUT.
set -uo pipefail

WT=/localdev/mstaletovic/2026_05_28/0104_mstaletovic_agent_eval/wt_origin_main
cd "$WT"
source python_env/bin/activate 2>/dev/null

# ccache must be controlled explicitly per-arm: scrub any ambient setting (this shell had
# TT_METAL_CCACHE_KERNEL_SUPPORT=1). OFF arms then simply omit it; ON arms set it.
unset TT_METAL_CCACHE_KERNEL_SUPPORT CCACHE_DIR

BENCH_DIR="$WT/scripts/precompile_bench"
SEL="${BENCH_SEL:-$BENCH_DIR/layernorm_75.txt}"
mapfile -t IDS < "$SEL"
PYEXE="$WT/python_env/bin/python"
PYTEST="$WT/python_env/bin/pytest"
SAMPLER="$BENCH_DIR/cpu_sampler.py"
TIMER="$BENCH_DIR/run_and_time.py"

# pytest base args: override pytest.ini addopts (drop -vvs/-rA/junitxml), run ALL (no -x)
PYTEST_BASE=(-o "addopts=--import-mode=importlib" -p no:cacheprovider -q --no-header)
WORKERS="${PRECOMPILE_WORKERS:-$(nproc)}"
REPEATS="${BENCH_REPEATS:-2}"
DISPATCH_TO="${SAFE_PYTEST_DISPATCH_TIMEOUT:-120}"

OUT="${BENCH_OUT:-/tmp/lnbench}"
JITC="$OUT/jitcache"     # TT_METAL_CACHE
CCD="$OUT/ccache"        # isolated CCACHE_DIR (ccache-on arm)
DESC=/tmp/tt_precompile_cluster_desc.yaml
FP=/tmp/tt_precompile_build_fingerprint.txt
LOGS="$OUT/logs"
MARKS="$OUT/marks.csv"
RESULTS="$OUT/phase_results.csv"

rm -rf "$OUT"; mkdir -p "$LOGS"
echo "$(nproc)" > "$OUT/nproc.txt"
echo "epoch,event,phase,label" > "$MARKS"
echo "label,repeat,method,ccache,ccstate,phase,wall_s,user_s,sys_s,cpu_pct,maxrss_mb,rc,extra" > "$RESULTS"

echo "BENCH start $(date)  host=$(hostname)  branch=$(git branch --show-current)" | tee "$OUT/run.log"
echo "BENCH tests=${#IDS[@]}  workers=$WORKERS  repeats=$REPEATS  nproc=$(nproc)  out=$OUT" | tee -a "$OUT/run.log"

mark(){ echo "$(date +%s.%N),$1,$2,$3" >> "$MARKS"; }

# parse phase-specific telemetry out of a log -> single "key=val;key=val" field (no commas)
_parse_extra(){
  local phase=$1 log=$2 out=""
  local jit
  jit=$(grep -aoE "JIT cache stats: [0-9]+/[0-9]+ hits \([0-9.]+%\) \[[0-9]+ cached, [0-9]+ build-once dedup" "$log" 2>/dev/null | tail -1)
  if [[ -n "$jit" ]]; then
    local hits total pct cached jitted
    hits=$(echo "$jit"   | sed -n 's#.*stats: \([0-9]*\)/.*#\1#p')
    total=$(echo "$jit"  | sed -n 's#.*/\([0-9]*\) hits.*#\1#p')
    pct=$(echo "$jit"    | sed -n 's#.*hits (\([0-9.]*\)%).*#\1#p')
    cached=$(echo "$jit" | sed -n 's#.*\[\([0-9]*\) cached.*#\1#p')
    jitted=$(( total - hits ))
    out="jit_total=${total};jit_hits=${hits};jit_jitted=${jitted};jit_hitpct=${pct};jit_cached=${cached}"
  fi
  if [[ "$phase" == "warmup" ]]; then
    local uniq comp np cw
    uniq=$(grep -aoE "\-> [0-9]+ unique programs" "$log" | tail -1 | grep -oE "[0-9]+")
    comp=$(grep -aoE "compiled [0-9]+ programs in [0-9.]+s" "$log" | tail -1)
    np=$(echo "$comp" | grep -oE "compiled [0-9]+" | grep -oE "[0-9]+")
    cw=$(echo "$comp" | grep -oE "in [0-9.]+s" | grep -oE "[0-9.]+")
    out="${out:+$out;}unique=${uniq:-?};compiled=${np:-?};compile_wall=${cw:-?}"
  fi
  local res
  res=$(grep -aoE "[0-9]+ (passed|failed|error|skipped|xfailed)" "$log" 2>/dev/null | tr '\n' ' ' | sed 's/ *$//;s/ /+/g')
  [[ -n "$res" ]] && out="${out:+$out;}result=${res}"
  echo "${out:-none}"
}

# run_phase <label> <method> <cc> <ccst> <repeat> <phase> ENV... -- CMD...
run_phase(){
  local label=$1 method=$2 cc=$3 ccst=$4 rep=$5 phase=$6; shift 6
  local envv=(); while [[ "$1" != "--" ]]; do envv+=("$1"); shift; done; shift
  local log="$LOGS/${label}.${phase}.log"
  local tfile="$OUT/.timing.$$"
  mark START "$phase" "$label"
  env "${envv[@]}" "$PYEXE" "$TIMER" "$tfile" "$log" "$@"
  mark END "$phase" "$label"
  local wall user sys maxrss rc cpu_pct maxrss_mb extra
  wall=$(sed -n 's/.*wall=\([0-9.]*\).*/\1/p' "$tfile")
  user=$(sed -n 's/.*user=\([0-9.]*\).*/\1/p' "$tfile")
  sys=$(sed -n 's/.*sys=\([0-9.]*\).*/\1/p' "$tfile")
  maxrss=$(sed -n 's/.*maxrss_kb=\([0-9]*\).*/\1/p' "$tfile")
  rc=$(sed -n 's/.*rc=\(-*[0-9]*\).*/\1/p' "$tfile")
  cpu_pct=$(awk -v u="$user" -v s="$sys" -v w="$wall" 'BEGIN{if(w>0)printf "%.0f",100*(u+s)/w; else print 0}')
  maxrss_mb=$(awk -v k="$maxrss" 'BEGIN{printf "%.0f",k/1024}')
  extra=$(_parse_extra "$phase" "$log")
  echo "$label,$rep,$method,$cc,$ccst,$phase,$wall,$user,$sys,$cpu_pct,$maxrss_mb,$rc,$extra" >> "$RESULTS"
  printf "  %-26s %-9s wall=%7.1fs cpu=%5s%% rss=%6sMB rc=%-3s %s\n" \
    "$label" "$phase" "$wall" "$cpu_pct" "$maxrss_mb" "$rc" "$extra" | tee -a "$OUT/run.log"
}

# _precompile_seq <label> <cc> <ccst> <rep> <ccache-env-tokens...>
_precompile_seq(){
  local label=$1 cc=$2 ccst=$3 rep=$4; shift 4
  local ccenv=("$@")
  rm -rf "$JITC"; mkdir -p "$JITC"

  run_phase "$label" precompile "$cc" "$ccst" "$rep" probe_real \
    "PYTHONPATH=$WT" "PRECOMPILE_FP=$FP" "TT_METAL_CACHE=$JITC" -- \
    "$PYEXE" "$BENCH_DIR/_probe_real.py"
  local rk force2 realkey
  rk=$(grep -aE '^RKEY ' "$LOGS/${label}.probe_real.log" | tail -1 | sed 's/^RKEY //')
  read -r force2 realkey <<< "$rk"
  echo "    -> real build_key=$realkey 2erisc=$force2" | tee -a "$OUT/run.log"

  run_phase "$label" precompile "$cc" "$ccst" "$rep" probe_mock \
    "TT_METAL_FORCE_2_ERISC_MODE=$force2" "TT_METAL_JIT_BUILD_FINGERPRINT=$FP" \
    TT_METAL_SLOW_DISPATCH_MODE=1 "TT_METAL_MOCK_CLUSTER_DESC_PATH=$DESC" "PYTHONPATH=$WT" -- \
    "$PYEXE" "$BENCH_DIR/_probe_mock.py"
  local mk
  mk=$(grep -aoE 'MKEY [0-9]+' "$LOGS/${label}.probe_mock.log" | tail -1 | awk '{print $2}')
  if [[ "$mk" != "$realkey" ]]; then
    echo "    !! build_key MISMATCH real=$realkey mock=$mk -> warm would NOT be reused (skipping warm legs)" | tee -a "$OUT/run.log"
    return 0
  fi
  echo "    -> mock build_key MATCHES ($mk) — warm will be reused" | tee -a "$OUT/run.log"

  # WARMUP: hardware-free collect + parallel compile (fills JITC). ccache applies here.
  run_phase "$label" precompile "$cc" "$ccst" "$rep" warmup \
    "TT_METAL_FORCE_2_ERISC_MODE=$force2" "TT_METAL_JIT_BUILD_FINGERPRINT=$FP" \
    TT_METAL_SLOW_DISPATCH_MODE=1 "TT_METAL_MOCK_CLUSTER_DESC_PATH=$DESC" \
    UP_FRONT_COLLECT=1 UP_FRONT_META_COLLECT=1 "UP_FRONT_COLLECT_WORKERS=$WORKERS" \
    LOGURU_LEVEL=ERROR "TT_METAL_CACHE=$JITC" "PYTHONPATH=$WT" "${ccenv[@]}" -- \
    "$PYTEST" "${IDS[@]}" -p up_front_collect_plugin "${PYTEST_BASE[@]}"

  # WARM run: real device, fast dispatch, reuse JITC (should be ~0 compiles).
  run_phase "$label" precompile "$cc" "$ccst" "$rep" warm \
    "TT_METAL_CACHE=$JITC" TT_LOGGER_LEVEL=Info "TT_METAL_OPERATION_TIMEOUT_SECONDS=$DISPATCH_TO" \
    "PYTHONPATH=$WT" "${ccenv[@]}" -- \
    "$PYTEST" "${IDS[@]}" "${PYTEST_BASE[@]}"
}

# cold run helper
_cold(){
  local label=$1 cc=$2 ccst=$3 rep=$4; shift 4
  local ccenv=("$@")
  rm -rf "$JITC"; mkdir -p "$JITC"
  run_phase "$label" cold "$cc" "$ccst" "$rep" cold \
    "TT_METAL_CACHE=$JITC" TT_LOGGER_LEVEL=Info "TT_METAL_OPERATION_TIMEOUT_SECONDS=$DISPATCH_TO" \
    "PYTHONPATH=$WT" "${ccenv[@]}" -- \
    "$PYTEST" "${IDS[@]}" "${PYTEST_BASE[@]}"
}

run_matrix_repeat(){
  local rep=$1
  # ccache OFF
  _cold           "r${rep}.cold.off"      off na      "$rep"
  _precompile_seq "r${rep}.pc.off"        off na      "$rep"
  # ccache ON: deleted (empty CCD) then warm (reuse CCD), per method
  rm -rf "$CCD"; mkdir -p "$CCD"
  _cold           "r${rep}.cold.on.del"   on  deleted "$rep" "TT_METAL_CCACHE_KERNEL_SUPPORT=1" "CCACHE_DIR=$CCD"
  _cold           "r${rep}.cold.on.warm"  on  warm    "$rep" "TT_METAL_CCACHE_KERNEL_SUPPORT=1" "CCACHE_DIR=$CCD"
  rm -rf "$CCD"; mkdir -p "$CCD"
  _precompile_seq "r${rep}.pc.on.del"     on  deleted "$rep" "TT_METAL_CCACHE_KERNEL_SUPPORT=1" "CCACHE_DIR=$CCD"
  _precompile_seq "r${rep}.pc.on.warm"    on  warm    "$rep" "TT_METAL_CCACHE_KERNEL_SUPPORT=1" "CCACHE_DIR=$CCD"
}

# ---- device lock for the whole matrix ----
exec 9>/tmp/tt-device.lock
echo "BENCH: waiting for device lock..." | tee -a "$OUT/run.log"
flock 9
echo "BENCH: device lock acquired $(date)" | tee -a "$OUT/run.log"
if [[ -f /tmp/tt-device.dirty ]]; then
  echo "BENCH: device dirty -> tt-smi -r" | tee -a "$OUT/run.log"
  tt-smi -r >>"$OUT/run.log" 2>&1 && rm -f /tmp/tt-device.dirty
fi

# ---- one-time cluster descriptor (HW-stable), reported separately ----
if [[ ! -f "$DESC" ]]; then
  echo "BENCH: capturing one-time cluster descriptor..." | tee -a "$OUT/run.log"
  d0=$(date +%s.%N)
  timeout 120 "$PYEXE" - "$DESC" <<'PY' >"$LOGS/descriptor.log" 2>&1
import sys, tt_umd
tt_umd.TopologyDiscovery.create_cluster_descriptor().serialize_to_file(sys.argv[1])
PY
  d1=$(date +%s.%N)
  echo "one-time-descriptor wall=$(awk -v a=$d0 -v b=$d1 'BEGIN{printf "%.1f",b-a}')s" | tee -a "$OUT/run.log"
else
  echo "BENCH: reusing cached descriptor $DESC (one-time cost not in steady-state totals)" | tee -a "$OUT/run.log"
fi

# ---- sampler across the whole run ----
"$PYEXE" "$SAMPLER" "$OUT/sampler.csv" $$ 0.25 &
SAMP=$!
trap 'kill -TERM $SAMP 2>/dev/null' EXIT

for r in $(seq 1 "$REPEATS"); do
  echo "" | tee -a "$OUT/run.log"
  echo "======== REPEAT $r / $REPEATS  $(date '+%T') ========" | tee -a "$OUT/run.log"
  run_matrix_repeat "$r"
done

kill -TERM $SAMP 2>/dev/null
echo "" | tee -a "$OUT/run.log"
echo "BENCH done $(date)" | tee -a "$OUT/run.log"
touch "$OUT/DONE"
