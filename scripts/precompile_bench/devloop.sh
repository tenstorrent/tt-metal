#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# devloop.sh — two device experiments about the *development* loop (editing the op you're on):
#
#  PART 1  JIT-cache staleness: does a warm TT_METAL_CACHE pick up a semantic kernel edit,
#          or stale-hit (run the old binary)? Method: fresh JIT -> run (populate) -> rerun
#          unchanged (expect all hits) -> edit layernorm.cpp -> rerun SAME cache, read the
#          JIT telemetry. jitted>0 for the edited kernel => correctly invalidated; 0 => stale.
#
#  PART 2  dev-loop e2e: with a WARM ccache (the dev's persistent compiler cache) and a fresh
#          JIT cache, after editing the op, is PRECOMPILE (parallel warmup + warm run) faster
#          than a plain COLD inline rerun? Two edit sizes: one compute kernel vs the whole
#          compute path. North-star = total wall.
#
# Holds the device flock. Reverts all kernel edits (trap). Reuses captured fingerprint/desc.
set -uo pipefail
WT=/localdev/mstaletovic/2026_05_28/0104_mstaletovic_agent_eval/wt_origin_main
cd "$WT"; source python_env/bin/activate 2>/dev/null
unset TT_METAL_CCACHE_KERNEL_SUPPORT CCACHE_DIR
BENCH_DIR="$WT/scripts/precompile_bench"
mapfile -t IDS < "$BENCH_DIR/layernorm_75.txt"
TIMER="$BENCH_DIR/run_and_time.py"; PYTEST="$WT/python_env/bin/pytest"
FP=/tmp/tt_precompile_build_fingerprint.txt; DESC=/tmp/tt_precompile_cluster_desc.yaml
KDIR=ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels
MAIN="$KDIR/compute/layernorm.cpp"; ALLC=$(ls $KDIR/compute/layernorm*.cpp)
PB=(-o "addopts=--import-mode=importlib" -p no:cacheprovider -q --no-header)
DTO=120
OUT="${OUT:-/tmp/ln_devloop}"; rm -rf "$OUT"; mkdir -p "$OUT"; RES="$OUT/results.txt"
JITC="$OUT/jit"; CCD="$OUT/cc"
edit(){ printf '\nnamespace { [[maybe_unused]] volatile int _dev_probe_marker = %s; }\n' "$RANDOM$RANDOM" >> "$1"; }
revert(){ git checkout -- $ALLC 2>/dev/null; }
trap revert EXIT

# tel <log> -> "jitted/total (hit%)"
tel(){ grep -aoE "JIT cache stats: [0-9]+/[0-9]+ hits \([0-9.]+%\)" "$1" 2>/dev/null | tail -1 \
       | sed -E 's#JIT cache stats: ([0-9]+)/([0-9]+) hits \(([0-9.]+)%\)#hit=\1/\2 (\3%)#'; }
# run <tag> ENV... -- CMD... ; prints wall + telemetry, returns wall via $WALL
run(){ local tag=$1; shift; local env=(); while [[ "$1" != "--" ]]; do env+=("$1"); shift; done; shift
  local log="$OUT/$tag.log" tf="$OUT/$tag.tf"
  env "${env[@]}" python "$TIMER" "$tf" "$log" "$@"
  WALL=$(sed -n 's/.*wall=\([0-9.]*\).*/\1/p' "$tf")
  local j; j=$(grep -aoE "[0-9]+ (passed|failed|error)" "$log" | tr '\n' ' ')
  printf "  %-22s wall=%6.1fs  %s  [%s]\n" "$tag" "$WALL" "$(tel "$log")" "$j" | tee -a "$RES"
}
warmup(){ local tag=$1 jit=$2 cc=$3 # cc='' for off
  local cce=(); [[ -n "$cc" ]] && cce=(TT_METAL_CCACHE_KERNEL_SUPPORT=1 "CCACHE_DIR=$cc")
  run "$tag" TT_METAL_FORCE_2_ERISC_MODE=1 "TT_METAL_JIT_BUILD_FINGERPRINT=$FP" TT_METAL_SLOW_DISPATCH_MODE=1 \
    "TT_METAL_MOCK_CLUSTER_DESC_PATH=$DESC" UP_FRONT_COLLECT=1 UP_FRONT_META_COLLECT=1 UP_FRONT_COLLECT_WORKERS=8 \
    "TT_METAL_CACHE=$jit" LOGURU_LEVEL=ERROR "PYTHONPATH=$WT" "${cce[@]}" -- \
    "$PYTEST" "${IDS[@]}" -p up_front_collect_plugin "${PB[@]}"
}
realrun(){ local tag=$1 jit=$2 cc=$3
  local cce=(); [[ -n "$cc" ]] && cce=(TT_METAL_CCACHE_KERNEL_SUPPORT=1 "CCACHE_DIR=$cc")
  run "$tag" "TT_METAL_CACHE=$jit" TT_LOGGER_LEVEL=Info "TT_METAL_OPERATION_TIMEOUT_SECONDS=$DTO" \
    "PYTHONPATH=$WT" "${cce[@]}" -- "$PYTEST" "${IDS[@]}" "${PB[@]}"
}

exec 9>/tmp/tt-device.lock; echo "devloop: waiting for device lock..."; flock 9; echo "devloop: lock acquired $(date)" | tee "$RES"
[[ -f /tmp/tt-device.dirty ]] && { tt-smi -r >/dev/null 2>&1 && rm -f /tmp/tt-device.dirty; }

echo ""; echo "######## PART 1: JIT-cache staleness on a kernel edit ########" | tee -a "$RES"
rm -rf "$JITC"; mkdir -p "$JITC"
realrun stale_1_populate  "$JITC" ""        # fresh JIT -> all jitted
realrun stale_2_rerun     "$JITC" ""        # same JIT, no edit -> expect all hits
edit "$MAIN"
realrun stale_3_afteredit "$JITC" ""        # same JIT, edited layernorm.cpp -> jitted>0? or stale(0)?
revert
echo "  (stale_3: jitted>0 => warm JIT correctly recompiles the edited kernel; ~0 => STALE hit)" | tee -a "$RES"
# Incremental warm-JIT rerun after editing the WHOLE compute path (the realistic 'big edit' rerun;
# repopulate the cache first since stale_3 left the edited-kernel build in it).
rm -rf "$JITC"; mkdir -p "$JITC"; realrun stale_4a_repopulate "$JITC" "" >/dev/null 2>&1 || true
for f in $ALLC; do edit "$f"; done
realrun stale_4_incr_editall "$JITC" ""     # same JIT, all compute edited -> incremental recompile
revert
echo "  (stale_4: incremental warm-JIT rerun after editing the whole compute path)" | tee -a "$RES"

echo ""; echo "######## PART 2: dev-loop e2e (warm ccache, fresh JIT)  COLD vs PRECOMPILE ########" | tee -a "$RES"
# warm the ccache with UNEDITED sources (the dev's persistent cache state before this edit)
rm -rf "$CCD"; mkdir -p "$CCD"; rm -rf "$JITC"; mkdir -p "$JITC"
warmup warm_ccache_unedited "$JITC" "$CCD" >/dev/null 2>&1 || true
echo "  (ccache warmed with unedited sources)" | tee -a "$RES"
# one-time probe cost (build_key is config-derived, invariant to kernel edits): measure pr+mk once
PR0=$(date +%s.%N)
PRECOMPILE_FP="$FP" PYTHONPATH="$WT" TT_METAL_CACHE="$JITC" python "$BENCH_DIR/_probe_real.py" >/dev/null 2>&1 || true
TT_METAL_FORCE_2_ERISC_MODE=1 TT_METAL_JIT_BUILD_FINGERPRINT="$FP" TT_METAL_SLOW_DISPATCH_MODE=1 \
  TT_METAL_MOCK_CLUSTER_DESC_PATH="$DESC" PYTHONPATH="$WT" python "$BENCH_DIR/_probe_mock.py" >/dev/null 2>&1 || true
PROBE=$(awk -v a=$PR0 -v b=$(date +%s.%N) 'BEGIN{printf "%.1f", b-a}')
echo "  (one-time precompile probe overhead pr+mk = ${PROBE}s, added to PRECOMPILE totals)" | tee -a "$RES"

devscenario(){ local name=$1; shift; local files="$*"
  echo "" | tee -a "$RES"; echo "=== edit: $name ===" | tee -a "$RES"
  for f in $files; do edit "$f"; done
  # COLD inline (warm ccache, fresh JIT, edited)
  rm -rf "$JITC"; mkdir -p "$JITC"
  realrun "$name.COLD" "$JITC" "$CCD"; local cold=$WALL
  # PRECOMPILE (warm ccache, fresh JIT, edited): warmup + warm run
  rm -rf "$JITC"; mkdir -p "$JITC"
  warmup "$name.warmup" "$JITC" "$CCD"; local wu=$WALL
  realrun "$name.warm" "$JITC" "$CCD"; local wm=$WALL
  revert
  local pct=$(awk -v p="$PROBE" -v u="$wu" -v m="$wm" 'BEGIN{printf "%.1f", p+u+m}')
  local spd=$(awk -v c="$cold" -v t="$pct" 'BEGIN{if(t>0)printf "%.2fx", c/t; else print "-"}')
  printf "  >> %-18s COLD=%.1fs  PRECOMPILE=%.1fs (pr+mk %ss + wu %.1fs + wm %.1fs)  speedup=%s\n" \
    "$name" "$cold" "$pct" "$PROBE" "$wu" "$wm" "$spd" | tee -a "$RES"
}
devscenario edit_one_kernel  "$MAIN"
devscenario edit_all_compute $ALLC

echo ""; echo "DONE_DEVLOOP $(date)" | tee -a "$RES"
