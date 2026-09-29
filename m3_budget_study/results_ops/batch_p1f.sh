#!/bin/bash
# P1-F (optional device top-up): packed forwards that separate per-request from per-2048-segment ops.
#
# The misc breakdown itself runs host-only on the P0-A zone profiles (tools/misc_breakdown.py): single and
# packed at W=4096 (2 slots x 1 segment) and W=8192 (4 slots x 1 segment), with per-op rows. The packed path
# (TtPrefillRuntime.prefill_segments) always cuts a forward into 2048-token segments, so B = W / 2048 there;
# "B=4 at W=4096" does not exist, and "B=2 at W=8192" is 2 slots x 2 segments. These two points tell whether an
# op scales with slots (requests) or with segments:
#   w4096_1x2   one slot, 2 consecutive segments (prose@139264+141312)            vs P0-A w4096_packed (2 x 1)
#   w8192_2x2   2 slots x 2 segments (prose@139264+141312, code@0+2048)           vs P0-A w8192_packed (4 x 1)
# Same harness, env and guards as batch_p0a.sh (layers 0-6 on the (2,4) stage 0, PREFIX_QUIET, warm point,
# PROFILE_PREFIX_READ_EVERY=0, M3_MOE_W_NDSHARD=1 M3_MOE_HYBRID_THRESHOLD=128, v1 dispatch/combine, 1d fabric),
# through run_prefill_profile.sh (it resets the galaxy with `env -u TT_VISIBLE_DEVICES tt-smi -glx_reset`).
# Rows: p1f_runs.csv, per_op.csv (appended), then misc_breakdown.py over p0a_runs.csv + p1f_runs.csv.
#
#   nohup m3_budget_study/results_ops/batch_p1f.sh > m3_budget_study/results_ops/logs/batch_p1f.out 2>&1 &
#   ONLY="w8192_2x2" ...   run a subset;   --dry-run / DRY_RUN=1: print the plan only
set -uo pipefail
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1
DRY_RUN="${DRY_RUN:-0}"
RES="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TT_METAL_HOME="${TT_METAL_HOME:-$(cd "$RES/../.." && pwd)}"
export TT_METAL_HOME
LOCK="$RES/.lock" LOCK_OWNER="owner=vmelnykov-ops-agent"
RUN_TIMEOUT="${RUN_TIMEOUT:-1200}" STALL_TIMEOUT="${STALL_TIMEOUT:-300}"
PREFIX="${PREFIX:-p1f}"
GOLDEN=/mnt/weka/model-cache/scratch/minimax/MiniMax-M3-cache/prefill/golden
PROSE="$GOLDEN/longbook_56320/metadata.json"
CODE="$RES/inputs/code_m3/metadata.json"
WRAPPER="$TT_METAL_HOME/models/demos/minimax_m3/scripts/run_prefill_profile.sh"
PARSE="$TT_METAL_HOME/models/demos/minimax_m3/tests/perf/parse_zone_perf.py"
PY="$TT_METAL_HOME/python_env/bin/python"
export NODE="${NODE:-$(ls ~/.vscode-server/cli/servers/*/server/node 2>/dev/null | head -1)}"
RUNS="$RES/p1f_runs.csv"

export STAGES=4 STAGE=0 FABRIC=1d M3_FABRIC=1d LAYER_IDS=0,1,2,3,4,5,6
export M3_MOE_W_NDSHARD=1 M3_MOE_HYBRID_THRESHOLD=128 M3_MOE_DISPATCH=v1 M3_MOE_COMBINE=v1
export M3_MOE_TOPOLOGY=linear M3_CCL_TOPOLOGY=linear EXPERT_DTYPE=bf4 LEVEL=2
export PROFILE_SKIP_COMPILE=1 PREFIX_QUIET=1 WARM_POINT="${WARM_POINT:-3}" PROFILE_PROGRESS_EVERY=8
export PROFILE_PREFIX_READ_EVERY=0
export HF_MODEL="${HF_MODEL:-/mnt/weka/model-weights/llm/minimax/MiniMax-M3}"
export TT_CACHE_PATH="${TT_CACHE_PATH:-/mnt/weka/model-cache/scratch/minimax/MiniMax-M3-cache/prefill}"
unset SKIP_PREFIX PROFILE_SKIP_PREFIX NOC_TRACES CACHE SEGMENTS MESH HARNESS

# name | W | h label | B (segments) | input | SEGMENTS | roofline segments
POINTS=(
  "w4096_1x2|4096|139264+141312|2|prose|prose@139264+141312|2048:139264,2048:141312"
  "w8192_2x2|8192|139264+141312+0+2048|4|prose+code|prose@139264+141312,code@0+2048|2048:139264,2048:141312,2048:0,2048:2048"
)

die () { echo "[p1f] ERROR: $*" >&2; exit 1; }
other_harness () {
  pgrep -u "$(id -u)" -af 'models/demos/minimax_m3/tests/perf/[a-z_]+\.py|tools/profile_4x2\.py|tracy-capture|prefill\.runners\.prefill_(runner|producer)' \
    | grep -v "^$$ " || true
}

run_point () {
  local name W h B input segs rsegs
  IFS='|' read -r name W h B input segs rsegs <<< "$1"
  local id="${PREFIX}_${name}"
  local log="$RES/logs/$id.log" envf="$RES/logs/$id.env" prof="$RES/profiles/$id" ptmp="$RES/profiler_tmp/$id"
  local -a knobs=(SRC_TRACE="$PROSE" RESULTS_DIR="$prof" LOGDIR="$prof" TT_METAL_PROFILER_DIR="$ptmp"
                  REPORTS="$ptmp/reports" PERF_WORKDIR="$ptmp/traces" SEGMENTS="$segs" INPUTS="prose=$PROSE;code=$CODE"
                  CHUNK="$W" TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT="${PROGRAMS_PACKED:-2400}")
  if [ "$DRY_RUN" = 1 ]; then echo "[p1f] DRY $id: ${knobs[*]} $WRAPPER"; return 0; fi
  if [ -e "$log" ]; then echo "[p1f] $id: $log exists, skipping"; return 0; fi
  grep -q "$LOCK_OWNER" "$LOCK" 2>/dev/null || die "$LOCK does not name $LOCK_OWNER; stopping before $id"
  local busy; busy="$(other_harness)"; [ -n "$busy" ] && die "another harness is running, stopping before $id:"$'\n'"$busy"
  mkdir -p "$prof" "$ptmp" "$RES/logs"
  { echo "run_id=$id"; echo "git_sha=$(git -C "$TT_METAL_HOME" rev-parse HEAD)"
    echo "dirty=$(git -C "$TT_METAL_HOME" status --porcelain -uno | wc -l)"; echo "date=$(date -Is)"
    printf '%s\n' "${knobs[@]}"
    env | grep -E '^(STAGES|STAGE|FABRIC|LAYER_IDS|LEVEL|WARM_POINT|PREFIX_QUIET|M3_|PROFILE_|TT_|EXPERT_|HF_)=' | sort
  } > "$envf"
  echo "[p1f] $(date '+%F %T') START $id (W=$W B=$B segments=$segs)"
  local t0; t0=$(date +%s)
  setsid env "${knobs[@]}" "$WRAPPER" > "$log" 2>&1 &
  local pid=$! status=""
  while kill -0 "$pid" 2>/dev/null; do
    sleep 15
    local now age; now=$(date +%s); age=$(( now - $(stat -c %Y "$log") ))
    if [ $(( now - t0 )) -gt "$RUN_TIMEOUT" ]; then status=TIMEOUT; break; fi
    if grep -q '\[zone-prof\] warmup / compile' "$log" && ! grep -q '\[zone-prof\] DONE' "$log" \
       && [ "$age" -gt "$STALL_TIMEOUT" ]; then status=HANG; break; fi
  done
  if [ -n "$status" ]; then
    kill -TERM -- "-$pid" 2>/dev/null; sleep 20; kill -KILL -- "-$pid" 2>/dev/null
    pkill -KILL -u "$(id -u)" -f 'models/demos/minimax_m3/tests/perf/profile_prefill.py' 2>/dev/null
    wait "$pid" 2>/dev/null
    env -u TT_VISIBLE_DEVICES tt-smi -glx_reset >> "$RES/logs/$id.reset" 2>&1
  else
    wait "$pid"; local rc=$?
    if [ "$rc" -eq 0 ] && grep -q '\[zone-prof\] DONE' "$log"; then status=OK
    elif grep -qiE 'out of memory|OOM' "$log"; then status=OOM
    else status="ERROR(rc=$rc)"; fi
  fi
  local elapsed=$(( $(date +%s) - t0 ))
  local wall; wall="$(grep -oE 'wall-clock: [0-9.]+ ms' "$log" | tail -1 | grep -oE '[0-9.]+' | head -1)"
  local mb; mb="$(du -sm "$ptmp" 2>/dev/null | cut -f1)"
  local csv; csv="$(find "$prof" -name 'ops_perf_results_*.csv' 2>/dev/null | sort | tail -1)"
  echo "[p1f] $(date '+%F %T') END $id status=$status elapsed=${elapsed}s wall=${wall:-?}ms csv=${csv:-none}"
  if [ "$status" = OK ] && [ -n "$csv" ]; then
    if "$PY" "$PARSE" "$csv" --json "$prof/zones.json" --per-device "$prof/per_device.json" \
         --html "$prof/zones.html" > "$prof/parse.log" 2>&1; then
      "$PY" "$RES/tools/zones_to_per_op.py" --zones "$prof/zones.json" --per-device "$prof/per_device.json" \
        --W "$W" --h "$h" --B "$B" --input "$input" --mesh 2x4 --segments "$rsegs" --run-id "$id" \
        --out "$RES/per_op.csv" >> "$prof/parse.log" 2>&1 || status="$status/per_op_failed"
    else
      status="$status/parse_failed"
    fi
  fi
  [ -f "$RUNS" ] || echo "run_id,status,elapsed_s,W,h,B,input,segments,wall_ms,capture_mb,csv" > "$RUNS"
  echo "$id,$status,$elapsed,$W,$h,$B,$input,\"$segs\",${wall:-},${mb:-},${csv:-}" >> "$RUNS"
}

for spec in "${POINTS[@]}"; do
  name="${spec%%|*}"
  if [ -n "${ONLY:-}" ] && [[ " $ONLY " != *" $name "* ]]; then continue; fi
  run_point "$spec"
done
[ "$DRY_RUN" = 1 ] && exit 0
"$PY" "$RES/tools/misc_breakdown.py" --runs "$RES/p0a_runs.csv,$RUNS" || echo "[p1f] misc_breakdown failed"
echo "[p1f] $(date '+%F %T') done"
