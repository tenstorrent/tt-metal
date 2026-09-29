#!/bin/bash
# P0-A: per-op zone profiles of MiniMax-M3 prefill on one (2,4) sub-mesh (SP=2, TP=4, EP=8), layers 0-6
# (3 dense + 4 sparse, contiguous from 0 so routing is real), real M3 tokens, real history (PROFILE_PREFIX_QUIET).
#
# Grid: W in {4096, 8192} x h in {0, 141312, 548864} x input in {prose, code}, single request, plus one packed
# forward per W mixing both documents. One device process per point, each through run_prefill_profile.sh
# (which resets the galaxy with `env -u TT_VISIBLE_DEVICES tt-smi -glx_reset` first), each with an in-process
# warm point (PROFILE_WARM_POINT: the first cache-read forward repeated before going deeper; avoids the W=8192
# "slow mode"). Fixed env: M3_MOE_W_NDSHARD=1 M3_MOE_HYBRID_THRESHOLD=128, v1 dispatch/combine, 1d fabric.
#
# Per run <id>: logs/<id>.log (wrapper stdout), logs/<id>.env (env + git SHA), profiles/<id>/ (ops CSV, wrapper
# log, zones.json, per_device.json, zones.html), profiler_tmp/<id>/ (tracy artifacts; its size is the capture
# size), a row in p0a_runs.csv, and the run's rows appended to per_op.csv (tools/zones_to_per_op.py).
#
# Guards: refuses to start a process unless .lock names owner=vmelnykov-ops-agent, or while another M3 harness
# process is running. Watchdog: RUN_TIMEOUT s total, STALL_TIMEOUT s without log output once the model is built
# (until the harness prints DONE) -> kill the process group, reset.
#
#   nohup m3_budget_study/results_ops/batch_p0a.sh > m3_budget_study/results_ops/logs/batch_p0a.out 2>&1 &
#   ONLY="w4096_h0_prose w4096_packed" ...   run a subset (run-id suffixes, see POINTS below)
#   MESH=4x2 PREFIX=p4x2 PER_OP_CSV=.. RUNS_CSV=..  another SPxTP sub-mesh (stage 0 of create_submeshes(SP, TP)); TP != 4
#                                             runs through tools/profile_4x2.py (multi-head KV cache)
set -uo pipefail
RES="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TT_METAL_HOME="${TT_METAL_HOME:-$(cd "$RES/../.." && pwd)}"
export TT_METAL_HOME
LOCK="$RES/.lock"
LOCK_OWNER="owner=vmelnykov-ops-agent"
RUN_TIMEOUT="${RUN_TIMEOUT:-1200}"
STALL_TIMEOUT="${STALL_TIMEOUT:-300}"
PREFIX="${PREFIX:-p0a}"
GOLDEN=/mnt/weka/model-cache/scratch/minimax/MiniMax-M3-cache/prefill/golden
PROSE="$GOLDEN/longbook_56320/metadata.json"
CODE="$RES/inputs/code_m3/metadata.json"
WRAPPER="${WRAPPER:-$TT_METAL_HOME/models/demos/minimax_m3/scripts/run_prefill_profile.sh}"
PARSE="$TT_METAL_HOME/models/demos/minimax_m3/tests/perf/parse_zone_perf.py"
PER_OP="$RES/tools/zones_to_per_op.py"
export NODE="${NODE:-$(ls ~/.vscode-server/cli/servers/*/server/node 2>/dev/null | head -1)}"
RUNS="${RUNS_CSV:-$RES/p0a_runs.csv}"
mkdir -p "$RES/logs" "$RES/profiles" "$RES/profiler_tmp"

MESH="${MESH:-2x4}"
if [ "$MESH" != 2x4 ]; then
  export MESH HARNESS="${HARNESS:-m3_budget_study/results_ops/tools/profile_4x2.py}"
fi

# Fixed configuration of every run. CFG_* override the transport (e.g. the 4x4 sub-torus with the v2 MoE ops;
# pass TT_VISIBLE_DEVICES / TT_MESH_GRAPH_DESC_PATH / PROFILE_PARENT_MESH through the environment).
export STAGES=4 STAGE=0 FABRIC="${CFG_FABRIC:-1d}" M3_FABRIC="${CFG_FABRIC:-1d}" LAYER_IDS=0,1,2,3,4,5,6
export M3_MOE_W_NDSHARD=1 M3_MOE_HYBRID_THRESHOLD=128
export M3_MOE_DISPATCH="${CFG_DISPATCH:-v1}" M3_MOE_COMBINE="${CFG_COMBINE:-v1}"
export M3_MOE_TOPOLOGY="${CFG_MOE_TOPOLOGY:-linear}" M3_CCL_TOPOLOGY="${CFG_CCL_TOPOLOGY:-linear}" EXPERT_DTYPE=bf4 LEVEL=2
export PROFILE_SKIP_COMPILE=1 PREFIX_QUIET=1 WARM_POINT="${WARM_POINT:-3}" PROFILE_PROGRESS_EVERY=8
# No drains in the un-profiled forwards: every marker read is a host zone in the .tracy, so a deep prefix drained
# every few forwards grows the capture with h (3.4 GB of tracy_ops_times.csv at h=0 already). The device buffer
# instead overflows (prefix markers dropped) and holds only what the last two forwards need: ~400 programs per
# chip per single forward; PROGRAMS_PACKED for the packed ones.
export PROFILE_PREFIX_READ_EVERY="${PROFILE_PREFIX_READ_EVERY:-0}"
PROGRAMS_SINGLE="${PROGRAMS_SINGLE:-1200}" PROGRAMS_PACKED="${PROGRAMS_PACKED:-2400}"
export HF_MODEL="${HF_MODEL:-/mnt/weka/model-weights/llm/minimax/MiniMax-M3}"
export TT_CACHE_PATH="${TT_CACHE_PATH:-/mnt/weka/model-cache/scratch/minimax/MiniMax-M3-cache/prefill}"
unset SKIP_PREFIX PROFILE_SKIP_PREFIX NOC_TRACES CACHE SEGMENTS

die () { echo "[p0a] ERROR: $*" >&2; exit 1; }
[ -f "$PROSE" ] || die "prose tokens missing: $PROSE"
if [ ! -f "$CODE" ]; then
  echo "[p0a] building the code corpus -> $CODE"
  "$TT_METAL_HOME/python_env/bin/python" "$RES/tools/make_code_corpus.py" --out "$CODE" || die "corpus build failed"
fi
[ -f "$RUNS" ] || echo "run_id,status,elapsed_s,W,h,B,input,segments,wall_ms,capture_mb,csv" > "$RUNS"

# name | W | h (history; packed: label) | B | input label | SRC_TRACE | SEGMENTS | roofline segments
POINTS=()
for W in 4096 8192; do
  for h in 0 141312 548864; do
    POINTS+=("w${W}_h${h}_prose|$W|$h|1|prose|$PROSE||$W:$h")
    POINTS+=("w${W}_h${h}_code|$W|$h|1|code|$CODE||$W:$h")
  done
  if [ "$W" = 4096 ]; then
    POINTS+=("w4096_packed|4096|141312+0|2|prose+code|$PROSE|prose@141312:2048,code@0:2048|2048:141312,2048:0")
  else
    POINTS+=("w8192_packed|8192|548864+141312+16384+0|4|prose+code+prose+code|$PROSE|prose@548864:2048,code@141312:2048,prose.1@16384:2048,code.1@0:2048|2048:548864,2048:141312,2048:16384,2048:0")
  fi
done

other_harness () {  # another M3 device harness running (sibling session / other agent)?
  pgrep -u "$(id -u)" -af 'models/demos/minimax_m3/tests/perf/[a-z_]+\.py|tools/profile_4x2\.py|tracy-capture' | grep -v "^$$ " || true
}

run_point () {
  local spec="$1" name W h B input src segs rsegs
  IFS='|' read -r name W h B input src segs rsegs <<< "$spec"
  local id="${PREFIX}_${name}"
  local log="$RES/logs/$id.log" envf="$RES/logs/$id.env" prof="$RES/profiles/$id" ptmp="$RES/profiler_tmp/$id"
  if [ -e "$log" ]; then echo "[p0a] $id: $log exists, skipping"; return 0; fi
  if ! grep -q "$LOCK_OWNER" "$LOCK" 2>/dev/null; then die "$LOCK does not name $LOCK_OWNER; stopping before $id"; fi
  local busy=""; [ "${BUSY_CHECK:-1}" = 1 ] && busy="$(other_harness)"
  if [ -n "$busy" ]; then die "another harness is running, stopping before $id:"$'\n'"$busy"; fi
  mkdir -p "$prof" "$ptmp"

  local -a knobs=(SRC_TRACE="$src" RESULTS_DIR="$prof" LOGDIR="$prof" TT_METAL_PROFILER_DIR="$ptmp"
                  REPORTS="$ptmp/reports" PERF_WORKDIR="$ptmp/traces")
  if [ -n "$segs" ]; then
    knobs+=(SEGMENTS="$segs" INPUTS="prose=$PROSE;code=$CODE" CHUNK="$W"
            TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT="$PROGRAMS_PACKED")
  else
    knobs+=(CHUNK="$W" CACHE="$h" TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT="$PROGRAMS_SINGLE")
    # the harness rounds the depth down to whole chunks: label per_op / the roofline with the depth profiled
    h=$(( h / W * W )); rsegs="$W:$h"
  fi
  { echo "run_id=$id"; echo "git_sha=$(git -C "$TT_METAL_HOME" rev-parse HEAD)"
    echo "dirty=$(git -C "$TT_METAL_HOME" status --porcelain -uno | wc -l)"; echo "date=$(date -Is)"
    printf '%s\n' "${knobs[@]}"
    env | grep -E '^((MESH|HARNESS|STAGES|STAGE|FABRIC|LAYER_IDS|LEVEL|WARM_POINT|PREFIX_QUIET)=|(M3_|PROFILE_|TT_|EXPERT_|HF_))' | sort
  } > "$envf"

  echo "[p0a] $(date '+%F %T') START $id (W=$W h=$h B=$B input=$input${segs:+ segments=$segs})"
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
    # python -m tracy starts the harness in a session of its own, outside our process group.
    pkill -KILL -u "$(id -u)" -f 'models/demos/minimax_m3/tests/perf/profile_prefill.py|tools/profile_4x2.py' 2>/dev/null
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
  echo "[p0a] $(date '+%F %T') END $id status=$status elapsed=${elapsed}s wall=${wall:-?}ms capture=${mb:-?}MB csv=${csv:-none}"

  if [ "$status" = OK ] && [ -n "$csv" ]; then
    if "$TT_METAL_HOME/python_env/bin/python" "$PARSE" "$csv" --json "$prof/zones.json" \
         --per-device "$prof/per_device.json" --html "$prof/zones.html" > "$prof/parse.log" 2>&1; then
      "$TT_METAL_HOME/python_env/bin/python" "$PER_OP" --zones "$prof/zones.json" --per-device "$prof/per_device.json" \
        --W "$W" --h "$h" --B "$B" --input "$input" --mesh "$MESH" --segments "$rsegs" --run-id "$id" \
        --out "${PER_OP_CSV:-$RES/per_op.csv}" >> "$prof/parse.log" 2>&1 || status="$status/per_op_failed"
    else
      status="$status/parse_failed"
    fi
    tail -3 "$prof/parse.log"
  fi
  echo "$id,$status,$elapsed,$W,$h,$B,$input,\"$segs\",${wall:-},${mb:-},${csv:-}" >> "$RUNS"
}

echo "[p0a] $(date '+%F %T') ${#POINTS[@]} points, results in $RES"
for spec in "${POINTS[@]}"; do
  name="${spec%%|*}"
  if [ -n "${ONLY:-}" ] && [[ " $ONLY " != *" $name "* ]]; then continue; fi
  run_point "$spec"
done
echo "[p0a] $(date '+%F %T') batch done; per-op rows in $RES/per_op.csv, run table $RUNS"
