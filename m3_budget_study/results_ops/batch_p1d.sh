#!/bin/bash
# P1-D: pipeline overheads on 4 x (2,4) (SP=2, TP=4, EP=8) with the common prefill runner, full 60 layers
# (even split 15/15/15/15; LAYER_COUNTS=1,1,1,57 gives the dense-heavy stage shapes instead).
#
# Sessions (one runner per session; producers run one after another against it, the last sends SHUTDOWN):
#   d_w{2048,4096,8192}_sync1   PREFILL_SYNC_PER_CHUNK=1: per-stage compute; cold and hot-139264 streams
#   ab_lease_w4096_sync0        un-synced throughput, lease mode (today's runner): K=2/4/open, cold + hot
#   ab_own_w4096_sync0          same streams, PREFILL_D2D_SHARE_FABRIC_LINKS=0 (async handoff: D2D services own
#                               their links, the send overlaps the next compute). Experimental, runs last.
# Every session has PREFILL_PP_TIMING=1 (default-off runner instrumentation): per-chunk lease waits (the
# blocking send = push + lease_out), input wait, enqueue, compute, wall stamps for the hop.
# Hot = PREFILL_PRODUCER_PREFIX_TOKENS=139264 (synthetic prefix, the history KV is never written).
# Fixed env: M3_MOE_W_NDSHARD=1 M3_MOE_HYBRID_THRESHOLD=128, v1 dispatch/combine; PREFILL_FABRIC_MODE=2d as in the
# 4-rank manifest (torus MoE ops do not apply at SP=2). Tokens: longbook_56320 (M3 ids).
#
# Per session <id>: pipeline/<id>/{binding.yaml,runner.log,<stream>.log,reset.log,session.env}, a row in runs.csv.
# Readout: tools/pipeline_overheads.py pipeline -> pipeline_overheads.txt (run at the end of the batch).
# Guards: .lock must name owner=vmelnykov-ops-agent; no other M3 harness / runner process running.
# Watchdog: RUNNER_UP_TIMEOUT s for the runner to come up, then STALL_TIMEOUT s with no runner/producer log
# output or RUN_TIMEOUT s per session in total -> kill, reset.
#
#   nohup m3_budget_study/results_ops/batch_p1d.sh > m3_budget_study/results_ops/logs/batch_p1d.out 2>&1 &
#   ONLY="d_w4096_sync1" ...     run a subset;   DRY_RUN=1 (or --dry-run): print the plan, write nothing
#   LAYER_COUNTS=1,1,1,57 PREFIX=p1d57 ...   the dense-heavy split (session ids get the PREFIX)
set -uo pipefail
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1
DRY_RUN="${DRY_RUN:-0}"
RES="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TT_METAL_HOME="${TT_METAL_HOME:-$(cd "$RES/../.." && pwd)}"
LOCK="$RES/.lock" LOCK_OWNER="owner=vmelnykov-ops-agent"
PIPE="$RES/pipeline" PY="$TT_METAL_HOME/python_env/bin/python"
PREFIX="${PREFIX:-}" LAYER_COUNTS="${LAYER_COUNTS:-}"
RUNNER_UP_TIMEOUT="${RUNNER_UP_TIMEOUT:-1200}" STALL_TIMEOUT="${STALL_TIMEOUT:-300}" RUN_TIMEOUT="${RUN_TIMEOUT:-1200}"
TT_CACHE_PATH=/mnt/weka/model-cache/scratch/minimax/MiniMax-M3-cache/prefill
TRACE_DIR="$TT_CACHE_PATH/golden/longbook_56320"
MAX_SEQ=557056   # capacity per slot (2 users), as in the SP2 Part B 4-rank sessions
HOT=139264
DEVS=("0,1,2,3,4,5,6,7" "24,25,26,27,28,29,30,31" "16,17,18,19,20,21,22,23" "8,9,10,11,12,13,14,15")
MGD=tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_4x4x2_z_chain_graph_descriptor.textproto

G=PREFILL_PRODUCER_MAX_IN_FLIGHT
sync_streams () {
  echo "warm|PREFILL_PRODUCER_MAX_REQUESTS=4 $G=10000"
  echo "cold_open|PREFILL_PRODUCER_MAX_REQUESTS=24 $G=10000"
  echo "h139_warm|PREFILL_PRODUCER_MAX_REQUESTS=4 PREFILL_PRODUCER_PREFIX_TOKENS=$HOT $G=10000"
  echo "h139_open|PREFILL_PRODUCER_MAX_REQUESTS=24 PREFILL_PRODUCER_PREFIX_TOKENS=$HOT $G=10000"
}
ab_streams () {  # K=2 first: at most 2 chunks in flight never lets a stage overwrite an unsent OWN-mode backing
  echo "warm|PREFILL_PRODUCER_MAX_REQUESTS=8 $G=10000"
  echo "cold_k2|PREFILL_PRODUCER_MAX_REQUESTS=48 $G=2"
  echo "cold_k4|PREFILL_PRODUCER_MAX_REQUESTS=48 $G=4"
  echo "cold_open|PREFILL_PRODUCER_MAX_REQUESTS=48 $G=10000"
  echo "h139_warm|PREFILL_PRODUCER_MAX_REQUESTS=4 PREFILL_PRODUCER_PREFIX_TOKENS=$HOT $G=10000"
  echo "h139_k4|PREFILL_PRODUCER_MAX_REQUESTS=48 PREFILL_PRODUCER_PREFIX_TOKENS=$HOT $G=4"
  echo "h139_open|PREFILL_PRODUCER_MAX_REQUESTS=48 PREFILL_PRODUCER_PREFIX_TOKENS=$HOT $G=10000"
}
pcc_streams () {  # KV read-back vs the golden (cold, real tokens [0, W)): only the last request per slot is checked
  echo "warm|PREFILL_PRODUCER_MAX_REQUESTS=4 $G=10000"
  echo "pcc_open|PREFILL_PRODUCER_MAX_REQUESTS=24 $G=10000 PREFILL_PRODUCER_CHECK_PCC=1"
}
h16k_streams () {  # P1-E (batch_p1e.sh): dense stages at h=16384, plus a cold control
  echo "warm|PREFILL_PRODUCER_MAX_REQUESTS=4 $G=10000"
  echo "cold_open|PREFILL_PRODUCER_MAX_REQUESTS=12 $G=10000"
  echo "h16k_warm|PREFILL_PRODUCER_MAX_REQUESTS=4 PREFILL_PRODUCER_PREFIX_TOKENS=16384 $G=10000"
  echo "h16k_open|PREFILL_PRODUCER_MAX_REQUESTS=24 PREFILL_PRODUCER_PREFIX_TOKENS=16384 $G=10000"
}
# pcc_* sessions: merged mock migration publishes the KV chunk table and per-rank device maps; the batch merges the
# maps (all 32 chips are local) so a single-process producer can read every stage's KV back.
PCC_ENV="PREFILL_ENABLE_MIGRATION=1 PREFILL_MOCK_MIGRATION=1 PREFILL_MIGRATION_DEVICE_MAP_PATH=@D@/kv_device_map.json"
# id | W | sync | share_fabric_links | stream generator [| extra K=V ... overriding the runner + producer env]
SESSIONS=(
  "d_w2048_sync1|2048|1|1|sync_streams"
  "d_w4096_sync1|4096|1|1|sync_streams"
  "d_w8192_sync1|8192|1|1|sync_streams"
  "ab_lease_w4096_sync0|4096|0|1|ab_streams"
  "pcc_lease_w4096_sync0|4096|0|1|pcc_streams|$PCC_ENV"
  "ab_own_w4096_sync0|4096|0|0|ab_streams"
  "pcc_own_w4096_sync0|4096|0|0|pcc_streams|$PCC_ENV"
)
# P1D_SESSIONS (newline-separated specs) replaces the list, e.g. from batch_p1e.sh.
if [ -n "${P1D_SESSIONS:-}" ]; then mapfile -t SESSIONS <<< "$P1D_SESSIONS"; fi

die () { echo "[p1d] ERROR: $*" >&2; exit 1; }
busy () {  # another M3 harness, a runner, or a producer running (sibling session / other agent)?
  pgrep -u "$(id -u)" -af 'models/demos/minimax_m3/tests/perf/[a-z_]+\.py|tracy-capture|prefill\.runners\.prefill_(runner|producer)|ttrun\.py' \
    | grep -v "^$$ " || true
}
kill_session () {
  pkill -TERM -u "$(id -u)" -f 'models.demos.common.prefill.runners.prefill_(runner|producer)' 2>/dev/null; sleep 15
  pkill -KILL -u "$(id -u)" -f 'models.demos.common.prefill.runners.prefill_(runner|producer)|ttnn/distributed/ttrun.py' 2>/dev/null
}

run_env () {  # $1 = W, $2 = sync, $3 = share, $4 = extra "K=V ..."; prints the runner's global_env, one K=V a line
  local -A e=() ; local -a keys=() kv
  for kv in PREFILL_FABRIC_MODE=2d PREFILL_SP=2 PREFILL_TP=4 PREFILL_NUM_LAYERS=60 PREFILL_MAX_SEQ_LEN=$MAX_SEQ \
            PREFILL_CHUNK_SIZE="$1" PREFILL_NUM_USERS=2 PREFILL_SYNC_PER_CHUNK="$2" PREFILL_TRACE_DIR="$TRACE_DIR" \
            TT_CACHE_PATH="$TT_CACHE_PATH" PREFILL_MODEL=minimax_m3 M3_INDEX_CACHE_BF16=1 \
            PREFILL_H2D_SERVICE_ID=ds_prefill PREFILL_PP_D2D_FIFO_BYTES=32768 LOGURU_LEVEL=INFO \
            PREFILL_PP_TIMING=1 PREFILL_D2D_SHARE_FABRIC_LINKS="$3" \
            M3_MOE_W_NDSHARD=1 M3_MOE_HYBRID_THRESHOLD=128 M3_MOE_DISPATCH=v1 M3_MOE_COMBINE=v1 EXPERT_DTYPE=bf4 \
            ${LAYER_COUNTS:+PREFILL_PP_LAYER_COUNTS=$LAYER_COUNTS} $4; do
    [ -z "${e[${kv%%=*}]+x}" ] && keys+=("${kv%%=*}")
    e[${kv%%=*}]="${kv#*=}"
  done
  for kv in "${keys[@]}"; do echo "$kv=${e[$kv]}"; done
}

write_binding () {  # $1 = file, $2 = W, $3 = sync, $4 = share, $5 = extra
  {
    echo "rank_bindings:"
    for r in 0 1 2 3; do
      echo "  - {rank: $r, mesh_id: $r, mesh_host_rank: 0, env_overrides: {TT_VISIBLE_DEVICES: \"${DEVS[$r]}\"}}"
    done
    echo; echo "mesh_graph_desc_path: $MGD"; echo; echo "global_env:"
    run_env "$2" "$3" "$4" "$5" | while IFS= read -r kv; do echo "  ${kv%%=*}: \"${kv#*=}\""; done
  } > "$1"
}

producer () {  # $1 = dir, $2 = name, $3 = binding.yaml (runner env), rest = stream env
  local d=$1 name=$2 bind=$3; shift 3
  echo "[p1d] $(date '+%F %T') producer $(basename "$d")/$name $*"
  local -a renv=()
  mapfile -t renv < <(sed -n '/^global_env:/,$p' "$bind" | tail -n +2 | sed 's/^  //; s/: "/=/; s/"$//' \
    | grep -E '^(PREFILL_(SP|TP|NUM_LAYERS|CHUNK_SIZE|MAX_SEQ_LEN|NUM_USERS|TRACE_DIR|MODEL|H2D_SERVICE_ID)|LOGURU_LEVEL)=')
  env "${renv[@]}" PREFILL_PRODUCER_INTERLEAVE=round_robin PREFILL_PRODUCER_CHUNKS=1 "$@" \
    timeout "${PRODUCER_TIMEOUT:-600}" python3 -m models.demos.common.prefill.runners.prefill_producer > "$d/$name.log" 2>&1
  local rc=$?
  echo "[p1d]   exit=$rc $(grep -h 'DONE wall' "$d/$name.log" | tail -1 | grep -oE 'DONE wall=[^ ]+ pushes=[0-9]+ .*tok/s')"
  return $rc
}

watchdog () {  # $1 = runner pid, $2 = dir, $3 = t0; kills the session on stall / timeout
  local rpid=$1 d=$2 t0=$3
  while kill -0 "$rpid" 2>/dev/null; do
    sleep 15
    local now newest age; now=$(date +%s)
    newest=$(stat -c %Y "$d"/*.log 2>/dev/null | sort -n | tail -1); age=$(( now - ${newest:-now} ))
    if [ $(( now - t0 )) -gt "$RUN_TIMEOUT" ]; then echo TIMEOUT > "$d/watchdog"; kill_session; return; fi
    if grep -q "\[h2d\] descriptor" "$d/runner.log" && [ "$age" -gt "$STALL_TIMEOUT" ]; then
      echo HANG > "$d/watchdog"; kill_session; return
    fi
  done
}

session () {  # $1 = spec
  local id W sync share gen extra
  IFS='|' read -r id W sync share gen extra <<< "$1"
  id="${PREFIX:+${PREFIX}_}$id"
  local d="$PIPE/$id"
  extra="${extra//@D@/$d}"
  mapfile -t S < <($gen)
  if [ "$DRY_RUN" = 1 ]; then
    echo "[p1d] DRY $id: W=$W sync=$sync share_fabric_links=$share layer_counts=${LAYER_COUNTS:-even} extra=${extra:-}"
    local tmp; tmp=$(mktemp); write_binding "$tmp" "$W" "$sync" "$share" "${extra:-}"; sed 's/^/      /' "$tmp"; rm -f "$tmp"
    printf '      stream %s\n' "${S[@]}"
    return 0
  fi
  if [ -e "$d/runner.log" ]; then echo "[p1d] $id: $d/runner.log exists, skipping"; return 0; fi
  grep -q "$LOCK_OWNER" "$LOCK" 2>/dev/null || die "$LOCK does not name $LOCK_OWNER; stopping before $id"
  local b; b="$(busy)"; [ -n "$b" ] && die "another harness is running, stopping before $id:"$'\n'"$b"
  mkdir -p "$d"; write_binding "$d/binding.yaml" "$W" "$sync" "$share" "${extra:-}"
  { echo "run_id=$id"; echo "git_sha=$(git -C "$TT_METAL_HOME" rev-parse HEAD)"
    echo "dirty=$(git -C "$TT_METAL_HOME" status --porcelain -uno | awk '{print $2}' | tr '\n' ' ')"; echo "date=$(date -Is)"
    sed -n '/^global_env:/,$p' "$d/binding.yaml" | tail -n +2 | sed 's/^  //; s/: "/=/; s/"$//'
    printf 'stream=%s\n' "${S[@]}"; } > "$d/session.env"
  local start; start=$(date -u +%FT%TZ)
  echo "[p1d] $(date '+%F %T') START $id (W=$W sync=$sync share_fabric_links=$share)"
  env -u TT_VISIBLE_DEVICES tt-smi -glx_reset > "$d/reset.log" 2>&1 || { echo "[p1d] reset failed"; return 1; }
  ( cd "$TT_METAL_HOME" && exec setsid ./models/demos/common/prefill/runners/run_pipeline_prefill.sh "$d/binding.yaml" \
      "$(hostname -s):4" ) > "$d/runner.log" 2>&1 &
  local rpid=$! t0; t0=$(date +%s)
  local status="" up=0
  until grep -q "\[h2d\] descriptor" "$d/runner.log"; do
    sleep 10
    if ! kill -0 "$rpid" 2>/dev/null; then status=RUNNER_DIED; break; fi
    if [ $(( $(date +%s) - t0 )) -gt "$RUNNER_UP_TIMEOUT" ]; then status=RUNNER_UP_TIMEOUT; break; fi
  done
  local notes=""
  if [ -z "$status" ]; then
    up=$(( $(date +%s) - t0 )); echo "[p1d] runner $id up after ${up}s"
    watchdog "$rpid" "$d" "$t0" & local wpid=$!
    local pmap=""
    if [[ "${extra:-}" == *PREFILL_MOCK_MIGRATION=1* ]]; then
      local k=0; until [ "$(ls "$d"/kv_device_map_r*.json 2>/dev/null | wc -l)" -ge 4 ] || [ $k -ge 30 ]; do sleep 2; k=$((k+1)); done
      if "$PY" - "$d" <<'PY'
import glob, json, sys
d = sys.argv[1]
m = {}
for f in sorted(glob.glob(f"{d}/kv_device_map_r*.json")):
    m.update(json.load(open(f)))
json.dump(m, open(f"{d}/kv_device_map_merged.json", "w"))
print(f"[p1d] merged {len(m)} chips into {d}/kv_device_map_merged.json")
PY
      then pmap="PREFILL_MIGRATION_DEVICE_MAP_PATH=$d/kv_device_map_merged.json"; fi
    fi
    local n=${#S[@]} i=0 spec
    for spec in "${S[@]}"; do
      i=$((i + 1)); local name=${spec%%|*} envs=${spec#*|}
      [ $i -eq "$n" ] && envs="$envs PREFILL_SEND_SHUTDOWN=1"
      [ -e "$d/watchdog" ] && break
      # shellcheck disable=SC2086
      producer "$d" "$name" "$d/binding.yaml" $envs $pmap; notes="$notes $name:$?"
      local pcc; pcc="$(grep -hoE 'kv_cache_pcc_complete .*|KV cache PCC (PASSED|below)[^(]*' "$d/$name.log" | head -2 | tr '\n' ' ')"
      [ -n "$pcc" ] && { echo "[p1d]   $pcc"; notes="$notes [$pcc]"; }
    done
    local t1; t1=$(date +%s)
    while kill -0 "$rpid" 2>/dev/null && [ $(( $(date +%s) - t1 )) -lt 300 ]; do sleep 5; done
    kill -0 "$rpid" 2>/dev/null && { echo "[p1d] runner $id still up; killing"; kill_session; }
    kill "$wpid" 2>/dev/null; wait "$wpid" 2>/dev/null
    status="$(cat "$d/watchdog" 2>/dev/null)"; [ -z "$status" ] && status=OK
    [[ "$notes" =~ :[1-9] ]] && [ "$status" = OK ] && status=PRODUCER_ERROR
  else
    kill_session
  fi
  wait "$rpid" 2>/dev/null
  env -u TT_VISIBLE_DEVICES tt-smi -glx_reset >> "$d/reset.log" 2>&1
  local end; end=$(date -u +%FT%TZ)
  echo "[p1d] $(date '+%F %T') END $id status=$status up=${up}s streams:${notes}"
  local sha envline
  sha="$(grep '^git_sha=' "$d/session.env" | cut -d= -f2)"
  local dirty; dirty="$(grep '^dirty=' "$d/session.env" | cut -d= -f2- | sed 's/ $//')"
  [ -n "$dirty" ] && sha="$sha dirty($(echo "$dirty" | wc -w): $dirty)"
  envline="$(grep -vE '^(run_id|git_sha|dirty|date|stream)=' "$d/session.env" | sort | tr '\n' ' ' | sed 's/ $//')"
  "$PY" - "$RES/runs.csv" "$id" "$start" "$end" "$sha" "$envline" \
    "run_pipeline_prefill.sh pipeline/$id/binding.yaml $(hostname -s):4 + producers ($(printf '%s;' "${S[@]}"))" \
    "STATUS=$status" "m3_budget_study/results_ops/pipeline/$id/runner.log" "P1D runner up ${up}s; streams$notes" <<'PY'
import csv, sys
out, rid, start, end, sha, env, cmd, status, log, notes = sys.argv[1:]
with open(out, "a", newline="") as f:
    csv.writer(f).writerow([rid, "P1D", start, end, sha, env, cmd, status, log, notes])
PY
}

ulimit -Su "$(ulimit -Hu)" 2>/dev/null
unset $(env | sed -n 's/^\(SLURM[^=]*\)=.*/\1/p') 2>/dev/null
export PRTE_MCA_ras="^slurm" PRTE_MCA_plm="^slurm"
cd "$TT_METAL_HOME" && source python_env/bin/activate && export PYTHONPATH="$TT_METAL_HOME"
export PREFILL_MANIFEST=models/demos/minimax_m3/tt/runners/manifests/minimax_m3.json PREFILL_MODEL=minimax_m3
unset TT_VISIBLE_DEVICES
mkdir -p "$PIPE" "$RES/logs"
echo "[p1d] $(date '+%F %T') ${#SESSIONS[@]} sessions, results in $PIPE$([ "$DRY_RUN" = 1 ] && echo ' (dry run)')"
for spec in "${SESSIONS[@]}"; do
  name="${spec%%|*}"
  if [ -n "${ONLY:-}" ] && [[ " $ONLY " != *" $name "* ]]; then continue; fi
  session "$spec"
done
[ "$DRY_RUN" = 1 ] && exit 0
"$PY" "$RES/tools/pipeline_overheads.py" "$PIPE" --out "$RES/pipeline_overheads.txt" || echo "[p1d] readout failed"
echo "[p1d] $(date '+%F %T') batch done"
