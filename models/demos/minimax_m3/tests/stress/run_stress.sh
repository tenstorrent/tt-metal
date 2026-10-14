#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
# MiniMax-M3 prefill stress test on one Blackhole galaxy (a slice of the SC24 P/D prefill pipeline). See README.md.
# Exit: 0 = no hang, 3 = hang captured, 1 = refused / failed to start.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARGS="$*"
export TT_METAL_HOME="${TT_METAL_HOME:-$(git -C "$HERE" rev-parse --show-toplevel)}"
WEKA=/mnt/weka/model-cache/scratch/minimax/MiniMax-M3-cache/prefill

PROFILE=paced STAGES=4 LAYERS="" FIRST_LAYER=37 LPS=2 USERS=90 MAXSEQ=1048576 CHUNK=2048 FABRIC=2d
DURATION=2h SEED=59542 WATCHER=0 STALL_S=240 OUT="" LABEL="" TCP_IFACE="" KCACHE="" RESET=1 HARD_STOP=""
READY_S=3600 EXIT_GRACE_S=600 COLD_STALL_S=120 DRY=0
CACHE_PATH="${TT_CACHE_PATH:-$WEKA}" TRACE_DIR="${PREFILL_TRACE_DIR:-$WEKA/golden/longbook_56320}"
EXTRA_ENV=()

usage() {
  cat <<EOF
Usage: $0 [options]        defaults = SC24 production slice (4 stages x 2 MoE layers, layers 37-44)

Pipeline
  --stages N              pipeline stages, one [2,4] mesh (8 chips) each, 1-4 (default $STAGES)
  --layers "A,B C,D ..."  absolute model layers per stage (overrides --first-layer / --layers-per-stage / --stages)
  --first-layer L         first absolute layer; stages take consecutive layers (default $FIRST_LAYER; 45 = SC24 ranks 24-27)
  --layers-per-stage N    (default $LPS)
  --users N               KV slots (PREFILL_NUM_USERS, default $USERS)
  --max-seq-len N         per-user KV capacity (default $MAXSEQ)
  --chunk-size N          (default $CHUNK)
  --fabric MODE           PREFILL_FABRIC_MODE (default $FABRIC)
Traffic
  --profile P             paced (production duty cycle) | agentx (saturated) | cold (cold-start cycles) | custom
                          (stock producer schedule, PREFILL_PRODUCER_* via --env). Default $PROFILE
  --duration T            load time, e.g. 5400, 90m, 2h (startup/shutdown not included; cold: total time). Default $DURATION
  --seed N                producer seed (default $SEED)
  --conc LO,HI            agentx: active sessions, redrawn every STRESS_CONC_PERIOD_S (default 56,90)
  --conc-phases A,B,..    paced: concurrency phases (default 2,8,30,90)
  --phase-s S             paced: seconds per phase (default 900)
  --depth MED,SIGMA,MIN,MAX  session start depth, lognormal tokens (default 300000,0.55,32768,746000)
  --fresh-prob P          share of sessions starting at 0 (default 0.1)
  --turn MED,SIGMA,MAX    new tokens per turn, lognormal (default 1600,1.0,30000)
  --think MED,MAX         agentx: seconds between a session's turns (default 2,60)
  --gap MED,SIGMA,MAX     paced: idle gap seconds between bursts, lognormal (default 3,1.0,120)
  --long-idle-prob P      paced: share of gaps that are long idles (default 0.006)
  --long-idle-s LO,HI     paced: long idle seconds (default 120,600)
  --cold TINY,NORMAL      cold: requests per cycle (default 6,3)
  --custom-depth LO,HI    custom: fresh requests start at a random depth in [LO,HI)
Run control
  --env K=V               extra env for runner (binding global_env) and producer; repeatable. Bisect knobs, e.g.
                          M3_MOE_HYBRID_THRESHOLD=0, PREFILL_OVERLAP_SHARED_EXPERT=0, any STRESS_* traffic knob
  --watcher               watcher env that works with fabric (see README) and its own kernel cache
  --watchdog-stall-s S    capture after S s without progress while work is outstanding; 0 = off (default $STALL_S)
  --hard-stop T           absolute end (epoch or HH:MM UTC today): everything is stopped and reset by then
  --out DIR               run directory (default \$TT_METAL_HOME/generated/m3_stress/<label>_<utc>)
  --label NAME            (default <profile>)
  --kernel-cache DIR      TT_METAL_CACHE (default per HEAD + tree diff + watcher under generated/m3_stress/kernel_cache)
  --cache-path DIR        TT_CACHE_PATH weight cache (default $CACHE_PATH)
  --trace-dir DIR         PREFILL_TRACE_DIR token source (default $TRACE_DIR)
  --tcp-interface IF      MPI TCP interface (default: the default-route interface)
  --ready-timeout-s S     runner startup limit (default $READY_S)
  --no-reset              skip tt-smi -glx_reset in preflight and at the end
  --dry-run               write binding/env, print the commands and the tt-run dry run; no device access
EOF
}

die() { echo "run_stress: $*" >&2; exit 1; }
seconds() {
  local t=$1
  case "$t" in *h) echo $(( ${t%h} * 3600 ));; *m) echo $(( ${t%m} * 60 ));; *s) echo "${t%s}";; *) echo "$t";; esac
}
add_env() { EXTRA_ENV+=("$1=$2"); }

while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) usage; exit 0;;
    --stages) STAGES=$2; shift;;
    --layers) LAYERS=$2; shift;;
    --first-layer) FIRST_LAYER=$2; shift;;
    --layers-per-stage) LPS=$2; shift;;
    --users) USERS=$2; shift;;
    --max-seq-len) MAXSEQ=$2; shift;;
    --chunk-size) CHUNK=$2; shift;;
    --fabric) FABRIC=$2; shift;;
    --profile) PROFILE=$2; shift;;
    --duration) DURATION=$2; shift;;
    --seed) SEED=$2; shift;;
    --conc) add_env STRESS_CONC "$2"; shift;;
    --conc-phases) add_env STRESS_CONC_PHASES "$2"; shift;;
    --phase-s) add_env STRESS_PHASE_S "$2"; shift;;
    --depth) add_env STRESS_DEPTH "$2"; shift;;
    --fresh-prob) add_env STRESS_FRESH_PROB "$2"; shift;;
    --turn) add_env STRESS_TURN "$2"; shift;;
    --think) add_env STRESS_THINK "$2"; shift;;
    --gap) add_env STRESS_GAP "$2"; shift;;
    --long-idle-prob) add_env STRESS_LONG_IDLE_PROB "$2"; shift;;
    --long-idle-s) add_env STRESS_LONG_IDLE_S "$2"; shift;;
    --cold) add_env STRESS_COLD "$2"; shift;;
    --custom-depth) add_env STRESS_CUSTOM_DEPTH "$2"; shift;;
    --env) [[ "$2" == *=* ]] || die "--env wants K=V, got '$2'"; EXTRA_ENV+=("$2"); shift;;
    --watcher) WATCHER=1;;
    --watchdog-stall-s) STALL_S=$2; shift;;
    --hard-stop) HARD_STOP=$2; shift;;
    --out) OUT=$2; shift;;
    --label) LABEL=$2; shift;;
    --kernel-cache) KCACHE=$2; shift;;
    --cache-path) CACHE_PATH=$2; shift;;
    --trace-dir) TRACE_DIR=$2; shift;;
    --tcp-interface) TCP_IFACE=$2; shift;;
    --ready-timeout-s) READY_S=$2; shift;;
    --no-reset) RESET=0;;
    --dry-run) DRY=1;;
    *) usage >&2; die "unknown option '$1'";;
  esac
  shift
done
case "$PROFILE" in paced|agentx|cold|custom) ;; *) die "unknown profile '$PROFILE'";; esac

# ---- environment
cd "$TT_METAL_HOME" || die "TT_METAL_HOME=$TT_METAL_HOME not found"
[ -z "${VIRTUAL_ENV:-}" ] && [ -f python_env/bin/activate ] && source python_env/bin/activate
export PYTHONPATH="$TT_METAL_HOME${PYTHONPATH:+:$PYTHONPATH}"
unset $(env | sed -n 's/^\(SLURM[^=]*\)=.*/\1/p')
export PRTE_MCA_ras="^slurm" PRTE_MCA_plm="^slurm"
ulimit -u unlimited 2>/dev/null || true

LOAD_S=$(seconds "$DURATION")
NOW=$(date +%s)
if [ -n "$HARD_STOP" ]; then
  case "$HARD_STOP" in *:*) STOP_AT=$(date -u -d "today $HARD_STOP" +%s);; *) STOP_AT=$HARD_STOP;; esac
else
  STOP_AT=$(( NOW + LOAD_S + READY_S + 2 * EXIT_GRACE_S + 3600 ))
fi
LABEL="${LABEL:-$PROFILE}"
OUT="$(realpath -m "${OUT:-$TT_METAL_HOME/generated/m3_stress/${LABEL}_$(date -u +%Y%m%dT%H%M%SZ)}")"
mkdir -p "$OUT" || die "cannot create $OUT"
MODE=$([ "$WATCHER" = 1 ] && echo watcher || echo prod)
TREE="$(git rev-parse --short=10 HEAD)-$(git diff HEAD | md5sum | cut -c1-8)"
KCACHE="${KCACHE:-$TT_METAL_HOME/generated/m3_stress/kernel_cache/$TREE-$MODE}"
SERVICE_ID="m3_stress_$(id -u)"
RUN_ID="m3_stress_${LABEL}_${NOW}_$$"
[ -z "$TCP_IFACE" ] && TCP_IFACE=$(ip -o route get 1.1.1.1 2>/dev/null | sed -n 's/.* dev \([^ ]*\).*/\1/p')
RESET_CMD="${STRESS_RESET_CMD:-env -u TT_VISIBLE_DEVICES tt-smi -glx_reset}"

event() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/events.log" >&2; }

gen_binding() {  # <run dir>
  local args=(--run-dir "$1" --tt-metal-home "$TT_METAL_HOME" --first-layer "$FIRST_LAYER" --layers-per-stage "$LPS"
              --stages "$STAGES" --users "$USERS" --max-seq-len "$MAXSEQ" --chunk-size "$CHUNK" --fabric "$FABRIC"
              --profile "$PROFILE" --seed "$SEED" --service-id "$SERVICE_ID" --run-id "$RUN_ID"
              --cache-path "$CACHE_PATH" --trace-dir "$TRACE_DIR" --kernel-cache "$KCACHE")
  [ -n "$LAYERS" ] && args+=(--layers "$LAYERS")
  [ "$WATCHER" = 1 ] && args+=(--watcher)
  local kv; for kv in "${EXTRA_ENV[@]}"; do args+=(--env "$kv"); done
  python3 "$HERE/make_binding.py" "${args[@]}"
}

runner_cmd() {  # <run dir> <stages>
  local tcp=(); [ -n "$TCP_IFACE" ] && tcp=(--tcp-interface "$TCP_IFACE")
  echo python3 "$TT_METAL_HOME/ttnn/ttnn/distributed/ttrun.py" "${tcp[@]}" --rank-binding "$1/binding.yaml" \
    --mpi-args "\"--host $(hostname -s):$2 --map-by slot --bind-to none --tag-output --allow-run-as-root -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH\"" \
    -- python3 "$HERE/stress_runner.py"
}

own_pids() {  # every process of this run (they all carry STRESS_RUN_ID)
  grep -lzx "STRESS_RUN_ID=$RUN_ID" /proc/[0-9]*/environ 2>/dev/null | cut -d/ -f3 | grep -vx "$$"
}

kill_own() {
  local p; p=$(own_pids | tr '\n' ' ')
  [ -z "${p// }" ] && return 0
  kill -TERM $p 2>/dev/null
  for _ in $(seq 30); do sleep 1; [ -z "$(own_pids)" ] && break; done
  p=$(own_pids | tr '\n' ' '); [ -n "${p// }" ] && { event "SIGKILL $p"; kill -KILL $p 2>/dev/null; sleep 3; }
  rm -f /dev/shm/*"$SERVICE_ID"* 2>/dev/null
}

device_holders() {
  local p
  for p in $(lsof -t /dev/tenstorrent/* 2>/dev/null | sort -u); do
    echo "  pid $p ($(ps -o user= -p "$p")): $(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | cut -c1-160)"
  done
}

reset_glx() {
  [ "$RESET" = 1 ] || return 0
  local h; h=$(device_holders)
  [ -n "$h" ] && { event "NOT resetting, devices held by:"; echo "$h" | tee -a "$OUT/events.log" >&2; return 1; }
  event "reset: $RESET_CMD"
  $RESET_CMD >> "$OUT/reset.log" 2>&1 || { event "reset FAILED (see $OUT/reset.log)"; return 1; }
}

preflight() {
  command -v lsof >/dev/null || event "WARNING: lsof not found, cannot check for other device users"
  local h; h=$(device_holders)
  [ -n "$h" ] && { event "PREFLIGHT REFUSED: devices held by:"; echo "$h" | tee -a "$OUT/events.log" >&2; return 1; }
  if pgrep -u "$(id -u)" -af "prterun|ttrun\.py|pytest" | grep -v pgrep | grep -q .; then
    event "PREFLIGHT REFUSED: another of your MPI/pytest jobs is running:"
    pgrep -u "$(id -u)" -af "prterun|ttrun\.py|pytest" | grep -v pgrep | tee -a "$OUT/events.log" >&2; return 1
  fi
  [ $(( STOP_AT - $(date +%s) )) -gt 900 ] || { event "PREFLIGHT REFUSED: less than 15 min to the hard stop"; return 1; }
  reset_glx
}

ncount() { local c; c=$(grep -c "$1" "$2" 2>/dev/null); echo "${c:-0}"; }
count() {  # <run dir> <stages> -> "<pushes> <rank0> <rank1> ..."
  local d=$1 out r
  out=$(ncount "\[producer\] push slot" "$d/producer.log")
  for r in $(seq 0 $(( $2 - 1 ))); do out="$out $(ncount "\[pp rank $r\] CHUNK_START" "$d/runner.log")"; done
  echo "$out"
}
all_equal() { [ "$1" -gt 0 ] && [ -z "$(tr ' ' '\n' <<< "$2" | grep -vx "$1")" ]; }  # <pushes> "<rank counts>"

capture() {  # <run dir> <why>
  python3 "$HERE/stall_watchdog.py" capture "$1" --why "$2" --events "$OUT/events.log" > "$1/capture.log" 2>&1
}

hang_dir_state() {  # <run dir> -> "" | busy | done
  local d
  for d in "$1"/hang_*; do
    [ -d "$d" ] || continue
    [ -e "$d/CAPTURE_DONE" ] && { echo done; return; }
    echo busy; return
  done
}

start_runner() {  # <run dir> <stages>; sets RPID
  rm -f /dev/shm/*"$SERVICE_ID"* 2>/dev/null
  eval "env STRESS_RUN_ID=$RUN_ID TT_METAL_CACHE=$KCACHE setsid $(runner_cmd "$1" "$2")" > "$1/runner.log" 2>&1 &
  RPID=$!
}

wait_ready() {  # <run dir> <stages> -> 0 ready, 1 runner died, 2 timeout, 3 hard stop
  local t0; t0=$(date +%s)
  while kill -0 "$RPID" 2>/dev/null; do
    [ "$(ncount 'request (unbounded) loop start' "$1/runner.log")" -ge "$2" ] && return 0
    [ $(( $(date +%s) - t0 )) -gt "$READY_S" ] && return 2
    [ "$(date +%s)" -ge "$STOP_AT" ] && return 3
    sleep 5
  done
  return 1
}

start_producer() {  # <run dir> <seconds>; sets PPID_
  env STRESS_RUN_ID="$RUN_ID" PREFILL_PRODUCER_DURATION_S="$2" setsid bash -c \
    "source '$1/producer.env' && exec python3 '$HERE/stress_producer.py'" > "$1/producer.log" 2>&1 &
  PPID_=$!
}

# ---- one long run (paced / agentx / custom). Sets RESULT.
run_load() {
  local d=$OUT n rc p rest t_pdone=0
  n=$(gen_binding "$d") || { RESULT=bad-config; return; }
  preflight || { RESULT=refused; return; }
  event "runner starting: $n stages, layers $(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["layers"])' "$d/run.json"), kernel cache $KCACHE"
  start_runner "$d" "$n"
  wait_ready "$d" "$n"; rc=$?
  [ $rc = 3 ] && { RESULT=hard-stop; return; }
  if [ $rc = 2 ]; then capture "$d" "runner not ready after ${READY_S}s"; RESULT=startup-stall; return; fi
  if [ $rc = 1 ]; then
    RESULT="startup-died: $(grep -m2 -E 'Error|Traceback|TT_FATAL' "$d/runner.log" | cut -c1-200 | tr '\n' ' ')"; return
  fi
  local load=$LOAD_S left=$(( STOP_AT - $(date +%s) - EXIT_GRACE_S - 900 ))
  [ "$load" -gt "$left" ] && load=$left
  [ "$load" -lt 60 ] && { RESULT=no-time; return; }
  event "runner ready ($(grep -m1 -o 'DRAM per chip after KV alloc.*' "$d/runner.log" | cut -c1-120)); load for ${load}s"
  if [ "$STALL_S" != 0 ]; then
    env STRESS_RUN_ID="$RUN_ID" setsid python3 "$HERE/stall_watchdog.py" watch "$d" --stall-s "$STALL_S" \
      --until-epoch "$STOP_AT" > "$d/watchdog.log" 2>&1 &
  fi
  start_producer "$d" "$load"
  while :; do
    sleep 10
    case "$(hang_dir_state "$d")" in
      done) RESULT=hang; return;;
      busy) [ "$(date +%s)" -lt "$STOP_AT" ] && continue;;  # never interrupt a capture before the hard stop
    esac
    [ "$(date +%s)" -ge "$STOP_AT" ] && { RESULT=hard-stop; return; }
    if ! kill -0 "$RPID" 2>/dev/null; then
      wait "$RPID"; rc=$?
      read -r p rest <<< "$(count "$d" "$n")"
      if [ "$rc" = 0 ] && all_equal "$p" "$rest"; then RESULT=ok; else RESULT="runner-exit rc=$rc pushes $p chunks $rest"; fi
      return
    fi
    if [ $t_pdone = 0 ] && ! kill -0 "$PPID_" 2>/dev/null; then
      t_pdone=$(date +%s); wait "$PPID_"; rc=$?
      [ "$rc" != 0 ] && { RESULT="producer-failed rc=$rc"; return; }
    fi
    if [ $t_pdone != 0 ] && [ $(( $(date +%s) - t_pdone )) -gt "$EXIT_GRACE_S" ]; then
      capture "$d" "runner did not exit ${EXIT_GRACE_S}s after the producer's SHUTDOWN"; RESULT=no-exit; return
    fi
  done
}

# ---- cold-start cycles: reset -> runner -> a few tiny/normal requests -> SHUTDOWN, until the time is up. Sets RESULT.
run_cold() {
  local end=$(( $(date +%s) + LOAD_S )) i=0 fails=0 c d n v rc p rest last cur t0 t_ready t_move t_pdone
  [ "$end" -gt $(( STOP_AT - 900 )) ] && end=$(( STOP_AT - 900 ))
  while [ $(( end - $(date +%s) )) -gt 600 ]; do
    i=$((i + 1)); c=$(printf "c%03d" $i); d="$OUT/$c"; t0=$(date +%s); t_pdone=0; v=""
    n=$(gen_binding "$d") || { RESULT=bad-config; return; }
    preflight || { echo "$c refused" >> "$OUT/cycles.txt"; RESULT=refused; return; }
    start_runner "$d" "$n"
    wait_ready "$d" "$n"; rc=$?
    [ $rc = 3 ] && { RESULT=hard-stop; return; }
    if [ $rc = 2 ]; then
      capture "$d" "runner not ready after ${READY_S}s"; echo "$c startup-stall" >> "$OUT/cycles.txt"; RESULT=hang; return
    elif [ $rc = 1 ]; then
      echo "$c startup-died" >> "$OUT/cycles.txt"; kill_own
      fails=$((fails + 1)); [ $fails -ge 3 ] && { RESULT="startup-died x3"; return; }
      continue
    fi
    fails=0; t_ready=$(( $(date +%s) - t0 ))
    start_producer "$d" 0
    last=$(count "$d" "$n"); t_move=$(date +%s)
    while :; do
      sleep 2
      cur=$(count "$d" "$n"); [ "$cur" != "$last" ] && { last=$cur; t_move=$(date +%s); }
      [ "$t_pdone" = 0 ] && ! kill -0 "$PPID_" 2>/dev/null && t_pdone=$(date +%s)
      if ! kill -0 "$RPID" 2>/dev/null; then
        wait "$RPID"; rc=$?
        read -r p rest <<< "$(count "$d" "$n")"
        if [ "$rc" = 0 ] && all_equal "$p" "$rest"; then v=ok; else v="runner-exit rc=$rc"; fi
        break
      fi
      [ "$t_pdone" = 0 ] && [ $(( $(date +%s) - t_move )) -gt "$COLD_STALL_S" ] && { v=stall; break; }
      [ "$t_pdone" != 0 ] && [ $(( $(date +%s) - t_pdone )) -gt 180 ] && { v=no-exit; break; }
      [ "$(date +%s)" -ge "$STOP_AT" ] && { v=hard-stop; break; }
    done
    read -r p rest <<< "$(count "$d" "$n")"
    echo "$c $v ready=${t_ready}s total=$(( $(date +%s) - t0 ))s pushes=$p chunks=${rest// //}" >> "$OUT/cycles.txt"
    case "$v" in
      ok) [ $((i % 5)) = 0 ] && event "cold: $i cycles, last ok";;
      stall|no-exit) capture "$d" "$v: pushes $p, CHUNK_START per rank $rest"; RESULT=hang; return;;
      hard-stop) RESULT=hard-stop; return;;
      *) event "cold $c: $v (see $d)";;
    esac
    kill_own
  done
  RESULT="ok ($i cycles)"
}

# ---- main
{
  echo "cmd: $0 $ARGS"
  echo "HEAD $(git rev-parse HEAD) ($(git status --porcelain --untracked-files=no | wc -l) tracked changes)"
  echo "profile=$PROFILE layers='${LAYERS:-$STAGES stages x $LPS from layer $FIRST_LAYER}' users=$USERS max_seq=$MAXSEQ chunk=$CHUNK"
  echo "fabric=$FABRIC load=${LOAD_S}s seed=$SEED watcher=$WATCHER stall_s=$STALL_S hard_stop=$(date -u -d "@$STOP_AT" +%FT%TZ)"
  echo "extra env: ${EXTRA_ENV[*]:-none}"
} > "$OUT/config.txt"

if [ "$DRY" = 1 ]; then
  n=$(gen_binding "$OUT") || exit 1
  for f in config.txt binding.yaml producer.env; do echo "== $OUT/$f"; cat "$OUT/$f"; done
  [ -f "$OUT/mesh_graph.textproto" ] && { echo "== $OUT/mesh_graph.textproto"; cat "$OUT/mesh_graph.textproto"; }
  echo "== runner:   $(runner_cmd "$OUT" "$n")"
  echo "== producer: (source $OUT/producer.env; PREFILL_PRODUCER_DURATION_S=$LOAD_S python3 $HERE/stress_producer.py)"
  echo "== tt-run --dry-run:"
  eval "$(runner_cmd "$OUT" "$n" | sed 's|ttrun.py |ttrun.py --dry-run |')" 2>&1 | tail -n 20
  exit 0
fi

RESULT=""
trap 'event "interrupted"; kill_own; reset_glx; exit 130' INT TERM HUP
event "start $LABEL ($OUT)"
if [ "$PROFILE" = cold ]; then run_cold; else run_load; fi
kill_own
[ "$RESULT" != refused ] && reset_glx
event "RESULT $RESULT"
python3 "$HERE/summarize_run.py" "$OUT"
case "$RESULT" in hang | no-exit | startup-stall) exit 3;; ok*) exit 0;; *) exit 1;; esac
