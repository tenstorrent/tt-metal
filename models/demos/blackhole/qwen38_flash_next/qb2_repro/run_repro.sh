#!/usr/bin/env bash
# Reproduce the Qwen3.8-Flash-Next QB2 perf numbers.   run_repro.sh STEP...
#   STEP: prewarm | prepare | acceptance | P1 | P2 | P3 | P4a | P4b | all
#   P1 plain | P2 --prefill-slab 2048 | P3 --mtp 4 --prefill-slab 2048 | P4a lanes 4 mtp 4 | P4b lanes 8 mtp 3 (wrapper)
# Env: HF_MODEL, TT_CACHE_PATH (required); QWEN38_PY; QWEN38_OUT (default ./qb2_repro_results/<utc stamp>);
#      QWEN38_PORT (8000); TT_METAL_HOME; QWEN38_DEVICE_LOCK (optional lock file, held with flock -n per server).
# No device lock, no other tt process may run during these steps. Never kills by name.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
TT_METAL_HOME="${TT_METAL_HOME:-$(cd -- "$HERE/../../../../.." && pwd -P)}"
export TT_METAL_HOME
CKPT="${HF_MODEL:?set HF_MODEL to the flat checkpoint directory}"
CACHE="${TT_CACHE_PATH:?set TT_CACHE_PATH}"
PY="${QWEN38_PY:-$TT_METAL_HOME/python_env_q38fn/bin/python}"
PORT="${QWEN38_PORT:-8000}"
OUT="${QWEN38_OUT:-$PWD/qb2_repro_results/$(date -u +%Y%m%dT%H%M%SZ)}"
LAUNCHER="$TT_METAL_HOME/models/demos/blackhole/qwen38_flash_next/tools/run_qwen38_chat_server.sh"
WRAPPER="$HERE/run_server_direct.sh"
BENCH="$HERE/bench_native.py"
export QWEN38_PY="$PY"
export TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0   # needed on some hosts (tt-metal#55957)
export PYTHONNOUSERSITE=1
READY_TIMEOUT=1200
IDLE_SECONDS=120

log() { printf '[repro %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }
die() { printf '[repro] ERROR: %s\n' "$*" >&2; exit 2; }

[[ $# -gt 0 ]] || { sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 2; }
[[ -d "$CKPT" ]] || die "HF_MODEL=$CKPT is not a directory"
[[ -x "$PY" ]] || die "no python at $PY (run setup_venv.sh or set QWEN38_PY)"
[[ -x "$LAUNCHER" ]] || die "launcher not found: $LAUNCHER (is this file inside the tt-metal tree? set TT_METAL_HOME)"
mkdir -p "$OUT"
OUT=$(cd -- "$OUT" && pwd -P)
log "TT_METAL_HOME=$TT_METAL_HOME PY=$PY CKPT=$CKPT CACHE=$CACHE PORT=$PORT OUT=$OUT"

SERVER_PID=
LOCK_OPEN=
descendants() {  # PID and all its descendants
    local p=$1 c
    echo "$p"
    for c in $(pgrep -P "$p" 2>/dev/null || true); do descendants "$c"; done
}
port_busy() {
    if command -v ss >/dev/null 2>&1; then [[ -n "$(ss -H -ltn "sport = :$PORT" 2>/dev/null)" ]]
    else (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; fi
}
wait_port_free() {
    local i
    for ((i = 0; i < 120; i++)); do port_busy || return 0; sleep 1; done
    log "WARNING: port $PORT still busy after 120 s"; return 1
}
stop_server() {
    [[ -n "$SERVER_PID" ]] || return 0
    local pid=$SERVER_PID pids p i alive
    SERVER_PID=
    pids=$(descendants "$pid" | sort -u) || true
    if kill -0 "$pid" 2>/dev/null; then
        log "stopping server pid(s): $(echo $pids) (SIGTERM; SIGINT does not stop it)"
        for p in $pids; do kill -TERM "$p" 2>/dev/null || true; done
        for ((i = 0; i < 60; i++)); do
            alive=
            for p in $pids; do kill -0 "$p" 2>/dev/null && alive=1; done
            [[ -n "$alive" ]] || break
            sleep 1
        done
        for p in $pids; do
            if kill -0 "$p" 2>/dev/null; then log "pid $p still alive after 60 s, kill -9"; kill -9 "$p" 2>/dev/null || true; fi
        done
    fi
    wait "$pid" 2>/dev/null || true
    wait_port_free || true
    if [[ -n "$LOCK_OPEN" ]]; then exec 9>&-; LOCK_OPEN=; fi
}
trap 'stop_server' EXIT
trap 'log "interrupted"; exit 130' INT TERM

take_lock() {
    [[ -n "${QWEN38_DEVICE_LOCK:-}" ]] || return 0
    exec 9>"$QWEN38_DEVICE_LOCK"
    flock -n 9 || die "device lock $QWEN38_DEVICE_LOCK is held by another process"
    LOCK_OPEN=1
}

page_cache_ok() {  # the 104 GB PLE table must stay in page cache
    local gb
    gb=$(free -g | awk '/^Mem:/ {print $6}')
    log "buff/cache = ${gb} GB"
    [[ "$gb" -ge 110 ]]
}
do_prewarm() {
    log "prewarm PLE table"
    (cd "$TT_METAL_HOME" && PYTHONPATH="$TT_METAL_HOME${PYTHONPATH:+:$PYTHONPATH}" \
        "$PY" -m models.demos.blackhole.qwen38_flash_next.tools.prewarm_ple_table --checkpoint "$CKPT") 2>&1 | tee "$OUT/prewarm.log"
}
ensure_ple_cache() { page_cache_ok || do_prewarm; page_cache_ok || log "WARNING: page cache still below 110 GB; results may be slower"; }

# start_server DIR CMD...  : background server, log to DIR/server.log, wait for /health ready
start_server() {
    local dir=$1; shift
    mkdir -p "$dir"
    port_busy && die "port $PORT is busy"
    ensure_ple_cache
    take_lock
    log "start: $* (log $dir/server.log)"
    "$@" >"$dir/server.log" 2>&1 9>&- &
    SERVER_PID=$!
    echo "$SERVER_PID" >"$dir/server.pid"
    local t0=$SECONDS
    while ! curl -sf "http://127.0.0.1:$PORT/health" >"$dir/health.json" 2>/dev/null; do
        kill -0 "$SERVER_PID" 2>/dev/null || { tail -n 30 "$dir/server.log" >&2; die "server exited before ready"; }
        ((SECONDS - t0 < READY_TIMEOUT)) || { tail -n 30 "$dir/server.log" >&2; die "server not ready after $READY_TIMEOUT s"; }
        sleep 5
    done
    log "server ready after $((SECONDS - t0)) s"
}
run_oneshot() {  # DIR CMD... : foreground-equivalent (background + wait so the trap works)
    local dir=$1; shift
    mkdir -p "$dir"
    ensure_ple_cache
    take_lock
    log "run: $*"
    "$@" >"$dir/run.log" 2>&1 9>&- &
    SERVER_PID=$!
    local rc=0
    wait "$SERVER_PID" || rc=$?
    SERVER_PID=
    if [[ -n "$LOCK_OPEN" ]]; then exec 9>&-; LOCK_OPEN=; fi
    log "exit code $rc (log $dir/run.log)"
    return "$rc"
}

FAILED=0
bench() {  # DIR SUITE [bench args...]
    local dir=$1 suite=$2; shift 2
    log "bench $suite $*"
    "$PY" "$BENCH" --url "http://127.0.0.1:$PORT" --tokenizer-dir "$CKPT" --out-dir "$dir/$suite" --suite "$suite" "$@" \
        2>&1 | tee "$dir/$suite.log" || { log "suite $suite FAILED"; FAILED=1; }
}
idle() { log "idle $IDLE_SECONDS s"; sleep "$IDLE_SECONDS"; }
ISLS=1024,4096,8192,16384,31000

step_prewarm() { do_prewarm; }
step_prepare() { run_oneshot "$OUT/prepare" "$LAUNCHER" --profile qb2 --checkpoint "$CKPT" --cache-root "$CACHE" --python "$PY" --prepare-only; }
step_acceptance() {
    local dir="$OUT/acceptance" rd
    run_oneshot "$dir" "$LAUNCHER" --profile qb2 --checkpoint "$CKPT" --cache-root "$CACHE" --python "$PY" \
        --acceptance --require-json-96 --acceptance-only || FAILED=1
    rd=$(sed -n 's/.* run \(\/[^ ]*\)$/\1/p' "$dir/run.log" | head -n1)
    if [[ -n "$rd" && -f "$rd/acceptance.json" ]]; then cp "$rd/acceptance.json" "$dir/acceptance.json"; log "copied $rd/acceptance.json"
    else log "acceptance.json not found (run dir '$rd')"; FAILED=1; fi
}
launcher_server() {  # DIR [launcher args...]
    local dir=$1; shift
    start_server "$dir" "$LAUNCHER" --profile qb2 --checkpoint "$CKPT" --cache-root "$CACHE" --python "$PY" \
        --host 127.0.0.1 --port "$PORT" "$@"
}
wrapper_server() { local dir=$1; shift; start_server "$dir" "$WRAPPER" --host 127.0.0.1 --port "$PORT" "$@"; }

step_P1() {
    local d="$OUT/P1_plain"; launcher_server "$d"
    bench "$d" decode --repeats 3; bench "$d" sampled --repeats 3
    bench "$d" concurrency --repeats 2 --concurrency-list 1,2,4
    idle; bench "$d" prefill --repeats 2 --isl-list "$ISLS"
    stop_server
}
step_P2() {
    local d="$OUT/P2_slab2048"; launcher_server "$d" --prefill-slab 2048
    bench "$d" decode --repeats 3
    idle; bench "$d" prefill --repeats 2 --isl-list "$ISLS"
    stop_server
}
step_P3() {
    local d="$OUT/P3_mtp4_slab2048"; launcher_server "$d" --mtp 4 --prefill-slab 2048
    bench "$d" decode --repeats 3; bench "$d" sampled --repeats 3
    idle; bench "$d" prefill --repeats 2 --isl-list 4096,16384,31000
    stop_server
}
step_P4a() {
    local d="$OUT/P4a_lanes4_mtp4"; wrapper_server "$d" --mtp 4 --lanes 4
    bench "$d" concurrency --repeats 2 --concurrency-list 1,2,4
    stop_server
}
step_P4b() {
    local d="$OUT/P4b_lanes8_mtp3"; wrapper_server "$d" --mtp 3 --lanes 8
    bench "$d" concurrency --repeats 2 --concurrency-list 1,2,4,8
    stop_server
}

steps=()
for s in "$@"; do
    case "$s" in
        all) steps+=(prepare acceptance P1 P2 P3 P4a P4b) ;;
        prewarm|prepare|acceptance|P1|P2|P3|P4a|P4b) steps+=("$s") ;;
        *) die "unknown step $s" ;;
    esac
done
for s in "${steps[@]}"; do
    log "=== step $s ==="
    "step_$s" || { log "step $s FAILED"; FAILED=1; }
done
log "results in $OUT; summarize with: $PY $HERE/summarize.py $OUT"
exit "$FAILED"
