#!/bin/bash
# tt_safe_run.sh - plumbing shared by run_safe_pytest.sh, tt-probe.sh and
# eval/eval_test_runner.sh (in the tt_ops_code_gen submodule)
#
# Source this file; do not execute it. It owns everything the runners used to
# duplicate: simulator detection, the --device/--mesh selector rules, card
# acquisition (via tt_device_pool.sh), venv activation, --dev instrumentation,
# the dispatch-timeout hang hook, signal forwarding, running the child under
# tee, hang detection, post-hang reset + log dump, the device-timing record,
# a single EXIT dispatcher, and the inline precompile / JIT-server machinery.
# The scripts keep only their own argument parsing, command construction and
# result policy.
#
# Conventions: every function prints with the "$TTRUN_PREFIX: " prefix
# (SAFE_PYTEST / TT_PROBE / EVAL_RUNNER). Globals the caller reads are
# UPPER_CASE and listed next to the function that sets them. Callers must not
# `set -e`.
#
# Usage sketch:
#   TTRUN_PREFIX=SAFE_PYTEST; source scripts/lib/tt_safe_run.sh
#   ttrun_init run_safe_pytest          # paths, timing record, EXIT dispatcher, sim detection
#   ttrun_precompile_defaults auto TT_METAL_JIT_SERVER_ENDPOINT
#   ... parse flags into DEVICE_SELECTOR / MESH_MODE / DEV_MODE; hand precompile
#       flags to ttrun_precompile_parse_flag ...
#   ttrun_resolve_selector || exit 3
#   ttrun_acquire_card || exit 3        # hardware only (no-op on sim); sets RUN_START
#   ttrun_activate_venv
#   ttrun_export_dev_env                # if DEV_MODE
#   ttrun_setup_hang_hook
#   ttrun_precompile_prepare abort|fallback; ttrun_precompile_cmd pytest ...   # -> PRECOMPILE_CMD
#   ttrun_mark_dirty
#   ttrun_run_child "$STDOUT_LOG" "${PRECOMPILE_CMD[@]}"   # sets EXIT_CODE
#   ttrun_precompile_record "$STDOUT_LOG"
#   ttrun_detect_hang "$STDOUT_LOG"        # sets IS_HANG
#   ttrun_reset_cards / ttrun_dump_hang_logs / ttrun_clear_dirty / ttrun_cleanup_tmp

TTRUN_PREFIX="${TTRUN_PREFIX:-TTRUN}"
TTPOOL_LOG_PREFIX="$TTRUN_PREFIX"
_TTRUN_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib/tt_device_pool.sh
source "${_TTRUN_LIB_DIR}/tt_device_pool.sh"

_ttrun_say() { echo "${TTRUN_PREFIX}: $*"; }

# ---------------------------------------------------------------------------
# EXIT dispatcher. Bash has ONE EXIT trap; the runners used to install three and
# the last one silently replaced the others (SAFE_PYTEST_TOTAL_RUNTIME never
# printed). Hooks run in registration order, all of them, and receive the
# script's exit status as $1 (inside a called function $? is the status of the
# previous statement, not of the exiting script). The status is preserved.
# ---------------------------------------------------------------------------
TTRUN_EXIT_HOOKS=()
ttrun_on_exit() { TTRUN_EXIT_HOOKS+=("$1"); }
_ttrun_run_exit_hooks() {
    local ec=$? hook
    for hook in "${TTRUN_EXIT_HOOKS[@]}"; do "$hook" "$ec"; done
    return $ec
}

# ---------------------------------------------------------------------------
# ttrun_init <source_name>
#   REPO_DIR, TRIAGE_SCRIPT, TRIAGE_LOG, legacy TRIAGE_* / WATCHER_LOG defaults
#   (re-pointed per card by ttrun_acquire_card), DISPATCH_TIMEOUT
#   (SAFE_PYTEST_DISPATCH_TIMEOUT, default 5), the device-timing record
#   (TT_TIMING_*), SIM_MODE (+ sim env exports), and the EXIT dispatcher.
# ---------------------------------------------------------------------------
ttrun_init() {
    REPO_DIR="$(cd "${_TTRUN_LIB_DIR}/../.." && pwd)"
    TRIAGE_SCRIPT="${REPO_DIR}/tools/tt-triage.py"
    TRIAGE_LOG="/tmp/tt-safe-run-triage-$$.log"
    WATCHER_LOG="${REPO_DIR}/generated/watcher/watcher.log"
    TRIAGE_OUT_DIR="${REPO_DIR}/generated/tt-triage"
    TRIAGE_REPORT="${TRIAGE_OUT_DIR}/triage.txt"
    TRIAGE_EXTRA_ARGS=""
    DISPATCH_TIMEOUT="${SAFE_PYTEST_DISPATCH_TIMEOUT:-5}"
    DEVICE_SELECTOR="${DEVICE_SELECTOR:-}"
    MESH_MODE="${MESH_MODE:-false}"
    DEV_MODE="${DEV_MODE:-false}"
    CHILD_PID=""
    EXIT_CODE=0
    IS_HANG=false

    # --- Device-lock contention profiling ---
    # When $TT_DEVICE_TIMING_LOG is set, on EXIT we append one JSON line:
    #   {source, pid, started_at_ms, wait_ms, run_ms, test_path, exit_code, precompile_*, cards}
    # wait_ms = script entry -> card acquired (contention); run_ms = acquired -> exit.
    # On sim there is no lock: the marker is seeded at entry (wait_ms 0, run_ms = wall).
    # precompile_* are filled by ttrun_precompile_record; tt-probe leaves the defaults.
    TT_TIMING_ENTRY_MS=$(date +%s%3N)
    TT_TIMING_LOCK_ACQUIRED_MS=0
    TT_TIMING_SOURCE="$1"
    TT_TIMING_TEST_PATH=""
    TT_TIMING_PRECOMPILE_MODE="off"
    TT_TIMING_PRECOMPILE_REASON="disabled"
    TT_TIMING_PRECOMPILE_S=0
    TT_TIMING_PRECOMPILE_PROGRAMS=-1
    TT_TIMING_PRECOMPILE_BUILT=-1
    ttrun_on_exit _ttrun_emit_device_timing
    trap _ttrun_run_exit_hooks EXIT

    # --- Simulator mode (TT_METAL_SIMULATOR set) ---
    SIM_MODE=false
    if [[ -n "${TT_METAL_SIMULATOR:-}" ]]; then
        SIM_MODE=true
        export TT_METAL_SLOW_DISPATCH_MODE=1
        export TT_METAL_DISABLE_SFPLOADMACRO=1
        # libttsim's own hang watchdog (clocks of no RISC-V / Tensix progress with
        # pending work before the sim _Exit(1)'s). User-set env wins.
        : "${TTSIM_HANG_WATCHDOG_CLOCKS:=50000}"
        export TTSIM_HANG_WATCHDOG_CLOCKS
        TT_TIMING_LOCK_ACQUIRED_MS=$TT_TIMING_ENTRY_MS
    fi
}

_ttrun_emit_device_timing() {
    local ec="${1:-$?}"
    if [[ -n "${TT_DEVICE_TIMING_LOG:-}" && "$TT_TIMING_LOCK_ACQUIRED_MS" -ne 0 ]]; then
        local end_ms wait_ms run_ms log_dir esc_path
        end_ms=$(date +%s%3N)
        wait_ms=$(( TT_TIMING_LOCK_ACQUIRED_MS - TT_TIMING_ENTRY_MS ))
        run_ms=$(( end_ms - TT_TIMING_LOCK_ACQUIRED_MS ))
        log_dir="$(dirname "$TT_DEVICE_TIMING_LOG")"
        [[ -n "$log_dir" ]] && mkdir -p "$log_dir" 2>/dev/null
        # JSON-escape test_path: backslash first, then double-quote.
        esc_path="${TT_TIMING_TEST_PATH//\\/\\\\}"
        esc_path="${esc_path//\"/\\\"}"
        printf '{"source":"%s","pid":%d,"started_at_ms":%s,"wait_ms":%d,"run_ms":%d,"test_path":"%s","exit_code":%d,"precompile_mode":"%s","precompile_reason":"%s","precompile_s":%d,"precompile_programs":%d,"precompile_built":%d,"cards":"%s"}\n' \
            "$TT_TIMING_SOURCE" "$$" "$TT_TIMING_ENTRY_MS" "$wait_ms" "$run_ms" "$esc_path" "$ec" \
            "$TT_TIMING_PRECOMPILE_MODE" "$TT_TIMING_PRECOMPILE_REASON" "$TT_TIMING_PRECOMPILE_S" \
            "$TT_TIMING_PRECOMPILE_PROGRAMS" "$TT_TIMING_PRECOMPILE_BUILT" "${TTPOOL_CARDS:-sim}" \
            >> "$TT_DEVICE_TIMING_LOG" 2>/dev/null || true
    fi
    return $ec
}

# ---------------------------------------------------------------------------
# ttrun_resolve_selector: turns MESH_MODE + DEVICE_SELECTOR (+ a pre-set
# TT_VISIBLE_DEVICES) into a validated DEVICE_SELECTOR (auto | mesh | N).
# ---------------------------------------------------------------------------
ttrun_resolve_selector() {
    if [[ "$MESH_MODE" == true && -n "$DEVICE_SELECTOR" ]]; then
        echo "${TTRUN_PREFIX}_ERROR: --mesh and --device are mutually exclusive"
        return 1
    fi
    if [[ "$MESH_MODE" == true ]]; then
        DEVICE_SELECTOR="mesh"
    elif [[ -z "$DEVICE_SELECTOR" ]]; then
        # A caller that already pinned a card via TT_VISIBLE_DEVICES gets that card's lock, so
        # lock and visibility can never disagree. A list means multi-device: require --mesh.
        if [[ -n "${TT_VISIBLE_DEVICES:-}" ]]; then
            if [[ "$TT_VISIBLE_DEVICES" =~ ^[0-9]+$ ]]; then
                DEVICE_SELECTOR="$TT_VISIBLE_DEVICES"
                _ttrun_say "TT_VISIBLE_DEVICES=${TT_VISIBLE_DEVICES} in env -> --device ${TT_VISIBLE_DEVICES}"
            else
                echo "${TTRUN_PREFIX}_ERROR: TT_VISIBLE_DEVICES='${TT_VISIBLE_DEVICES}' is a list; pass --mesh (all cards) or --device N and unset it"
                return 1
            fi
        else
            DEVICE_SELECTOR="auto"
        fi
    fi
    if [[ "$DEVICE_SELECTOR" != auto && "$DEVICE_SELECTOR" != mesh && ! "$DEVICE_SELECTOR" =~ ^[0-9]+$ ]]; then
        echo "${TTRUN_PREFIX}_ERROR: --device wants a UMD card id or 'auto' (got: $DEVICE_SELECTOR)"
        return 1
    fi
    # Canonical decimal: "00" and "0" are the same card and must take the same lock.
    [[ "$DEVICE_SELECTOR" =~ ^[0-9]+$ ]] && DEVICE_SELECTOR=$((10#$DEVICE_SELECTOR))
    return 0
}

# ---------------------------------------------------------------------------
# ttrun_acquire_card: hardware only (no-op on sim). Blocks until the selected
# card(s) are held, exports the per-card env, re-points TRIAGE_* / WATCHER_LOG /
# TRIAGE_EXTRA_ARGS, sets RUN_START and the timing marker, resets a dirty card.
# ---------------------------------------------------------------------------
ttrun_acquire_card() {
    if [[ "$SIM_MODE" == true ]]; then
        RUN_START=$(date +%s)
        return 0
    fi
    if ! ttpool_acquire "$DEVICE_SELECTOR"; then
        echo "${TTRUN_PREFIX}_ERROR: could not acquire device(s) for selector '${DEVICE_SELECTOR}'"
        return 1
    fi
    TT_TIMING_LOCK_ACQUIRED_MS=$(date +%s%3N)
    # The total-run clock starts the moment we own the device; queueing behind other
    # runners is deliberately excluded.
    RUN_START=$(date +%s)
    ttpool_setup_env "$REPO_DIR" || return 1
    WATCHER_LOG="$TTPOOL_WATCHER_LOG"
    TRIAGE_OUT_DIR="$TTPOOL_TRIAGE_OUT_DIR"
    TRIAGE_REPORT="$TTPOOL_TRIAGE_REPORT"
    TRIAGE_EXTRA_ARGS="$TTPOOL_TRIAGE_EXTRA_ARGS"
    _ttrun_say "card(s)=${TTPOOL_CARDS} (${TTPOOL_LABEL}) logs=${TTPOOL_LOGS_DIR}"
    if ! ttpool_reset_if_dirty; then
        echo "${TTRUN_PREFIX}_ERROR: Device reset (tt-smi -r ${TTPOOL_CARDS}) failed"
        return 1
    fi
}

ttrun_activate_venv() {
    cd "$REPO_DIR" || return 1
    if [[ -f python_env/bin/activate ]]; then
        # shellcheck disable=SC1091
        source python_env/bin/activate || _ttrun_say "WARNING: Failed to activate python_env virtual environment"
    else
        _ttrun_say "WARNING: python_env not found; using system Python"
    fi
}

# ---------------------------------------------------------------------------
# ttrun_export_dev_env: --dev instrumentation. Exported before the child starts
# so a precompile step inside the session sees them: LIGHTWEIGHT_KERNEL_ASSERTS,
# LLK_ASSERTS and WATCHER_NOINLINE are compile-time flags (they change the
# kernel build key), so compile step and real pass must agree on them.
#   * Lightweight asserts compile ASSERT() as ebreak, halting the core at the
#     exact instruction; the dispatch timeout then runs triage, which captures
#     callstacks from ALL cores (assert site + anything blocked on it).
#   * LLK asserts enable LLK_ASSERT() in the compute API / LLK layer.
#   * Polling watcher: NoC sanitizer, waypoints, CB sanitization. Its own assert
#     handling is disabled so the ebreak asserts above reach triage.
#   * NoC sanitizer is off on sim (tuned for HW; false positives under libttsim).
# ---------------------------------------------------------------------------
ttrun_export_dev_env() {
    export TT_METAL_LIGHTWEIGHT_KERNEL_ASSERTS=1
    export TT_METAL_LLK_ASSERTS=1
    export TT_METAL_WATCHER=1
    export TT_METAL_WATCHER_NOINLINE=1
    export TT_METAL_WATCHER_DISABLE_ASSERT=1
    export TT_METAL_WATCHER_DISABLE_DISPATCH=1
    if [[ "$SIM_MODE" == true ]]; then
        export TT_METAL_WATCHER_DISABLE_NOC_SANITIZE=1
    fi
}

# One-line mode banner; <extra> is appended (e.g. "workers=16").
ttrun_print_mode_banner() {
    local extra="${1:+ $1}"
    if [[ "$DEV_MODE" == true && "$SIM_MODE" == true ]]; then
        _ttrun_say "[sim+dev] asserts=ebreak llk_asserts=ON watcher=polling(noc_sanitize=OFF) watchdog=${TTSIM_HANG_WATCHDOG_CLOCKS} clocks${extra}"
    elif [[ "$DEV_MODE" == true ]]; then
        _ttrun_say "[dev] asserts=ebreak llk_asserts=ON watcher=polling triage=ON timeout=${DISPATCH_TIMEOUT}s${extra}"
    elif [[ "$SIM_MODE" == true ]]; then
        _ttrun_say "[sim] watchdog=${TTSIM_HANG_WATCHDOG_CLOCKS} clocks${extra}"
    else
        _ttrun_say "dispatch_timeout=${DISPATCH_TIMEOUT}s${extra}"
    fi
}

# ---------------------------------------------------------------------------
# ttrun_setup_hang_hook: hardware only. On dispatch timeout the runtime runs
# tt-triage scoped to our card(s) (the hook inherits TT_VISIBLE_DEVICES; the
# extra args point it at our inspector port / log dir). Zero overhead for
# passing tests. On sim there is no wall-clock hang detection (kHz clocks);
# sim hangs are caught by the libttsim watchdog (see ttrun_detect_hang).
# A non-empty TRIAGE_LOG afterwards is the hang signal, so without tt-exalens
# the hook still writes a marker line.
# ---------------------------------------------------------------------------
ttrun_setup_hang_hook() {
    rm -f "$TRIAGE_LOG"
    # Clear any stale report: downstream consumers treat its presence as the hang signal.
    rm -f "$TRIAGE_REPORT"
    [[ "$SIM_MODE" == false ]] || return 0
    export TT_METAL_OPERATION_TIMEOUT_SECONDS="$DISPATCH_TIMEOUT"
    mkdir -p "$TRIAGE_OUT_DIR"
    if python3 -c "import ttexalens" 2>/dev/null; then
        export TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE="python3 ${TRIAGE_SCRIPT} --disable-progress --skip-version-check ${TRIAGE_EXTRA_ARGS} --llm-output --llm-output-path=${TRIAGE_REPORT} > ${TRIAGE_LOG} 2>&1"
    else
        # Requires tt-exalens: uv pip install -r tools/triage/requirements.txt. The warning is
        # deferred to EXIT so it is not buried in the child's output.
        export TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE="echo HANG_NO_TRIAGE > ${TRIAGE_LOG}"
        ttrun_on_exit _ttrun_warn_missing_ttexalens
    fi
}

_ttrun_warn_missing_ttexalens() {
    echo ""
    _ttrun_say "WARNING: tt-exalens not installed — triage on hang is unavailable."
    _ttrun_say "Install with: uv pip install -r tools/triage/requirements.txt"
}

ttrun_mark_dirty()  { [[ "$SIM_MODE" == false ]] && ttpool_mark_dirty; return 0; }
ttrun_clear_dirty() { [[ "$SIM_MODE" == false ]] && ttpool_clear_dirty; return 0; }

# ---------------------------------------------------------------------------
# Signal forwarding: if this script is killed (parent SIGTERM, watchdog, ...)
# SIGKILL the child and its descendants. Otherwise the child is orphaned holding
# the lock fd -> /tmp/tt-device-<N>.lock and /dev/tenstorrent/*, blocking that
# card for good. The card stays marked dirty so the next taker resets it.
# ---------------------------------------------------------------------------
_ttrun_signal_cleanup() {
    local sig=$1
    echo ""
    _ttrun_say "Caught SIG${sig} — killing child, marking card(s) ${TTPOOL_CARDS:-} dirty"
    ttrun_mark_dirty
    if [[ -n "$CHILD_PID" ]]; then
        pkill -KILL -P "$CHILD_PID" 2>/dev/null || true
        kill -KILL "$CHILD_PID" 2>/dev/null || true
    fi
    pkill -KILL -P $$ 2>/dev/null || true
    exit 143
}

# ttrun_run_child <stdout_log> <cmd...>: runs the command in the background and
# waits, so the signal traps can fire (bash defers signals while a foreground
# command runs). stdout+stderr are mirrored to <stdout_log> via process
# substitution, which keeps $! pointing at the child itself, not tee. Sets EXIT_CODE.
ttrun_run_child() {
    local stdout_log="$1"; shift
    trap '_ttrun_signal_cleanup TERM' SIGTERM
    trap '_ttrun_signal_cleanup HUP'  SIGHUP
    trap '_ttrun_signal_cleanup INT'  SIGINT
    "$@" > >(tee "$stdout_log") 2>&1 &
    CHILD_PID=$!
    wait "$CHILD_PID"
    EXIT_CODE=$?
    wait 2>/dev/null  # let tee flush before anyone greps the log
    CHILD_PID=""
    # Kill any orphans the child left behind.
    pkill -9 -P $$ 2>/dev/null || true
}

# ttrun_detect_hang <stdout_log>: sets IS_HANG.
#   HW:  TRIAGE_LOG non-empty = the dispatch-timeout hook ran.
#   Sim: "hang watchdog fired" in the child's output = libttsim _Exit(1)'d; the
#        message is staged into TRIAGE_LOG so the dump below treats it uniformly.
ttrun_detect_hang() {
    local stdout_log="$1"
    IS_HANG=false
    if [[ -s "$TRIAGE_LOG" ]]; then
        IS_HANG=true
    elif [[ "$SIM_MODE" == true ]] && grep -q "hang watchdog fired" "$stdout_log" 2>/dev/null; then
        IS_HANG=true
        grep -A4 "hang watchdog fired" "$stdout_log" > "$TRIAGE_LOG"
    fi
}

# ttrun_reset_cards: hardware only. Resets the held card(s), clears the dirty
# flags on success, and points the legacy triage path at this run's report.
ttrun_reset_cards() {
    [[ "$SIM_MODE" == false ]] || return 0
    _ttrun_say "Resetting card(s) ${TTPOOL_CARDS}..."
    if ttpool_reset; then
        sleep 2
        ttpool_clear_dirty
        _ttrun_say "Device reset complete"
    else
        _ttrun_say "Device reset FAILED; leaving card(s) ${TTPOOL_CARDS} marked dirty"
    fi
    ttpool_publish_triage_report "$REPO_DIR"
}

# ttrun_dump_hang_logs: the triage log, the watcher log tail under --dev, and
# the report path as its own line so machine readers can find it.
ttrun_dump_hang_logs() {
    echo ""
    echo "=== TRIAGE LOG ==="
    cat "$TRIAGE_LOG"
    echo "=== END TRIAGE LOG ==="
    echo ""
    if [[ "$DEV_MODE" == true && -f "$WATCHER_LOG" ]]; then
        echo "=== WATCHER LOG (last 50 lines) ==="
        tail -50 "$WATCHER_LOG"
        echo "=== END WATCHER LOG ==="
        echo ""
    fi
    if [[ -s "$TRIAGE_REPORT" ]]; then
        _ttrun_say "triage report: ${TRIAGE_REPORT}"
    fi
}

ttrun_cleanup_tmp() { rm -f "$TRIAGE_LOG" "$@"; }

# Registered by run_safe_pytest.sh: total wall-clock from card acquired to exit,
# always the last line. Nothing is printed for exits before a run began.
ttrun_print_total_runtime() {
    [[ -z "${RUN_START:-}" ]] && return 0
    local elapsed=$(( $(date +%s) - RUN_START ))
    echo "========================================" >&2
    printf '%s_TOTAL_RUNTIME: %dm%02ds (%ds total, device-lock-acquired -> exit)\n' \
        "$TTRUN_PREFIX" $((elapsed / 60)) $((elapsed % 60)) "$elapsed" >&2
}

# ===========================================================================
# Precompile (tests/plugins/up_front_collect, INLINE: collect pass + parallel compile +
# real pass in ONE pytest session) + JIT compile server, shared by run_safe_pytest.sh and
# eval_test_runner.sh. Defaults, flags, the server probe, the route decision, the plugin
# environment, the command wrapping and the RESULT parsing that feeds the device-timing
# record all live here; a caller keeps only its unreachable-server policy (abort vs
# fallback) and what it does with the session's exit status.
#
# State (globals, set by ttrun_precompile_defaults, refined by the flag parser):
#   PRECOMPILE                    auto | true | false
#   PRECOMPILE_WORKERS            compile threads (default nproc)
#   PRECOMPILE_FARM_MIN_PROGRAMS  route the compile step to the farm at >= N programs
#   JIT_SERVER_ENDPOINT           host:port or empty; JIT_SERVER_DISABLED true|false
# ===========================================================================

# ttrun_precompile_defaults <initial PRECOMPILE> <endpoint env var names...>
#   Endpoint = first non-empty of the named env vars (e.g. EVAL_JIT_SERVER_ENDPOINT
#   TT_METAL_JIT_SERVER_ENDPOINT). Also scrubs TT_METAL_JIT_SERVER_ENABLE from the
#   ambient env: the server-enable bit is raised only inside the compile step.
ttrun_precompile_defaults() {
    PRECOMPILE="$1"; shift
    PRECOMPILE_WORKERS="${PRECOMPILE_WORKERS:-${EVAL_PRECOMPILE_WORKERS:-$(nproc 2>/dev/null || echo 8)}}"
    PRECOMPILE_FARM_MIN_PROGRAMS="${PRECOMPILE_FARM_MIN_PROGRAMS:-10}"
    JIT_SERVER_ENDPOINT=""
    local v
    for v in "$@"; do
        if [[ -n "${!v:-}" ]]; then JIT_SERVER_ENDPOINT="${!v}"; break; fi
    done
    JIT_SERVER_DISABLED=false
    unset TT_METAL_JIT_SERVER_ENABLE
}

# The flag help snippet, so both usage lines stay identical.
TTRUN_PRECOMPILE_USAGE="[--precompile|--no-precompile] [--precompile-workers N] [--farm-min-programs N] [--jit-server[=host:port]|--no-jit-server]"

# ttrun_precompile_parse_flag "$@": if $1 is a precompile/JIT flag, apply it and set
# TTRUN_CONSUMED to the number of argv words used (1 or 2); return 0. Return 1 if $1 is
# not one of ours (caller handles it). Return 2 on a malformed value (message printed).
ttrun_precompile_parse_flag() {
    TTRUN_CONSUMED=1
    case "$1" in
        --precompile) PRECOMPILE=true ;;
        --no-precompile) PRECOMPILE=false ;;
        --precompile-workers)
            [[ $# -ge 2 && "$2" =~ ^[1-9][0-9]*$ ]] || { echo "${TTRUN_PREFIX}_ERROR: --precompile-workers requires a positive integer argument"; return 2; }
            PRECOMPILE_WORKERS="$2"; TTRUN_CONSUMED=2 ;;
        --farm-min-programs)
            [[ $# -ge 2 && "$2" =~ ^[0-9]+$ ]] || { echo "${TTRUN_PREFIX}_ERROR: --farm-min-programs requires a non-negative integer argument"; return 2; }
            PRECOMPILE_FARM_MIN_PROGRAMS="$2"; TTRUN_CONSUMED=2 ;;
        --jit-server)
            [[ $# -ge 2 ]] || { echo "${TTRUN_PREFIX}_ERROR: --jit-server requires a host:port argument"; return 2; }
            JIT_SERVER_ENDPOINT="$2"; JIT_SERVER_DISABLED=false; TTRUN_CONSUMED=2 ;;
        --jit-server=*) JIT_SERVER_ENDPOINT="${1#*=}"; JIT_SERVER_DISABLED=false ;;
        --no-jit-server) JIT_SERVER_DISABLED=true ;;
        *) TTRUN_CONSUMED=0; return 1 ;;
    esac
    return 0
}

# ttrun_jit_server_status: 0 = configured, enabled and reachable (TCP connect within 5 s);
# 1 = configured and enabled but unreachable; 2 = no endpoint / disabled.
ttrun_jit_server_status() {
    [[ -n "$JIT_SERVER_ENDPOINT" && "$JIT_SERVER_DISABLED" == false ]] || return 2
    local h="${JIT_SERVER_ENDPOINT%:*}" p="${JIT_SERVER_ENDPOINT##*:}"
    timeout 5 bash -c "exec 3<>/dev/tcp/${h}/${p}" 2>/dev/null && return 0
    return 1
}

# ttrun_precompile_prepare <abort|fallback>
#   The whole pre-run decision, shared verbatim by both runners:
#     * simulator -> precompile off (reason=sim; no warm benefit at kHz clocks)
#     * route: farm only if a JIT server is configured, enabled AND reachable, else local
#     * unreachable server: <abort> prints the error and returns 4 (run_safe_pytest.sh:
#       a silent local fallback would hide a down farm from an interactive user);
#       <fallback> warns and compiles locally (eval_test_runner.sh: a graded run must
#       never be lost to a down farm). Either way the record says jit_unreachable.
#     * announces the decision and fills PRECOMPILE_ENV for the plugin session.
#   Sets PRECOMPILE (may flip to false), PRECOMPILE_ROUTE (off|local|farm), PRECOMPILE_ENV,
#   PRECOMPILE_JIT_DOWN.
ttrun_precompile_prepare() {
    local policy="$1" jit
    PRECOMPILE_ROUTE=off; PRECOMPILE_ENV=(); PRECOMPILE_JIT_DOWN=false
    if [[ "$PRECOMPILE" == true && "$SIM_MODE" == true ]]; then
        PRECOMPILE=false
        TT_TIMING_PRECOMPILE_REASON="sim"
    fi
    if [[ "$PRECOMPILE" != true ]]; then
        echo "PRECOMPILE: off (${TT_TIMING_PRECOMPILE_REASON}) — kernels compile on demand" >&2
        return 0
    fi
    PRECOMPILE_ROUTE=local
    ttrun_jit_server_status; jit=$?
    if [[ $jit -eq 0 ]]; then
        PRECOMPILE_ROUTE=farm
    elif [[ $jit -eq 1 ]]; then
        TT_TIMING_PRECOMPILE_REASON="jit_unreachable"
        if [[ "$policy" == abort ]]; then
            echo "${TTRUN_PREFIX}_ERROR: JIT server '${JIT_SERVER_ENDPOINT}' unreachable — aborting." >&2
            echo "${TTRUN_PREFIX}_ERROR: start the server, fix --jit-server, or pass --no-jit-server to compile locally." >&2
            return 4
        fi
        PRECOMPILE_JIT_DOWN=true
        _ttrun_say "WARNING: JIT server '${JIT_SERVER_ENDPOINT}' unreachable — compile step runs LOCALLY (slower). Start the farm to restore speed." >&2
    fi
    if [[ "$PRECOMPILE_ROUTE" == farm ]]; then
        echo "PRECOMPILE: on — compile step -> JIT server ${JIT_SERVER_ENDPOINT} when >= ${PRECOMPILE_FARM_MIN_PROGRAMS} programs, else local (real pass always local)" >&2
    else
        echo "PRECOMPILE: on — collect + parallel compile (x${PRECOMPILE_WORKERS}) inside the pytest session, local" >&2
    fi
    # Plugin environment. PYTHONPATH is PREPENDED with the repo root so `tests.plugins.up_front_collect`
    # resolves to this clone's plugin without shadowing the caller's own entries. The plugin scopes
    # UP_FRONT_COLLECT=1 to its collect pass itself. The server-enable bit is raised by the plugin
    # only around its compile step, and only when the collected program count reaches
    # PRECOMPILE_FARM_MIN_PROGRAMS (below that the per-kernel round trips cost more than they save);
    # nothing outside the compile step can ever hit the farm.
    PRECOMPILE_ENV=(UP_FRONT_INLINE=1 UP_FRONT_REAL_ALLOC=1 UP_FRONT_COLLECT_WORKERS="$PRECOMPILE_WORKERS"
                    PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:$PYTHONPATH}")
    if [[ "$PRECOMPILE_ROUTE" == farm ]]; then
        PRECOMPILE_ENV+=(UP_FRONT_INLINE_JIT_SERVER=1 UP_FRONT_INLINE_FARM_MIN_PROGRAMS="$PRECOMPILE_FARM_MIN_PROGRAMS"
                         TT_METAL_JIT_SERVER_ENDPOINT="$JIT_SERVER_ENDPOINT" TT_METAL_JIT_PREPROCESS=1 TT_METAL_JIT_SERVER_KEEPALIVE=1)
    fi
}

# ttrun_precompile_cmd <pytest command...>: fills PRECOMPILE_CMD with the command to actually
# run: the given command wrapped in the plugin env with the collector loaded when precompile
# is on, or the command unchanged when it is off.
ttrun_precompile_cmd() {
    if [[ "$PRECOMPILE" == true ]]; then
        PRECOMPILE_CMD=(env "${PRECOMPILE_ENV[@]}" "$@" -p tests.plugins.up_front_collect)
    else
        PRECOMPILE_CMD=("$@")
    fi
}

# ttrun_precompile_record <stdout_log>: after the session, fill the device-timing record's
# precompile fields from the plugin's RESULT line (no-op when precompile was off).
ttrun_precompile_record() {
    [[ "$PRECOMPILE" == true ]] || return 0
    ttrun_precompile_parse_result "$1"
    TT_TIMING_PRECOMPILE_MODE="inline_${TTRUN_PC_ROUTE:-$PRECOMPILE_ROUTE}"
    if [[ "$PRECOMPILE_JIT_DOWN" == true ]]; then
        TT_TIMING_PRECOMPILE_REASON="jit_unreachable"
    else
        TT_TIMING_PRECOMPILE_REASON="${TTRUN_PC_REASON:-no_result_line}"
    fi
}

# ttrun_precompile_parse_result <log>
#   Parses the plugin's one-line `UP_FRONT_COLLECT_RESULT: k=v ...` (last one wins) into
#   TT_TIMING_PRECOMPILE_PROGRAMS / _BUILT / _S and the raw fields TTRUN_PC_REASON /
#   TTRUN_PC_ROUTE for the caller's mode/reason vocabulary. Missing line -> the -1 / empty
#   sentinels. Also echoes the plugin's UP_FRONT_* progress lines, prefixed "PRECOMPILE: ",
#   to stderr so both runners report the compile step identically. Best effort: never fails.
ttrun_precompile_parse_result() {
    local log="$1" line c_s="" k_s=""
    TTRUN_PC_REASON=""; TTRUN_PC_ROUTE=""
    line=$(grep -a '^UP_FRONT_COLLECT_RESULT:' "$log" 2>/dev/null | tail -1 || true)
    [[ "$line" =~ reason=([^[:space:]]+) ]] && TTRUN_PC_REASON="${BASH_REMATCH[1]}"
    [[ "$line" =~ route=([^[:space:]]+) ]] && TTRUN_PC_ROUTE="${BASH_REMATCH[1]}"
    [[ "$line" =~ programs=([0-9]+) ]] && TT_TIMING_PRECOMPILE_PROGRAMS="${BASH_REMATCH[1]}" || TT_TIMING_PRECOMPILE_PROGRAMS=-1
    [[ "$line" =~ built=([0-9]+) ]] && TT_TIMING_PRECOMPILE_BUILT="${BASH_REMATCH[1]}" || TT_TIMING_PRECOMPILE_BUILT=-1
    [[ "$line" =~ collect_s=([0-9]+(\.[0-9]+)?) ]] && c_s="${BASH_REMATCH[1]}"
    [[ "$line" =~ compile_s=([0-9]+(\.[0-9]+)?) ]] && k_s="${BASH_REMATCH[1]}"
    TT_TIMING_PRECOMPILE_S=$(awk -v a="${c_s:-0}" -v b="${k_s:-0}" 'BEGIN{printf "%d", a+b+0.5}')
    grep -a '^UP_FRONT_INLINE:\|^UP_FRONT_COLLECT:' "$log" 2>/dev/null | sed 's/^/PRECOMPILE: /' >&2
}
