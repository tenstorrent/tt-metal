#!/bin/bash
# run_safe_pytest.sh - Cooperative device-aware test runner
#
# Uses one flock PER CARD (scripts/lib/tt_device_pool.sh) so independent jobs can
# run on independent cards concurrently. By default a run takes the lowest free
# card; --device N pins a card; --mesh takes every card for multi-device tests.
# Uses TT_METAL_OPERATION_TIMEOUT_SECONDS for precise hang detection at the
# dispatch layer (does not penalize setup/compilation time).
# Automatically resets the card(s) it held after hangs, ensuring the next runner
# always gets a clean device — without touching the other cards.
#
# Simulator mode (TT_METAL_SIMULATOR set):
#   Exports TT_METAL_SLOW_DISPATCH_MODE=1 and TT_METAL_DISABLE_SFPLOADMACRO=1.
#   Does not set TT_METAL_QUASAR_NOC_API_VERSION (default is 2).
#   ttsim does not implement NOC API v2: for Quasar ttsim, export
#   TT_METAL_QUASAR_NOC_API_VERSION=1 yourself.
#   Skips flock, device resets, and HW dispatch-timeout triage (these require
#   real hardware).
#   Enables the libttsim hang watchdog via TTSIM_HANG_WATCHDOG_CLOCKS (default
#   50000; pre-existing env wins). On hang the watchdog _Exit(1)'s the child;
#   we classify that as HANG and dump the watchdog message.
#
# Usage: scripts/run_safe_pytest.sh [--device N|auto] [--mesh] [--dev] [--run-all] [--profile] [--sim-workers N] [--precompile|--no-precompile] [--farm-min-programs N] [--jit-server[=host:port]|--no-jit-server] <test_path> [extra_pytest_args...]
#
# Wrapper flags below are position-independent: they may appear before or after the test path and
# may be interleaved with pytest args. Everything the wrapper does not recognize (the test path,
# -k/-m filters, ::nodeids, ...) is forwarded to pytest verbatim, in the order given.
#
# Options:
#   --device N|auto  Which card to run on. DEFAULT is auto: the lowest FREE card is
#                    taken (card 0 if free, else 1, ...); if all are busy the run waits
#                    and re-scans every second, still lowest-first. --device N waits for
#                    that specific card. N is the UMD logical id (`tt-smi -ls`, same as
#                    tt-smi -r / TT_VISIBLE_DEVICES), NOT /dev/tenstorrent/<n>. A single
#                    integer already in $TT_VISIBLE_DEVICES is honoured as --device N.
#                    The run sees its card as device 0; logs, triage report and profiler
#                    output go under generated/dev<N>/.
#   --mesh           Take EVERY card (multi-device / CCL tests). Waits for the cards in
#                    ascending order, holding each as it frees up, so a mesh job steadily
#                    drains the pool. All cards visible; logs under generated/mesh/; a hang
#                    resets all cards. Mutually exclusive with --device.
#   --dev            Enables polling watcher (NoC sanitizer, waypoints, CB
#                    sanitization), lightweight ebreak asserts, and auto-triage
#                    on hang with full triage + watcher log dump.
#   --run-all        Run all tests instead of stopping on first failure (-x).
#                    Useful for eval scoring where you need full pass/fail counts.
#   --profile        Run under the Tracy device profiler (python -m tracy -r). Emits
#                    a per-op CSV (generated/<dev|mesh>/profiler/reports/<ts>/ops_perf_results*.csv)
#                    and prints its path as "SAFE_PYTEST: PROFILER CSV: <path>" next to
#                    the result line. Requires a Tracy-enabled build. NOTE: the tracy
#                    wrapper masks pytest's exit code, so a profiled run is reported
#                    PASS as long as profiling completed, regardless of the underlying
#                    test result. Hangs are still detected and still reset the device.
#   --sim-workers N  Sim only: pytest-xdist worker count. Each worker dlopens
#                    its own libttsim (no shared device state). Default is 16.
#                    Pass 1 to serialize (e.g. when DPRINT ordering matters or
#                    you suspect cross-worker contention). Errors out if used
#                    outside sim mode.
#   --precompile     Force-on. Precompile = the tests/plugins/up_front_collect.py plugin loaded
#                    INTO the real pytest session: a collect pass runs every selected body under
#                    NO_DISPATCH to gather the distinct programs, compiles them once in parallel,
#                    then the real pass runs warm. One process, one import, one collection, one
#                    device open. Hardware only. Tune parallelism with --precompile-workers N
#                    (default: nproc).
#   --no-precompile  Force-off: plain pytest, kernels compile on demand (inline & serial).
#                    AUTO (neither flag given): decided from argv alone, no pre-pass. A whole
#                    directory or whole test file -> ON. NARROW (a ::nodeid or a -k filter) -> a
#                    handful of programs, the collect pass cannot pay for itself -> OFF. Sim: OFF.
#   --jit-server[=host:port] / --no-jit-server
#                    Route the precompile COMPILE STEP to a remote JIT server (default endpoint:
#                    $TT_METAL_JIT_SERVER_ENDPOINT). Used only when the collected program count is
#                    at least --farm-min-programs N (default 10, or $PRECOMPILE_FARM_MIN_PROGRAMS):
#                    below that the per-kernel round trips cost more than they save (+2.5s on a
#                    1-program session, -3.4s at 24, -47s at 401). The real pass always compiles
#                    locally. A configured-but-unreachable server aborts loudly (exit 4).
#
# Modes:
#   default  - Dispatch timeout only. Lean, no debug overhead.
#   --dev    - Debug mode with watcher, asserts, and triage (see above).
#   --profile - Tracy device profiling with per-op CSV report (see above).
#
# Exit codes:
#   0 - All tests passed
#   1 - Test failure (normal pytest failure, no hang)
#   2 - Hang detected (dispatch timeout fired)
#   3 - Setup error (missing args, etc.)
#
# Hang triage report: generated/dev<N>/tt-triage/triage.txt (or generated/mesh/...),
# printed as "SAFE_PYTEST: triage report: <path>". The legacy path
# generated/tt-triage/triage.txt is kept as a symlink to the most recent hang report.
#
# Total runtime:
#   Always prints SAFE_PYTEST_TOTAL_RUNTIME as the very last line (on every exit path).
#   It is the wall-clock time from "device lock acquired" (idle lock-wait queueing is
#   deliberately excluded) to script exit, so it covers the whole run — device reset and
#   the pytest run itself (incl. the precompile passes inside it). Under simulator
#   there is no lock, so the clock starts at the equivalent point.

set -o pipefail

TTRUN_PREFIX="SAFE_PYTEST"
# shellcheck source=scripts/lib/tt_safe_run.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/tt_safe_run.sh"
# Sets REPO_DIR, DISPATCH_TIMEOUT, TRIAGE_*, WATCHER_LOG, SIM_MODE (+ sim env), the
# device-timing record and the EXIT dispatcher.
ttrun_init run_safe_pytest
PYTEST_STDOUT_LOG="/tmp/safe-pytest-stdout-$$.log"
# --profile: per-card profiler artifacts dir + explicit Tracy capture port (set on hardware
# after the card is acquired; the legacy default only survives on the simulator).
PROFILE_REPORTS_DIR="${REPO_DIR}/generated/profiler/reports"
TRACY_PORT_ARGS=()

# --- Parse flags ---
DEV_MODE=false
FAIL_FAST=true
PROFILE_MODE=false
# Card selection (DEVICE_SELECTOR / MESH_MODE): auto (default, lowest free card), a UMD id,
# or mesh (every card). Validated by ttrun_resolve_selector after parsing.
SIM_WORKERS=""
SIM_WORKERS_GIVEN=false
# Precompile (inline, see scripts/lib/tt_safe_run.sh): defaults + the JIT endpoint from
# $TT_METAL_JIT_SERVER_ENDPOINT. AUTO turns it on for broad selections and off for narrow ones
# (::nodeid / -k); decided from argv below, after parsing.
ttrun_precompile_defaults auto TT_METAL_JIT_SERVER_ENDPOINT

# Wrapper flags are POSITION-INDEPENDENT: they may appear anywhere in argv, before or after the
# test path, interleaved with pytest args. Anything the wrapper does not recognize is collected
# verbatim (in order) into PYTEST_ARGS and forwarded to pytest untouched.
# This loop used to `break` at the first unrecognized argument, which made position silently
# significant: a `--dev` / `--run-all` / `--profile` written after the test path was forwarded to
# pytest, which has no such option, so the run died with a generic "usage error" that blamed the
# path — and `--run-all` in that position also failed to disable -x.
PYTEST_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dev)
            DEV_MODE=true
            shift
            ;;
        --run-all)
            FAIL_FAST=false
            shift
            ;;
        --device)
            if [[ $# -lt 2 ]]; then
                echo "SAFE_PYTEST_ERROR: --device requires an argument (UMD card id or 'auto')"
                exit 3
            fi
            DEVICE_SELECTOR="$2"
            shift 2
            ;;
        --device=*)
            DEVICE_SELECTOR="${1#*=}"
            shift
            ;;
        --mesh)
            MESH_MODE=true
            shift
            ;;
        --profile)
            # Run the real pytest under the Tracy device profiler (see PYTEST_CMD below).
            PROFILE_MODE=true
            shift
            ;;
        --sim-workers)
            if [[ $# -lt 2 ]]; then
                echo "SAFE_PYTEST_ERROR: --sim-workers requires an integer argument"
                exit 3
            fi
            SIM_WORKERS="$2"
            SIM_WORKERS_GIVEN=true
            shift 2
            ;;
        --precompile|--no-precompile|--precompile-workers|--farm-min-programs|--jit-server|--jit-server=*|--no-jit-server)
            # Shared precompile / JIT-server flags (scripts/lib/tt_safe_run.sh).
            ttrun_precompile_parse_flag "$@" || exit 3
            shift "$TTRUN_CONSUMED"
            ;;
        *)
            PYTEST_ARGS+=("$1")
            shift
            ;;
    esac
done

# --- Validate card selection (auto | N | mesh, honours a single-int TT_VISIBLE_DEVICES) ---
ttrun_resolve_selector || exit 3

# --- Validate --sim-workers ---
if [[ "$SIM_WORKERS_GIVEN" == true ]]; then
    if [[ "$SIM_MODE" == false ]]; then
        echo "SAFE_PYTEST_ERROR: --sim-workers is only valid when TT_METAL_SIMULATOR is set"
        exit 3
    fi
    if ! [[ "$SIM_WORKERS" =~ ^[1-9][0-9]*$ ]]; then
        echo "SAFE_PYTEST_ERROR: --sim-workers must be a positive integer (got: $SIM_WORKERS)"
        exit 3
    fi
fi

# --- Default sim worker count: 16 ---
if [[ "$SIM_MODE" == true && -z "$SIM_WORKERS" ]]; then
    SIM_WORKERS=16
fi

# --- Argument validation ---
if [[ ${#PYTEST_ARGS[@]} -eq 0 ]]; then
    echo "SAFE_PYTEST_ERROR: No test path provided" >&2
    echo "Usage: scripts/run_safe_pytest.sh [--device N|auto] [--mesh] [--dev] [--run-all] [--profile] [--sim-workers N] ${TTRUN_PRECOMPILE_USAGE} <test_path> [extra_pytest_args...]" >&2
    exit 3
fi

# Best-effort test path, used ONLY for the device-timing record and log lines: the first pytest arg
# that is neither an option nor the value of a value-taking option. Informational by construction —
# the pytest command lines below forward PYTEST_ARGS verbatim, so a mis-detection here can never
# change which tests run. (Previously TEST_PATH was literally $1 and was spliced back into the
# command, so `-k` written before the path became the "test path".)
TEST_PATH=""
_skip_next=false
for _arg in "${PYTEST_ARGS[@]}"; do
    if [[ "$_skip_next" == true ]]; then _skip_next=false; continue; fi
    case "$_arg" in
        # Separated-value options: the NEXT token is their value, not a path.
        -k|-m|-n|-p|-c|-o|-W|--deselect|--ignore|--rootdir|--junitxml|--maxfail|--timeout|--log-level)
            _skip_next=true; continue ;;
        -*) continue ;;
    esac
    TEST_PATH="$_arg"
    break
done
if [[ -z "$TEST_PATH" ]]; then
    echo "SAFE_PYTEST_ERROR: No test path found in: $(printf '%q ' "${PYTEST_ARGS[@]}")" >&2
    echo "SAFE_PYTEST_ERROR: pass a file, directory or ::nodeid — a bare filter (e.g. only -k) would collect the whole repo" >&2
    exit 3
fi
TT_TIMING_TEST_PATH="$TEST_PATH"

# --- Precompile AUTO decision (argv only, no pre-pass) ---
# A ::nodeid or a -k filter selects a handful of programs. For those the collect pass costs more
# than the parallel compile saves (the compile is ~2s for one program, cold), so leave them plain.
# Explicit --precompile / --no-precompile always win.
PRECOMPILE_NARROW=false
if [[ "$TEST_PATH" == *"::"* ]]; then
    PRECOMPILE_NARROW=true
else
    for _arg in "${PYTEST_ARGS[@]}"; do
        if [[ "$_arg" == "-k" || "$_arg" == -k=* || "$_arg" == "--keyword" ]]; then
            PRECOMPILE_NARROW=true
            break
        fi
    done
fi
if [[ "$PRECOMPILE" == auto ]]; then
    if [[ "$PRECOMPILE_NARROW" == true ]]; then
        PRECOMPILE=false
        TT_TIMING_PRECOMPILE_REASON="narrow"
    else
        PRECOMPILE=true
    fi
fi


# --- Profiler CSV reporting (--profile) ---
# Newest ops_perf_results CSV before the run; snapshotted just before pytest (below).
PROFILE_CSV_BEFORE=""

# Print this run's per-op CSV path. `python -m tracy -r` writes a fresh
# reports/<ts>/ subdir per run, so we report the newest only if it differs from
# the pre-run snapshot — otherwise this run produced none and it's a stale leftover.
# Plain echo (stdout) so it lands next to the SAFE_PYTEST_RESULT line.
emit_profiler_csv() {
    [[ "$PROFILE_MODE" == true ]] || return 0
    local csv
    csv=$(ls -t "${PROFILE_REPORTS_DIR}"/*/ops_perf_results*.csv 2>/dev/null | head -1)
    if [[ -n "$csv" && "$csv" != "$PROFILE_CSV_BEFORE" ]]; then
        echo "SAFE_PYTEST: PROFILER CSV: ${csv}"
    else
        echo "SAFE_PYTEST: WARNING: --profile set but this run produced no ops_perf_results CSV"
    fi
}

# --- Total-run timer: always the last line, on every exit path ---
ttrun_on_exit ttrun_print_total_runtime

# --- Acquire a card (hardware only; no-op on sim) ---
# auto = lowest free card; N = that card; mesh = every card. Exports the per-card env,
# re-points the triage / watcher paths, resets the card if a previous run left it dirty.
ttrun_acquire_card || exit 3
if [[ "$SIM_MODE" == false ]]; then
    PROFILE_REPORTS_DIR="${TTPOOL_PROFILER_DIR}/reports"
fi
ttrun_activate_venv

# --- Profiling preflight ---
# `python -m tracy` needs a Tracy-enabled build and tracy deps (e.g. websockets).
# Probe the import it does at startup (tracy.__main__ -> tracy.serve_wasm) so a
# missing dep fails fast here instead of as a confusing mid-run traceback.
if [[ "$PROFILE_MODE" == true ]]; then
    if ! python3 -c "import tracy.serve_wasm" 2>/dev/null; then
        echo "SAFE_PYTEST_ERROR: --profile requested but 'python -m tracy' is unavailable"
        echo "SAFE_PYTEST: Ensure a Tracy-enabled build and tracy deps (e.g. 'pip install websockets')"
        exit 3
    fi
    if [[ "$SIM_MODE" == false ]]; then
        # Per-card artifacts dir (both tools/tracy and the C++ profiler honour it): the default
        # generated/profiler is shared and concurrent runs corrupt each other's device CSV and
        # .tracy capture. Explicit capture port: tracy's get_available_port() is check-then-bind,
        # so simultaneous runs all pick 8086 and only one of them captures.
        export TT_METAL_PROFILER_DIR="$TTPOOL_PROFILER_DIR"
        TRACY_PORT_ARGS=(-t "$TTPOOL_TRACY_PORT")
        # tools/tracy resolves tracy-capture under $TT_METAL_HOME/build; a TT_METAL_HOME pointing
        # at another checkout pairs a foreign capture binary with this build's Tracy client and
        # the capture silently produces nothing ("capture output file was not generated").
        if [[ -n "${TT_METAL_HOME:-}" && "$(readlink -f "$TT_METAL_HOME")" != "$(readlink -f "$REPO_DIR")" ]]; then
            echo "SAFE_PYTEST: WARNING: TT_METAL_HOME='${TT_METAL_HOME}' is not this checkout; using ${REPO_DIR} for this profiled run so tracy-capture matches the client"
            export TT_METAL_HOME="$REPO_DIR"
        fi
    fi
fi

# --- Pre-flight: pytest-xdist required for sim parallelism (SIM_WORKERS > 1) ---
if [[ "$SIM_MODE" == true && "$SIM_WORKERS" -gt 1 ]]; then
    if ! python3 -c "import xdist" 2>/dev/null; then
        echo "SAFE_PYTEST_ERROR: --sim-workers=${SIM_WORKERS} requires pytest-xdist."
        echo "                   Install with: pip install pytest-xdist"
        echo "                   Or pass --sim-workers 1 to run serially."
        exit 3
    fi
fi

# --- Debug/sim mode env (asserts + watcher) ---
# Exported before pytest starts, so the precompile compile step inside the session sees the
# compile-time flags (see ttrun_export_dev_env for the full rationale).
[[ "$DEV_MODE" == true ]] && ttrun_export_dev_env
_banner_extra=""
[[ "$SIM_MODE" == true ]] && _banner_extra="workers=${SIM_WORKERS}"
ttrun_print_mode_banner "$_banner_extra"

# --- XIP disassembly dump (default OFF; kept under --dev) ---
# Must be exported before pytest starts: the dump fires on every kernel BINARY
# LOAD, including the precompile compile step inside the session.
#
# tt_memory.cpp re-writes the XIP-transformed ELF as <kernel>.xip.elf on every
# kernel BINARY LOAD — ~514 KB per kernel, so ~1.4 GB on a 2753-kernel golden
# run, MEASURED at ~12% of wall on a cold farm build (51s -> 45s on a 173-kernel
# bench). The XIP transform itself is one line (`segments.front().address = 0`),
# so the file's entire information content is a single rebase.
#
# Its ONLY consumer is tools/triage/check_binary_integrity.py, which byte-compares
# L1 against the loaded image. dump_callstacks.py — the script that actually
# diagnoses hangs — never reads any ELF, and the field is already optional
# (kernel_xip_path is None for NCRISC on wormhole, and the check is guarded).
# So: keep the dump under --dev, where triage is the whole point, and drop it for
# ordinary runs. Set TT_METAL_DISABLE_XIP_DUMP=0 to force it back on.
if [[ "$DEV_MODE" != true ]]; then
    export TT_METAL_DISABLE_XIP_DUMP="${TT_METAL_DISABLE_XIP_DUMP:-1}"
fi

# --- Hang detection setup (hardware only) ---
# On dispatch timeout the runtime runs tt-triage scoped to our card(s); see ttrun_setup_hang_hook.
ttrun_setup_hang_hook

# --- Build the pytest command ---
# Wrapper-injected options go FIRST and the user's args LAST, verbatim and contiguous. Both halves
# of that ordering are load-bearing:
#   * Nothing may be spliced INTO the user's args. -x used to be inserted straight after the test
#     path, so `-k` written before the path became the "test path" and -x landed where the filter
#     expression belonged: `pytest -k -x '<expr>' <path>` -> "argument -k: expected one argument".
#   * User args last means an explicit `--maxfail=N` still wins over our -x, as it did before.
# pytest options are order-independent, so leading -x / -n behave exactly as trailing ones did.
#
# --profile: wrap pytest in the Tracy profiler. `python -m tracy -r` runs pytest
#   as a child and post-processes results into ops_perf_results*.csv on pass or
#   fail. Its exit-code masking is handled at the result check below.
if [[ "$PROFILE_MODE" == true ]]; then
    PYTEST_CMD=(python -m tracy -r "${TRACY_PORT_ARGS[@]}" -m pytest)
else
    PYTEST_CMD=(pytest)
fi
# --- Precompile: load the collector into this session ---
# Shared decision (sim rule, farm/local route, plugin env); this runner's policy on an
# unreachable JIT server is ABORT (exit 4). The collect pass, compile step and real pass are one
# process, so they share the user's TT_METAL_CACHE (and, under --profile, one build key).
ttrun_precompile_prepare abort || exit $?
ttrun_precompile_cmd "${PYTEST_CMD[@]}"
PYTEST_CMD=("${PRECOMPILE_CMD[@]}")
# -x: stop on first failure (avoids running tests after a hang bricks the device)
# --run-all: skip -x to get full pass/fail counts (for eval scoring)
if [[ "$FAIL_FAST" == true ]]; then
    PYTEST_CMD+=(-x)
fi
# Sim parallelism via pytest-xdist. Each worker dlopens its own libttsim;
# DRAM is MAP_PRIVATE per-process so workers don't interfere. Skip when
# workers=1 to avoid xdist setup overhead.
if [[ "$SIM_MODE" == true && "$SIM_WORKERS" -gt 1 ]]; then
    PYTEST_CMD+=(-n "$SIM_WORKERS")
fi
PYTEST_CMD+=("${PYTEST_ARGS[@]}")

# Echo the EXACT command, shell-quoted (printf %q). The old line interpolated "$*", which drops all
# quoting: `-k "a or b"` printed as `-k a or b`, so readers concluded the expression had been
# word-split even on runs where it was passed through intact.
echo "SAFE_PYTEST: $(printf '%q ' "${PYTEST_CMD[@]}")"
echo "========================================"

# --- Mark device dirty before running tests (hardware only) ---
# Pessimistic: assume the device will get corrupted. If the script is killed at any
# point (SIGKILL, OOM, etc.), the flag persists and the next runner will reset.
# Cleared on clean exit or after a successful inline reset.
ttrun_mark_dirty

# --- Run pytest ---

# Snapshot the newest CSV now, so emit_profiler_csv can tell this run's report
# from a pre-existing one afterward.
if [[ "$PROFILE_MODE" == true ]]; then
    PROFILE_CSV_BEFORE=$(ls -t "${PROFILE_REPORTS_DIR}"/*/ops_perf_results*.csv 2>/dev/null | head -1)
fi

# Signal forwarding + run-under-tee live in ttrun_run_child (sets EXIT_CODE).
ttrun_run_child "$PYTEST_STDOUT_LOG" "${PYTEST_CMD[@]}"

echo "========================================"

# --- Handle result ---
# The triage-log guard matters in profile mode: the tracy wrapper exits 0 even
# when the underlying test failed OR hung, so without it a hang would be reported
# PASS and skip the device reset. An empty triage log means no hang fired.
ttrun_precompile_record "$PYTEST_STDOUT_LOG"   # timing-record fields only; never affects the verdict

if [[ $EXIT_CODE -eq 0 && ! -s "$TRIAGE_LOG" ]]; then
    ttrun_clear_dirty
    ttrun_cleanup_tmp "$PYTEST_STDOUT_LOG"
    emit_profiler_csv
    echo "SAFE_PYTEST_RESULT: PASS"
    exit 0
fi

# Pytest setup errors (no device touched):
#   4 = usage error (bad args, nonexistent path)
#   5 = no tests collected (typo in path, bad marker filter, etc.)
if [[ $EXIT_CODE -eq 4 || $EXIT_CODE -eq 5 ]]; then
    ttrun_clear_dirty
    ttrun_cleanup_tmp "$PYTEST_STDOUT_LOG"
    if [[ $EXIT_CODE -eq 4 ]]; then
        echo "SAFE_PYTEST_ERROR: Pytest usage error (invalid path or arguments)"
    else
        echo "SAFE_PYTEST_ERROR: No tests collected"
    fi
    exit 3
fi

# Hang? (HW: triage log non-empty; sim: libttsim watchdog message in the pytest output.)
ttrun_detect_hang "$PYTEST_STDOUT_LOG"

# Only reset device when the failure might have left it dirty.
# Hangs and crashes corrupt device state. Normal test failures (PCC mismatch,
# assertion errors) and collection errors don't touch the device.
if [[ "$IS_HANG" == true ]]; then
    ttrun_reset_cards
    echo "SAFE_PYTEST_RESULT: HANG (exit code: $EXIT_CODE)"
    ttrun_dump_hang_logs
    ttrun_cleanup_tmp "$PYTEST_STDOUT_LOG"
    exit 2
fi

ttrun_clear_dirty
ttrun_cleanup_tmp "$PYTEST_STDOUT_LOG"
# Note: $EXIT_CODE here is pytest's internal exit code (e.g. 1 = test failure,
# 2 = collection error / user interrupt). The wrapper's own exit code is
# always 1 for this branch — exit 2 is reserved for real dispatch-timeout
# hangs (handled above). The label keeps both visible to readers and to
# hooks parsing this output.
echo "SAFE_PYTEST_RESULT: FAIL (pytest exit code: $EXIT_CODE; wrapper exit: 1)"
exit 1
