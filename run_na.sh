#!/usr/bin/env bash
# Single-log dev loop for the 3D neighborhood attention work.
#
#   Terminal A (leave running):  tail -f generated/na_run.log
#   Terminal B:                  ./run_na.sh [-b] [-k EXPR] [-t SECS]
#
# EVERYTHING goes in the log, unfiltered and unbuffered: build output, every device log line,
# the full pytest failure, the watcher summary. Nothing is grepped away. The log is truncated
# in place at the start of each run, so the tail pane always shows exactly the current attempt
# and `tail -f` survives across runs.
#
#   -b        rebuild + install the ttnn extension first. Needed after any HOST C++ change
#             (program factory, device op, plan, nanobind). Kernel changes under
#             device/kernels/ are JIT-compiled at run time and need no rebuild.
#   -k EXPR   pytest -k selector, e.g. -k "single_chunk and stride_one"
#   -f FILE   pytest file to run (default: the correctness test)
#   -p SCRIPT plain python script to run instead of pytest (e.g. the DiffVAE decode)
#   -w        enable the watcher. Off by default: it prints 32 "checking device" lines every
#             10s, drowning the log, and it slows every op. Turn it on to debug a hang.
#   -t SECS   timeout, default 120. A device-side deadlock sits forever otherwise, holding
#             CHIP_IN_USE_0_PCIe and blocking every later run.
#   -s SECS   stall watchdog, default 40. After the test body starts (compile/warmup), if the
#             log shows no probe-time / PASSED / FAILED for this long, kill the job. These
#             NA hangs show up in ~40s; do not sit on -t. 0 disables.
#
# Watcher waypoints, when a hang is dumped:
#   CWFW = waiting on a circular buffer front      CRBW = waiting to reserve a CB
#   MWDD = math waiting on DEST                    W    = finished
# The five fields per core are BRISC, NCRISC, TRISC0, TRISC1, TRISC2.

set -uo pipefail
cd "$(dirname "$0")"

LOG=generated/na_run.log
TEST=models/tt_dit/tests/unit/test_neighborhood_sdpa.py
TIMEOUT=120
STALL=40
SELECTOR=""
REBUILD=0
WATCHER=0
SCRIPT=""

while getopts "bk:t:f:p:ws:" option; do
    case $option in
        b) REBUILD=1 ;;
        k) SELECTOR="$OPTARG" ;;
        t) TIMEOUT="$OPTARG" ;;
        s) STALL="$OPTARG" ;;
        f) TEST="$OPTARG" ;;
        p) SCRIPT="$OPTARG" ;;
        w) WATCHER=1 ;;
        *) echo "usage: $0 [-b] [-k EXPR] [-t SECS] [-s STALL]" >&2; exit 2 ;;
    esac
done

mkdir -p generated
: > "$LOG"   # truncate, keeping the inode so an existing `tail -f` keeps working

# Everything written from here on goes to both the terminal and the log, line buffered.
exec > >(stdbuf -oL tee -a "$LOG") 2>&1

echo "=== $(date '+%H:%M:%S')  neighborhood attention run ==========================="
echo "target   : ${SCRIPT:-$TEST}"
[ -n "$SELECTOR" ] && echo "selector : $SELECTOR"
echo "timeout  : ${TIMEOUT}s     stall: ${STALL}s     rebuild: $REBUILD     watcher: $WATCHER"
echo

if fuser -s /dev/tenstorrent/0 2>/dev/null; then
    echo "--- killing stale device holder(s) ---"
    fuser -k -v /dev/tenstorrent/0
    sleep 2
fi

if [ "$REBUILD" = "1" ]; then
    echo "--- rebuild (full output) ---"
    if stdbuf -oL cmake --build build_Release --target ttnn; then
        cp build_Release/ttnn/_ttnn.so ttnn/ttnn/_ttnn.so
        echo "--- rebuild ok, installed ---"
    else
        echo "!!! BUILD FAILED (exit $?) -- not running the test"
        exit 1
    fi
    echo
fi

rm -rf generated/watcher   # so the summary below can only describe THIS run

[ "$WATCHER" = "1" ] && export TT_METAL_WATCHER=10

if [ -n "$SCRIPT" ]; then
    echo "--- python $SCRIPT (full output, nothing filtered) ---"
    PYTHONUNBUFFERED=1 stdbuf -oL -eL timeout "$TIMEOUT" ./python_env/bin/python "$SCRIPT" &
    CHILD=$!
else
    echo "--- pytest (full output, nothing filtered) ---"
    # -s disables pytest's output capture, so device logs stream live instead of appearing
    # all at once when the process ends.
    PYTEST_ARGS=(-q -s --no-header -p no:cacheprovider)
    [ -n "$SELECTOR" ] && PYTEST_ARGS+=(-k "$SELECTOR")
    PYTHONUNBUFFERED=1 \
        stdbuf -oL -eL timeout "$TIMEOUT" ./python_env/bin/python -m pytest "$TEST" "${PYTEST_ARGS[@]}" &
    CHILD=$!
fi

WATCHDOG=""
if [ "$STALL" != "0" ]; then
    python3 ./na_stall_watchdog.py --log "$LOG" --pid "$CHILD" --stall "$STALL" &
    WATCHDOG=$!
fi
wait "$CHILD"
STATUS=$?
if [ -n "$WATCHDOG" ]; then
    kill "$WATCHDOG" 2>/dev/null || true
    wait "$WATCHDOG" 2>/dev/null || true
fi

echo
if [ "$STATUS" = "124" ] || [ "$STATUS" = "137" ] || [ "$STATUS" = "143" ]; then
    if [ "$STATUS" = "124" ]; then
        echo "!!! TIMED OUT after ${TIMEOUT}s -- device-side deadlock."
    else
        echo "!!! KILLED (exit $STATUS) -- stall watchdog or signal; treating as device deadlock."
    fi
    WATCHER=generated/watcher/watcher.log
    if [ -f "$WATCHER" ]; then
        echo "--- kernels in this program ---"
        grep neighborhood generated/watcher/kernel_names.txt 2>/dev/null
        echo "--- device 0 worker states (BRISC,NCRISC,TRISC0,TRISC1,TRISC2), last dump ---"
        LAST=$(grep -n "Dump #" "$WATCHER" | tail -2 | head -1 | cut -d: -f1)
        sed -n "${LAST},\$p" "$WATCHER" | grep "^Device  0 worker" \
            | sed 's/.*virtual([^)]*): //; s/ *rmsg.*//' | sort | uniq -c | sort -rn
        echo "--- full watcher log: $WATCHER ---"
    else
        echo "(no watcher log -- hung before the device opened?)"
    fi
    echo
    echo "CWFW=waiting on CB front  CRBW=waiting to reserve  MWDD=math waiting on DEST  W=done"
    echo
    # Killing the host process does NOT stop the RISCs -- they keep spinning on the deadlock,
    # and the next run then fails at device init with "failed to initialize FW". Reset here so
    # the next attempt starts from a clean board instead of inheriting this one's wreckage.
    echo "--- resetting board (a deadlock leaves the cores spinning) ---"
    timeout 600 tt-smi -r all 2>&1 | tail -5
elif [ "$STATUS" = "0" ]; then
    echo "=== PASSED ==="
else
    echo "=== FAILED (exit $STATUS) ==="
fi

echo "=== end $(date '+%H:%M:%S') ==========================================="
exit "$STATUS"
