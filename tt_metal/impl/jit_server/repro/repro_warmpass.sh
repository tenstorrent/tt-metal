#!/usr/bin/env bash
# Focused reproduction: run ONLY the JIT warm pass (up_front_collect plugin) against the
# remote JIT server, under the shared device flock + a timeout. Avoids the heavy real run.
#
# Wedge signature: prints "UP_FRONT_COLLECT: ... distinct ..." but NEVER prints
# "UP_FRONT_COLLECT: compiled N programs in Xs" -> timeout kills it (exit 124/137).
# Healthy: prints "compiled N programs in Xs" and exits 0 well within the timeout.
#
# Usage: repro_warmpass.sh [WORKERS] [TIMEOUT_S] [NUM_CASES]
set -u
cd "$(git rev-parse --show-toplevel)"

WORKERS="${1:-128}"
TIMEOUT_S="${2:-360}"
NUM_CASES="${3:-500}"

C=bgdepyc01-special-mstaletovic-for-reservation-24729
REPRO_DIR=tt_metal/impl/jit_server/repro
LOCK_FILE=/tmp/tt-device.lock

source python_env/bin/activate 2>/dev/null
export PYTHONPATH="$PWD:$PWD/tt_metal/third_party/tt_ops_code_gen"

# Client -> remote server
export TT_METAL_JIT_SERVER_ENABLE=1
export TT_METAL_JIT_SERVER_ENDPOINT=bgdepyc01:54210
export TT_METAL_JIT_PREPROCESS=1

# Cold local cache
COLD="$PWD/$REPRO_DIR/cold_cache_$$"
rm -rf "$COLD"; mkdir -p "$COLD"
export TT_METAL_CACHE="$COLD"

# Warm-pass plugin knobs (mirror run_safe_pytest precompile_warm)
export UP_FRONT_COLLECT=1 UP_FRONT_REAL_ALLOC=1 UP_FRONT_COLLECT_WORKERS="$WORKERS"
export LOGURU_LEVEL=ERROR

echo "REPRO: dewarming server rms_norm cache..."
ssh -o ConnectTimeout=10 bgdepyc01 "docker exec $C bash -c 'find /tmp/tt-metal-cache -type d -name \"rms_norm*\" -exec rm -rf {} + ; echo dewarmed'" 2>&1 | tail -1

# Node-id list is regenerable (and too big to commit) — generate it on first use.
GOLDEN=tt_metal/third_party/tt_ops_code_gen/eval/golden_tests/rms_norm/test_golden.py
if [[ ! -s "$REPRO_DIR/all_golden_nodeids.txt" ]]; then
    echo "REPRO: generating golden node-id list..."
    python "$REPRO_DIR/collect_ids.py" "$GOLDEN" "$REPRO_DIR/all_golden_nodeids.txt"
fi
mapfile -t IDS < <(head -"$NUM_CASES" "$REPRO_DIR/all_golden_nodeids.txt")
echo "REPRO: ${#IDS[@]} node ids, $WORKERS workers, timeout ${TIMEOUT_S}s, cold $COLD"

run_warm() {
    t0=$(date +%s)
    timeout --signal=INT --kill-after=20 "$TIMEOUT_S" \
        pytest "${IDS[@]}" -p tests.plugins.up_front_collect -p no:cacheprovider -q
    rc=$?
    t1=$(date +%s)
    # Classify: a WEDGE is a timeout-kill (124/137) where the warm pass never finished. Any other
    # exit means the warm pass RAN to completion — confirm by checking for the completion marker.
    # (exit 1 is normal here: the NO_DISPATCH pass-1 test bodies fail; that is NOT a wedge.)
    if [[ $rc -eq 124 || $rc -eq 137 ]]; then
        echo "REPRO: warm-pass TIMED OUT (exit=$rc) after $((t1-t0))s => WEDGE (grep log: no 'compiled N programs')"
    else
        echo "REPRO: warm-pass exit=$rc after $((t1-t0))s => RAN TO COMPLETION (exit 1 = expected"
        echo "       NO_DISPATCH body failures, NOT a wedge). Confirm via 'UP_FRONT_COLLECT: compiled N programs' in log."
    fi
    return $rc
}

# Serialize against other agents via the same lock run_safe_pytest uses.
exec 9>"$LOCK_FILE"
echo "REPRO: acquiring device lock $LOCK_FILE ..."
flock 9
echo "REPRO: lock acquired."
touch /tmp/tt-device.dirty   # warm pass opens the device; let next run_safe_pytest reset it
run_warm
rc=$?
rm -rf "$COLD"
exit $rc
