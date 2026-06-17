#!/usr/bin/env bash
# Reproduce the remote-JIT-server warm-pass hang (see SERVER_HANG_HANDOFF.md).
#
# Runs the --precompile warm pass for the first N rms_norm golden cases against the
# remote JIT server with a COLD local cache and a DEWARMED server .elf cache, so every
# kernel must compile + round-trip. Under high concurrency a response is lost and the
# warm pass wedges forever (no "warmup complete"); `timeout` then kills it.
#
# Usage: repro_server_hang.sh [WORKERS] [TIMEOUT_S] [NUM_CASES]
set -u
cd "$(git rev-parse --show-toplevel)"

WORKERS="${1:-128}"
TIMEOUT_S="${2:-420}"
NUM_CASES="${3:-500}"

C=bgdepyc01-special-mstaletovic-for-reservation-24729
REPRO_DIR=tt_metal/impl/jit_server/repro

source python_env/bin/activate 2>/dev/null
export PYTHONPATH="$PWD:$PWD/tt_metal/third_party/tt_ops_code_gen"

# --- Client points at the remote server ---
export TT_METAL_JIT_SERVER_ENABLE=1
export TT_METAL_JIT_SERVER_ENDPOINT=bgdepyc01:54210
export TT_METAL_JIT_PREPROCESS=1

# --- Cold local cache (fresh each run) ---
COLD="$PWD/$REPRO_DIR/cold_cache_$$"
rm -rf "$COLD"; mkdir -p "$COLD"
export TT_METAL_CACHE="$COLD"

# --- Dewarm the server .elf cache so it must compile fresh ---
echo "REPRO: dewarming server rms_norm cache..."
ssh -o ConnectTimeout=10 bgdepyc01 "docker exec $C bash -c 'find /tmp/tt-metal-cache -type d -name \"rms_norm*\" -exec rm -rf {} + ; echo dewarmed'" 2>&1 | tail -1

# --- Node IDs ---
mapfile -t IDS < <(head -"$NUM_CASES" "$REPRO_DIR/all_golden_nodeids.txt")
echo "REPRO: ${#IDS[@]} golden node ids, $WORKERS workers, timeout ${TIMEOUT_S}s, cold cache $COLD"

t0=$(date +%s)
timeout --signal=INT --kill-after=20 "$TIMEOUT_S" \
    scripts/run_safe_pytest.sh --precompile --precompile-workers "$WORKERS" --run-all "${IDS[@]}"
rc=$?
t1=$(date +%s)
echo "REPRO: exit=$rc after $((t1-t0))s (124/137 => timeout-killed => WEDGE reproduced)"
exit $rc
