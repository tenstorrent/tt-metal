#!/bin/bash
# precompile_warm.sh — standalone, CI-friendly JIT-cache warm (no flock / no device reset / no triage).
#
# Warms WHATEVER cache TT_METAL_CACHE points at (or tt-metal's default) for a given pytest selection,
# hardware-free & in parallel, so a *subsequent* `pytest <same selection>` in the same environment
# reuses it instead of compiling inline. This is the warm half of `run_safe_pytest.sh --precompile`,
# factored out so CI can call it as its own step (the device test step that follows is unchanged).
#
# Usage:  scripts/precompile_warm.sh <test_path> [extra pytest args matching the real run...]
#
# Contract: ALWAYS exits 0. Every failure path degrades to "no warm" -> the real run just compiles
# cold (correct, only slower). A definitive build_key PRE-FLIGHT skips a doomed warm before any work
# and says exactly why. The warm cache holds content-hashed kernels (byte-identical to a cold run),
# so results are never affected. Honors PRECOMPILE_WORKERS (default: nproc).
set -o pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:$PYTHONPATH}"
WORKERS="${PRECOMPILE_WORKERS:-$(nproc 2>/dev/null || echo 8)}"
DESC="${PRECOMPILE_DESC:-/tmp/tt_precompile_cluster_desc.yaml}"

if [[ $# -eq 0 ]]; then
    echo "PRECOMPILE: no test path given -> nothing to warm (real run will be COLD)." >&2
    exit 0
fi
TEST_PATH="$1"; shift
EXTRA_ARGS=("$@")

echo "PRECOMPILE: ===== warming JIT cache (hardware-free, ${WORKERS}-way) for: ${TEST_PATH} ${EXTRA_ARGS[*]} =====" >&2

# 1. cluster descriptor from UMD topology (HW-stable -> capture once)
if [[ ! -f "$DESC" ]]; then
    timeout 120 python3 - "$DESC" >"/tmp/precompile_desc_$$.log" 2>&1 <<'PY' || true
import sys, tt_umd
tt_umd.TopologyDiscovery.create_cluster_descriptor().serialize_to_file(sys.argv[1])
PY
fi
if [[ ! -f "$DESC" ]]; then
    echo "PRECOMPILE: ✗ cluster-descriptor capture failed/timed out -> COLD. See /tmp/precompile_desc_$$.log" >&2
    exit 0
fi

# 2. real device fingerprint: resolved 2-erisc + the build_key the real run will use.
#    MUST open the device the way tests do (open_mesh_device) — build_key differs from the single-device path.
timeout 180 python3 - >"/tmp/precompile_real_$$.log" 2>&1 <<'PY' || true
import ttnn
md = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
try:
    f = 1 if ttnn.cluster.get_enable_2_erisc_mode() else 0
    k = ttnn.cluster.get_build_key()
finally:
    ttnn.close_mesh_device(md)
print(f"RKEY {f} {k}")
PY
probe="$(grep '^RKEY ' "/tmp/precompile_real_$$.log" 2>/dev/null | tail -1 | sed 's/^RKEY //')"
if [[ -z "$probe" ]]; then
    echo "PRECOMPILE: ✗ couldn't read the device build_key -> COLD (device unhealthy, or build predates the" >&2
    echo "PRECOMPILE:   get_build_key/get_enable_2_erisc_mode bindings). See /tmp/precompile_real_$$.log" >&2
    exit 0
fi
force2="${probe%% *}"; realkey="${probe##* }"

# 3. hardware-free mock fingerprint build_key
timeout 120 env TT_METAL_FORCE_2_ERISC_MODE="$force2" TT_METAL_SLOW_DISPATCH_MODE=1 \
    TT_METAL_MOCK_CLUSTER_DESC_PATH="$DESC" \
    python3 - >"/tmp/precompile_mock_$$.log" 2>&1 <<'PY' || true
import ttnn
md = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
try:
    print("MKEY", ttnn.cluster.get_build_key())
finally:
    ttnn.close_mesh_device(md)
PY
mockkey="$(grep '^MKEY ' "/tmp/precompile_mock_$$.log" 2>/dev/null | tail -1 | sed 's/^MKEY //')"
if [[ -z "$mockkey" ]]; then
    echo "PRECOMPILE: ✗ couldn't compute the hardware-free build_key -> COLD. See /tmp/precompile_mock_$$.log" >&2
    exit 0
fi

# 4. PRE-FLIGHT: will the warm actually be reused by the real run?
if [[ "$mockkey" != "$realkey" ]]; then
    echo "PRECOMPILE: ✗ build_key MISMATCH — device uses ${realkey}, hardware-free fingerprint produces ${mockkey}." >&2
    echo "PRECOMPILE:   => warm would NOT be reused, so it is SKIPPED (no wasted work); real run is COLD but CORRECT." >&2
    echo "PRECOMPILE:   Cause: stale descriptor ($DESC) from another host, or a multi-device/Blackhole config the" >&2
    echo "PRECOMPILE:   (1,1) hardware-free fingerprint didn't reproduce (harvesting / dispatch_core / 2-erisc / arch)." >&2
    exit 0
fi
echo "PRECOMPILE: ✓ fingerprint matches the device (build_key ${realkey}) — the warm cache WILL be reused." >&2

# 5. hardware-free meta-collect over the SAME selection -> warms the shared cache in parallel.
#    Parallelism model: the session-end up_front_compile uses its OWN thread pool
#    (UP_FRONT_COLLECT_WORKERS, 0 => hardware_concurrency) — that is the dominant cost and needs no
#    xdist. If pytest-xdist IS installed we additionally shard the collect body-run across workers
#    (each then compiles its shard single-threaded); if not (e.g. CI venv), we collect in one process
#    and let the compile pool use all cores. Either way the *compile* is fully parallel.
declare -a PAR_ENV PAR_ARGS
if python3 -c "import xdist" >/dev/null 2>&1; then
    PAR_ENV=(UP_FRONT_COLLECT_WORKERS=1); PAR_ARGS=(-n "$WORKERS")
    echo "PRECOMPILE: xdist present -> ${WORKERS}-way parallel collect+compile" >&2
else
    PAR_ENV=(UP_FRONT_COLLECT_WORKERS=0); PAR_ARGS=()
    echo "PRECOMPILE: no xdist -> single-process collect, all-core parallel compile (hardware_concurrency)" >&2
fi
t0=$(date +%s)
env TT_METAL_FORCE_2_ERISC_MODE="$force2" TT_METAL_SLOW_DISPATCH_MODE=1 \
    TT_METAL_MOCK_CLUSTER_DESC_PATH="$DESC" \
    UP_FRONT_COLLECT=1 UP_FRONT_META_COLLECT=1 "${PAR_ENV[@]}" \
    LOGURU_LEVEL=ERROR \
    pytest "$TEST_PATH" "${EXTRA_ARGS[@]}" -p up_front_collect_plugin "${PAR_ARGS[@]}" -s \
    >"/tmp/precompile_collect_$$.log" 2>&1 || true
echo "PRECOMPILE: ✓ warm pass complete in $(($(date +%s)-t0))s (build_key ${realkey}). Real run below reuses it." >&2
grep -E "UP_FRONT_COLLECT:|compiled [0-9]+ programs" "/tmp/precompile_collect_$$.log" 2>/dev/null | tail -3 >&2 || true
exit 0
