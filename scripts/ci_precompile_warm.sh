#!/bin/bash
# CI-friendly hardware-free JIT warm with build-fingerprint replay (the precompile-blackhole-fix path),
# as a standalone CI step (no flock / device reset / triage — those are the local run_safe_pytest's job).
# Inherits the container env (PYTHONPATH=/work etc.) so the warm uses the SAME ttnn the test step uses.
#
# Exercises the actual fix: capture the real device's JitBuildFingerprint (num_l1_banks, dispatch core
# type/axis, 2-erisc) -> replay it in the hardware-free (slow-dispatch) mock via TT_METAL_JIT_BUILD_FINGERPRINT
# so the warm-compiled kernels match the real fast-dispatch run. build_key PRE-FLIGHT gates reuse.
# ALWAYS exits 0 -> any failure degrades to a cold run (correct, just not warmed).
#
# Usage: scripts/ci_precompile_warm.sh <test_path> [extra pytest args matching the real run...]
set -uo pipefail
WORKERS="${PRECOMPILE_WORKERS:-$(nproc 2>/dev/null || echo 8)}"
DESC="${PRECOMPILE_DESC:-/tmp/tt_precompile_cluster_desc.yaml}"
FP="${PRECOMPILE_FINGERPRINT:-/tmp/tt_precompile_build_fingerprint.txt}"
[[ $# -eq 0 ]] && { echo "PRECOMPILE: no test path -> COLD"; exit 0; }
TEST_PATH="$1"; shift; EXTRA=("$@")

echo "PRECOMPILE: ===== warming (hardware-free, fingerprint replay) for: $TEST_PATH ${EXTRA[*]} ====="
echo "PRECOMPILE: TT_METAL_CACHE=${TT_METAL_CACHE:-<default>}  workers=$WORKERS"

# 1. cluster descriptor from UMD topology
timeout 120 python3 - "$DESC" >/tmp/ci_desc.log 2>&1 <<'PY' || true
import sys, tt_umd
tt_umd.TopologyDiscovery.create_cluster_descriptor().serialize_to_file(sys.argv[1])
PY
[[ -f "$DESC" ]] || { echo "PRECOMPILE: ✗ descriptor capture failed -> COLD"; cat /tmp/ci_desc.log; exit 0; }

# 2. real device probe: 2-erisc + build_key + CAPTURE the build fingerprint
timeout 180 env PRECOMPILE_FP="$FP" python3 - >/tmp/ci_real.log 2>&1 <<'PY' || true
import os, ttnn
md = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
try:
    f = 1 if ttnn.cluster.get_enable_2_erisc_mode() else 0
    k = ttnn.cluster.get_build_key()
    ttnn.cluster.capture_jit_build_fingerprint(os.environ["PRECOMPILE_FP"])
finally:
    ttnn.close_mesh_device(md)
print(f"RKEY {f} {k}")
PY
probe="$(grep '^RKEY ' /tmp/ci_real.log 2>/dev/null | tail -1 | sed 's/^RKEY //')"
[[ -z "$probe" ]] && { echo "PRECOMPILE: ✗ real build_key/fingerprint probe failed -> COLD"; tail -8 /tmp/ci_real.log; exit 0; }
read -r force2 realkey <<< "$probe"
echo "PRECOMPILE: real device: 2erisc=$force2 build_key=$realkey  fingerprint=$(cat "$FP" 2>/dev/null)"

# 3. hardware-free mock probe (slow dispatch + mock desc + fingerprint replay) -> build_key
timeout 120 env TT_METAL_FORCE_2_ERISC_MODE="$force2" TT_METAL_JIT_BUILD_FINGERPRINT="$FP" \
    TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_MOCK_CLUSTER_DESC_PATH="$DESC" \
    python3 - >/tmp/ci_mock.log 2>&1 <<'PY' || true
import ttnn
md = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
try: print("MKEY", ttnn.cluster.get_build_key())
finally: ttnn.close_mesh_device(md)
PY
mockkey="$(grep '^MKEY ' /tmp/ci_mock.log 2>/dev/null | tail -1 | sed 's/^MKEY //')"
[[ -z "$mockkey" ]] && { echo "PRECOMPILE: ✗ mock build_key probe failed -> COLD"; tail -8 /tmp/ci_mock.log; exit 0; }

# 4. PRE-FLIGHT: with the fingerprint replay these must match (the BH fix); pre-fix BH would MISMATCH here
if [[ "$mockkey" != "$realkey" ]]; then
    echo "PRECOMPILE: ✗ build_key MISMATCH (real=$realkey mock=$mockkey) -> COLD (fix not effective here)"
    exit 0
fi
echo "PRECOMPILE: ✓ build_key MATCHES ($realkey) with fingerprint replay — warm cache WILL be reused"

# 5. hardware-free meta-collect (fingerprint replayed) -> warms the shared TT_METAL_CACHE.
# SINGLE-PROCESS: the COMPILE is parallelized by the plugin's in-process C++ thread pool
# (UP_FRONT_COLLECT_WORKERS=N). An xdist multi-process prewarm loses ~half the cache (concurrent
# writers + per-worker dedup): measured full conv2d 47.6% (xdist) vs 99.8% (single-process), same build.
nf=(); cw="$WORKERS"; echo "PRECOMPILE: single proc x ${WORKERS} compile-threads"
t0=$(date +%s)
env TT_METAL_FORCE_2_ERISC_MODE="$force2" TT_METAL_JIT_BUILD_FINGERPRINT="$FP" \
    TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_MOCK_CLUSTER_DESC_PATH="$DESC" \
    UP_FRONT_COLLECT=1 UP_FRONT_META_COLLECT=1 UP_FRONT_COLLECT_WORKERS="$cw" LOGURU_LEVEL=ERROR \
    pytest "$TEST_PATH" "${EXTRA[@]}" -p up_front_collect_plugin "${nf[@]}" >/tmp/ci_warm_collect.log 2>&1
cs=$?
if [[ $cs -ne 0 ]]; then
    echo "PRECOMPILE: ✗ warm collect FAILED (pytest exit $cs) after $(($(date +%s)-t0))s -> warmed NOTHING; COLD"
    grep -iE "error|unrecognized|no tests" /tmp/ci_warm_collect.log | head -3
    exit 0
fi
echo "PRECOMPILE: ✓ warm complete in $(($(date +%s)-t0))s — real run reuses it"
grep -aE "UP_FRONT_COLLECT:|compiled [0-9]+ programs" /tmp/ci_warm_collect.log | tail -3
exit 0
