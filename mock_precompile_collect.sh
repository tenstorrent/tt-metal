#!/bin/bash
# Hardware-free, parallel precompile COLLECT via mock mode.
#
# Warms the on-disk JIT cache with kernels keyed by the SAME build_key the real device uses,
# WITHOUT a device, parallelized across cores via pytest-xdist. A subsequent real run then hits
# that cache. This removes the device as the scarce resource and parallelizes the collect beyond
# device count (the serial single-device collect was the precompile bottleneck).
#
# How the build_key is made to match the real device (see compute_build_key / get_compile_hash_string):
#   * dispatch_core_type ETH    -> mock honors get_default_dispatch_core_type() (ETH on this N300)
#   * harvesting_mask           -> capture the REAL cluster descriptor from UMD (TopologyDiscovery)
#                                  and point mock at it (generic descriptors have wrong harvesting)
#   * enable_2_erisc_mode       -> mock force-disables it. CAPTURE the RESOLVED value off a real
#                                  device (ttnn.cluster.get_enable_2_erisc_mode, post firmware-cap
#                                  resolution) and replay it via TT_METAL_FORCE_2_ERISC_MODE=0|1.
#                                  General on any arch (incl. a Blackhole downgraded to single-erisc
#                                  on old eth-fw, which mock cannot re-derive). Requires the
#                                  rtoptions.cpp + ttnn cluster-getter patches built into the .so.
#   * slow dispatch             -> avoids the mock fast-dispatch CQ-config segfault; build_key is
#                                  dispatch-mode-invariant on real HW, so this is free.
# Only the explicit detail::CompileProgram path (up_front_collect_plugin) compiles op kernels under
# mock; the op LAUNCH path does not. So this MUST use the plugin, not a bare suite run.
#
# Usage: mock_precompile_collect.sh <cache_dir> <xdist_n> <pytest target/args...>
set -uo pipefail
WT=/localdev/mstaletovic/2026_05_28/0104_mstaletovic_agent_eval/wt_origin_main
CACHE="$1"; shift
NJ="$1"; shift
DESC=/tmp/real_cluster_desc.yaml
ERISC2F=/tmp/real_2erisc.txt
cd "$WT"

# 1a. Capture the real device's cluster descriptor ONCE (harvesting_mask; brief real-HW open).
if [ ! -f "$DESC" ]; then
    echo "[mock-collect] capturing real cluster descriptor from UMD ..."
    cat > /tmp/_dump_desc.py <<'PY'
import tt_umd
def test_dump():
    p = tt_umd.TopologyDiscovery.create_cluster_descriptor().serialize_to_file("/tmp/real_cluster_desc.yaml")
    print("DUMPED", p)
PY
    PYTHONPATH="$WT" scripts/run_safe_pytest.sh /tmp/_dump_desc.py -s 2>&1 | grep -E "DUMPED|passed|failed"
fi

# 1b. Capture the real device's RESOLVED enable_2_erisc_mode ONCE (brief real-HW open). This is the
#     value AFTER firmware-capability resolution, which mock cannot re-derive (no eth-fw query); we
#     replay it verbatim via TT_METAL_FORCE_2_ERISC_MODE so the build_key matches on any arch.
if [ ! -f "$ERISC2F" ]; then
    echo "[mock-collect] capturing real resolved enable_2_erisc_mode ..."
    cat > /tmp/_dump_erisc2.py <<'PY'
import ttnn
def test_dump():
    dev = ttnn.open_device(device_id=0)
    try:
        v = ttnn.cluster.get_enable_2_erisc_mode()
    finally:
        ttnn.close_device(dev)
    open("/tmp/real_2erisc.txt", "w").write("1" if v else "0")
    print("ERISC2", v)
PY
    PYTHONPATH="$WT" scripts/run_safe_pytest.sh /tmp/_dump_erisc2.py -s 2>&1 | grep -E "ERISC2|passed|failed"
fi
FORCE2="$(cat "$ERISC2F" 2>/dev/null || echo 1)"
echo "[mock-collect] replaying enable_2_erisc_mode=$FORCE2 (captured from real device)"

# 2. Hardware-free, xdist-parallel collect into the real-build_key cache.
rm -rf "$CACHE" && mkdir -p "$CACHE"
echo "[mock-collect] hardware-free collect -n $NJ into $CACHE ..."
TT_METAL_FORCE_2_ERISC_MODE="$FORCE2" TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_MOCK_CLUSTER_DESC_PATH="$DESC" \
    UP_FRONT_COLLECT=1 UP_FRONT_FAST_COLLECT=1 UP_FRONT_COLLECT_WORKERS=1 LOGURU_LEVEL=INFO \
    TT_METAL_CACHE="$CACHE" PYTHONPATH="$WT" \
    scripts/run_safe_pytest.sh --run-all "$@" -p up_front_collect_plugin -n "$NJ"
