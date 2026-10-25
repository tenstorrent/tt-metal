#!/bin/bash
# Coverage job on bh-lb-120-a08u24 (broker job: no resets, output streamed, heartbeat; DO NOT run outside the broker).
# Logs the router compile and link commands of the one config type the CPU-only mock runs (coverage.py mock) cannot
# reach: channel-trimming replay, 1D line. test_tt_fabric's `enable_channel_trimming: true` runs the test twice:
# capture (records channel usage, exported at device close to generated/reports/channel_trimming_capture.yaml), then
# replay (rebuilds the routers from that profile: trimmed routers). The 2D replay builds are already logged (job 271,
# fabric_exp/results/cov_trim_2d.log), so this job runs the 1D suite only.
#
# Safety on the shared host:
#   - The traffic is deterministic all-to-all (coverage/trim_1d.yaml explains why). Job 271's random pairing gave the
#     replay different traffic from the capture, so it sent on trimmed channels and hung.
#   - A stale profile from an earlier run is removed first, so the replay can only use this run's capture.
#   - One suite with a 240 s timeout (SIGKILL 15 s later; timeout signals its whole process group, test_tt_fabric
#     included). The compile commands are logged before the replay launches. The job ends well
#     under the broker's 1,500 s cap.
# The tree is not modified (no swap): only the commands and the generated headers are wanted; coverage.py builds the
# variants offline. Afterwards (offline):
#   python3 fabric_exp/consolidation/coverage.py ingest fabric_exp/results/cov_trim_1d.log bh_1d_trimreplay
#   then `coverage.py build ... --cases bh_1d_trimreplay`. Keep the cache (~/.tt-metal-cache-cov-trim_1d) until then.
cd /home/aagarwal/tt-metal
(while sleep 60; do echo "heartbeat $(date +%T)"; done) & HB=$!
trap 'kill $HB 2>/dev/null' EXIT
PROFILE=generated/reports/channel_trimming_capture.yaml
run() {  # label yaml
    rm -rf /home/aagarwal/.tt-metal-cache-cov-$1
    rm -f $PROFILE
    env -u TT_METAL_SLOW_DISPATCH_MODE TT_FABRIC_KERNEL_VERSION=2 timeout -k 15 240 fabric_exp/run_exp.sh "cov_$1" "$2" \
        TT_FABRIC_NO_BW_TELEMETRY=1 TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT=1000 TT_FABRIC_EXP_FLAGS=13129088 \
        TT_METAL_CACHE=/home/aagarwal/.tt-metal-cache-cov-$1 TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1 \
        | grep --line-buffered -v 'Running Test'
    local log=fabric_exp/results/cov_$1.log
    local res=$(grep -oE 'Tests: .*executed, [0-9]+ failed' $log 2>/dev/null)
    local n=$(grep -c 'compile cmd: .*kernels/fabric_erisc_router_v2/' $log 2>/dev/null)
    local rep=$(grep -c 'Running Test: .*_replay' $log 2>/dev/null)
    echo "cov_$1: ${res:-NO RESULT}, $n router compile commands logged, replay started: $rep"
}
run trim_1d fabric_exp/consolidation/coverage/trim_1d.yaml
