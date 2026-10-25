#!/bin/bash
# Does channel-trimming replay work at all here? (RESULTS §94: the tree's v2 hung in trimmed replay with deterministic
# all-to-all traffic, job 276.) The 1D trimming suite (coverage/trim_1d.yaml: capture, then replay of the captured
# profile), each run in a fresh cache with a 240 s timeout that kills the test's process group; on a hang the
# read-only capture (hang_capture.py) records every router, then the job stops.
#   1. main-like: v1 router (TT_FABRIC_KERNEL_VERSION=1), study flags 0, firmware keepalive      label trimdiag_v1
#   2. only if 1 passes: the lead candidate v3l3be1c in v2's place, study settings               label trimdiag_m
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
V=fabric_exp/variants
PROFILE=generated/reports/channel_trimming_capture.yaml
trim_run() {  # label kernel-version flags keepalive
    local l=$1 c=/home/aagarwal/.tt-metal-cache-trimdiag-$1
    rm -rf "${c:?}"; rm -f $PROFILE
    echo "=== $l: kernel v$2, flags $3, keepalive $4 ==="
    env -u TT_METAL_SLOW_DISPATCH_MODE TT_FABRIC_KERNEL_VERSION=$2 timeout -k 15 240 fabric_exp/run_exp.sh "$l" \
        $CON/coverage/trim_1d.yaml TT_FABRIC_NO_BW_TELEMETRY=1 TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT=$4 \
        TT_FABRIC_EXP_FLAGS=$3 TT_METAL_CACHE=$c TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1 \
        | grep --line-buffered -v 'Running Test'
    local log=fabric_exp/results/$l.log
    local res=$(grep -oE 'Tests: .*executed, [0-9]+ failed' $log 2>/dev/null)
    echo "$l: ${res:-NO RESULT}; capture finished: $(grep -c 'CovTrim1d_capture Finished' $log), replay started: $(grep -c 'Running Test: CovTrim1d_replay' $log)"
    if [ -z "$res" ]; then
        sleep 5
        echo "=== $l: replay hung; read-only capture ==="
        timeout 260 python_env/bin/python $CON/hang_capture.py fabric_exp/results/${l}_hangcap --time-budget 200 \
            --meta label=$l --meta kernel=v$2 --meta flags=$3 --meta keepalive=$4 2>&1 | tail -15
        cp $PROFILE fabric_exp/results/${l}_hangcap/ 2>/dev/null
        return 1
    fi
    echo "$res" | grep -q ", 0 failed"
}
trim_run trimdiag_v1 1 0 0 || { echo "main-like router hangs too: trimming replay issue is not the study's routers"; rm -f $M; echo ALL_DONE; exit 0; }
python3 $CON/swap.py apply $V/v3l3be1c $BK || { echo STOPPED; exit 1; }
trim_run trimdiag_m 2 13129088 1000
python3 $CON/swap.py restore $BK
rm -f $M
echo ALL_DONE
