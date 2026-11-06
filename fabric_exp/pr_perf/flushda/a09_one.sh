#!/bin/bash
# One test_tt_fabric run on a09u32 through the tt-device broker (coordinator 2026-11-05; local): streamed output with a
# heartbeat, result line parsed, log at fabric_exp/results/LABEL.log, JIT cache per label (deleted after a pass).
#   a09_one.sh LABEL TREE YAML [TIMEOUT] [test_tt_fabric args...]
cd /home/aagarwal/tt-metal || exit 1
L=$1 W=$2 Y=$3 TO=${4:-900}; shift 4 2>/dev/null || shift $#
B=$W/build_Release; LOG=fabric_exp/results/$L.log; C=/home/aagarwal/.tt-metal-cache-$L
[ -f $Y ] && [ -x $B/test/tt_metal/tt_fabric/test_infra/test_tt_fabric ] || { echo "!!! no yaml $Y or no binary in $W"; exit 1; }
[ -e $LOG ] && { echo "!!! $LOG exists"; exit 1; }
echo "tree $W: $(git -C $W log --oneline -1); local diff: $(git -C $W diff --stat | tail -1)"
(while sleep 60; do echo "heartbeat $(date +%T)"; done) & HB=$!
trap 'kill $HB 2>/dev/null' EXIT
rm -rf "${C:?}"
(cd $W && env -u TT_METAL_SLOW_DISPATCH_MODE TT_METAL_HOME=$W TT_METAL_RUNTIME_ROOT=$W TT_METAL_CACHE=$C \
    TT_FABRIC_NO_BW_TELEMETRY=1 LD_LIBRARY_PATH=$B/tt_metal:$B/tt_stl:$B/lib:$B/tt_metal/third_party/umd/lib \
    timeout $TO $B/test/tt_metal/tt_fabric/test_infra/test_tt_fabric --test_config $Y "$@" 2>&1) \
    | tee $LOG | grep --line-buffered -aE "Running Test:|Receiver on|TT_FATAL|TT_THROW|FAILED|MISMATCH|Tests: [0-9]+ executed"
rc=${PIPESTATUS[0]}
res=$(sed 's/\x1b\[[0-9;]*m//g' $LOG | grep -aoE 'Tests: [0-9]+ executed, [0-9]+ failed' | tail -1)
echo "$L: exit $rc; ${res:-NO RESULT}"
[ $rc = 0 ] && echo "$res" | grep -q ", 0 failed" && { rm -rf "${C:?}"; exit 0; }
exit 1
