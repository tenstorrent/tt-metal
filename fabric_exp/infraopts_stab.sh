#!/bin/bash
# Stability slices for the test-infra optimizations (aagarwal/fabric-test-infra-opts cherry-picked onto the PR head
# 8047cfa60ca, tree /home/aagarwal/tt-metal-infra-opts). Broker job: output streamed, heartbeat, no resets.
# usage: infraopts_stab.sh short|mid|full_short [on|off] [label-suffix ENV=VALUE...]   (on = the knobs the handoff makes default)
set -u
which=$1; knobs=${2:-on}; suffix=${3:-}; shift $(( $# < 3 ? $# : 3 )); EXTRA=("$@")
T=/home/aagarwal/tt-metal-infra-opts
export TT_METAL_HOME=$T TT_METAL_CLEAR_L1=1 TT_METAL_DEVICE_PROFILER=0
unset TT_METAL_SLOW_DISPATCH_MODE
cd $T
out=/home/aagarwal/tt-metal/fabric_exp/results; mkdir -p $out
label=infraopts_stab_${which}_${knobs}${suffix:+_$suffix}
K=()
if [ "$knobs" = on ]; then
  for k in SHARED_KERNELS PARALLEL_BUILD BATCHED_HOST_WRITES SENDER_HOT SENDER_SLOT_CACHE SENDER_BATCH SFPU_PAYLOAD SFPU_VERIFY; do
    K+=(TT_FABRIC_TEST_$k=1); done
  K+=(TT_FABRIC_TEST_MUX_FULL_SIZE_ITERS=8)
fi
(while sleep 60; do echo "heartbeat $(date +%T)"; done) & HB=$!
trap 'kill $HB 2>/dev/null' EXIT
echo "=== $label: ${K[*]:-no knobs} ${EXTRA[*]:-} ($(git log --oneline -1 | cut -c1-70)) ==="
t0=$(date +%s)
if [ "$which" = full_short ]; then  # the CI suite: test_fabric_stability.yaml *_short, 10 iterations
  ARGS=(--test_config $T/tests/tt_metal/tt_fabric/test_infra/test_yamls/test_fabric_stability.yaml --filter 'name.*_short')
elif [ "$which" = 10m_linear ] || [ "$which" = 10m_mesh ]; then  # 10M packets per flow, one group per job
  grp=Linear; [ "$which" = 10m_mesh ] && grp=Mesh
  ARGS=(--test_config /home/aagarwal/tt-metal/fabric_exp/infraopts/stab_10m.yaml --filter "name.FlowControl${grp}_10m")
elif [ "$which" = torus_y ]; then  # the _short test on a Y torus, unicast only (fabric_exp/infraopts)
  ARGS=(--test_config /home/aagarwal/tt-metal/fabric_exp/infraopts/stab_torus_y_short.yaml)
else
  ARGS=(--test_config $T/tests/tt_metal/tt_fabric/test_infra/handoff/yamls/stab_${which}_1it.yaml)
fi
env "${K[@]}" "${EXTRA[@]}" TT_FABRIC_NO_BW_TELEMETRY=1 TT_METAL_OPERATION_TIMEOUT_SECONDS=0 timeout 1380 \
    ./build_Release/test/tt_metal/tt_fabric/test_infra/test_tt_fabric "${ARGS[@]}" 2>&1 |
  tee $out/$label.log | grep --line-buffered -E "Running Test:|FATAL|error:|HUNG|mismatch|Mismatch|PASS|FAIL|Tests:"
st=${PIPESTATUS[0]}
echo "=== $label: exit $st, wall $(( $(date +%s)-t0 )) s ==="
grep -oE 'Tests: .*' $out/$label.log | tail -1
exit $st
