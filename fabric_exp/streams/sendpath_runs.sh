#!/bin/bash
# Before/after runs for moving the existing benchmark send paths onto the faster loops.
cd /home/aagarwal/tt-metal
P=fabric_exp/streams/port
C="TT_FABRIC_NO_BW_TELEMETRY=1 TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT=1000"
F=TT_FABRIC_EXP_FLAGS=13129088
W32="TT_FABRIC_EXP_WORKER_SLOTS=32 TT_FABRIC_EXP_MAX_SENDER_SLOTS=8"
SAN=/home/aagarwal/tt-metal-streams/tests/tt_metal/tt_fabric/test_infra/test_yamls
check() { echo "$1: $(grep -oE 'Tests: .*executed, [0-9]+ failed' $2) worker-bound=$(grep -c 'was worker-bound' $2)"; }
run_b() { timeout 1500 fabric_exp/streams/run_branch.sh $1 $2 > /dev/null || { pkill -x test_tt_fabric; tt-smi -r all > /dev/null 2>&1; }; check $1 fabric_exp/streams/results/$1.log; }
run_e() { l=$1; y=$2; shift 2; timeout 1500 fabric_exp/run_exp.sh $l $y $C "$@" > /dev/null || { pkill -x test_tt_fabric; tt-smi -r all > /dev/null 2>&1; }; check $l fabric_exp/results/$l.log; }
tt-smi -r all > /dev/null 2>&1
# stock, new send paths
run_b sp_stock_gate fabric_exp/streams/stream_gate.yaml
run_b sp_stock_2d fabric_exp/exp_2d_4352.yaml
run_b sp_stock_1d fabric_exp/streams/acct_1d_ring.yaml
run_b sp_stock_mesh_sanity $SAN/test_fabric_mesh_sanity_common.yaml
run_b sp_stock_stream_sanity $SAN/test_fabric_stream_sanity.yaml
# fast build, new send paths
run_e sp_new_13129088_gate fabric_exp/streams/stream_gate.yaml $F
run_e sp_new_13129088_w32_gate fabric_exp/streams/stream_gate.yaml $F $W32
run_e sp_new_13129088_2d fabric_exp/exp_2d_4352.yaml $F
run_e sp_new_13129088_1d fabric_exp/streams/acct_1d_ring.yaml $F
# fast build, old send paths (swap the previous patch in, then restore)
git apply -R $P/streams_p3f.patch && git apply $P/streams_p3c.patch && echo "swapped to p3c" || echo "SWAP FAILED"
CCACHE_BASEDIR=/home/aagarwal ninja -C build_Release test_tt_fabric > /dev/null 2>&1 && echo "old build ok"
run_e sp_old_13129088_2d fabric_exp/exp_2d_4352.yaml $F
run_e sp_old_13129088_1d fabric_exp/streams/acct_1d_ring.yaml $F
git apply -R $P/streams_p3c.patch && git apply $P/streams_p3f.patch && echo "restored p3f" || echo "RESTORE FAILED"
CCACHE_BASEDIR=/home/aagarwal ninja -C build_Release test_tt_fabric > /dev/null 2>&1 && echo "new build ok"
echo SENDPATH_RUNS_DONE
