#!/bin/bash
# Bit 22 grouped transaction-ID flush (exp_flush_noc_trids_grouped, noinline) + 16-ID guard: validation, then
# benchmarks against the same-day baselines (b_*/x_* from bit2425_runs.sh). Labels: g_<flags>_*.
cd /home/aagarwal/tt-metal
KA="TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT=1000"
W="TT_FABRIC_EXP_WORKER_SLOTS=32 TT_FABRIC_EXP_MAX_SENDER_SLOTS=8"
SAN=tests/tt_metal/tt_fabric/test_infra/test_yamls
ok_cfgs=""
run() {  # label yaml [env...]; returns 1 on failure or hang
    timeout 1500 fabric_exp/run_exp.sh "$1" "$2" TT_FABRIC_NO_BW_TELEMETRY=1 $KA "${@:3}" > /dev/null 2>&1
    local status=$?
    local res=$(grep -oE 'Tests: .*executed, [0-9]+ failed' fabric_exp/results/$1.log 2>/dev/null)
    echo "$1: ${res:-NO RESULT (exit $status)} worker-bound=$(grep -c 'was worker-bound' fabric_exp/results/$1.log 2>/dev/null)"
    if [ $status -ne 0 ] || [ -z "$res" ] || ! echo "$res" | grep -q ", 0 failed"; then
        pkill -x test_tt_fabric; sleep 2; tt-smi -r all > /dev/null 2>&1
        return 1
    fi
    return 0
}
tt-smi -r all > /dev/null 2>&1
echo "== phase 1: validation"
for f in 13129088 4740480; do
    run g_${f}_mesh $SAN/test_fabric_mesh_sanity_common.yaml TT_FABRIC_EXP_FLAGS=$f || { echo "RULED OUT: $f (mesh sanity)"; continue; }
    run g_${f}_stream $SAN/test_fabric_stream_sanity.yaml TT_FABRIC_EXP_FLAGS=$f || { echo "RULED OUT: $f (stream sanity)"; continue; }
    ok_cfgs="$ok_cfgs $f"
done
echo "validated:$ok_cfgs"
[ -z "$ok_cfgs" ] && { echo ALL_DONE; exit 0; }
echo "== phase 2: benchmarks"
for f in $ok_cfgs; do
    run g_${f}_mix fabric_exp/streams/p3_mixes.yaml TT_FABRIC_EXP_FLAGS=$f
    run g_${f}_mix_trim fabric_exp/streams/p3_mixes_trim.yaml TT_FABRIC_EXP_FLAGS=$f
done
case " $ok_cfgs " in *" 13129088 "*)
    run g_13129088_4352 fabric_exp/exp_2d_4352.yaml TT_FABRIC_EXP_FLAGS=13129088
    run g_13129088_4352_trim fabric_exp/exp_2d_trim_4352.yaml TT_FABRIC_EXP_FLAGS=13129088
    run g_15226240w32_mix fabric_exp/streams/p3_mixes.yaml TT_FABRIC_EXP_FLAGS=15226240 $W
    run g_15226240w32_mix_trim fabric_exp/streams/p3_mixes_trim.yaml TT_FABRIC_EXP_FLAGS=15226240 $W
    ;;
esac
# Control without bit 22 (code path compiled out): should match m_546176_mix.
run g_546176_mix fabric_exp/streams/p3_mixes.yaml TT_FABRIC_EXP_FLAGS=546176
echo ALL_DONE
