#!/bin/bash
# Bits 26 (forwarding-channel mask gate), 27 (shared forwarding loop) and 28 (E1 partial-batch hold) on top of
# 13129088: validation, then benchmarks and latency against today's ACK-flush baselines (ak_*, byte-identical code).
# Labels: fm_<flags>_*.
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
has() { case " $ok_cfgs " in *" $1 "*) return 0;; esac; return 1; }
tt-smi -r all > /dev/null 2>&1
echo "== phase 1: validation"
for f in 80237952 147346816 281564544 348673408 415782272; do
    run fm_${f}_mesh $SAN/test_fabric_mesh_sanity_common.yaml TT_FABRIC_EXP_FLAGS=$f || { echo "RULED OUT: $f (mesh sanity)"; continue; }
    run fm_${f}_stream $SAN/test_fabric_stream_sanity.yaml TT_FABRIC_EXP_FLAGS=$f || { echo "RULED OUT: $f (stream sanity)"; continue; }
    ok_cfgs="$ok_cfgs $f"
done
echo "validated:$ok_cfgs"
echo "== phase 2: benchmarks"
for f in 80237952 147346816 281564544; do
    has $f || continue
    run fm_${f}_mix fabric_exp/streams/p3_mixes.yaml TT_FABRIC_EXP_FLAGS=$f
    run fm_${f}_4352 fabric_exp/exp_2d_4352.yaml TT_FABRIC_EXP_FLAGS=$f
    run fm_${f}_mix_trim fabric_exp/streams/p3_mixes_trim.yaml TT_FABRIC_EXP_FLAGS=$f
    run fm_${f}_4352_trim fabric_exp/exp_2d_trim_4352.yaml TT_FABRIC_EXP_FLAGS=$f
done
for f in 348673408 415782272; do
    has $f || continue
    run fm_${f}_mix fabric_exp/streams/p3_mixes.yaml TT_FABRIC_EXP_FLAGS=$f
    run fm_${f}_4352 fabric_exp/exp_2d_4352.yaml TT_FABRIC_EXP_FLAGS=$f
done
# Blind push W32 (benchmark only): needs bit 26 / 27 validated on the stock-RX build.
has 80237952 && { run fm_82335104w32_4352 fabric_exp/exp_2d_4352.yaml TT_FABRIC_EXP_FLAGS=82335104 $W;
                  run fm_82335104w32_mix fabric_exp/streams/p3_mixes.yaml TT_FABRIC_EXP_FLAGS=82335104 $W; }
has 147346816 && { run fm_149443968w32_4352 fabric_exp/exp_2d_4352.yaml TT_FABRIC_EXP_FLAGS=149443968 $W;
                   run fm_149443968w32_mix fabric_exp/streams/p3_mixes.yaml TT_FABRIC_EXP_FLAGS=149443968 $W; }
echo "== phase 3: small packets and latency"
run fm_13129088_small fabric_exp/exp_2d_trim_small.yaml TT_FABRIC_EXP_FLAGS=13129088
has 281564544 && run fm_281564544_small fabric_exp/exp_2d_trim_small.yaml TT_FABRIC_EXP_FLAGS=281564544
run fm_13129088_lat fabric_exp/exp_latency.yaml TT_FABRIC_EXP_FLAGS=13129088
for f in 80237952 147346816 281564544; do
    has $f && run fm_${f}_lat fabric_exp/exp_latency.yaml TT_FABRIC_EXP_FLAGS=$f
done
echo ALL_DONE
