#!/bin/bash
# Stream mixes (p3_mixes.yaml, trimmed and untrimmed) on the §29/§31 builds, experiment tree, labels m_*.
cd /home/aagarwal/tt-metal
KA="TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT=1000"
W="TT_FABRIC_EXP_WORKER_SLOTS=32 TT_FABRIC_EXP_MAX_SENDER_SLOTS=8"
run() {  # label yaml [env...]
    timeout 1500 fabric_exp/run_exp.sh "$1" "$2" TT_FABRIC_NO_BW_TELEMETRY=1 "${@:3}" > /dev/null 2>&1
    if [ ! -f "fabric_exp/results/$1.csv" ]; then echo "!!! STOPPING: $1 failed"; pkill -x test_tt_fabric; tt-smi -r all > /dev/null 2>&1; exit 1; fi
    echo "$1: $(grep -oE 'Tests: .*executed, [0-9]+ failed' fabric_exp/results/$1.log) worker-bound=$(grep -c 'was worker-bound' fabric_exp/results/$1.log)"
}
tt-smi -r all > /dev/null 2>&1
for t in "" _trim; do
    Y=fabric_exp/streams/p3_mixes${t}.yaml; L=mix${t}
    run m_stockka8000_$L $Y
    run m_stock_$L $Y $KA
    run m_546176_$L $Y $KA TT_FABRIC_EXP_FLAGS=546176
    run m_4740480_$L $Y $KA TT_FABRIC_EXP_FLAGS=4740480
    run m_13129088_$L $Y $KA TT_FABRIC_EXP_FLAGS=13129088
    run m_13129088w32_$L $Y $KA TT_FABRIC_EXP_FLAGS=13129088 $W
    run m_15226240w32_$L $Y $KA TT_FABRIC_EXP_FLAGS=15226240 $W
done
echo ALL_DONE
