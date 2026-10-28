#!/bin/bash
# VC1 functions for consolidation jobs (source after jobs_lib.sh). The P150x8 split into two 2x2 meshes
# (fabric_exp/vc1, RESULTS §84): multi-mesh sanity (data checked) then intra-mesh benchmarks
# (fabric_exp/vc1/vc1_split.yaml), two MPI ranks via ttrun.py, a fresh kernel cache per run, compile commands logged.
# The per-test CSV is kept as results/<label>_13129088_vc1_tests.csv. VC1_YAML runs another yaml (e.g. vc1/vc1_inter.yaml),
# VC1_TIMEOUT sets the run's timeout (600 s). VC1CAP_EXTRA_ENV (as vc1_capture_lib.sh) overrides run settings.
out=fabric_exp/results
run_vc1() {  # label cache opt|auto
    local l=$1 c=$2 o=$3 optenv=""
    [ "$o" = auto ] || optenv="TT_METAL_FABRIC_OPT_LEVEL=$o"
    rm -rf "${c:?}"; mkdir -p $c
    rm -f generated/fabric/bandwidth_summary_results_blackhole.csv generated/fabric/bandwidth_results_blackhole.csv \
        $out/${l}_13129088_vc1.csv $out/${l}_13129088_vc1_tests.csv
    echo "=== ${l}_13129088_vc1: opt $o, cache $c ==="
    env -u TT_METAL_SLOW_DISPATCH_MODE TT_METAL_HOME=/home/aagarwal/tt-metal TT_METAL_CLEAR_L1=1 \
        TT_METAL_DEVICE_PROFILER=0 TT_FABRIC_KERNEL_VERSION=2 TT_FABRIC_NO_BW_TELEMETRY=1 \
        ${KA_ENV-TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT=1000} TT_FABRIC_EXP_FLAGS=13129088 TT_METAL_CACHE=$c \
        TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1 $optenv ${VC1CAP_EXTRA_ENV:-} \
        timeout ${VC1_TIMEOUT:-600} python_env/bin/python ttnn/ttnn/distributed/ttrun.py \
        --rank-binding fabric_exp/vc1/p150x8_2x2_split_rank_bindings.yaml \
        ./build/test/tt_metal/tt_fabric/test_infra/test_tt_fabric --test_config $PWD/${VC1_YAML:-fabric_exp/vc1/vc1_split.yaml} \
        > $out/${l}_13129088_vc1.log 2>&1
    local rc=$?
    local fin=$(sed 's/\x1b\[[0-9;]*m//g' $out/${l}_13129088_vc1.log | grep -ac 'Finished\. (test_tt_fabric')
    local bad=$(sed 's/\x1b\[[0-9;]*m//g' $out/${l}_13129088_vc1.log | grep -acE 'TT_FATAL|TT_THROW|HUNG|timed out|Aborted|mismatch')
    echo "exit $rc; $fin test finishes over both ranks; $bad error lines"
    [ -f generated/fabric/bandwidth_results_blackhole.csv ] && cp generated/fabric/bandwidth_results_blackhole.csv $out/${l}_13129088_vc1_tests.csv
    [ $rc -eq 0 ] && [ $bad -eq 0 ] || { echo "!!! ${l} failed"; sed 's/\x1b\[[0-9;]*m//g' $out/${l}_13129088_vc1.log | grep -aE 'TT_FATAL|TT_THROW|HUNG|what\(\)' | head -5; return 1; }
}
vc1_variant() {  # name variant-dir|tree label-prefix level...   (levels: auto, O2, O3, Os)
    local n=$1 v=$2 l=$3 o; shift 3
    if [ "$v" != tree ]; then python3 $CON/swap.py apply $v $BK || { echo STOPPED; exit 1; }; fi
    for o in "$@"; do
        ${VC1_RUN:-run_vc1} ${l}_$(echo $o | tr A-Z a-z) /home/aagarwal/.tt-metal-cache-p1vc1-$n-$o $o || { echo STOPPED; exit 1; }
    done
    if [ "$v" != tree ]; then python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }; fi
}
# vc1_repeat NAME VARIANT-DIR|tree LABEL LEVEL N: run the VC1 sanity group (fabric_exp/vc1/vc1_sanity.yaml) up to N times
# at one level in one cache (the first run compiles), stopping at the first failure or hang (no repeated hangs on the
# shared device). Prints the count of clean runs. Labels LABEL_r<i>_13129088_vc1.log.
vc1_repeat() {
    local n=$1 v=$2 l=$3 o=$4 N=$5 i c optenv="" ok=0
    [ "$o" = auto ] || optenv="TT_METAL_FABRIC_OPT_LEVEL=$o"
    c=/home/aagarwal/.tt-metal-cache-p1vc1rep-$n-$o
    rm -rf "${c:?}"; mkdir -p $c
    if [ "$v" != tree ]; then python3 $CON/swap.py apply $v $BK || { echo STOPPED; exit 1; }; fi
    for i in $(seq 1 $N); do
        local lab=${l}_r${i}
        echo "=== ${lab}: $n, opt $o, run $i/$N ==="
        env -u TT_METAL_SLOW_DISPATCH_MODE TT_METAL_HOME=/home/aagarwal/tt-metal TT_METAL_CLEAR_L1=1 \
            TT_METAL_DEVICE_PROFILER=0 TT_FABRIC_KERNEL_VERSION=2 TT_FABRIC_NO_BW_TELEMETRY=1 \
            ${KA_ENV-TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT=1000} TT_FABRIC_EXP_FLAGS=13129088 TT_METAL_CACHE=$c $optenv \
            timeout 300 python_env/bin/python ttnn/ttnn/distributed/ttrun.py \
            --rank-binding fabric_exp/vc1/p150x8_2x2_split_rank_bindings.yaml \
            ./build/test/tt_metal/tt_fabric/test_infra/test_tt_fabric --test_config $PWD/fabric_exp/vc1/vc1_sanity.yaml \
            > $out/${lab}_13129088_vc1.log 2>&1
        local rc=$?
        local fin=$(sed 's/\x1b\[[0-9;]*m//g' $out/${lab}_13129088_vc1.log | grep -ac 'Finished\. (test_tt_fabric')
        local bad=$(sed 's/\x1b\[[0-9;]*m//g' $out/${lab}_13129088_vc1.log | grep -acE 'TT_FATAL|TT_THROW|HUNG|timed out|Aborted|mismatch')
        echo "exit $rc; $fin test finishes over both ranks (24 expected); $bad error lines"
        if [ $rc -ne 0 ] || [ $bad -ne 0 ] || [ $fin -ne 24 ]; then
            echo "!!! $n run $i failed: last tests $(sed 's/\x1b\[[0-9;]*m//g' $out/${lab}_13129088_vc1.log | grep -a 'Running Test:' | tail -2 | grep -o 'Running Test: [A-Za-z0-9_]*' | tr '\n' ' ')"
            break
        fi
        ok=$((ok+1))
    done
    echo "vc1_repeat $n -$o: $ok clean runs of $N"
    if [ "$v" != tree ]; then python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }; fi
}
