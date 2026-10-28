#!/bin/bash
# Shared functions for consolidation broker jobs on bh-lb-120-a08u24 (no resets, output streamed, heartbeat).
# Source it from a round script, then call `build`. Every build swaps its variant into the tree with swap.py
# (any files: router, headers, test kernels; fabric_erisc_router_v3.cpp runs in v2's place) and restores the tree
# right after; the EXIT trap restores it too. Router ELFs get the stack check after the build's first suite and are
# saved with their generated headers under fabric_exp/consolidation/builds/<build>/<config hash>/.
# Flags 13129088 (v2 reads only bit 22; v3 none), fast dispatch, BW telemetry off, keepalive 1000.
# KA_ENV= (set and empty) runs without the study's keepalive override (TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT), so the
# router gets the host's setting (ETH_TXQ_SEQ_UPDATE_TIMEOUT, RESULTS §95); v1, v2 and v3a then keep the firmware's 8000.
# EXTRA_ENV="VAR=value ..." adds study settings to every run (e.g. TT_FABRIC_EXP_WORKER_SLOTS=20, RESULTS §98).
cd /home/aagarwal/tt-metal
[ -n "${KA_ENV+x}" ] && [ -z "$KA_ENV" ] && unset TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT
SAN=tests/tt_metal/tt_fabric/test_infra/test_yamls
CON=fabric_exp/consolidation
OUT=$CON/builds
OD=runtime/sfpi/compiler/bin/riscv-tt-elf-objdump
NM=runtime/sfpi/compiler/bin/riscv-tt-elf-nm
BK=/home/aagarwal/.tt-metal-p1-swap-backup
C3=/home/aagarwal/.tt-metal-cache-p0
C2=/home/aagarwal/.tt-metal-cache-p0-O2
CS=/home/aagarwal/.tt-metal-cache-p0-Os
VAL="stream mesh val1d"
BENCH="4352 mix flushmix flushmix4 1d"

python3 $CON/swap.py restore $BK || { echo "!!! leftover swap could not be restored"; exit 1; }
(while sleep 60; do echo "heartbeat $(date +%T)"; done) & HB=$!
trap 'kill $HB 2>/dev/null; python3 $CON/swap.py restore $BK' EXIT

new_elfs() {  # cache marker-file
    find $1/*/kernels/fabric_erisc_router_v2 -newer $2 -name '*.elf' ! -name '*.xip.elf' 2>/dev/null
}
save_builds() {  # build cache marker-file
    local n=0 d
    rm -rf "${OUT:?}/${1:?}"
    for d in $(new_elfs $2 $3 | xargs -r -n1 dirname | sort -u); do
        local h=$(basename $(dirname $d)) k=$(basename $d) c=$(dirname $(dirname $d))
        mkdir -p $OUT/$1/$h && cp $d/$k.elf $OUT/$1/$h/ && cp $c/$h/*.h $c/$h/*.hpp $OUT/$1/$h/ && n=$((n+1))
    done
    echo "saved $n ELFs for $1"
}
stack_check() {  # cache marker-file
    local min=99999 n=0 e
    for e in $(new_elfs $1 $2); do
        local f=$($OD -d $e | grep -oE 'addi\s+sp,sp,-[0-9]+' | grep -oE '[0-9]+$' | sort -rn | head -3 | paste -sd+ | bc)
        local free=$(( 0xFFB02000 - 0x$($NM $e | awk '$3=="__ldm_bss_end"{print $1}') ))
        local m=$((free - f - 256)); n=$((n+1))
        [ $m -lt $min ] && min=$m && echo "   $(basename $e) $(basename $(dirname $(dirname $e))): frames $f, free $free"
    done
    echo "stack check: $n ELFs, smallest margin $min B"
    [ $n -gt 0 ] || { echo "   no new router ELFs (unchanged router, cache hit; e.g. a test-kernel-only variant)"; return 0; }
    [ $min -ge 0 ]
}
run() {  # label yaml cache opt
    env -u TT_METAL_SLOW_DISPATCH_MODE TT_FABRIC_KERNEL_VERSION=2 timeout 1200 fabric_exp/run_exp.sh "$1" "$2" \
        TT_FABRIC_NO_BW_TELEMETRY=1 ${KA_ENV-TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT=1000} TT_FABRIC_EXP_FLAGS=13129088 $EXTRA_ENV \
        TT_METAL_CACHE=$3 TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1 TT_METAL_FABRIC_OPT_LEVEL=$4 \
        | grep --line-buffered -v 'Running Test'
    local res=$(grep -oE 'Tests: .*executed, [0-9]+ failed' fabric_exp/results/$1.log 2>/dev/null)
    echo "$1: ${res:-NO RESULT}"
    [ -n "$res" ] && echo "$res" | grep -q ", 0 failed"
}
suite_yaml() {
    case $1 in
        stream) echo $SAN/test_fabric_stream_sanity.yaml ;; mesh) echo $SAN/test_fabric_mesh_sanity_common.yaml ;;
        val1d) echo $SAN/test_fabric_sanity_at_least_2x2_mesh.yaml ;;
        4352) echo fabric_exp/exp_2d_4352.yaml ;; mix) echo fabric_exp/streams/p3_mixes.yaml ;;
        flushmix) echo fabric_exp/streams/flush_mix.yaml ;; flushmix4) echo fabric_exp/streams/flush_mix4.yaml ;;
        1d) echo fabric_exp/exp_1d_4352.yaml ;;
        lat) echo fabric_exp/exp_latency.yaml ;;  # latency mode (RESULTS §19): results/<label>_latency.csv
        burst) echo fabric_exp/exp_burst.yaml ;;  # bursty senders: stream idle segments (RESULTS §98)
        burst612) echo fabric_exp/exp_burst612.yaml ;;  # bursts of 6 and 12 (not multiples of the completion group)
        val1d_m1) echo fabric_exp/v4/val1d_m1.yaml ;;  # v4 bring-up: val1d without rings, mux, telemetry, trimming
        1d_m1) echo fabric_exp/v4/exp_1d_m1.yaml ;;  # v4 bring-up: the 1D benchmark without its ring row
    esac
}
M=$(mktemp)
# build NAME VARIANT_DIR CACHE OPT LABEL_PREFIX SUITE...: stops the job on a failed suite or stack check.
build() {
    local b=$1 v=$2 c=$3 o=$4 l=$5 first=1 s; shift 5
    mkdir -p $c
    touch $M; sleep 1
    python3 $CON/swap.py apply $v $BK || { echo STOPPED; exit 1; }
    sleep 1
    echo "== build $b: $v, -$o, cache $c"
    for s in "$@"; do
        run ${l}_13129088_$s $(suite_yaml $s) $c $o || { echo STOPPED; exit 1; }
        if [ $first = 1 ]; then stack_check $c $M || { echo STOPPED; exit 1; }; first=0; fi
    done
    save_builds $b $c $M
    python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
}
