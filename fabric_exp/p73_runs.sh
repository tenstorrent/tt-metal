#!/bin/bash
# §73: where E0's ~72 cycles of channel-visit overhead go. Variant e0seg = the tree's router (the user's s71) with
# segment timers on phase IDs E0 never closes otherwise, plus a profiler copy that counts only the timed phase
# (fabric_phase_profiler.hpp). Level 2, one mask per run, no per-iteration timing. New E0 segments: loop top 16,
# worker can-send 32, worker completion step 64, worker connection check 128, forwarding channels 256, whole iteration
# 512, worker visit 2048; existing: send 1, completion 4, connection 8, all visits 1024. Plus one unprofiled run (its
# ELFs saved to compare .text with s71).
#   a: streams/p72_profile.yaml   labels p73_p72_<run>
#   b: exp_2d_4352.yaml           labels p73_4352_<run>
#   c: streams/p72_profile.yaml, worker pre-send segment (mask 2: packet + credit test passed -> send start, added to
#      e0seg after a/b) and can-send again (32) in the same build   labels p73c_p72_m<mask>
# Broker job: no resets, line-buffered output, heartbeat. Flags 13129088, fast dispatch, telemetry off. Router and
# profiler header restored to s71 on exit, checked with cmp.
cd /home/aagarwal/tt-metal
R=tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router_v2.cpp
P=tt_metal/fabric/hw/inc/edm_fabric/telemetry/fabric_phase_profiler.hpp
V=fabric_exp/variants
C=/home/aagarwal/.tt-metal-cache/tt-metal-cache8310703137207683420/kernels/fabric_erisc_router_v2
swap() { cp $V/$1/fabric_erisc_router_v2.cpp $R && cp $V/$1/fabric_phase_profiler.hpp $P && echo "== router + profiler: $1"; }
restore() { swap s71 > /dev/null
    cmp -s $V/s71/fabric_erisc_router_v2.cpp $R && cmp -s $V/s71/fabric_phase_profiler.hpp $P \
    && echo "tree restored to s71" || echo "!!! RESTORE FAILED"; }
save_elfs() {  # variant marker-file
    local n=0
    rm -rf $V/$1/elf
    for d in $(find $C -newer $2 -name '*.elf' ! -name '*.xip.elf' -printf '%h\n' | sort -u); do
        local h=$(basename $(dirname $d)) k=$(basename $d)
        mkdir -p $V/$1/elf/$h && cp $d/$k.elf $V/$1/elf/$h/ && cp $C/$h/named_ct_arg_map_generated.h $V/$1/elf/$h/ && n=$((n+1))
    done
    echo "saved $n ELFs for $1"
}
(while sleep 60; do echo "heartbeat $(date +%T)"; done) & HB=$!
trap 'kill $HB 2>/dev/null; restore' EXIT
run() {  # label yaml [ENV=VALUE ...]
    env -u TT_METAL_SLOW_DISPATCH_MODE TT_FABRIC_KERNEL_VERSION=2 timeout 600 fabric_exp/run_exp.sh "$1" "$2" \
        TT_FABRIC_NO_BW_TELEMETRY=1 TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT=1000 TT_FABRIC_EXP_FLAGS=13129088 "${@:3}" \
        | grep --line-buffered -v 'Running Test'
    local res=$(grep -oE 'Tests: .*executed, [0-9]+ failed' fabric_exp/results/$1.log 2>/dev/null)
    echo "$1: ${res:-NO RESULT}"
    [ -n "$res" ] && echo "$res" | grep -q ", 0 failed"
}
M=$(mktemp)
case "${1:-a}" in
a) n=p72; Y=fabric_exp/streams/p72_profile.yaml ;;
b) n=4352; Y=fabric_exp/exp_2d_4352.yaml ;;
c) swap e0seg; sleep 1
   for m in 2 32; do
       run p73c_p72_m$m fabric_exp/streams/p72_profile.yaml TT_FABRIC_PHASE_PROFILE=2 TT_FABRIC_PHASE_MASK=$m \
           TT_FABRIC_PHASE_NO_ITER=1 || { echo STOPPED; exit 1; }
   done
   echo ALL_DONE; exit 0 ;;
esac
swap e0seg; sleep 1
run p73_${n}_noprof $Y || { echo STOPPED; exit 1; }
[ "$1" = a ] && save_elfs e0seg $M
for m in 16 32 64 128 256 512 2048 1 4 8 1024; do
    run p73_${n}_m$m $Y TT_FABRIC_PHASE_PROFILE=2 TT_FABRIC_PHASE_MASK=$m TT_FABRIC_PHASE_NO_ITER=1 || { echo STOPPED; exit 1; }
done
rm -f $M
echo ALL_DONE
