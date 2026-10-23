#!/bin/bash
# §66 follow-up (broker job): §63 rebuilt + the !any_sender_channel_serviced guard (variants/s63g), labels a8s63g_13129088_<suite>,
# to check the guard leaves the ERISC code unchanged (ELFs saved to variants/s63g/elf/). Tree restored to §66 on exit.
cd /home/aagarwal/tt-metal
R=tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router_v2.cpp
T=tt_metal/fabric/hw/inc/edm_fabric/fabric_edm_packet_transmission.hpp
V=fabric_exp/variants
C=/home/aagarwal/.tt-metal-cache/tt-metal-cache8310703137207683420/kernels/fabric_erisc_router_v2
swap() { cp $V/$1/fabric_erisc_router_v2.cpp $R && cp $V/$1/fabric_edm_packet_transmission.hpp $T && echo "== router: $1"; }
restore() { swap s66 > /dev/null; cmp -s $V/s66/fabric_erisc_router_v2.cpp $R && cmp -s $V/s66/fabric_edm_packet_transmission.hpp $T \
    && echo "tree restored to s66" || echo "!!! RESTORE FAILED"; }
save_elfs() {  # variant marker-file
    local n=0
    for d in $(find $C -newer $2 -name '*.elf' ! -name '*.xip.elf' -printf '%h\n' | sort -u); do
        local h=$(basename $(dirname $d)) k=$(basename $d)
        mkdir -p $V/$1/elf/$h && cp $d/$k.elf $V/$1/elf/$h/ && cp $C/$h/named_ct_arg_map_generated.h $V/$1/elf/$h/ && n=$((n+1))
    done
    echo "saved $n ELFs for $1"
}
(while sleep 60; do echo "heartbeat $(date +%T)"; done) & HB=$!
trap 'kill $HB 2>/dev/null; restore' EXIT
run() {  # label yaml
    env -u TT_METAL_SLOW_DISPATCH_MODE TT_FABRIC_KERNEL_VERSION=2 timeout 1200 fabric_exp/run_exp.sh "$1" "$2" \
        TT_FABRIC_NO_BW_TELEMETRY=1 TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT=1000 TT_FABRIC_EXP_FLAGS=13129088
    local status=$?
    local res=$(grep -oE 'Tests: .*executed, [0-9]+ failed' fabric_exp/results/$1.log 2>/dev/null)
    echo "$1: ${res:-NO RESULT (exit $status)}"
    [ $status -eq 0 ] && [ -n "$res" ] && echo "$res" | grep -q ", 0 failed"
}
for pair in s63g:a8s63g; do
    v=${pair%%:*} l=${pair##*:}
    rm -rf $V/$v/elf; swap $v; M=$(mktemp); sleep 1
    run ${l}_13129088_4352 fabric_exp/exp_2d_4352.yaml || { echo STOPPED; exit 1; }
    run ${l}_13129088_mix fabric_exp/streams/p3_mixes.yaml || { echo STOPPED; exit 1; }
    run ${l}_13129088_flushmix fabric_exp/streams/flush_mix.yaml || { echo STOPPED; exit 1; }
    save_elfs $v $M; rm -f $M
done
echo ALL_DONE
