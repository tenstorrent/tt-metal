#!/bin/bash
# §78: the user's s78 on bh-lb-120-a08u24 (broker job: no resets, output streamed, heartbeat). s78 = s77s (compile-time
# free-slots stream ids) plus: the pre-send payload size read as the whole 4 B word at header offset 40 and masked after
# the TX-queue busy test (s78w = this part alone), and the 4 B LeanStrideSenderEthChannel with the compile-time slot
# advance in sender_channel_try_send and the v2 speedy path (s78l = this part alone). s78n / s77si = s78 / s77s with
# run_fabric_edm_main_loop noinline instead of FORCE_INLINE (on both ERISCs).
#   a: s78 validation (stream, mesh, 1D sanity) + 4352 / mix / flushmix / flushmix4 / 1D, then s78n the same
#                                                                                   labels dq78_, dq78n_
#   b: s77s (control), s78w, s78l benchmark suites; s77si validation + benchmark suites
#                                                                                   labels dq77s3_, dq78w_, dq78l_, dq77si_
#   c: s78n0 / s77si0 = s78 / s77s with only E0 running the main loop out of line (an if constexpr on MY_ERISC_ID picks
#      a noinline wrapper; E1 keeps it inlined): validation + benchmark suites each; then an s78 repeat
#                                                                                   labels dq78n0_, dq77si0_, dq78r2_
# A noinline build is stopped after its first suite if any new router ELF's three largest stack frames, plus 256 B
# for the firmware, exceed the local memory left above its .bss (top of local memory 0xFFB02000).
# Flags 13129088, fast dispatch, telemetry off. Tree restored to s78 on exit, checked with cmp.
cd /home/aagarwal/tt-metal
SAN=tests/tt_metal/tt_fabric/test_infra/test_yamls
R=tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router_v2.cpp
H=tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_datamover_channels.hpp
P=tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_speedy_path.hpp
V=fabric_exp/variants
OD=runtime/sfpi/compiler/bin/riscv-tt-elf-objdump
NM=runtime/sfpi/compiler/bin/riscv-tt-elf-nm
swap() { cp $V/$1/fabric_erisc_router_v2.cpp $R && cp $V/$1/fabric_erisc_datamover_channels.hpp $H \
    && cp $V/$1/fabric_erisc_router_speedy_path.hpp $P && echo "== router: $1"; }
restore() { swap s78 > /dev/null
    cmp -s $V/s78/fabric_erisc_router_v2.cpp $R && cmp -s $V/s78/fabric_erisc_datamover_channels.hpp $H \
    && cmp -s $V/s78/fabric_erisc_router_speedy_path.hpp $P && echo "tree restored to s78" || echo "!!! RESTORE FAILED"; }
new_elfs() {  # marker-file
    find /home/aagarwal/.tt-metal-cache/*/kernels/fabric_erisc_router_v2 -newer $1 -name '*.elf' ! -name '*.xip.elf'
}
save_elfs() {  # variant marker-file
    local n=0
    rm -rf $V/$1/elf
    for d in $(new_elfs $2 | xargs -r -n1 dirname | sort -u); do
        local h=$(basename $(dirname $d)) k=$(basename $d) c=$(dirname $(dirname $d))
        mkdir -p $V/$1/elf/$h && cp $d/$k.elf $V/$1/elf/$h/ && cp $c/$h/named_ct_arg_map_generated.h $V/$1/elf/$h/ \
            && n=$((n+1))
    done
    echo "saved $n ELFs for $1"
}
stack_check() {  # marker-file
    local min=99999 n=0
    for e in $(new_elfs $1); do
        local f=$($OD -d $e | grep -oE 'addi\s+sp,sp,-[0-9]+' | grep -oE '[0-9]+$' | sort -rn | head -3 | paste -sd+ | bc)
        local free=$(( 0xFFB02000 - 0x$($NM $e | awk '$3=="__ldm_bss_end"{print $1}') ))
        local m=$((free - f - 256)); n=$((n+1))
        [ $m -lt $min ] && min=$m && echo "   $(basename $e) $(basename $(dirname $(dirname $e))): frames $f, free $free"
    done
    echo "stack check: $n ELFs, smallest margin $min B"
    [ $min -ge 0 ] || { echo STOPPED; exit 1; }
}
(while sleep 60; do echo "heartbeat $(date +%T)"; done) & HB=$!
trap 'kill $HB 2>/dev/null; restore' EXIT
run() {  # label yaml
    env -u TT_METAL_SLOW_DISPATCH_MODE TT_FABRIC_KERNEL_VERSION=2 timeout 1200 fabric_exp/run_exp.sh "$1" "$2" \
        TT_FABRIC_NO_BW_TELEMETRY=1 TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT=1000 TT_FABRIC_EXP_FLAGS=13129088 \
        | grep --line-buffered -v 'Running Test'
    local res=$(grep -oE 'Tests: .*executed, [0-9]+ failed' fabric_exp/results/$1.log 2>/dev/null)
    echo "$1: ${res:-NO RESULT}"
    [ -n "$res" ] && echo "$res" | grep -q ", 0 failed"
}
suites() {  # label-prefix suite...
    local l=$1; shift
    for s in "$@"; do
        case $s in
            stream) y=$SAN/test_fabric_stream_sanity.yaml ;; mesh) y=$SAN/test_fabric_mesh_sanity_common.yaml ;;
            val1d) y=$SAN/test_fabric_sanity_at_least_2x2_mesh.yaml ;;
            4352) y=fabric_exp/exp_2d_4352.yaml ;; mix) y=fabric_exp/streams/p3_mixes.yaml ;;
            flushmix) y=fabric_exp/streams/flush_mix.yaml ;; flushmix4) y=fabric_exp/streams/flush_mix4.yaml ;;
            1d) y=fabric_exp/exp_1d_4352.yaml ;;
        esac
        run ${l}_13129088_$s $y || { echo STOPPED; exit 1; }
    done
}
B="4352 mix flushmix flushmix4 1d"
M=$(mktemp)
variant() {  # variant label-prefix suite...   (a noinline variant must start with a short suite: stream)
    local v=$1 l=$2; shift 2
    touch $M; sleep 1
    swap $v; sleep 1
    if grep -q '__attribute__((noinline)) void run_fabric_edm_main_loop' $V/$v/fabric_erisc_router_v2.cpp; then
        suites $l $1; stack_check $M; shift
    fi
    suites $l "$@"
    [ -d $V/$v/elf ] && [ "$KEEP_ELFS" = 1 ] || save_elfs $v $M
}
case "${1:-a}" in
a)  variant s78 dq78 stream mesh val1d $B
    variant s78n dq78n stream mesh val1d $B ;;
b)  variant s77s dq77s3 $B
    variant s78w dq78w $B
    variant s78l dq78l $B
    variant s77si dq77si stream mesh val1d $B ;;
c)  variant s78n0 dq78n0 stream mesh val1d $B
    variant s77si0 dq77si0 stream mesh val1d $B
    KEEP_ELFS=1 variant s78 dq78r2 $B ;;
esac
rm -f $M
echo ALL_DONE
