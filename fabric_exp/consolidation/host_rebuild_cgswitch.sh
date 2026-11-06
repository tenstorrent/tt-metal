#!/bin/bash
# Host rebuild for the completion-group study switch (SLOT_ALLOCATOR.md section 12.10, 2026-11-04): the tree already
# carries variants/host_dynslots/3_completion_group.patch (erisc_datamover_builder.cpp and the transaction-ID
# allocator's group override, TT_FABRIC_EXP_COMPLETION_GROUP; unset = today's sizes). Rebuilds libtt_metal (nice 15,
# JOBS parallel compiles, default 16) with CCACHE_DISABLE=1 (ccache lives on /data, which runs out of inodes). On a
# failure the three files go back to their pre-switch copies (~/.tt-metal-host-rebuild-backup/cgswitch-pre) and the
# library is rebuilt from them. Run as a broker job so no device job uses the host library meanwhile.
cd /home/aagarwal/tt-metal || exit 1
export CCACHE_DISABLE=1
BUILD="nice -n 15 cmake --build build_Release --target tt_metal -j ${JOBS:-16}"
BK=/home/aagarwal/.tt-metal-host-rebuild-backup/cgswitch-pre
grep -q TT_FABRIC_EXP_COMPLETION_GROUP tt_metal/fabric/erisc_datamover_builder.cpp || { echo "the switch is not in the tree"; exit 1; }
grep -q group_override tt_metal/fabric/builder/fabric_trid_allocator.hpp || { echo "the allocator override is not in the tree"; exit 1; }
for f in erisc_datamover_builder.cpp fabric_trid_allocator.hpp fabric_trid_allocator.cpp; do
    [ -f $BK/$f ] || { echo "no backup $BK/$f"; exit 1; }
done
restore() {
    cp -p $BK/erisc_datamover_builder.cpp tt_metal/fabric/erisc_datamover_builder.cpp
    cp -p $BK/fabric_trid_allocator.hpp $BK/fabric_trid_allocator.cpp tt_metal/fabric/builder/
    $BUILD
    echo RESTORED
    exit 1
}
$BUILD || restore
ls -la build_Release/tt_metal/libtt_metal.so
for k in TT_FABRIC_EXP_COMPLETION_GROUP TT_FABRIC_EXP_TRID_ALLOC vc0_receiver_floor "receiver transaction IDs"; do
    grep -aq "$k" build_Release/tt_metal/libtt_metal.so && echo "library has: $k" || { echo "!!! library lacks: $k"; exit 1; }
done
echo HOST_REBUILT
