#!/bin/bash
# Host rebuild for the transaction-ID allocation (TRID_ALLOCATION.md, 2026-11-04), on top of the dynamic slot allocator
# (host_rebuild_dynslots.sh, broker 648), run as a broker job so no device job uses the host library meanwhile. The
# rule's own files (tt_metal/fabric/builder/fabric_trid_allocator.{hpp,cpp}) are already in the tree (new files, built
# by nothing until sources.cmake lists them); this applies variants/host_dynslots/2_trid_alloc.patch to two files:
#   erisc_datamover_builder.cpp   per receiver channel RX_CH_<c>_COMPLETION_GROUP_SIZE / _NUM_TRIDS / _TRID_START:
#                                 TT_FABRIC_EXP_TRID_ALLOC=1 assign_transaction_ids, else today's uniform rule (logged
#                                 with TT_FABRIC_EXP_LOG_SLOTS=1); main's router and v4txa11 do not read them
#   sources.cmake                 builds fabric_trid_allocator.cpp
# The build also compiles the tree's current slot allocator (2026-11-04: the 2D VC0 receiver floor and the
# minimum-first rules, fabric_slot_allocator.{hpp,cpp}) and patch 1 as it now stands (1_dyn_slots.patch, already
# applied in the tree, with shape.two_d); the checks below refuse to build without them.
# No header changes, so no class layout changes: libtt_metal alone is rebuilt and the test binaries stay valid. Then
# rebuilds libtt_metal (nice 15, JOBS parallel compiles, default 16). On any failure the sources are restored and the
# library rebuilt from them. Same pattern as host_rebuild_dynslots.sh.
cd /home/aagarwal/tt-metal || exit 1
BUILD="nice -n 15 cmake --build build_Release --target tt_metal -j ${JOBS:-16}"
BK=/home/aagarwal/.tt-metal-host-rebuild-backup/tridalloc
P=fabric_exp/variants/host_dynslots/2_trid_alloc.patch
FILES="tt_metal/fabric/erisc_datamover_builder.cpp tt_metal/fabric/sources.cmake"
for f in tt_metal/fabric/builder/fabric_trid_allocator.hpp tt_metal/fabric/builder/fabric_trid_allocator.cpp; do
    [ -f $f ] || { echo "missing $f"; exit 1; }
done
grep -q "builder/fabric_slot_allocator.cpp" tt_metal/fabric/sources.cmake || { echo "the dyn-slots patch is not applied"; exit 1; }
grep -q "shape.two_d = " tt_metal/fabric/builder/fabric_static_sized_channels_allocator.cpp ||
    { echo "patch 1 is not the 2026-11-04 one (no shape.two_d)"; exit 1; }
grep -q '"vc0_receiver_floor"' tt_metal/fabric/builder/fabric_slot_allocator.cpp ||
    { echo "the slot allocator has no VC0 receiver floor"; exit 1; }
[ -e $BK ] && { echo "backup $BK exists; refusing"; exit 1; }
mkdir -p $BK
for f in $FILES; do cp -p $f $BK/$(basename $f); done
restore() {
    for f in $FILES; do cp -p $BK/$(basename $f) $f; done
    CCACHE_DISABLE=1 $BUILD
    echo RESTORED
    exit 1
}
patch -p1 --dry-run < $P > /dev/null || { echo "does not apply: $P"; rm -rf $BK; exit 1; }
patch -p1 < $P || restore
grep -c "TT_FABRIC_EXP_TRID_ALLOC" tt_metal/fabric/erisc_datamover_builder.cpp
CCACHE_DISABLE=1 $BUILD || restore
ls -la build_Release/tt_metal/libtt_metal.so
for k in TT_FABRIC_EXP_DYN_SLOTS_POLICY vc0_receiver_floor TT_FABRIC_EXP_TRID_ALLOC "receiver transaction IDs"; do
    grep -aq "$k" build_Release/tt_metal/libtt_metal.so && echo "library has: $k" || { echo "!!! library lacks: $k"; exit 1; }
done
echo HOST_REBUILT
