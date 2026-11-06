#!/bin/bash
# Host rebuild for the dynamic slot allocator (SLOT_ALLOCATOR.md, 2026-11-03), run as a broker job so no device job
# uses the host library meanwhile. The allocator's own files (tt_metal/fabric/builder/fabric_slot_allocator.{hpp,cpp})
# are already in the tree (new files, built by nothing until sources.cmake lists them); this applies
# variants/host_dynslots/1_dyn_slots.patch to two existing files:
#   builder/fabric_static_sized_channels_allocator.cpp   TT_FABRIC_EXP_DYN_SLOTS=1: the allocator's depths, logged (the
#                                                        mesh facts read from the control plane only then)
#   sources.cmake                                        builds fabric_slot_allocator.cpp
# No header changes, so no class layout changes: libtt_metal alone is rebuilt and the test binaries stay valid.
# Off by default: without TT_FABRIC_EXP_DYN_SLOTS every depth is what it was. Then rebuilds libtt_metal. On any failure
# the sources are restored and the library rebuilt from them. Same pattern as host_rebuild_20261102b.sh. The build runs
# at nice 15 with JOBS (default 16) parallel compiles, since a09u32 is shared.
cd /home/aagarwal/tt-metal || exit 1
BUILD="nice -n 15 cmake --build build_Release --target tt_metal -j ${JOBS:-16}"
BK=/home/aagarwal/.tt-metal-host-rebuild-backup/dynslots
P=fabric_exp/variants/host_dynslots/1_dyn_slots.patch
FILES="tt_metal/fabric/builder/fabric_static_sized_channels_allocator.cpp tt_metal/fabric/sources.cmake"
for f in tt_metal/fabric/builder/fabric_slot_allocator.hpp tt_metal/fabric/builder/fabric_slot_allocator.cpp; do
    [ -f $f ] || { echo "missing $f"; exit 1; }
done
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
grep -c "TT_FABRIC_EXP_DYN_SLOTS" tt_metal/fabric/builder/fabric_static_sized_channels_allocator.cpp
CCACHE_DISABLE=1 $BUILD || restore
ls -la build_Release/tt_metal/libtt_metal.so
echo HOST_REBUILT
