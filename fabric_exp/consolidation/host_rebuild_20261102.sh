#!/bin/bash
# Host rebuild (2026-11-02, sender ownership pass stage X-1), run as a broker job so no device job uses the host library
# meanwhile. Applies the pinned patch variants/host_rebuild_20261102/1_worker_role_first_channel.patch to
# tt_metal/fabric/erisc_datamover_builder.cpp: SENDER_CH_<i>_IS_WORKER_CHANNEL (and the live-check skip, which shares
# the rule) marks the first sender channel of each VC whose routing mode has a worker (VC0's worker-connected channel,
# VC2's first channel), not every VC2 channel; then rebuilds libtt_metal. On any failure the source is restored and the
# library rebuilt from it. Same pattern as host_rebuild_20261101.sh.
cd /home/aagarwal/tt-metal || exit 1
F=tt_metal/fabric/erisc_datamover_builder.cpp
BK=/home/aagarwal/.tt-metal-host-rebuild-backup
P="$(ls fabric_exp/variants/host_rebuild_20261102/*.patch)"
mkdir -p $BK
[ -e $BK/erisc_datamover_builder.cpp.pre20261102 ] && { echo "backup exists; refusing"; exit 1; }
cp -p $F $BK/erisc_datamover_builder.cpp.pre20261102
restore() { cp -p $BK/erisc_datamover_builder.cpp.pre20261102 $F; CCACHE_DISABLE=1 cmake --build build_Release --target tt_metal; echo RESTORED; exit 1; }
for p in $P; do patch -p1 --dry-run < $p > /dev/null || { echo "does not apply: $p"; exit 1; }; patch -p1 < $p || restore; done
grep -c 'vc2_first' $F
CCACHE_DISABLE=1 cmake --build build_Release --target tt_metal || restore
ls -la build_Release/tt_metal/libtt_metal.so
echo HOST_REBUILT
