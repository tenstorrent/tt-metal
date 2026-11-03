#!/bin/bash
# Batched host rebuild (2026-11-01), run as a broker job so no device job uses the host library meanwhile. Applies:
#   variants/v4rxo3p1/host_per_vc_args.patch          ENABLE_FIRST_LEVEL_ACK_VC2 = 0, VC2_CROSSOVER_TO_VC1 = 0
#   variants/v4txo_host/sender_worker_channel_arg.patch SENDER_CH_<i>_IS_WORKER_CHANNEL (user: option (a))
#   variants/host_vc1_o3/vc1_opt_level.patch          the study's router builds VC1 at O3 (user; job 367)
# to tt_metal/fabric/erisc_datamover_builder.cpp and rebuilds libtt_metal. Every arg is additive (main's router and
# older variants ignore them). On any failure the source is restored and the library rebuilt from it.
cd /home/aagarwal/tt-metal || exit 1
F=tt_metal/fabric/erisc_datamover_builder.cpp
BK=/home/aagarwal/.tt-metal-host-rebuild-backup
V=fabric_exp/variants
P="$(ls $V/host_rebuild_20261101/*.patch)"  # pinned copies (job 386 failed: an agent removed v4rxo3p1)
mkdir -p $BK
[ -e $BK/erisc_datamover_builder.cpp.pre20261101 ] && { echo "backup exists; refusing"; exit 1; }
cp -p $F $BK/erisc_datamover_builder.cpp.pre20261101
restore() { cp -p $BK/erisc_datamover_builder.cpp.pre20261101 $F; CCACHE_DISABLE=1 cmake --build build_Release --target tt_metal; echo RESTORED; exit 1; }
for p in $P; do patch -p1 --dry-run < $p > /dev/null || { echo "does not apply: $p"; exit 1; }; patch -p1 < $p || restore; done
grep -c 'ENABLE_FIRST_LEVEL_ACK_VC2\|IS_WORKER_CHANNEL\|study_router' $F
CCACHE_DISABLE=1 cmake --build build_Release --target tt_metal || restore
ls -la build_Release/tt_metal/libtt_metal.so
echo HOST_REBUILT
