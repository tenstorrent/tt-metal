#!/bin/bash
# Host rebuild (2026-11-02, sender ownership pass stage X4), run as a broker job so no device job uses the host library
# meanwhile. Applies the pinned patch variants/host_rebuild_20261102b/1_worker_credit_cmd_bufs.patch to
# tt_metal/fabric/erisc_datamover_builder.cpp: SENDER_CH_<i>_ACK_CMD_BUF_ID gives each worker channel a command buffer
# of its own (VC0's: WR, VC2's: WR_REG; the others keep AT); main's router does not use the arg. Then rebuilds
# libtt_metal. On any failure the source is restored and the library rebuilt from it. Same pattern as
# host_rebuild_20261101.sh.
cd /home/aagarwal/tt-metal || exit 1
F=tt_metal/fabric/erisc_datamover_builder.cpp
BK=/home/aagarwal/.tt-metal-host-rebuild-backup
P="$(ls fabric_exp/variants/host_rebuild_20261102b/*.patch)"
mkdir -p $BK
[ -e $BK/erisc_datamover_builder.cpp.pre20261102b ] && { echo "backup exists; refusing"; exit 1; }
cp -p $F $BK/erisc_datamover_builder.cpp.pre20261102b
restore() { cp -p $BK/erisc_datamover_builder.cpp.pre20261102b $F; CCACHE_DISABLE=1 cmake --build build_Release --target tt_metal; echo RESTORED; exit 1; }
for p in $P; do patch -p1 --dry-run < $p > /dev/null || { echo "does not apply: $p"; exit 1; }; patch -p1 < $p || restore; done
grep -c "ack_cmd_buf = i == get_worker_connected_sender_channel" $F
CCACHE_DISABLE=1 cmake --build build_Release --target tt_metal || restore
ls -la build_Release/tt_metal/libtt_metal.so
echo HOST_REBUILT
