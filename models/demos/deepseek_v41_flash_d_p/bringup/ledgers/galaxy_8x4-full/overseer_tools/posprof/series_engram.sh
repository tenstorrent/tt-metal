#!/usr/bin/env bash
# 5-position profile of the current default (block MoE, fused head+tail, FABRIC_2D_TORUS_XY) with the engram tables
# resident in host RAM as bfp8 (owner 2026-10-28; 2x2 commits up to c3d41c620db). Pytest defaults ENGRAM_TABLE to
# random (F222), so resident is forced. Device time (real-time profiler) then the F196 warm host wall.
D=/scratch/dnijemcevic/posprof; L=$D/series.log
st() { echo "$(date '+%F %T') STAGE $*" >> $L; }
export DS41_ENGRAM_TABLE=resident
st "ENG device (resident engram) @ $(git -C $HOME/repos/tt-metal rev-parse --short HEAD)"; TAG=ENG $D/run.sh
st "ENG wall";   TAG=ENG $D/run_wall.sh
st engdone
