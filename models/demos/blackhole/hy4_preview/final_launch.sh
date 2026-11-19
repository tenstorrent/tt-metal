#!/bin/bash
# final_launch.sh <tag> [VAR=val ...]: boot the FINAL Hy4 stack (FINAL_FLAGS.txt) on THIS host (under a claim; mega_gate.sh does claim/boot/chain for a gate run).
cd /mnt/tt-data/ssinghal/hy4-logs/program/product17; TAG=${1:?tag}; shift
B=$(grep '^BUNDLE_FLAGS=' FINAL_FLAGS.txt | sed 's/^BUNDLE_FLAGS=//; s/^"//; s/"$//')
exec env LIBDIR=$(cat MUXL1_LIBDIR.txt) BUNDLE="$B" ./launch.sh $TAG "$@"
