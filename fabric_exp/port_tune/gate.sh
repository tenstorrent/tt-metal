#!/bin/bash
# Offline gate of the receiver tuning sweep (RECEIVER_TUNING.md §2): router-only rebuilds of every mocked case
# (cov/cov.py; captures in /home/aagarwal/.tt-metal-port-tune-cov/cases from the patched host, env unset) and loadable-
# byte comparisons:
#   base   the unpatched port (alt tree /home/aagarwal/tt-metal-port-tune-base, clean at 22cdb1e55c3), no defines
#   none   the patched tree, no defines (env unset)
#   d4_32  the patched tree with the default arm's defines (idle passes 4, proof checks 32)
#   i1 i2 i8 (proof 32), c8 c128 (idle 4): the other arms
#   tx16 rp16: the delay hooks at idle 4 (they must compile and change images; not compared for identity)
# Gate 1: none = base and d4_32 = base on every image. Gate 2: the G 1 configs (rings, tori) identical across every
# arm. The per-case summary of which images change per arm goes to gate.txt beside this script.
set -uo pipefail
cd "$(dirname "$0")/cov"
C="python3 cov.py"
build() { echo "== build $1 ${3:-}"; $C build --router-only --jobs ${JOBS:-32} --name $1 $2 ${3:+--defines "$3"} | tail -2; }
build base "--alt-tree /home/aagarwal/tt-metal-port-tune-base"
build none ""
build d4_32 "" "-DFABRIC_EXP_IDLE_PASSES=4 -DFABRIC_EXP_PROOF_CHECKS=32"
for i in 1 2 8; do build i$i "" "-DFABRIC_EXP_IDLE_PASSES=$i -DFABRIC_EXP_PROOF_CHECKS=32"; done
for c in 8 128; do build c$c "" "-DFABRIC_EXP_IDLE_PASSES=4 -DFABRIC_EXP_PROOF_CHECKS=$c"; done
build tx16 "" "-DFABRIC_EXP_IDLE_PASSES=4 -DFABRIC_EXP_PROOF_CHECKS=32 -DFABRIC_EXP_TX_DELAY=16"
build rp16 "" "-DFABRIC_EXP_IDLE_PASSES=4 -DFABRIC_EXP_PROOF_CHECKS=32 -DFABRIC_EXP_DRISC_DELAY=1048576"
echo GATE_BUILDS_DONE
