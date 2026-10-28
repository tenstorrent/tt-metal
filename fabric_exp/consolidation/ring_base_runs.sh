#!/bin/bash
# Baseline for v4 rings and tori: the tree's own router (main, kernel v1) on the ring and Y-torus validation yamls,
# to learn which tests this P150x8 runs and how long they take. Hang capture armed.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source /home/aagarwal/tt-metal/fabric_exp/consolidation/cap_lib.sh
C=/home/aagarwal/.tt-metal-cache-v1-ringval
rm -rf "${C:?}"; mkdir -p $C
run_cap v1_val_ring fabric_exp/v4/val_ring.yaml $C 540 TT_FABRIC_KERNEL_VERSION=1
run_cap v1_val_torusy fabric_exp/v4/val_torus_y.yaml $C 540 TT_FABRIC_KERNEL_VERSION=1
rm -f $M
