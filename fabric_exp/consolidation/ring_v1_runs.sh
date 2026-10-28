#!/bin/bash
# Main's router (kernel v1) on the ring / Y-torus tests the mock probe passed: validation (data-checked) and the
# benchmark rows, as the baseline for v4. Hang capture armed.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source /home/aagarwal/tt-metal/fabric_exp/consolidation/cap_lib.sh
C=/home/aagarwal/.tt-metal-cache-v1-ringval
run_cap v1_val_ring_p150 fabric_exp/v4/val_ring_p150.yaml $C 400 TT_FABRIC_KERNEL_VERSION=1 || { echo STOPPED; exit 1; }
run_cap v1_val_torusy_p150 fabric_exp/v4/val_torus_y_p150.yaml $C 300 TT_FABRIC_KERNEL_VERSION=1 || { echo STOPPED; exit 1; }
run_cap v1_13129088_ring fabric_exp/v4/exp_ring.yaml $C 300 TT_FABRIC_KERNEL_VERSION=1
run_cap v1_13129088_torusy fabric_exp/v4/exp_torus_y.yaml $C 300 TT_FABRIC_KERNEL_VERSION=1
rm -f $M
