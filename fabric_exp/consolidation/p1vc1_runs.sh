#!/bin/bash
# Phase 1: VC1 baselines on the P150x8 split into two 2x2 meshes (fabric_exp/vc1, RESULTS §84). For each router:
# default level (the builder picks -Os because VC1 is active) and -O2, each in a fresh cache, compile commands logged.
# fabric_exp/vc1/vc1_split.yaml: multi-mesh sanity (data checked), then intra-mesh 1-hop E/W, N/S and multicast
# benchmarks (per-test CSV kept as results/<label>_13129088_vc1_tests.csv).
#   s79mc (the tree's router)    labels p1vc1s_auto_, p1vc1s_o2_
#   v3a (swapped in, v2's place) labels p1vc1a_auto_, p1vc1a_o2_
# Usage: p1vc1_runs.sh [a]   (variant list below)
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source $CON/vc1_lib.sh
case "${1:-a}" in
a)  vc1_variant s79mc tree p1vc1s auto O2
    vc1_variant v3a fabric_exp/variants/v3a p1vc1a auto O2 ;;
esac
rm -f $M
echo ALL_DONE
