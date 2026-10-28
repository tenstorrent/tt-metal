#!/bin/bash
# v4gi: v4vc with completion design B, decision 1 only (the mechanism; timing as v4vc): one transaction ID per aligned
# completion group, so one status read proves a group free over both command buffers; slot records hold only the
# sender channel; release_completions as due / proof / credit, one path for every group size. Against v4vc in the same
# job (labels v4vcg_*):
#   1. the VC1 split suite, hang capture armed: v4gi at -O3 and the builder's level (auto = -Os), v4vc at -O3;
#   2. VC0-only builds: v4gi data-checked (mesh, 1D, ring, Y-torus), then every suite and the latency rows for both,
#      as v4x_runs.sh.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source $CON/vc1_lib.sh
source $CON/vc1_capture_lib.sh
VC1_RUN=run_vc1_cap vc1_variant v4gi fabric_exp/variants/v4gi v4gi O3 auto
VC1_RUN=run_vc1_cap vc1_variant v4vcg fabric_exp/variants/v4vc v4vcg O3
bash $CON/v4x_runs.sh v4gi v4vc:v4vcg || { echo STOPPED; exit 1; }
rm -f $M
echo ALL_DONE
