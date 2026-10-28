#!/bin/bash
# v4vc: v4 (v4uf) with VC1 (V4_VC1_DESIGN.md steps 2-3): VC1 forwarding channels on E0, the VC1 receiver channel on E1
# (state in a record, one credit frame for both channels), downstreams per VC with crossover, intermesh landing and
# exits; init code made smaller (VC1 E0 + E1 at -O3: 24,424 B of 26,624).
#   1. the VC1 split suite (multi-mesh sanity, data-checked, then the intra-mesh rows on VC1 routers), hang capture
#      armed, at -O3 (the design's level), the builder's level (auto = -Os) and -O2; labels v4vc_<level>;
#   2. VC0-only builds (their loops are v4uf's code): data-checked validation, then every suite and the latency rows,
#      with v4uf as the same-job control (labels v4vc_13129088_*, v4ufv_13129088_*), as v4x_runs.sh.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source $CON/vc1_lib.sh
source $CON/vc1_capture_lib.sh
VC1_RUN=run_vc1_cap vc1_variant v4vc fabric_exp/variants/v4vc v4vc O3 auto O2
bash $CON/v4x_runs.sh v4vc v4uf:v4ufv || { echo STOPPED; exit 1; }
rm -f $M
echo ALL_DONE
