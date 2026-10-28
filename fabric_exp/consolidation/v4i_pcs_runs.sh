#!/bin/bash
# Receiver-gap profiles of the frozen v4i: 2D multicast and scatter, 1D (24 worker slots) 3-hop and multicast.
CON=/home/aagarwal/tt-metal/fabric_exp/consolidation
V=fabric_exp/variants
bash $CON/pcsv_runs.sh $V/v4i v4ir2d 2D mcast ewscatter
EXTRA_ENV="TT_FABRIC_EXP_LINEAR_SPARE_SLOTS=1 TT_FABRIC_EXP_WORKER_SLOTS=24" bash $CON/pcsv_runs.sh $V/v4i v4ir1d 1D lin3 linmcast
