#!/bin/bash
# Round-trip latency of one packet (exp_latency.yaml, RESULTS §19) for the one-hop 4352 B write regression (§98):
# v3tk (first and last), v3vc, v3vb, v3v4 at -O3. Labels p18k, p18c, p18b, p18v4, p18k2; per-sample CSVs
# results/<label>_13129088_lat_latency.csv (gitignored), summary: lat_compare.py.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
V=fabric_exp/variants
build v3tk_lat $V/v3tk $C3 O3 p18k lat
build v3vc_lat $V/v3vc $C3 O3 p18c lat
build v3vb_lat $V/v3vb $C3 O3 p18b lat
build v3v4_lat $V/v3v4 $C3 O3 p18v4 lat
build v3tk_lat2 $V/v3tk $C3 O3 p18k2 lat
rm -f $M
echo ALL_DONE
