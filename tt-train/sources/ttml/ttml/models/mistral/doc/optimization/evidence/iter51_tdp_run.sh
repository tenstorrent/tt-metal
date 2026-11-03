#!/bin/bash
# The raised-TDP experiment: set the per-chip limit (default 175 W; the board limit is 550 W for two chips, the current
# limit 500 A and the thermal limit 110 C stay in force), rerun the power-cap repro and its figure, then the 12-step
# full-model MFU run with the 20 Hz telemetry after a cool-down, and restore the default limit at the end whatever
# happened. Usage: iter51_tdp_run.sh [watts]
set -u
W=${1:-175}
cd /home/ttuser/agobeljic/tt-metal/tt-train
export TT_METAL_HOME=/home/ttuser/agobeljic/tt-metal TT_METAL_RUNTIME_ROOT=/home/ttuser/agobeljic/tt-metal
export TT_MESH_GRAPH_DESC_PATH=/home/ttuser/agobeljic/tt-metal/tt-train/configs/mgd/bh_p300x2_1_4_ring_ring.textproto
O=/home/ttuser/agobeljic/tt-metal/tt-train/sources/ttml/ttml/models/mistral/doc/optimization/evidence
PY=../python_env/bin/python
LUWEN=/home/ttuser/.tenstorrent-venv/bin/python
TAG=iter51_tdp${W}
restore() { echo "=== restoring the default TDP limit $(date -u +%FT%TZ)"; $LUWEN $O/iter51_set_tdp.py restore; }
trap restore EXIT
echo "=== $TAG set the TDP limit to ${W} W $(date -u +%FT%TZ)"
$LUWEN $O/iter51_set_tdp.py $W || exit 1
rm -f /dev/shm/sm_segment.* 2>/dev/null
echo "=== $TAG power-cap repro $(date -u +%FT%TZ)"
timeout 900 $PY $O/iter51_power_cap_repro.py $O/${TAG}_power_cap 30 > $O/${TAG}_power_cap_repro.log 2>&1; echo "rc=$?"
grep -E "^(HiFi)" $O/${TAG}_power_cap_repro.log
$PY $O/iter51_power_cap_plot.py $O/${TAG}_power_cap
echo "=== $TAG cool-down 300 s $(date -u +%FT%TZ)"; sleep 300
rm -f /dev/shm/sm_segment.* 2>/dev/null
echo "=== $TAG TP=4 MFU 12 steps with 20 Hz telemetry $(date -u +%FT%TZ)"
$LUWEN $O/iter51_telemetry_sampler.py $O/${TAG}_telemetry.csv 900 20 &
SAMPLER=$!
timeout 3600 $PY sources/examples/mistral_train/train.py --tp 4 --layers 0 --steps 12 --warmup-steps 2 --random-init --random-data --loss-log $O/${TAG}_mfu_tp4.json > $O/${TAG}_mfu_tp4.log 2>&1; echo "rc=$?"
sleep 3; kill $SAMPLER 2>/dev/null; wait $SAMPLER 2>/dev/null
grep -E "layers," $O/${TAG}_mfu_tp4.log | tail -1
python3 $O/iter51_clock_summary.py $O/${TAG}_telemetry.csv $O/${TAG}_mfu_tp4.log
echo "=== $TAG done $(date -u +%FT%TZ)"
