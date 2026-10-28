#!/usr/bin/env bash
. /scratch/dnijemcevic/node_env.sh
# F196 warm wall per position: warm until no new program-cache entry, then 10 timed runs with misses refused.
umask 022
cd /home/dnijemcevic/repos/tt-metal
export PATH=$HOME/.local/bin:$PATH
source python_env/bin/activate
D=/scratch/dnijemcevic/posprof
unset TT_METAL_DEVICE_PROFILER
for P in ${POSITIONS:-0 51200 102400 204800 512000}; do
  echo "=== WALL $P $(date -u +%T)" >> $D/${TAG:-}wall.log
  PYTHONPATH=$PWD:$D BRINGUP_SPEC=models/demos/deepseek_v41_flash_d_p/bringup/ledgers/galaxy_8x4-full/spec.yaml \
  BRINGUP_TASK=${TAG:-}WALL$P BRINGUP_RESULTS_DIR=$D/results BRINGUP_POS=$P BRINGUP_PROFILE_WALL=1 \
  timeout 5400 scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/common/bringup/tests/test_profile.py -p pos_profile_plugin -s > $D/${TAG:-}wall$P.log 2>&1
  echo "rc=$? $(date -u +%T)" >> $D/${TAG:-}wall.log
  grep -a "PROFILE wall" $D/${TAG:-}wall$P.log | tail -1 >> $D/${TAG:-}wall.log
done
echo DONE >> $D/${TAG:-}wall.log
