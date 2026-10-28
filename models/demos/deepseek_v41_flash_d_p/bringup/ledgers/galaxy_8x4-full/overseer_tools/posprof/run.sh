#!/usr/bin/env bash
. /scratch/dnijemcevic/node_env.sh
# One process per position (a fresh mesh each; the 500k state is the largest). Results in posprof/results/POS<p>*.json.
umask 022
cd /home/dnijemcevic/repos/tt-metal
export PATH=$HOME/.local/bin:$PATH
source python_env/bin/activate
D=/scratch/dnijemcevic/posprof
for P in ${POSITIONS:-0 51200 102400 204800 512000}; do
  echo "=== POS $P $(date -u +%T)" >> $D/${TAG:-}run.log
  PYTHONPATH=$PWD:$D BRINGUP_SPEC=models/demos/deepseek_v41_flash_d_p/bringup/ledgers/galaxy_8x4-full/spec.yaml \
  BRINGUP_TASK=${TAG:-POS}$P BRINGUP_RESULTS_DIR=$D/results BRINGUP_POS=$P \
  timeout 2400 scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/common/bringup/tests/test_profile.py -p pos_profile_plugin -s > $D/${TAG:-}pos$P.log 2>&1
  echo "rc=$? $(date -u +%T)" >> $D/${TAG:-}run.log
done
echo DONE >> $D/${TAG:-}run.log
