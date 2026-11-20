#!/usr/bin/env bash
# Galaxy end-to-end check of the 2x2 engram host-path commits (owner 2026-10-28): ladder 56k + Paris through tt-d-gen.
. /scratch/dnijemcevic/node_env.sh
umask 022
cd $HOME/repos/tt-metal
export PATH=$HOME/.local/bin:$PATH PYTHONPATH=$HOME/repos/tt-metal
source python_env/bin/activate
S=models/demos/deepseek_v41_flash_d_p/bringup/ledgers/galaxy_8x4-full/spec.yaml
LOG=/scratch/dnijemcevic/engram_check.log
echo "$(date '+%F %T') START verify L.s56320 @ $(git rev-parse --short HEAD)" >> $LOG
python -u -m models.demos.common.bringup.core.verify --spec $S --tasks L.s56320 >> $LOG 2>&1; echo "$(date '+%F %T') L.s56320 exit $?" >> $LOG
python -u -m models.demos.common.bringup gate K.1 --spec $S --unit 'K.1#test_runner_smoke' >> $LOG 2>&1; echo "$(date '+%F %T') K.1#test_runner_smoke exit $?" >> $LOG
echo "$(date '+%F %T') DONE" >> $LOG
