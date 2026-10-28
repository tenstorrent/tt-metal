#!/usr/bin/env bash
# Owner 2026-10-28: (1) Paris with the default (resident) engram tables; only if it passes: (2) the 5-position profile
# with the tables resident; (3) the orchestrator for perf task P.6 ENGRAM_ASYNC only (--until P.6).
. /scratch/dnijemcevic/node_env.sh
umask 022
cd $HOME/repos/tt-metal
export PATH=$HOME/.local/bin:$PATH PYTHONPATH=$HOME/repos/tt-metal
source python_env/bin/activate
S=models/demos/deepseek_v41_flash_d_p/bringup/ledgers/galaxy_8x4-full/spec.yaml
LOG=/scratch/dnijemcevic/engram_check.log
echo "$(date '+%F %T') CHAIN START Paris (resident) @ $(git rev-parse --short HEAD)" >> $LOG
python -u -m models.demos.common.bringup gate K.1 --spec $S --unit 'K.1#test_runner_smoke' > /scratch/dnijemcevic/paris_resident.log 2>&1
cat /scratch/dnijemcevic/paris_resident.log | tail -6 >> $LOG
if ! grep -q "K.1#test_runner_smoke: PASS" /scratch/dnijemcevic/paris_resident.log; then echo "$(date '+%F %T') CHAIN STOP: Paris not PASS" >> $LOG; exit 1; fi
echo "$(date '+%F %T') CHAIN profile series" >> $LOG
/scratch/dnijemcevic/posprof/series_engram.sh
echo "$(date '+%F %T') CHAIN orchestrator --until P.6" >> $LOG
echo "$(date "+%F %T") launch: orchestrator resume --until P.6" >> /scratch/dnijemcevic/bringup/deepseek_v41_flash_d_p/runs/galaxy_8x4-full/run1/orchestrator.log
python -u -m models.demos.common.bringup.orchestrator resume --until P.6 --spec $S >> /scratch/dnijemcevic/bringup/deepseek_v41_flash_d_p/runs/galaxy_8x4-full/run1/orchestrator.log 2>&1
echo "$(date '+%F %T') CHAIN DONE orchestrator exit $?" >> $LOG
