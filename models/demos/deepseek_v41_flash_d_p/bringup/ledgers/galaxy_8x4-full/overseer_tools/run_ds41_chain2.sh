#!/usr/bin/env bash
# Owner 2026-10-28: FIX.10 (resident engram tables fit the Galaxy host) -> if PASS the resident 5-position profile ->
# orchestrator --until P.6.
. /scratch/dnijemcevic/node_env.sh
umask 022
cd $HOME/repos/tt-metal
export PATH=$HOME/.local/bin:$PATH PYTHONPATH=$HOME/repos/tt-metal
source python_env/bin/activate
S=models/demos/deepseek_v41_flash_d_p/bringup/ledgers/galaxy_8x4-full/spec.yaml
OL=/scratch/dnijemcevic/bringup/deepseek_v41_flash_d_p/runs/galaxy_8x4-full/run1/orchestrator.log
LOG=/scratch/dnijemcevic/engram_check.log
echo "$(date '+%F %T') CHAIN2 orchestrator --until FIX.10 @ $(git rev-parse --short HEAD)" >> $LOG
echo "$(date "+%F %T") launch: orchestrator resume --until FIX.10" >> $OL
python -u -m models.demos.common.bringup.orchestrator resume --until FIX.10 --spec $S >> $OL 2>&1
if ! python -m models.demos.common.bringup status --spec $S 2>/dev/null | grep -qE "^FIX\.10 +PASS"; then echo "$(date '+%F %T') CHAIN2 STOP: FIX.10 not PASS" >> $LOG; exit 1; fi
echo "$(date '+%F %T') CHAIN2 profile series" >> $LOG
/scratch/dnijemcevic/posprof/series_engram.sh
echo "$(date '+%F %T') CHAIN2 DONE (P.6 waits for the overseer to set its bar)" >> $LOG
