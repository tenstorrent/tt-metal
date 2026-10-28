#!/usr/bin/env bash
# K.1#test_runner_smoke ONCE (owner 2026-10-27): keep the hang's tt-triage dump. Never kill this while tt-triage writes;
# when the gate exits, generated/{tt-triage,inspector} and the unit log go to /data before any re-run or box test.
. /scratch/dnijemcevic/node_env.sh
umask 022
cd $HOME/repos/tt-metal
export PATH=$HOME/.local/bin:$PATH PYTHONPATH=$HOME/repos/tt-metal
source python_env/bin/activate
LOG=/scratch/dnijemcevic/k1_smoke.log
KEEP=/data/dnijemcevic/k1_smoke_$(date +%Y%m%d_%H%M)
echo "$(date '+%F %T') STAGE K.1 unit test_runner_smoke (keep -> $KEEP)" >> $LOG
python -u -m models.demos.common.bringup gate K.1 --unit 'K.1#test_runner_smoke' --spec models/demos/deepseek_v41_flash_d_p/bringup/ledgers/galaxy_8x4-full/spec.yaml >> $LOG 2>&1
rc=$?
echo "$(date '+%F %T') STAGE gate exit $rc; copying generated/ to $KEEP" >> $LOG
mkdir -p $KEEP && cp -a generated/tt-triage generated/inspector $KEEP/ 2>>$LOG
R=/scratch/dnijemcevic/bringup/deepseek_v41_flash_d_p/runs/galaxy_8x4-full/run1/logs
cp -a $R/K.1.units.log $R/K.1*.triage.txt $LOG $KEEP/ 2>/dev/null
echo "$(date '+%F %T') STAGE KEPT $(du -sh $KEEP | cut -f1) in $KEEP" >> $LOG
