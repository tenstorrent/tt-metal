#!/bin/bash
# usage (ON the device host, background it): adapt_run.sh <logname> "<ENV=val ...>" "<pytest -k expr>"
# One pytest process of demo/text_demo.py under the per-host flock + a hangwatch (kills + triages + resets after 25 min of log silence).
LOGN=$1; HW_MIN=${HW_MIN:-25}; ENVS=$2; KEXPR=$3; W=/mnt/tt-data/ssinghal/wt/spec_adapt; H=$(hostname -s | tr -dc 0-9 | tail -c 2)
LOG=/mnt/tt-data/ssinghal/dsv4-logs/spec_adapt_${LOGN}_h${H}.log
exec $W/run.sh "
export DSV41_MEMLOG=1 DSV41_ENGRAM_RAM=1 DSV41_TRACE_REGION=1900000000 DSV41_SPEC_PRINT=2 $ENVS
echo ADAPT_ENV \$(hostname -s) \$(date +%FT%T) head=\$(git -C $W rev-parse --short=11 HEAD)+wt >> $LOG
env | grep -E '^(DSV41_|MOE_COMPUTE_|TT_METAL_)' | sort | sed 's/^/ADAPT_ENV /' >> $LOG
timeout 43200 pytest -x -s -q -o junit_suite_name=spec_adapt models/demos/blackhole/deepseek_v41_flash/demo/text_demo.py -k '$KEXPR' >> $LOG 2>&1 &
P=\$!
$W/models/demos/blackhole/deepseek_v41_flash/tools/hangwatch.sh \$P $LOG ${HW_MIN:-25}
wait \$P
echo ADAPT_DONE rc=\$? >> $LOG
"
