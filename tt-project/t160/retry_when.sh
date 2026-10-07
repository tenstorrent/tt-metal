#!/bin/bash
# exits 0 when the weight copy ended (ok or not) or the exabox build job 127515 ended non-zero; 1 otherwise.
/home/smarton/fasth3/tt-metal/tt-project/harness/bin/ttp detach --check /home/smarton/fasth3/tt-metal/tt-project/state/runs/724/t160-pcopy.rc >/dev/null 2>&1 && exit 0
timeout 40 ssh -o BatchMode=yes -o ConnectTimeout=15 exabox-login \
  'f=/data/smarton/fasth3/t160/job-127515.rc; [ -e $f ] && ! grep -q "JOB_RC=0 " $f' && exit 0
exit 1
