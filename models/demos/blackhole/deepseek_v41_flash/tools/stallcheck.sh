#!/bin/bash
# Lists device jobs per host with CPU use, and logs untouched for >10 min whose job is still running. Read-only.
L=/mnt/tt-data/ssinghal/dsv4-logs; now=$(date +%s)
for h in 30 31 32 33 34 35 40 41 42 43 44 45 46 47 48; do
  echo "== .$h"
  ssh -o BatchMode=yes -o ConnectTimeout=10 10.82.97.$h 'ps -eo pid,etime,pcpu,args | grep "[p]ytest" | grep -v "flock\|bash -c\|timeout" | sed "s#/mnt/tt-data/ssinghal/tests/tt-metal/python_env/bin/##;s#models/demos/blackhole/deepseek_v41_flash/tests/##" | cut -c1-120; echo "lock waiters: $(ps -eo args | grep -c "[f]lock -w")"' 2>&1 | grep -v Warn
done
echo "== logs updated <60 min ago but silent >10 min (candidates for stuck runs)"
for f in $(ls -t $L/*.log 2>/dev/null | head -60); do a=$(( (now - $(stat -c %Y $f))/60 )); [ $a -ge 10 ] && [ $a -lt 60 ] && echo "${a}m silent: $(basename $f)"; done
