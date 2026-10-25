#!/bin/bash
S=/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/v3a
cd /home/aagarwal/tt-metal
V=fabric_exp/variants
b() { local name=$1; shift; local t0=$(date +%s); python3 $S/ob.py build "$@" --jobs 20 > $S/logs/$name.log 2>&1; echo "$name rc=$? $(( $(date +%s)-t0 ))s $(tail -n 1 $S/logs/$name.log)" >> $S/logs/final.txt; }
: > $S/logs/final.txt
( b f_O3 --variant $V/v3a --out $S/fO3; b f_hook --variant $S/probe/hook --out $S/fprobe_hook ) &
( b f_O2 --variant $V/v3a --out $S/fO2 --opt O2; b f_statefield --variant $S/probe/statefield --out $S/fprobe_statefield ) &
( b f_Os --variant $V/v3a --out $S/fOs --opt Os ) &
wait
echo ALL DONE >> $S/logs/final.txt
