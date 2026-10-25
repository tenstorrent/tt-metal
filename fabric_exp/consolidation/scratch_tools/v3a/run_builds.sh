#!/bin/bash
# Three parallel streams of offline builds from the frozen snapshot.
S=/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/v3a
cd /home/aagarwal/tt-metal
V=fabric_exp/variants
b() { local name=$1; shift; local t0=$(date +%s); python3 $S/ob.py build "$@" --jobs 20 > $S/logs/$name.log 2>&1; echo "$name rc=$? $(( $(date +%s)-t0 ))s $(tail -1 $S/logs/$name.log)" >> $S/logs/summary.txt; }
mkdir -p $S/logs; : > $S/logs/summary.txt
( b v3a_O3 --variant $V/v3a --out $S/O3; b v3a_O2 --variant $V/v3a --out $S/O2 --opt O2; b v3a_Os --variant $V/v3a --out $S/Os --opt Os; b probe_hook --variant $S/probe/hook --out $S/probe_hook ) &
( b nohint_O3 --variant $S/variants/s79mc_nohint --out $S/nohint; b v2hdr_O3 --variant $S/variants/v2hdr --out $S/v2hdr; b base_O2 --variant $V/s79mc --out $S/base_O2 --opt O2; b base_Os --variant $V/s79mc --out $S/base_Os --opt Os ) &
( b v1orig_O3 --variant $S/variants/v1orig --out $S/v1orig; b v1hdr_O3 --variant $S/variants/v1hdr --out $S/v1hdr; b probe_statefield --variant $S/probe/statefield --out $S/probe_statefield ) &
wait
echo ALL DONE >> $S/logs/summary.txt
