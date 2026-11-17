#!/bin/bash
# soak_pfx.sh <host>: A) 2 x (4 concurrent multi-turn sessions x 8 turns) = 64 requests, B) accept_probe all T0 + T0.7 = 54, C) mix 48 random prefix-sharing requests, 4 threads
cd /mnt/tt-data/ssinghal/hy4-logs/program/pfx; H=$1; PY=/mnt/tt-data/ssinghal/tests/tt-metal/python_env/bin/python
st() { echo "$(date -Is) pfx soak: $*" >> ../state/pfx.status.md; echo "$(date -Is) $*" >> soak.log; }
for base in 10 20; do
  for s in 1 2 3 4; do SEED=$((base+s)) $PY multiturn_s.py $H 8 190 > ev_soak_mt_${base}_$s.out 2>&1 & done; wait
  st "A$base: $(grep -h MULTITURN ev_soak_mt_${base}_*.out | tr '\n' ' ')"
done
OUT=ev_soak_ap_T0.json timeout -s TERM 3000 $PY ../product12/accept_probe.py $H all > ev_soak_ap_T0.out 2>&1; st "B T0 rc=$?"
TEMP=0.7 OUT=ev_soak_ap_T07.json timeout -s TERM 3000 $PY ../product12/accept_probe.py $H all > ev_soak_ap_T07.out 2>&1; st "B T0.7 rc=$?"
$PY mix.py $H 48 4 > ev_soak_mix.out 2>&1; st "C: $(tail -n1 ev_soak_mix.out)"
