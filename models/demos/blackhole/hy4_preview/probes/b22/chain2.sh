#!/bin/bash
cd /mnt/tt-data/ssinghal/hy4-logs/program/b22
while pgrep -f "chain1.sh" >/dev/null; do sleep 5; done
X="HY4_ATTN_FUSE=1 HY4_TP24_PLACE=east HY4_KV_FUSE=1 HY4_RING_KB=128"
C="CTXS=65536 PAGES=128 NGEN=12 $X"
WD=4500 ./nl_b22.sh n78a 78 $C
WD=4500 ./nl_b22.sh n78b 78 $C HY4_NORM_FUSE=1
NL=78 WD=5400 ./se_b22.sh s78a NR=60 $X
NL=78 WD=5400 ./se_b22.sh s78b NR=60 $X HY4_NORM_FUSE=1
NL=78 WD=7200 ./q_b22.sh nf TFREP=2 GSMN=0 $X HY4_NORM_FUSE=1
WD=4500 ./nl_b22.sh n78a2 78 $C
WD=4500 ./nl_b22.sh n78b2 78 $C HY4_NORM_FUSE=1
