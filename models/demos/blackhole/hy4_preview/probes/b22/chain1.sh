#!/bin/bash
cd /mnt/tt-data/ssinghal/hy4-logs/program/b22
C="CTXS=65536 PAGES=128 NGEN=12 HY4_ATTN_FUSE=1 HY4_TP24_PLACE=east HY4_KV_FUSE=1 HY4_RING_KB=128"
WD=2400 ./nl_b22.sh n16base 16 $C
WD=2400 ./nl_b22.sh n16nf 16 $C HY4_NORM_FUSE=1
WD=2400 ./nl_b22.sh n16base2 16 $C
WD=2400 ./nl_b22.sh n16nf2 16 $C HY4_NORM_FUSE=1
