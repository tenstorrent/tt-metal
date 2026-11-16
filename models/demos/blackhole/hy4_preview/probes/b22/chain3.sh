#!/bin/bash
cd /mnt/tt-data/ssinghal/hy4-logs/program/b22
while pgrep -f "chain2.sh" >/dev/null; do sleep 5; done
X="HY4_ATTN_FUSE=1 HY4_TP24_PLACE=east HY4_KV_FUSE=1 HY4_RING_KB=128"
NL=78 WD=5400 ./se_b22.sh s78a NR=60 $X
NL=78 WD=5400 ./se_b22.sh s78b NR=60 $X HY4_NORM_FUSE=1
# step 4 unit test (slot 0 only, whole host is ours)
export TT_VISIBLE_DEVICES=0,1,2,3,24,25,26,27
TMO=900 ./../b22_run.sh bo1 bo_test SS=1,2,4 NCALL=4 TIME=1
