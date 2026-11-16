#!/bin/bash
# token accuracy vs BF16 reference: RS 4 links alone vs RS 4 links + matmul configs
export RS_LINKS=4 RS_WORKERS=2
K="performance and ci-token-matching" bash ~/teja/run_tp8dp4.sh acc_rs4
K="performance and ci-token-matching" MM_CFG='{"1024x8192":[1,4],"8192x3584":[16,0],"8192x1280":[32,0]}' \
  bash ~/teja/run_tp8dp4.sh acc_rs4_mm
grep -iE "top-?1|top-?5" ~/teja/runs/tp8dp4_acc_rs4.log ~/teja/runs/tp8dp4_acc_rs4_mm.log | grep -v DEBUG | tail -8
echo ACCDONE
