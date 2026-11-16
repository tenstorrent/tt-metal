#!/bin/bash
# matmul config step on top of RS 4 links / 2 workers; value = [in0_block_w, cores_per_bank]
export RS_LINKS=4 RS_WORKERS=2
MM_CFG='{"1024x8192":[1,4]}' bash ~/teja/tp8_step.sh mm_wo
MM_CFG='{"8192x3584":[16,0]}' bash ~/teja/tp8_step.sh mm_w1w3
MM_CFG='{"1024x8192":[1,4],"8192x3584":[16,0],"8192x1280":[32,0]}' bash ~/teja/tp8_step.sh mm_all
RS_WORKERS=3 bash ~/teja/tp8_step.sh rs_l4_w3_retry
echo MMDONE
