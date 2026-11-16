#!/bin/bash
# matmul configs without w1w3 (w1w3 config was slower in the model profile)
export RS_LINKS=4 RS_WORKERS=2
MM_CFG='{"1024x8192":[1,4],"8192x1280":[32,0]}' bash ~/teja/tp8_gate.sh mm_no_w1w3
MM_CFG='{"1024x8192":[1,4]}' bash ~/teja/tp8_gate.sh mm_wo_only
echo "GATES3 FINISHED"
