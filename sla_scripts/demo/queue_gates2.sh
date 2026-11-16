#!/bin/bash
# gated steps on top of rs4 + matmul configs (both passed the gate)
export MM_CFG='{"1024x8192":[1,4],"8192x3584":[16,0],"8192x1280":[32,0]}'
export RS_LINKS=4
G=~/teja/tp8_gate.sh
RS_WORKERS=2 ATTN_AG_LINKS=4 bash $G attnag4
ATTN_AG_LINKS=4 CCL_CFG_FROM=Llama-3.1-8B bash $G ccl8b
RS_WORKERS=2 AGMM=1 bash $G agmm
CCL_CFG_FROM=Llama-3.1-8B AGMM=1 bash $G ccl8b_agmm
echo "GATES2 FINISHED"
