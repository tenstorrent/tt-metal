#!/bin/bash
while pgrep -f queue_gates3.sh >/dev/null; do sleep 20; done
export RS_LINKS=4 RS_WORKERS=2
MM_CFG='{"1024x8192":[1,4],"8192x1280":[32,0]}' SILU_FUSE=1 bash ~/teja/tp8_gate.sh silu_fuse
echo "GATES4 FINISHED"
