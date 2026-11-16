#!/bin/bash
# 128 users (DP4 x 32): stock, RS 4 links, RS 4 links + matmul configs
bash ~/teja/tp8_step128.sh baseline
RS_LINKS=4 RS_WORKERS=2 bash ~/teja/tp8_step128.sh rs4
RS_LINKS=4 RS_WORKERS=2 MM_CFG='{"1024x8192":[1,4],"8192x3584":[16,0],"8192x1280":[32,0]}' \
  bash ~/teja/tp8_step128.sh rs4_mm
echo B128DONE
