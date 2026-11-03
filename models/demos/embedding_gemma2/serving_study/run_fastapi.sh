#!/bin/bash
# FastAPI template server (emb_serve_fastapi.py), 1 worker on the comparison chip, port 8101. Prints pid + log.
source "$(dirname "${BASH_SOURCE[0]}")/env_emb.sh"
L=$E/logs/fastapi_$(date -u +%Y%m%dT%H%M%S).log
cd $EMB_SCRIPTS
nohup python3 emb_serve_fastapi.py --dp 1 --chips $TT_VISIBLE_DEVICES --port 8101 ${EMB_FASTAPI_ARGS:-} > $L 2>&1 < /dev/null &
echo "pid $! log $L url http://127.0.0.1:8101"
