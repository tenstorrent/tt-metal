#!/bin/bash
# DP FastAPI server: one worker per chip (TT_VISIBLE_DEVICES=0..DP-1), least-loaded dispatch.
# FRONTENDS=K (default 8) runs K server processes on one port (SO_REUSEPORT), each owning DP/K consecutive chips: one
# Python event loop serves ~700 req/s, so a single front-end caps 32 chips at ~1/7 of what they can do.
# Front-end 0 starts first and fills the weight/kernel caches; the others start once it is ready.
# Env: DP (32), FRONTENDS (8), PORT (8132), MAXB (most texts per forward per chip, default 32; a chip gets a second
# text only once every chip holds one, so light load still runs batch 1), EMBEDDING_GEMMA2_SERVE_BATCHES (trace
# ladder, default 1,2,4,8,16,32: a batch pads to the next rung, and at 512 tokens padding 2 texts to 8 rows costs more
# than the batching saves), EMB_FASTAPI_ARGS (extra flags, e.g. --debug-chip).
# Needs the SDPA reader write barrier (sdpa: wait for K/V chain write acks): without it traced batch-8 forwards hang.
# Prints pids + logs; stop with: pkill -f "emb_serve_fastapi.py --dp".
source "$(dirname "${BASH_SOURCE[0]}")/env_emb.sh"
DP=${DP:-32}; K=${FRONTENDS:-8}; PORT=${PORT:-8132}; MAXB=${MAXB:-32}
export EMBEDDING_GEMMA2_SERVE_BATCHES=${EMBEDDING_GEMMA2_SERVE_BATCHES:-1,2,4,8,16,32}
(( DP % K == 0 )) || { echo "DP ($DP) must be a multiple of FRONTENDS ($K)"; exit 2; }
PER=$((DP / K)); TS=$(date -u +%Y%m%dT%H%M%S)
cd $EMB_SCRIPTS
EXTRA=""; [ $K -gt 1 ] && EXTRA="--reuse-port --server-phys 0"
for f in $(seq 0 $((K - 1))); do
  CH=$(seq -s, $((f * PER)) $((f * PER + PER - 1)))
  L=$E/logs/fastapi_dp${DP}_k${K}_f${f}_$TS.log
  nohup python3 emb_serve_fastapi.py --dp $PER --chips $CH --port $PORT --max-batch $MAXB $EXTRA ${EMB_FASTAPI_ARGS:-} \
    > $L 2>&1 < /dev/null &
  echo "frontend $f pid $! chips $CH log $L"
  if [ $f -eq 0 ] && [ $K -gt 1 ]; then  # warm caches before the rest build
    for i in $(seq 1 180); do grep -qE "Embedding pool ready|failed|Traceback" $L && break; sleep 5; done
    grep -q "Embedding pool ready" $L || { echo "frontend 0 did not come up: $L"; exit 1; }
  fi
done
echo "url http://127.0.0.1:$PORT  (GET /health answers for whichever front-end takes the connection)"
