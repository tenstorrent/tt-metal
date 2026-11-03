#!/bin/bash
# Repro of the 2026-10-29 FastAPI --max-batch 32 hang: only the 32-in-flight x 128-token load point.
# Watchdog: if the load client is not done 90 s after it starts, py-spy dump the chip worker and stop everything.
source "$(dirname "${BASH_SOURCE[0]}")/env_emb.sh"
R=$E/results/fastapi_bs32_repro_$(date -u +%Y%m%dT%H%M); mkdir -p $R
[ -z "$(fuser $EMB_DEV 2>/dev/null)" ] || { echo "chip busy"; exit 3; }
out=$(EMB_BATCH_LOG=1 EMB_FASTAPI_ARGS="--max-batch ${MAXB:-32}" $EMB_SCRIPTS/run_fastapi.sh); echo "$out"
pid=$(echo "$out" | sed -n 's/^pid \([0-9]*\).*/\1/p'); log=$(echo "$out" | sed -n 's/.* log \([^ ]*\).*/\1/p')
for i in $(seq 72); do [ "$(curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:8101/health)" = 200 ] && break; kill -0 $pid || { echo died; tail $log; exit 1; }; sleep 5; done
echo "up after $((i*5)) s"
(cd $EMB_SCRIPTS && EMB_LOAD_SECONDS=30 timeout 90 python3 emb_load.py http://127.0.0.1:8101 $R/load_c${CONCS:-32}_t${TOK:-128}.json ${TOK:-128} ${CONCS:-32}) > $R/load.log 2>&1; rc=$?
grep '^{' $R/load.log; echo "load exit $rc (124 = watchdog fired)"
worker=$(pgrep -P $pid -f spawn_main)
if [ $rc -ne 0 ]; then ${PY_SPY:-/home/ttuser/.local/bin/py-spy} dump --pid $worker > $R/worker_stack.txt 2>&1; head -20 $R/worker_stack.txt; fi
grep -c "^\[batch\]" $log | sed 's/^/forwards logged: /'; grep "^\[batch\]" $log | awk '{print $2}' | sort | uniq -c | sort -k2 -t= -n | head -40 | tr '\n' ' '; echo; echo "last forwards:"; grep "^\[batch\]" $log | tail -5
kill -TERM $pid; sleep 10; for p in $worker $pid; do kill -0 $p 2>/dev/null && kill -KILL $p; done; sleep 3
cp $log $R/; fuser $EMB_DEV 2>/dev/null && echo "chip 31 still held" || echo "chip 31 released"; echo "-> $R"
