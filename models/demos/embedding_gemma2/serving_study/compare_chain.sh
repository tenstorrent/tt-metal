#!/bin/bash
# Full comparison: bare (timing off, on), then each server (timing off, on) with the same client. Results in $E/results/.
source "$(dirname "${BASH_SOURCE[0]}")/env_emb.sh"
R=$E/results/$(date -u +%Y%m%dT%H%M); mkdir -p $R; REPS=${REPS:-20}
free_chip() { [ -z "$(fuser $EMB_DEV 2>/dev/null)" ]; }
wait_up() { # url pid log
  for i in $(seq 1 360); do
    code=$(curl -s -m 5 -o /dev/null -w '%{http_code}' -H "Authorization: Bearer ${API_KEY:-your-secret-key}" $1/health)
    [ "$code" = 200 ] && { echo "up after $((i*5)) s"; return 0; }
    grep -q "initiating server shutdown" $3 2>/dev/null && { echo "server fatal"; tail -30 $3; return 1; }
    kill -0 $2 2>/dev/null || { echo "server died"; tail -30 $3; return 1; }
    sleep 5
  done; echo "not up after 30 min"; return 1; }
stop() { kill -TERM $1 2>/dev/null; for i in $(seq 1 60); do kill -0 $1 2>/dev/null || break; sleep 2; done; kill -0 $1 2>/dev/null && kill -KILL $1; sleep 5; }
free_chip || { echo "chip busy"; exit 3; }
echo "== bare $(date -u +%H:%MZ)"
(cd $EMB_SCRIPTS && python3 emb_bare.py $R/bare.json $REPS) > $R/bare.log 2>&1; grep '^{' $R/bare.log
(cd $EMB_SCRIPTS && EMB_TIMING_LOG=$R/timing_bare.jsonl python3 emb_bare.py $R/bare_timed.json $REPS) > $R/bare_timed.log 2>&1
for srv in ${SRVS:-fastapi media_py media_cpp}; do
  for timed in 0 1; do
    echo "== $srv timed=$timed $(date -u +%H:%MZ)"
    free_chip || { echo "chip busy"; exit 3; }
    if [ $timed = 1 ]; then export EMB_TIMING_LOG=$R/timing_$srv.jsonl; else unset EMB_TIMING_LOG; fi
    out=$($EMB_SCRIPTS/run_$srv.sh); echo "$out"
    pid=$(echo "$out" | sed -n 's/^pid \([0-9]*\).*/\1/p'); log=$(echo "$out" | sed -n 's/.* log \([^ ]*\).*/\1/p'); url=$(echo "$out" | sed -n 's/.* url \(.*\)$/\1/p')
    if wait_up $url $pid $log; then
      c=$R/client_${srv}_t$timed.log
      (cd $EMB_SCRIPTS && python3 emb_client.py $url $R/${srv}_t$timed.json $REPS $R/bare.json) > $c 2>&1
      grep '^{' $c || { echo "client failed:"; tail -5 $c; }
    fi
    cp $log $R/ 2>/dev/null; stop $pid
  done
done
unset EMB_TIMING_LOG; echo "chain done $(date -u +%FT%TZ) -> $R"
