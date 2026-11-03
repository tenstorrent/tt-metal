#!/bin/bash
# Sustained load per server (timing off): closed-loop single-text requests at 1/8/32 in flight, 128 and 512 tokens.
# Usage: load_chain.sh <results dir>   env SRVS (default "fastapi media_py media_cpp"), EMB_LOAD_SECONDS.
source "$(dirname "${BASH_SOURCE[0]}")/env_emb.sh"
R=${1:?results dir}; mkdir -p $R
free_chip() { [ -z "$(fuser $EMB_DEV 2>/dev/null)" ]; }
wait_up() { for i in $(seq 1 360); do
    [ "$(curl -s -m 5 -o /dev/null -w '%{http_code}' -H "Authorization: Bearer ${API_KEY:-your-secret-key}" $1/health)" = 200 ] && { echo "up after $((i*5)) s"; return 0; }
    grep -q "initiating server shutdown" $3 2>/dev/null && { echo "server fatal"; tail -30 $3; return 1; }
    kill -0 $2 2>/dev/null || { echo "server died"; tail -30 $3; return 1; }; sleep 5; done; echo "not up after 30 min"; return 1; }
stop() { kill -TERM $1 2>/dev/null; for i in $(seq 1 60); do kill -0 $1 2>/dev/null || break; sleep 2; done; kill -0 $1 2>/dev/null && kill -KILL $1; sleep 5; }
unset EMB_TIMING_LOG
for srv in ${SRVS:-fastapi media_py media_cpp}; do
  echo "== $srv load $(date -u +%H:%MZ)"; free_chip || { echo "chip busy"; exit 3; }
  out=$($EMB_SCRIPTS/run_$srv.sh); echo "$out"
  pid=$(echo "$out" | sed -n 's/^pid \([0-9]*\).*/\1/p'); log=$(echo "$out" | sed -n 's/.* log \([^ ]*\).*/\1/p'); url=$(echo "$out" | sed -n 's/.* url \(.*\)$/\1/p')
  if wait_up $url $pid $log; then
    for tok in 128 512; do
      c=$R/load_${srv}_${tok}.log
      (cd $EMB_SCRIPTS && python3 emb_load.py $url $R/load_${srv}_${tok}.json $tok 1 8 32) > $c 2>&1
      grep '^{' $c | sed "s/^/$srv /" || { echo "load failed:"; tail -5 $c; }
    done
  fi
  cp $log $R/ 2>/dev/null; stop $pid
done
echo "load done $(date -u +%FT%TZ) -> $R"
