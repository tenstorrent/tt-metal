#!/bin/bash
# Free = no device holders, no pytest/vllm, and no other user's driver (anything under /home/ttuser/gtobar or other users' dirs)
# for two consecutive checks 120 s apart. Waits up to $1 minutes (default 240). Exit 0 free, 1 timeout.
max=${1:-240}; start=$(date +%s)
busy() {
  (lsof /dev/tenstorrent/* 2>/dev/null; fuser -v /dev/tenstorrent/* 2>&1 | grep -E 'tenstorrent.*[0-9]';
   ps -eo pid,cmd | grep -E 'pytest|vllm|run_vllm_api_server|/home/ttuser/(gtobar|work)/.*\.(sh|py)|tt_per_layer|tt-smi' \
     | grep -vE 'grep|gate2|serve_wasm|http.server|clang|cmake|ninja') | head -3
}
while :; do
  o=$(busy)
  if [ -z "$o" ]; then sleep 120; o=$(busy); [ -z "$o" ] && exit 0; fi
  echo "BUSY $(date +%T): $(echo "$o" | head -1 | cut -c1-160)"
  [ $(( $(date +%s) - start )) -ge $((max*60)) ] && exit 1
  sleep 60
done
