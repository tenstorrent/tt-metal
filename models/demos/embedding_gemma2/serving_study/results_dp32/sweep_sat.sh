#!/bin/bash
# Saturation sweep on .40: DP=32 batched (MAXB=32, ladder 1-32), 32..4096 requests in flight (1..128 per chip), at 128
# and 512 tokens, with 8 and 16 front-ends. Per point: reset every front-end's /metrics, 20 s closed-loop load, then
# collect the texts-per-forward histogram (the batch the chips really ran). vmstat logs host CPU for the whole run.
# Output: ~/gtobar/embeddinggemma/results/sweep_sat/{load,batch}_k<K>_t<T>_c<C>.json, vmstat.txt
set -u
source ~/gtobar/embeddinggemma/env.sh; cd $EMB_SCRIPTS
R=$E/results/sweep_sat; mkdir -p $R
CONCS="32 64 128 256 512 1024 2048 4096"
export EMB_LOAD_SECONDS=20 EMB_LOAD_WARMUP_S=5
busy=$(for d in /dev/tenstorrent/[0-9]*; do fuser $d 2>/dev/null; done | wc -w)
[ "$busy" -eq 0 ] || { echo "chips busy ($busy holders): not starting"; exit 3; }
vmstat -t 5 > $R/vmstat.txt 2>&1 &
VM=$!
stop_servers() { pkill -f "[e]mb_serve_fastapi"; sleep 10; }

for K in 8 16; do
  DP=32 FRONTENDS=$K MAXB=32 EMBEDDING_GEMMA2_SERVE_BATCHES=1,2,4,8,16,32 bash run_fastapi_dp.sh > $R/launch_k$K.txt
  TS=$(ls -t $E/logs/fastapi_dp32_k${K}_f0_*.log | head -1 | sed "s/.*_f0_//")
  for i in $(seq 1 120); do n=$(grep -l "Embedding pool ready" $E/logs/fastapi_dp32_k${K}_f*_$TS 2>/dev/null | wc -l); [ $n -ge $K ] && break; sleep 5; done
  echo "K=$K front-ends ready: $n/$K (logs *_$TS)"
  if [ $n -lt $K ]; then echo "INVALID: not all front-ends up"; stop_servers; continue; fi
  for T in 128 512; do
    for C in $CONCS; do
      python3 emb_fe_metrics.py http://127.0.0.1:8132 --frontends $K --reset > /dev/null
      echo "$(date -u +%H:%M:%S) start k$K t$T c$C" >> $R/points.txt
      python3 emb_load_mp.py http://127.0.0.1:8132 $R/load_k${K}_t${T}_c$C.json --tokens $T --conc $C --procs 16 --tag k$K 2>&1 | grep "^{"
      python3 emb_fe_metrics.py http://127.0.0.1:8132 --frontends $K --out $R/batch_k${K}_t${T}_c$C.json
    done
  done
  grep -hE "ERROR|failed" $E/logs/fastapi_dp32_k${K}_f*_$TS | head -3
  stop_servers
done
kill $VM
echo SWEEP_DONE
