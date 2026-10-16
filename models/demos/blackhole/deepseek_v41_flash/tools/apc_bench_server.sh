#!/bin/bash
# usage (ON the host): apc_server.sh <name> <apc 0|1> <spec_k> <max_model_len> <mnbt> [client args...]
# boots a 40-layer vLLM 0.31 server from tm_apc + plugin_apc under the device lock (B=16), runs apc_client.py, stops the server, resets the board. logs: dsv4-logs/apc/<name>.*
N=$1; APC=$2; K=$3; L=$4; MN=$5; shift 5
D=/mnt/tt-data/ssinghal/dsv4-logs/apc
exec flock -w 86400 /tmp/dsv4_dev.lock bash -c "
export TT=/mnt/tt-data/ssinghal/wt/tm_apc PLUGIN=/mnt/tt-data/ssinghal/wt/plugin_apc DSV41_VLLM_CHUNK=512 MNBT=$MN TRACE=1600000000
if [ $APC = 1 ]; then export DSV41_APC=1 DSV41_APC_FILE=$D/$N.switch; echo 1 > $D/$N.switch; EXTRA='--enable-prefix-caching'; else EXTRA=''; fi
echo \$(hostname) \$(date -u +%FT%T) > $D/$N.info
setsid /mnt/tt-data/ssinghal/dsv4-logs/v031/vllm_serve031_tr.sh 16 $L $K \$EXTRA > $D/$N.server.log 2>&1 &
SP=\$!
echo \"server pid \$SP start \$(date +%T)\" > $D/$N.result
OK=0
for i in \$(seq 1 300); do
  sleep 10
  kill -0 \$SP 2>/dev/null || { echo 'server died' >> $D/$N.result; break; }
  if curl -s -m 5 localhost:8000/health >/dev/null 2>&1; then OK=1; echo \"healthy at \$(date +%T)\" >> $D/$N.result; break; fi
done
if [ \$OK = 1 ]; then
  python3 $D/apc_client.py $D/$N.json $* >> $D/$N.client.log 2>&1
  echo \"client rc=\$? \$(date +%T)\" >> $D/$N.result
fi
kill -TERM -\$SP 2>/dev/null; sleep 30; kill -KILL -\$SP 2>/dev/null; sync
echo \"stopped \$(date +%T); resetting board\" >> $D/$N.result
(tt-smi -glx_reset >/dev/null 2>&1; echo \"reset rc=\$?\"; M=/home/ttuser/ssinghal/tt-metal-cache; sudo umount -l \$M 2>/dev/null; echo 1 | sudo tee /sys/bus/pci/rescan >/dev/null; sleep 8; sudo mount \$M 2>&1 | head -2; echo \"mount: \$(grep -c tt-metal-cache /proc/mounts)\") > $D/$N.reset.log 2>&1
echo done >> $D/$N.result"
