#!/bin/bash
# structured-outputs hold: server of image rc2 with the structured plugin/model worktrees mounted; run under flock /tmp/dsv4_dev.lock
# boots, stays up until srv/<name>/STOP or HOLD_MAX s (default 5400) after ready, then logs, stop, board reset (nolock variant).
N=$1
IMAGE=${IMAGE:-dsv41-flash-local:rc2-95a82b1cadf3-8c3d3d6f-fd7ed0c}
C=/home/ttuser/ssinghal/tt-metal-cache; D=/mnt/tt-data/ssinghal/dsv4-logs/structured_m2b/srv/$N; CKPT=/mnt/tt-data/ssinghal/deepseek-v41-flash-1017; CACHE=$C/1017; STATE=$C/structured/state
mkdir -p $D
rm -f $D/STOP
A=$(free -g | awk "/Mem:/{print \$7}"); echo "mem avail $A GB" > $D/mem; [ "$A" -lt 300 ] && { echo "LOW MEM $A" >> $D/info; exit 1; }
echo "HOST $(hostname) IMAGE $IMAGE START $(date)" > $D/info
sync
/mnt/tt-data/ssinghal/dsv4-logs/base/reset_host_nolock.sh > $D/reset0.log 2>&1
sleep 45
grep -q 'mount: 1' $D/reset0.log || { echo "NVME NOT MOUNTED" >> $D/info; exit 1; }
mkdir -p $STATE/cache_root_$N $STATE/logs_$N $STATE/traces_$N $CACHE/kernels
R=$STATE/hf/hub/models--deepseek-ai--DeepSeek-V4.1-Flash; mkdir -p $R/refs $R/snapshots/local && echo -n local > $R/refs/main
cp $CKPT/config.json $CKPT/tokenizer.json $CKPT/tokenizer_config.json $R/snapshots/local/
sync
docker rm -f st_$N >/dev/null 2>&1
docker run -d --name st_$N --network host --ipc host --user $(id -u):$(id -g) --ulimit memlock=-1 \
  $EXTRA_DOCKER --device /dev/tenstorrent:/dev/tenstorrent --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
  -v $CKPT:/ckpt:ro -v $CACHE:/cache -v /etc/passwd:/etc/passwd:ro -v $STATE/hf:/hf \
  -v $STATE/cache_root_$N:/home/container_app_user/cache_root -v $STATE/logs_$N:/home/container_app_user/logs \
  -v $STATE/traces_$N:/home/container_app_user/tt-metal/build/profiler/build_wasm/traces \
  -e MODEL_WEIGHTS_DIR=/ckpt -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 -e HOME=/tmp -e USER=$(id -un) -e LOGNAME=$(id -un) \
  $IMAGE --model deepseek-ai/DeepSeek-V4.1-Flash --tt-device blackhole_galaxy --no-auth > $D/container_id 2>&1
echo "CONTAINER $(cat $D/container_id)" >> $D/info
( docker logs -f st_$N > $D/server.log 2>&1 & )
T0=$(date +%s); UP=0
while true; do
  curl -sf http://127.0.0.1:8000/health >/dev/null && { UP=1; break; }
  [ "$(docker inspect -f '{{.State.Running}}' st_$N 2>/dev/null)" = true ] || break
  [ $(( $(date +%s) - T0 )) -gt ${BOOT_TIMEOUT:-3000} ] && { echo "BOOT_TIMEOUT" >> $D/info; break; }
  sleep 10
done
if [ $UP = 1 ]; then
  echo "SERVER_READY $(date) boot $(( $(date +%s) - T0 )) s" >> $D/info
  { echo "== free -g at ready"; free -g | head -2; echo "== MEMREPORT"; grep -a "MEMREPORT" $D/server.log | sed -e 's/^.*MEMREPORT/MEMREPORT/' | cut -c1-260; } > $D/boot_metrics.txt 2>&1
  T1=$(date +%s)
  [ -n "$LOAD" ] && { nohup /mnt/tt-data/ssinghal/dsv4-logs/structured_m2b/${RUNLOAD:-run_load.sh} $N $LOAD > $D/load.out 2>&1 < /dev/null & }
  while [ ! -f $D/STOP ] && [ "$(docker inspect -f '{{.State.Running}}' st_$N 2>/dev/null)" = true ] && [ $(( $(date +%s) - T1 )) -lt ${HOLD_MAX:-5400} ]; do sleep 10; done
  [ -f $D/STOP ] && echo "STOP requested $(date)" >> $D/info || echo "STOP/DIED/HOLD_MAX $(date)" >> $D/info
else
  echo "NO_SERVER $(date)" >> $D/info
fi
docker inspect -f 'state={{.State.Status}} exit={{.State.ExitCode}} oom={{.State.OOMKilled}}' st_$N > $D/evidence.txt 2>&1
{ echo '== dmesg tail'; (dmesg 2>/dev/null || sudo -n dmesg 2>&1) | tail -150; echo '== tt-smi -ls'; timeout 90 tt-smi -ls 2>&1 | tail -40; echo '== free'; free -g; echo '== ls /dev/tenstorrent'; ls /dev/tenstorrent | wc -l; } >> $D/evidence.txt 2>&1
docker logs st_$N > $D/server_full.log 2>&1
docker stop -t 30 st_$N >/dev/null 2>&1; docker rm -f st_$N >/dev/null 2>&1
echo "RUN_STOPPED $(date); resetting board" >> $D/info
sync
/mnt/tt-data/ssinghal/dsv4-logs/base/reset_host_nolock.sh > $D/reset.log 2>&1
echo "HOLD_DONE $(date)" >> $D/info
