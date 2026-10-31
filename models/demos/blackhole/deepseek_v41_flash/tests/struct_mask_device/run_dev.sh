#!/bin/bash
# ON the host (.34), detached: run_dev.sh <name> <timeout s> <cmd-src-dir> [K=V ...]
# ONE pytest process (tests/test_b16_server_like.py, the interactive adapter harness) inside the rc2 image with the worktree wt/structured_model mounted over the model sources,
# under /tmp/dsv4_dev.lock (the hold is the whole boot + the command files; announced <= 90 min). Command files <cmd-src-dir>/*.py are copied to the run's cmd dir; a `stop` file is written when the
# file named in $LAST_CMD (default: the last one, sorted) printed CMD_DONE, or at timeout.
N=$1; TMO=$2; SRC=$3; shift 3
IMAGE=${IMAGE:-dsv41-flash-local:rc2-95a82b1cadf3-8c3d3d6f-fd7ed0c}
WT=${WT:-/mnt/tt-data/ssinghal/wt/structured_model}
C=/home/ttuser/ssinghal/tt-metal-cache; D=/mnt/tt-data/ssinghal/dsv4-logs/structured_model; CKPT=/mnt/tt-data/ssinghal/deepseek-v41-flash-1017; CACHE=$C/1017; STATE=$C/structured_model/state
MD=$WT/models/demos/blackhole/deepseek_v41_flash
EXTRAE=""; for kv in "$@"; do EXTRAE="$EXTRAE -e $kv"; done
CMDD=$D/runs/cmd_$N; rm -f $D/runs/$N.log; mkdir -p $CMDD; cp $SRC/*.py $CMDD/
LASTF=${LAST_CMD:-$(ls $CMDD | sort | tail -n 1)}
exec 9>/tmp/dsv4_dev.lock; flock -w 43200 9 || { echo "no lock" > $D/runs/$N.log; exit 1; }
A=$(free -g | awk "/Mem:/{print \$7}"); [ "$A" -lt 300 ] && { echo "LOW MEM $A" > $D/runs/$N.log; exit 1; }
grep -q tt-metal-cache /proc/mounts || { echo "cache drive not mounted" > $D/runs/$N.log; exit 78; }
echo "HOST $(hostname) START $(date) $* head=$(git -C $WT rev-parse --short=11 HEAD)" > $D/runs/$N.info
{ echo "== pre-run $(date)"; ls /dev/tenstorrent | wc -l; timeout 90 tt-smi -ls 2>&1 | tail -45; (sudo -n dmesg 2>&1 || dmesg 2>&1) | tail -50; } > $D/runs/${N}_evidence0.txt 2>&1
sync; /mnt/tt-data/ssinghal/dsv4-logs/base/reset_host_nolock.sh > $D/runs/${N}_reset0.log 2>&1
grep -q 'mount: 1' $D/runs/${N}_reset0.log || { echo "NVME NOT MOUNTED" >> $D/runs/$N.info; exit 78; }
mkdir -p $STATE/gen_$N $STATE/traces_$N $STATE/cache_root_$N $STATE/logs_$N $CACHE/kernels; sync
docker rm -f sm_$N >/dev/null 2>&1
( while ! grep -q "CMD_DONE $LASTF" $D/runs/$N.log 2>/dev/null && ! grep -q "^RC=" $D/runs/$N.log 2>/dev/null; do sleep 10; done; touch $CMDD/stop ) &
timeout $TMO docker run --name sm_$N --network host --ipc host --user $(id -u):$(id -g) --ulimit memlock=-1 \
  --device /dev/tenstorrent:/dev/tenstorrent --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
  --mount type=bind,src=$MD,dst=/home/container_app_user/tt-metal/models/demos/blackhole/deepseek_v41_flash,readonly \
  -v $D:/out -v $CKPT:/ckpt:ro -v $CACHE:/cache -v /etc/passwd:/etc/passwd:ro \
  -v $STATE/cache_root_$N:/home/container_app_user/cache_root -v $STATE/logs_$N:/home/container_app_user/logs -v $STATE/gen_$N:/home/container_app_user/tt-metal/generated -v $STATE/traces_$N:/home/container_app_user/tt-metal/build/profiler/build_wasm/traces \
  -e MESH_DEVICE="(4, 8)" -e ARCH_NAME=blackhole -e DSV41_LAYERS=0-39 -e DSV41_ENGRAM_RAM=1 -e DSV41_POOL_DTYPE=fp8 -e DSV41_TRACE_REGION=800000000 \
  -e DSV41_CKPT=/ckpt -e DSV41_WEIGHT_CACHE=/cache/weights -e DSV41_UNIFIED_CACHE=/cache/unified -e TT_METAL_CACHE=/cache/kernels \
  -e DSV41_BUILD_SLOTS=0 -e DSV41_VISION=1 -e DSV41_VLLM_CHUNK=512 -e DSV41_SPEC_MAX_CTX=262144 -e DSV41_PF_PAGED=read \
  -e DSV41_VLLM_SPEC_K=5 -e DSV41_VLLM_STATS_EVERY=16000 -e DSV41_STRUCTURED_SPEC=1 \
  -e ISO_B=16 -e ISO_CTX=262144 -e ISO_IDS=/out/ids.json -e ISO_CMD_DIR=/out/runs/cmd_$N \
  -e HOME=/tmp -e USER=$(id -un) -e LOGNAME=$(id -un) $EXTRAE \
  --entrypoint /bin/bash $IMAGE -c 'source ${PYTHON_ENV_DIR}/bin/activate && cd /home/container_app_user/tt-metal && python -X faulthandler -m pytest -x -s -q models/demos/blackhole/deepseek_v41_flash/tests/test_b16_server_like.py' > $D/runs/$N.log 2>&1
RC=$?; echo "RC=$RC $(date)" >> $D/runs/$N.log
touch $CMDD/stop
{ echo "== post-run rc=$RC $(date)"; ls /dev/tenstorrent | wc -l; timeout 90 tt-smi -ls 2>&1 | tail -45; (sudo -n dmesg 2>&1 || dmesg 2>&1) | tail -50; } > $D/runs/${N}_evidence1.txt 2>&1
docker rm -f sm_$N >/dev/null 2>&1
sync; /mnt/tt-data/ssinghal/dsv4-logs/base/reset_host_nolock.sh > $D/runs/${N}_reset1.log 2>&1
echo "DONE $(date) $(grep -c 'mount: 1' $D/runs/${N}_reset1.log)" >> $D/runs/$N.info
