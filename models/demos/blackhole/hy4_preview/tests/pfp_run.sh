#!/bin/bash
# pfp_run.sh <tag> <script under hy4_preview/tests> [K=V env...] : RUN ON the claimed host (ssh there); SLOT=<k> (default 2). Image 13cc5087d5f7 (stock ttnn), worktree WT mounted read-only. TERM-only watchdog.
TAG=$1; SCRIPT=$2; shift 2
SLOT=${SLOT:-2}; HOST=${HOST:-10.82.97.45}
IMAGE=${IMAGE:-13cc5087d5f7}; P=/mnt/tt-data/ssinghal/hy4-logs/program/pool-prefill; C=/home/ttuser/ssinghal/tt-metal-cache; O=$C/pfpool/${HOST}_slot$SLOT${CACHE:+_$CACHE}; TM=/home/container_app_user/tt-metal
WT=${WT:-/mnt/tt-data/ssinghal/wt/hy4_pfpool}
eval "$(/mnt/tt-data/ssinghal/hy4-logs/program/bin/slot_env.sh $SLOT $HOST | grep TT_VISIBLE)"
mkdir -p $O/traces $O/kern $O/gen $O/root $O/logs $P/logs; chmod -R 777 $O
EXTRAE=""; for kv in "$@"; do EXTRAE="$EXTRAE -e $kv"; done
docker rm -f pfp_$TAG >/dev/null 2>&1
timeout -s TERM -k 120 ${TMO:-1800} docker run --name pfp_$TAG --network host --ipc host --user $(id -u):$(id -g) --ulimit memlock=-1 --ulimit core=0 \
  --device /dev/tenstorrent:/dev/tenstorrent --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
  --mount type=bind,src=$WT/models/demos/blackhole/hy4_preview,dst=$TM/models/demos/blackhole/hy4_preview,readonly \
  -v /mnt/tt-data/Hy4-Preview:/ckpt:ro -v /mnt/tt-data/ssinghal/hy4-logs/layer-attn:/gold:ro -v $P:/out \
  -v /etc/passwd:/etc/passwd:ro -v $O/traces:$TM/build/profiler/build_wasm/traces -v $O/kern:/kcache -v $O/gen:$TM/generated -v $O/root:/home/container_app_user/cache_root -v $O/logs:/home/container_app_user/logs \
  -e TT_VISIBLE_DEVICES=$TT_VISIBLE_DEVICES -e TT_METAL_CACHE=/kcache -e PYTHONDONTWRITEBYTECODE=1 -e HOME=/tmp -e USER=$(id -un) -e LOGNAME=$(id -un) -e ARCH_NAME=blackhole -e TT_METAL_INSPECTOR_RPC_SERVER_ADDRESS=127.0.0.1:$((50100+SLOT)) $EXTRAE \
  --entrypoint /bin/bash $IMAGE -c "source \${PYTHON_ENV_DIR}/bin/activate && cd $TM && PYTHONPATH=$TM python -X faulthandler $TM/models/demos/blackhole/hy4_preview/tests/$SCRIPT" > $P/logs/run_$TAG.log 2>&1
echo "RC=$? $(date)" >> $P/logs/run_$TAG.log
docker rm -f pfp_$TAG >/dev/null 2>&1
