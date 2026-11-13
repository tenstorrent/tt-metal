#!/bin/bash
# run_probe.sh <tag> [CASES=...] : ON host .45. DSV4 image 13cc5087d5f7 (stock ttnn, no overlay), probe script mounted read-only. one chip (1x1 mesh).
SCRIPT=${SCRIPT:-probe_sparse_sdpa.py}
TAG=$1; shift
IMAGE=13cc5087d5f7
P=/mnt/tt-data/ssinghal/hy4-logs/program/attn-op-probe
C=/home/ttuser/ssinghal/tt-metal-cache; O=$C/attn_op_probe/$TAG
TM=/home/container_app_user/tt-metal
mkdir -p $O/traces $O/kern $O/gen $O/root $O/logs; chmod -R 777 $O
EXTRAE=""; for kv in "$@"; do EXTRAE="$EXTRAE -e $kv"; done
exec 9>/tmp/dsv4_dev.lock; flock -w 600 9 || { echo NOLOCK; exit 1; }
docker rm -f probe_$TAG >/dev/null 2>&1
timeout ${TMO:-2400} docker run --name probe_$TAG --network host --ipc host --user $(id -u):$(id -g) --ulimit memlock=-1 --ulimit core=0 \
  --device /dev/tenstorrent:/dev/tenstorrent --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
  -v $P:/probe:ro --mount type=bind,src=/mnt/tt-data/ssinghal/wt/hy4_attn/models/demos/blackhole/deepseek_v41_flash/tt,dst=$TM/models/demos/blackhole/deepseek_v41_flash/tt,readonly -v /etc/passwd:/etc/passwd:ro -v $O/traces:$TM/build/profiler/build_wasm/traces -v $O/kern:/kcache -v $O/gen:$TM/generated -v $O/root:/home/container_app_user/cache_root -v $O/logs:/home/container_app_user/logs \
  -e TT_METAL_CACHE=/kcache -e PYTHONDONTWRITEBYTECODE=1 -e HOME=/tmp -e USER=$(id -un) -e LOGNAME=$(id -un) -e ARCH_NAME=blackhole -e SCRIPT=$SCRIPT $EXTRAE \
  --entrypoint /bin/bash $IMAGE -c "source \${PYTHON_ENV_DIR}/bin/activate && cd $TM && python -X faulthandler /probe/$SCRIPT" > $P/probe_$TAG.log 2>&1
echo "RC=$? $(date)" >> $P/probe_$TAG.log
docker rm -f probe_$TAG >/dev/null 2>&1
