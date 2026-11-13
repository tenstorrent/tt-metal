#!/bin/bash
# run_attn.sh <tag> [VAR=val ...]: ON host .45 (claim layer-attn first). DSV4 image 13cc5087d5f7 (stock ttnn), worktree hy4_preview + deepseek tt dir mounted read-only. 4x8 mesh.
SCRIPT=${SCRIPT:-run_attn_device.py}
TAG=$1; shift
IMAGE=13cc5087d5f7
W=/mnt/tt-data/ssinghal/wt/hy4_lattn
P=/mnt/tt-data/ssinghal/hy4-logs/program/layer-attn
C=/home/ttuser/ssinghal/tt-metal-cache; O=$C/layer_attn/$TAG
TM=/home/container_app_user/tt-metal
mkdir -p $O/traces $O/kern $O/gen $O/root $O/logs; chmod -R 777 $O
EXTRAE=""; for kv in "$@"; do EXTRAE="$EXTRAE -e $kv"; done
exec 9>/tmp/dsv4_dev.lock; flock -w 600 9 || { echo NOLOCK; exit 1; }
docker rm -f lattn_$TAG >/dev/null 2>&1
timeout ${TMO:-3000} docker run --name lattn_$TAG --network host --ipc host --user $(id -u):$(id -g) --ulimit memlock=-1 --ulimit core=0 \
  --device /dev/tenstorrent:/dev/tenstorrent --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
  --mount type=bind,src=$W/models/demos/blackhole/hy4_preview,dst=$TM/models/demos/blackhole/hy4_preview,readonly \
  --mount type=bind,src=$W/models/demos/blackhole/deepseek_v41_flash/tt,dst=$TM/models/demos/blackhole/deepseek_v41_flash/tt,readonly \
  -v /mnt/tt-data/Hy4-Preview:/ckpt:ro -v /mnt/tt-data/ssinghal/hy4-logs/layer-attn:/gold:ro \
  -v /etc/passwd:/etc/passwd:ro -v $O/traces:$TM/build/profiler/build_wasm/traces -v $O/kern:/kcache -v $O/gen:$TM/generated -v $O/root:/home/container_app_user/cache_root -v $O/logs:/home/container_app_user/logs \
  -e TT_METAL_CACHE=/kcache -e PYTHONDONTWRITEBYTECODE=1 -e HOME=/tmp -e USER=$(id -un) -e LOGNAME=$(id -un) -e ARCH_NAME=blackhole $EXTRAE \
  --entrypoint /bin/bash $IMAGE -c "source \${PYTHON_ENV_DIR}/bin/activate && cd $TM && python -X faulthandler $TM/models/demos/blackhole/hy4_preview/tests/$SCRIPT" > $P/run_$TAG.log 2>&1
echo "RC=$? $(date)" >> $P/run_$TAG.log
docker rm -f lattn_$TAG >/dev/null 2>&1
