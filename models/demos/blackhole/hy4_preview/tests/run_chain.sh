#!/bin/bash
# run_chain.sh <tag> [VAR=val ...]: ON host .45 under an all-slot claim (chain). DSV4 image 13cc5087d5f7, hy4_chain worktree mounted read-only, whole 4x8 mesh.
SCRIPT=${SCRIPT:-run_chain.py}
TAG=$1; shift
IMAGE=13cc5087d5f7
W=/mnt/tt-data/ssinghal/wt/hy4_chain
P=/mnt/tt-data/ssinghal/hy4-logs/program/chain
CKID=ccbbd8d38d03f5ab06e516311b2e38005cffd226cb84ad3a832fbad29f3f17f5
C=/home/ttuser/ssinghal/tt-metal-cache; O=$C/hy4_chain_run/$TAG; HC=$C/hy4/$CKID
TM=/home/container_app_user/tt-metal
mkdir -p $O/traces $O/kern $O/gen $O/root $O/logs $HC $P; chmod -R 777 $O $HC
EXTRAE=""; for kv in "$@"; do EXTRAE="$EXTRAE -e $kv"; done
exec 9>/tmp/dsv4_dev.lock; flock -w 600 9 || { echo NOLOCK; exit 1; }
docker rm -f chain_$TAG >/dev/null 2>&1
timeout ${TMO:-14000} docker run --name chain_$TAG --network host --ipc host --user $(id -u):$(id -g) --ulimit memlock=-1 --ulimit core=0 \
  --device /dev/tenstorrent:/dev/tenstorrent --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
  --mount type=bind,src=$W/models/demos/blackhole/hy4_preview,dst=$TM/models/demos/blackhole/hy4_preview,readonly \
  --mount type=bind,src=$W/models/demos/blackhole/deepseek_v41_flash/tt,dst=$TM/models/demos/blackhole/deepseek_v41_flash/tt,readonly \
  -v $P:/out -v /mnt/tt-data/Hy4-Preview:/ckpt:ro -v /mnt/tt-data/ssinghal/hy4-logs/ref_dumps/dump_v1:/dumps:ro -v $HC:/hy4cache \
  -v /etc/passwd:/etc/passwd:ro -v $O/traces:$TM/build/profiler/build_wasm/traces -v $O/kern:/kcache -v $O/gen:$TM/generated -v $O/root:/home/container_app_user/cache_root -v $O/logs:/home/container_app_user/logs \
  -e TT_METAL_CACHE=/kcache -e PYTHONDONTWRITEBYTECODE=1 -e HOME=/tmp -e USER=$(id -un) -e LOGNAME=$(id -un) -e ARCH_NAME=blackhole \
  -e HY4_WEIGHT_CACHE=/hy4cache/moe_bfp4 -e HY4_CKPT=/ckpt $EXTRAE \
  --entrypoint /bin/bash $IMAGE -c "source \${PYTHON_ENV_DIR}/bin/activate && cd $TM && python -X faulthandler $TM/models/demos/blackhole/hy4_preview/tests/$SCRIPT" > $P/run_$TAG.log 2>&1
echo "RC=$? $(date)" >> $P/run_$TAG.log
docker rm -f chain_$TAG >/dev/null 2>&1
