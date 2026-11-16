#!/bin/bash
# b22_run.sh <tag> <module under hy4_preview.probes.b22> [K=V env...] : RUN ON .44 under the b22 claim; slot via TT_VISIBLE_DEVICES env. TERM-only watchdog (docker stop on timeout is the caller's job).
TAG=$1; MOD=$2; shift 2
IMAGE=${IMAGE:-13cc5087d5f7}; P=/mnt/tt-data/ssinghal/hy4-logs/program/b22/logs; C=/home/ttuser/ssinghal/combo_cache; O=$C/b22_run/$TAG; TM=/home/container_app_user/tt-metal
WT=${WT:-/mnt/tt-data/ssinghal/wt/hy4_b22}
mkdir -p $O/traces $C/b22_kern $O/gen $O/root $O/logs $P; chmod -R 777 $O
EXTRAE=""; for kv in "$@"; do EXTRAE="$EXTRAE -e $kv"; done
docker rm -f b22_$TAG >/dev/null 2>&1
timeout -s TERM ${TMO:-1500} docker run --name b22_$TAG --network host --ipc host --user $(id -u):$(id -g) --ulimit memlock=-1 --ulimit core=0 \
  --device /dev/tenstorrent:/dev/tenstorrent --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
  --mount type=bind,src=$WT/models/demos/blackhole/hy4_preview,dst=$TM/models/demos/blackhole/hy4_preview,readonly \
  -v /etc/passwd:/etc/passwd:ro -v $O/traces:$TM/build/profiler/build_wasm/traces -v $C/b22_kern:/kcache -v $O/gen:$TM/generated -v $O/root:/home/container_app_user/cache_root -v $O/logs:/home/container_app_user/logs \
  -e TT_METAL_CACHE=/kcache -e PYTHONDONTWRITEBYTECODE=1 -e HOME=/tmp -e USER=$(id -un) -e LOGNAME=$(id -un) -e ARCH_NAME=blackhole -e HY4_WEIGHT_CACHE=0 -e TT_VISIBLE_DEVICES=${TT_VISIBLE_DEVICES:?} $EXTRAE \
  --entrypoint /bin/bash $IMAGE -c "source \${PYTHON_ENV_DIR}/bin/activate && cd $TM && python -X faulthandler -m models.demos.blackhole.hy4_preview.probes.b22.$MOD" > $P/run_$TAG.log 2>&1
echo "RC=$? $(date)" >> $P/run_$TAG.log
docker rm -f b22_$TAG >/dev/null 2>&1
