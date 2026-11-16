#!/bin/bash
# run_full.sh <tag> [script=run_pool_decode.py] [VAR=val ...]: RUN ON THE HOST (ssh first), whole 4x8 mesh, ALL FOUR slots held by task integ. Image 13cc5087d5f7 (stock ttnn, JIT kernels only unless MCK_SO set);
# wt/hy4_integ hy4_preview mounted over the image tree. SIGTERM watchdog only (WD seconds). HOST decides the cache dir (the hy4 bfp4 weight cache must be on this host's cache NVMe).
TAG=$1; shift; SCRIPT=run_pool_decode.py; case "$1" in *.py) SCRIPT=$1; shift;; esac
IMAGE=${IMAGE:-13cc5087d5f7}
W=${WT:-/mnt/tt-data/ssinghal/wt/hy4_b10}
PI=/mnt/tt-data/ssinghal/hy4-logs/program/integ; P=/mnt/tt-data/ssinghal/hy4-logs/program/b10/logs
CKID=ccbbd8d38d03f5ab06e516311b2e38005cffd226cb84ad3a832fbad29f3f17f5
C=/home/ttuser/ssinghal/combo_cache; O=$C/b10_run/$TAG; HC=$C/hy4cache; TC=/mnt/tt-data/ssinghal/hy4-logs/program/combo/tp32v1
TM=/home/container_app_user/tt-metal
OPS=ttnn/cpp/ttnn/operations/experimental
# LIBDIR=<dir with _ttnncpp.so + _ttnn.so> mounts rebuilt libs (MOE_RING_FAST needs them); KOVER=1 mounts the JIT kernel dirs of unified_routed_expert_ffn + moe_compute from the integ worktree
LM=""; [ -n "$LIBDIR" ] && LM="--mount type=bind,src=$LIBDIR/_ttnncpp.so,dst=$TM/build_Release/lib/_ttnncpp.so,readonly --mount type=bind,src=$LIBDIR/_ttnn.so,dst=$TM/ttnn/ttnn/_ttnn.so,readonly"
KM=""; [ "${KOVER:-0}" = "1" ] && KM="--mount type=bind,src=$W/$OPS/deepseek_prefill/unified_routed_expert_ffn/device/kernels,dst=$TM/$OPS/deepseek_prefill/unified_routed_expert_ffn/device/kernels,readonly --mount type=bind,src=$W/$OPS/ccl/moe_compute/device/kernels,dst=$TM/$OPS/ccl/moe_compute/device/kernels,readonly"
mkdir -p $C/b10_kern $HC $O/traces $O/gen $O/root $O/logs /mnt/tt-data/ssinghal/hy4-logs/program/b10/logs; chmod -R 777 $O
EXTRAE=""; for kv in "$@"; do EXTRAE="$EXTRAE -e $kv"; done
exec 9>/tmp/dsv4_dev.lock; flock -w 60 9 || { echo NOLOCK; exit 1; }
docker rm -f b10_$TAG >/dev/null 2>&1
docker run --name b10_$TAG --network host --ipc host --user $(id -u):$(id -g) --ulimit memlock=-1 --ulimit core=0 \
  --device /dev/tenstorrent:/dev/tenstorrent --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
  --mount type=bind,src=$W/models/demos/blackhole/hy4_preview,dst=$TM/models/demos/blackhole/hy4_preview,readonly \
  $LM $KM -v $PI:/in:ro -v $P:/out -v $P:/rout -v /mnt/tt-data/ssinghal/hy4-logs/program/layer-moe/acts:/mnt/tt-data/ssinghal/hy4-logs/program/layer-moe/acts:ro -v /mnt/tt-data/Hy4-Preview:/ckpt:ro -v /mnt/tt-data/ssinghal/hy4-logs/ref_dumps/dump_v1:/dumps:ro -v /mnt/tt-data/ssinghal/hy4-logs/program/chain:/chain:ro -v $HC:/hy4cache -v $TC:/tp32v1:ro -e HY4_TP32_CACHE_DIR=/tp32v1 \
  -v /etc/passwd:/etc/passwd:ro -v $O/traces:$TM/build/profiler/build_wasm/traces -v $C/b10_kern:/kcache -v $O/gen:$TM/generated -v $O/root:/home/container_app_user/cache_root -v $O/logs:/home/container_app_user/logs \
  -e TT_METAL_CACHE=/kcache -e PYTHONDONTWRITEBYTECODE=1 -e HOME=/tmp -e USER=$(id -un) -e LOGNAME=$(id -un) -e ARCH_NAME=blackhole \
  -e HY4_WEIGHT_CACHE=/hy4cache/moe_bfp4 -e HY4_CKPT=/ckpt -e PROMPTS=/in/decode_prompts.pt -e HY4_POOL=1 -e HY4_KV_FMT=scaled_fp8 -e OUT=/out $EXTRAE \
  --entrypoint /bin/bash $IMAGE -c "source \${PYTHON_ENV_DIR}/bin/activate && cd $TM && timeout -s TERM ${WD:-3600} python -X faulthandler $TM/models/demos/blackhole/hy4_preview/tests/$SCRIPT" > /mnt/tt-data/ssinghal/hy4-logs/program/b10/logs/run_$TAG.log 2>&1
echo "RC=$? $(date)" >> /mnt/tt-data/ssinghal/hy4-logs/program/b10/logs/run_$TAG.log
docker rm -f b10_$TAG >/dev/null 2>&1
