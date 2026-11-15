#!/bin/bash
# run_pfi_full.sh <tag> <script> [VAR=val ...]: RUN ON THE HOST (ssh first), whole 4x8 mesh, ALL FOUR slots held by task pf-integrate. Same as integ/run_full2.sh but wt/hy4_pfint, logs in program/pool-prefill/logs,
# the integ MOE_RING_FAST libs (LIBDIR default = integ/libs) and the recommended flag set + HY4_POOL_PF=1 (direct-to-pool prefill). SIGTERM watchdog only (WD seconds).
TAG=$1; SCRIPT=$2; shift 2
IMAGE=${IMAGE:-13cc5087d5f7}
W=${WT:-/mnt/tt-data/ssinghal/wt/hy4_pfint}
P=/mnt/tt-data/ssinghal/hy4-logs/program/pool-prefill
LIBDIR=${LIBDIR:-/mnt/tt-data/ssinghal/hy4-logs/program/integ/libs}
CKID=ccbbd8d38d03f5ab06e516311b2e38005cffd226cb84ad3a832fbad29f3f17f5
C=/home/ttuser/ssinghal/tt-metal-cache; O=$C/pfint_run/$TAG; HC=$C/hy4/$CKID
TM=/home/container_app_user/tt-metal
OPS=ttnn/cpp/ttnn/operations/experimental
LM="--mount type=bind,src=$LIBDIR/_ttnncpp.so,dst=$TM/build_Release/lib/_ttnncpp.so,readonly --mount type=bind,src=$LIBDIR/_ttnn.so,dst=$TM/ttnn/ttnn/_ttnn.so,readonly"
KM="--mount type=bind,src=$W/$OPS/deepseek_prefill/unified_routed_expert_ffn/device/kernels,dst=$TM/$OPS/deepseek_prefill/unified_routed_expert_ffn/device/kernels,readonly --mount type=bind,src=$W/$OPS/ccl/moe_compute/device/kernels,dst=$TM/$OPS/ccl/moe_compute/device/kernels,readonly"
mkdir -p $O/traces $O/kern $O/gen $O/root $O/logs /mnt/tt-data/ssinghal/hy4-logs/program/pf-integrate/logs; chmod -R 777 $O
EXTRAE=""; for kv in "$@"; do EXTRAE="$EXTRAE -e $kv"; done
exec 9>/tmp/dsv4_dev.lock; flock -w 600 9 || { echo NOLOCK; exit 1; }
docker rm -f pfi_$TAG >/dev/null 2>&1
docker run --name pfi_$TAG --network host --ipc host --user $(id -u):$(id -g) --ulimit memlock=-1 --ulimit core=0 \
  --device /dev/tenstorrent:/dev/tenstorrent --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
  --mount type=bind,src=$W/models/demos/blackhole/hy4_preview,dst=$TM/models/demos/blackhole/hy4_preview,readonly \
  $LM $KM -v $P:/out -v /mnt/tt-data/Hy4-Preview:/ckpt:ro -v /mnt/tt-data/ssinghal/hy4-logs/ref_dumps/dump_v1:/dumps:ro -v /mnt/tt-data/ssinghal/hy4-logs/program/chain:/chain:ro -v $HC:/hy4cache \
  -v /etc/passwd:/etc/passwd:ro -v $O/traces:$TM/build/profiler/build_wasm/traces -v $O/kern:/kcache -v $O/gen:$TM/generated -v $O/root:/home/container_app_user/cache_root -v $O/logs:/home/container_app_user/logs \
  -e TT_METAL_CACHE=/kcache -e PYTHONDONTWRITEBYTECODE=1 -e HOME=/tmp -e USER=$(id -un) -e LOGNAME=$(id -un) -e ARCH_NAME=blackhole \
  -e HY4_WEIGHT_CACHE=/hy4cache/moe_bfp4 -e HY4_CKPT=/ckpt -e HY4_POOL=1 -e HY4_POOL_PF=${POOL_PF:-1} -e HY4_KV_FMT=scaled_fp8 -e HY4_OPT=all -e HY4_IHC_FUSED=1 -e HY4_FUSE=sparse,xchg -e HY4_XM_NSPLIT=8 \
  -e HY4_SKINNY=shared,router,oproj,hproj -e HY4_PF_OPT=all,moeall -e MOE_RING_FAST=1 -e OUT=/out $EXTRAE \
  --entrypoint /bin/bash $IMAGE -c "source \${PYTHON_ENV_DIR}/bin/activate && cd $TM && timeout -s TERM -k 120 ${WD:-3600} python -X faulthandler $TM/models/demos/blackhole/hy4_preview/tests/$SCRIPT" > /mnt/tt-data/ssinghal/hy4-logs/program/pf-integrate/logs/run_$TAG.log 2>&1
echo "RC=$? $(date)" >> /mnt/tt-data/ssinghal/hy4-logs/program/pf-integrate/logs/run_$TAG.log
docker rm -f pfi_$TAG >/dev/null 2>&1
