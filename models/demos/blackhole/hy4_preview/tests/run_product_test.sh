#!/bin/bash
# run_product_test.sh <tag> <script.py> [VAR=val ...]: RUN ON THE HOST (whole 4x8 mesh, ALL FOUR slots held by task product-integ). Image 13cc5087d5f7, worktree hy4_preview mounted over the image tree, rebuilt libs
# (integ/libs: MOE_RING_FAST) + JIT kernel dirs of the worktree, product flags. TERM-only watchdog (WD seconds). Log: program/product-integ/logs/run_<tag>.log
TAG=$1; SCRIPT=$2; shift; shift
IMAGE=${IMAGE:-13cc5087d5f7}
W=${W:-/mnt/tt-data/ssinghal/wt/hy4_product}
PR=/mnt/tt-data/ssinghal/hy4-logs/program; P=$PR/product-integ
CKID=ccbbd8d38d03f5ab06e516311b2e38005cffd226cb84ad3a832fbad29f3f17f5
C=/home/ttuser/ssinghal/tt-metal-cache; O=$C/hy4_product_run/test_$TAG; HC=$C/hy4/$CKID
TM=/home/container_app_user/tt-metal; OPS=ttnn/cpp/ttnn/operations/experimental
LIBDIR=${LIBDIR-$PR/integ/libs}
FLAGS_DEFAULT="HY4_POOL=1 HY4_POOL_PF=1 HY4_KV_FMT=scaled_fp8 HY4_OPT=all HY4_IHC_FUSED=1 HY4_FUSE=sparse,xchg HY4_XM_NSPLIT=8 HY4_SKINNY=shared,router,oproj,hproj HY4_PF_OPT=all,moeall MOE_RING_FAST=1"
FLAGS=${FLAGS-$FLAGS_DEFAULT}
LM=""; [ -n "$LIBDIR" ] && LM="--mount type=bind,src=$LIBDIR/_ttnncpp.so,dst=$TM/build_Release/lib/_ttnncpp.so,readonly --mount type=bind,src=$LIBDIR/_ttnn.so,dst=$TM/ttnn/ttnn/_ttnn.so,readonly --mount type=bind,src=$W/$OPS/deepseek_prefill/unified_routed_expert_ffn/device/kernels,dst=$TM/$OPS/deepseek_prefill/unified_routed_expert_ffn/device/kernels,readonly --mount type=bind,src=$W/$OPS/ccl/moe_compute/device/kernels,dst=$TM/$OPS/ccl/moe_compute/device/kernels,readonly"
mkdir -p $O/traces $O/kern $O/gen $O/root $O/logs $P/logs; chmod -R 777 $O
EXTRAE=""; for kv in $FLAGS "$@"; do EXTRAE="$EXTRAE -e $kv"; done
exec 9>/tmp/dsv4_dev.lock; flock -w 600 9 || { echo NOLOCK; exit 1; }
docker rm -f hy4prodt_$TAG >/dev/null 2>&1
docker run --name hy4prodt_$TAG --network host --ipc host --user $(id -u):$(id -g) --ulimit memlock=-1 --ulimit core=0 \
  --device /dev/tenstorrent:/dev/tenstorrent --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
  --mount type=bind,src=$W/models/demos/blackhole/hy4_preview,dst=$TM/models/demos/blackhole/hy4_preview,readonly \
  $LM -v $P:/out -v /mnt/tt-data/Hy4-Preview:/ckpt:ro -v $HC:/hy4cache \
  -v /etc/passwd:/etc/passwd:ro -v $O/traces:$TM/build/profiler/build_wasm/traces -v $O/kern:/kcache -v $O/gen:$TM/generated -v $O/root:/home/container_app_user/cache_root -v $O/logs:/home/container_app_user/logs \
  -e TT_METAL_CACHE=/kcache -e PYTHONDONTWRITEBYTECODE=1 -e HOME=/tmp -e USER=$(id -un) -e LOGNAME=$(id -un) -e ARCH_NAME=blackhole \
  -e HY4_WEIGHT_CACHE=/hy4cache/moe_bfp4 -e HY4_CKPT=/ckpt -e OUT=/out -e TT_METAL_INSPECTOR_RPC=1 $EXTRAE \
  --entrypoint /bin/bash $IMAGE -c "source \${PYTHON_ENV_DIR}/bin/activate && cd $TM && timeout -s TERM ${WD:-3600} python -X faulthandler $TM/models/demos/blackhole/hy4_preview/tests/$SCRIPT" > $P/logs/run_$TAG.log 2>&1
echo "RC=$? $(date)" >> $P/logs/run_$TAG.log
docker rm -f hy4prodt_$TAG >/dev/null 2>&1
