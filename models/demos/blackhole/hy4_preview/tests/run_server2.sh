#!/bin/bash
# run_server2.sh <tag> [VAR=val ...]  (serve-long): Hy4 vLLM + TT plugin server, INTEG pool model, on a host under an ALL-SLOT claim. ONE container, --network host, image 13cc5087d5f7 (plugin a8962b0),
# worktree python mounted read-only over the image (hy4_preview + deepseek_v41_flash/tt), rebuilt C++ libs (integ/libs, MOE_RING_FAST) + JIT kernel dirs of the worktree. Foreground (holds the device flock);
# stop with `docker stop -t 180 hy4slong_<tag>` (SIGTERM, never kill -9). Env of this script: MAXLEN (default 65536) NSEQ (4) W (worktree) LIBDIR (default integ/libs, "" = stock) FLAGS (default recommended set, see below).
TAG=$1; shift
IMAGE=${IMAGE:-13cc5087d5f7}
W=${W:-/mnt/tt-data/ssinghal/wt/hy4_slong}
PR=/mnt/tt-data/ssinghal/hy4-logs/program
P=$PR/serve-long
CKID=ccbbd8d38d03f5ab06e516311b2e38005cffd226cb84ad3a832fbad29f3f17f5
C=/home/ttuser/ssinghal/tt-metal-cache; O=$C/hy4_slong_run/srv_$TAG; HC=$C/hy4/$CKID
TM=/home/container_app_user/tt-metal
MD=models/demos/blackhole
OPS=ttnn/cpp/ttnn/operations/experimental
MAXLEN=${MAXLEN:-65536}; NSEQ=${NSEQ:-4}
LIBDIR=${LIBDIR-$PR/integ/libs}
FLAGS_DEFAULT="HY4_POOL=1 HY4_KV_FMT=scaled_fp8 HY4_OPT=all HY4_IHC_FUSED=1 HY4_FUSE=sparse,xchg HY4_XM_NSPLIT=8 HY4_SKINNY=shared,router,oproj,hproj HY4_PF_OPT=all,moeall MOE_RING_FAST=1"
FLAGS=${FLAGS-$FLAGS_DEFAULT}
mkdir -p $O/traces $O/gen $O/root $O/logs $P/logs
[ -d $O/kern ] || cp -r $C/integ_run/m1/kern $O/kern
chmod -R 777 $O $HC
EXTRAE=""; for kv in $FLAGS "$@"; do EXTRAE="$EXTRAE -e $kv"; done
LM=""; [ -n "$LIBDIR" ] && LM="--mount type=bind,src=$LIBDIR/_ttnncpp.so,dst=$TM/build_Release/lib/_ttnncpp.so,readonly --mount type=bind,src=$LIBDIR/_ttnn.so,dst=$TM/ttnn/ttnn/_ttnn.so,readonly --mount type=bind,src=$W/$OPS/deepseek_prefill/unified_routed_expert_ffn/device/kernels,dst=$TM/$OPS/deepseek_prefill/unified_routed_expert_ffn/device/kernels,readonly --mount type=bind,src=$W/$OPS/ccl/moe_compute/device/kernels,dst=$TM/$OPS/ccl/moe_compute/device/kernels,readonly"
PF="$MD/hy4_preview/tt/generator_vllm.py $MD/hy4_preview/tt/hy4_serve_pool.py $MD/hy4_preview/tt/hy4_pool.py $MD/hy4_preview/tt/hy4_pool_kernels/slice_copy_masked.cpp $MD/hy4_preview/tt/hy4_pool_model.py $MD/hy4_preview/tt/hy4_chain.py $MD/hy4_preview/vllm_metadata.json $MD/hy4_preview/vllm_config/config.json"
( cd $W && sha256sum $PF; git rev-parse HEAD; git status --short | grep -v '^??' | head ) > $P/mountproof_host_$TAG.txt 2>&1
exec 9>/tmp/dsv4_dev.lock; flock -w 600 9 || { echo NOLOCK; exit 1; }
docker rm -f hy4slong_$TAG >/dev/null 2>&1
VLLM_CMD="python -m vllm.entrypoints.openai.api_server \
  --model /ckpt --served-model-name tencent/Hy4-preview --hf-config-path $TM/$MD/hy4_preview/vllm_config --tokenizer /ckpt \
  --hf-overrides '{\"architectures\":[\"TTHYV4ForCausalLM\"]}' \
  --block-size 128 --max-model-len $MAXLEN --max-num-seqs $NSEQ --max-num-batched-tokens $((MAXLEN*NSEQ)) --seed 9472 --max-log-len 32 \
  --no-enable-prefix-caching --chat-template-content-format openai \
  --additional-config '{\"tt\": {\"fabric_config\": \"FABRIC_1D_RING\", \"l1_small_size\": 16384, \"trace_region_size\": 700000000}}' \
  --enable-auto-tool-choice --tool-call-parser hy_v4 --reasoning-parser hy_v4 --enable-prompt-tokens-details \
  --default-chat-template-kwargs '{\"reasoning_effort\": \"high\"}' --host 0.0.0.0 --port 8000"
docker run --name hy4slong_$TAG --network host --ipc host --user $(id -u):$(id -g) --ulimit memlock=-1 --ulimit core=0 \
  --device /dev/tenstorrent:/dev/tenstorrent --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
  --mount type=bind,src=$W/$MD/hy4_preview,dst=$TM/$MD/hy4_preview,readonly \
  --mount type=bind,src=$W/$MD/deepseek_v41_flash/tt,dst=$TM/$MD/deepseek_v41_flash/tt,readonly \
  $LM -v $P:/out -v /mnt/tt-data/Hy4-Preview:/ckpt:ro -v $HC:/hy4cache \
  -v /etc/passwd:/etc/passwd:ro -v $O/traces:$TM/build/profiler/build_wasm/traces -v $O/kern:/kcache -v $O/gen:$TM/generated -v $O/root:/home/container_app_user/cache_root -v $O/logs:/home/container_app_user/logs \
  -e TT_METAL_CACHE=/kcache -e PYTHONDONTWRITEBYTECODE=1 -e HOME=/tmp -e USER=$(id -un) -e LOGNAME=$(id -un) -e ARCH_NAME=blackhole \
  -e HY4_WEIGHT_CACHE=/hy4cache/moe_bfp4 -e HY4_CKPT=/ckpt -e MESH_DEVICE="(4, 8)" -e EXTRA_MODELS_DIR=$TM/$MD \
  -e VLLM_TARGET_DEVICE=tt -e VLLM_CONFIGURE_LOGGING=1 -e VLLM_RPC_TIMEOUT=7200000 -e TORCHDYNAMO_DISABLE=1 -e HF_HUB_OFFLINE=1 -e HF_HOME=/tmp/hf \
  $EXTRAE \
  --entrypoint /bin/bash $IMAGE -c "source \${PYTHON_ENV_DIR}/bin/activate && cd $TM && (cd $TM && sha256sum $PF > /out/mountproof_container_$TAG.txt) && exec $VLLM_CMD" > $P/logs/server_$TAG.log 2>&1
echo "RC=$? $(date)" >> $P/logs/server_$TAG.log
docker rm -f hy4slong_$TAG >/dev/null 2>&1
