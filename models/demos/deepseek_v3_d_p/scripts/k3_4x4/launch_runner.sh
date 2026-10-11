#!/bin/bash
# K3 prefill through the real runner and producer on one galaxy: two 4x4 stages (default), one 4x4 stage,
# or one 8x4 stage, traced, with KV-cache PCC against the golden.
# usage: launch_runner.sh <fabric_mode 2d|2d_torus_x|2d_torus_y|2d_torus_xy> <chunk_size> <outdir> [use_trace 0|1]
# env:
#   NRANKS            2 (two 4x4 stages) or 1 (one stage)
#   NUM_LAYERS        24, or 12 for one 4x4 stage
#   SEQ               request length, default 56320 (the golden length)
#   SP, TP            mesh shape, default 4 x 4; use SP=8 for one 8x4 stage
#   CACHE_ROOT        runner weight cache (default $HOME/k3_cache_4x4)
#   BINDING_OVERRIDE  rank binding to use instead of the 2-rank 4x4 binding (e.g. binding_1rank_top_2d.yaml)
#   MGD_OVERRIDE      mesh graph descriptor override for the 2-rank binding
#   PREFILL_KV_PERIOD_TOKENS  block-cyclic KV period (default 5120); set it equal to the chunk size to run
#                     other chunk sizes (see utils/chunk_config.py)
#   PRODUCER_EXTRA    extra shell exports for the producer, e.g. "export PREFILL_PRODUCER_CHUNKS=2;"
#   WARMUP            producer warm-up chunks (2)
#   PREFILL_TCP_INTERFACE  MPI interface (ens5f0np0)
# Prints the producer/runner exit codes and per-rank KV PCC; timing CSVs land in <outdir>/timing
# (summarize with cadence.py).
set -uo pipefail
MODE=$1; CHUNK=$2; OUT=$3; TRACE=${4:-1}
NRANKS=${NRANKS:-2}; NUM_LAYERS=${NUM_LAYERS:-24}; SEQ=${SEQ:-56320}
IFACE=${PREFILL_TCP_INTERFACE:-ens5f0np0}
KV_PERIOD=${PREFILL_KV_PERIOD_TOKENS:-}
REPO=$(cd "$(dirname "$0")/../../../../.." && pwd)
cd "$REPO"; source python_env/bin/activate
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
unset TT_METAL_CCACHE_KERNEL_SUPPORT  # build.cpp passes ccache settings inline; needs ccache >= 4.8
mkdir -p $OUT/ranklogs $OUT/timing $OUT/pcc
H=$(hostname -s)
BIND=models/demos/common/prefill/runners/topology_configuration/pipeline_prefill_request_intragalaxy_2rank_4x4_torus_x.$H.yaml
[ -f "$BIND" ] || BIND=models/demos/common/prefill/runners/topology_configuration/pipeline_prefill_request_intragalaxy_2rank_4x4_torus_x.yaml
MGD_DIR=models/demos/deepseek_v3_d_p/experimental_descriptors
case $MODE in
  2d_torus_y) MGD=$MGD_DIR/single_bh_galaxy_2x4x4_z_torus_y_graph_descriptor.textproto ;;
  *)          MGD=$MGD_DIR/single_bh_galaxy_2x4x4_z_torus_x_graph_descriptor.textproto ;;
esac
[ -n "${MGD_OVERRIDE:-}" ] && MGD=$MGD_OVERRIDE
B=$OUT/binding.yaml
if [ -n "${BINDING_OVERRIDE:-}" ]; then cp "$BINDING_OVERRIDE" $B; else
  sed -e "s|^mesh_graph_desc_path:.*|mesh_graph_desc_path: $MGD|" -e "s|PREFILL_FABRIC_MODE: .*|PREFILL_FABRIC_MODE: \"$MODE\"|" $BIND > $B; fi
# The runner ranks get their environment from the binding's global_env.
[ -n "$KV_PERIOD" ] && sed -i "s|^global_env:|global_env:\n  PREFILL_KV_PERIOD_TOKENS: \"$KV_PERIOD\"|" $B
export PREFILL_MANIFEST=$PWD/models/demos/deepseek_v3_d_p/tt/runners/manifests/kimi_k3.json
export PREFILL_HF_MODEL=${PREFILL_HF_MODEL:-/mnt/weka/model-weights/llm/moonshotai/Kimi-K3-dequantized}
export PREFILL_TTNN_CACHE=${CACHE_ROOT:-$HOME/k3_cache_4x4}
export PREFILL_NUM_LAYERS=$NUM_LAYERS PREFILL_MAX_SEQ_LEN=$SEQ PREFILL_NUM_USERS=1
export PREFILL_SP=${SP:-4} PREFILL_TP=${TP:-4} PREFILL_FABRIC_MODE=$MODE
export PREFILL_CHUNK_SIZE=$CHUNK PREFILL_USE_TRACE=$TRACE PREFILL_LAYER_ACK_D2H=1
export PREFILL_SYNC_PER_CHUNK=${PREFILL_SYNC_PER_CHUNK:-1} PREFILL_TIMING_DIR=$OUT/timing
export PREFILL_ENABLE_MIGRATION=1 PREFILL_MOCK_MIGRATION=1 PREFILL_MIGRATION_TABLE_PATH=$OUT/kv_chunk_table.pb
export PREFILL_MIGRATION_DEVICE_MAP_PATH=$OUT/device_map.json
export TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0 PYTHONUNBUFFERED=1 LOGURU_LEVEL=INFO
export PP_TT_METAL_CACHE=/tmp/tt-metal-cache-pp
setsid ./models/demos/common/prefill/runners/run_pipeline_prefill.sh $B $H:$NRANKS $IFACE > $OUT/runner.log 2>&1 &
RPID=$!
echo "runner pid $RPID"
for _ in $(seq 1 1440); do
  [ -f $OUT/kv_chunk_table.pb ] && break
  kill -0 $RPID 2>/dev/null || { echo "RUNNER EXITED before publishing KV table"; exit 1; }
  sleep 5
done
[ -f $OUT/kv_chunk_table.pb ] || { echo "KV table not published in 2h"; kill -- -$RPID; exit 1; }
echo "KV table published, starting producer"
REAL_CHUNKS=$(( (SEQ + CHUNK - 1) / CHUNK ))
PERIOD_EXPORT=""; [ -n "$KV_PERIOD" ] && PERIOD_EXPORT="export PREFILL_KV_PERIOD_TOKENS=$KV_PERIOD;"
mpirun -np $NRANKS --host $H:$NRANKS --map-by slot --bind-to none --tag-output --allow-run-as-root \
  --output-filename $OUT/ranklogs/producer --mca btl self,tcp --mca btl_tcp_if_include $IFACE \
  -x PATH -x LD_LIBRARY_PATH \
  bash -lc "cd $PWD; export PYTHONPATH=$PWD TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0 PYTHONUNBUFFERED=1; \
    export PREFILL_CHUNK_SIZE=$CHUNK PREFILL_MAX_SEQ_LEN=$SEQ PREFILL_NUM_USERS=1 PREFILL_NUM_LAYERS=$NUM_LAYERS; \
    export PREFILL_PRODUCER_CHUNKS=$REAL_CHUNKS PREFILL_PRODUCER_WARMUP_CHUNKS=${WARMUP:-2}; \
    export PREFILL_MIGRATION_TABLE_PATH=$OUT/kv_chunk_table.pb PREFILL_MIGRATION_DEVICE_MAP_PATH=$OUT/device_map.json; \
    export PREFILL_PCC_SUMMARY_DIR=$OUT/pcc PREFILL_PRODUCER_CHECK_PCC=1 PREFILL_SEND_SHUTDOWN=1; \
    export PREFILL_STANDALONE_CHUNKED_PCC=0.85 PREFILL_H2D_CONNECT_TIMEOUT=120; \
    export PREFILL_PRODUCER_MANIFEST=$PREFILL_MANIFEST PREFILL_SP=${SP:-4} PREFILL_TP=${TP:-4}; \
    export PREFILL_TRACE_DIR=/mnt/weka/model-cache/stable/deepseek-prefill-cache/golden/k3_vllm_code_debug_1M PREFILL_PCC_GOLDEN_LEN=$SEQ; \
    $PERIOD_EXPORT ${PRODUCER_EXTRA:-} export LOGURU_LEVEL=INFO; exec python3 -m models.demos.common.prefill.runners.prefill_producer" > $OUT/producer.log 2>&1
PROD_RC=$?
echo "producer rc=$PROD_RC"
timeout 180 tail --pid=$RPID -f /dev/null || { echo "runner did not shut down"; kill -- -$RPID; }
wait $RPID; RRC=$?
echo "runner rc=$RRC"
for f in $OUT/pcc/rank*.json; do echo "$(basename $f): $(cat $f)"; done
exit $(( PROD_RC || RRC ))
