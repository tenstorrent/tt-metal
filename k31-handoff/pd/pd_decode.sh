#!/bin/bash
# K3.1 P->D, DECODE quad (Quad8): Ryan's test_k31_quad_decode_from_migrated on the 16-rank 4x2 quad ring, publishing
# its table for tt-d-gen's KV Manager (file export). Based on /data/sjameel/q8_gen.sh. Run on c09u02.
# Usage: pd_decode.sh NAME   (env from pd.env; log in $OUT/NAME.decode.log)
set -o pipefail
name=${1:?usage: pd_decode.sh NAME}
source /data/bklockiewicz/k31run/pd/pd.env
HOSTS=bh-glx-120-c09u02,bh-glx-120-c09u08,bh-glx-120-c09u14,bh-glx-120-c09u20
T=/data/bklockiewicz/k31-migration-blaze
cd $T || exit 3
source tt-metal/python_env/bin/activate
mkdir -p $OUT; LOG=$OUT/$name.decode.log TRI=$OUT/$name.triage
export TT_METAL_HOME=$T/tt-metal TT_METAL_RUNTIME_ROOT=$T/tt-metal
export TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_ALLOCATOR_MODE_HYBRID=1 TT_METAL_DISABLE_PRECOMPILED_FW=1 \
  TT_METAL_FW_SRC_BRISC=blaze TT_METAL_FW_HEADER_BRISC=$T/blaze/firmware/runtime_reload.h \
  TT_METAL_FABRIC_ROUTER_SYNC_TIMEOUT_MS=300000 TT_METAL_RISCV_DEBUG_INFO=1 \
  K3_WEIGHTS_TYPE=real HF_MODEL=$K31_HF_MODEL KIMI_K3_MODEL=$K31_HF_MODEL \
  BLAZE_TENSORBIN_CACHE=$DECODE_CACHE \
  TT_METAL_CACHE=$HOME/k31pd_cache TMPDIR=$HOME/k31pd_tmp PYTHONUNBUFFERED=1 \
  TT_K3_MAX_SEQ_LEN=$K3_MAX_SEQ_LEN \
  K31_PD_PROMPT_DIR=$PROMPT_DIR K31_PD_PREFILL_LEN=$P K31_PD_TOKENS=$N_TOKENS K31_PD_MATCH_TOKENS=$MATCH_TOKENS \
  K31_PD_DONE_FILE=$DONE_FILE K31_PD_HANDOFF=$HANDOFF K31_PD_TABLE_PATH=$DECODE_TABLE K31_PD_OUT=$OUT/$name \
  TT_MIGRATION_EXPORT_TO_FILE=1 TT_MIGRATION_DEVICE_MAP_PATH=$DECODE_DEVICE_MAP
export LD_LIBRARY_PATH=$T/tt-metal/build/lib:${LD_LIBRARY_PATH:-} PYTHONPATH=$T:$T/tests/pipeline_builder:$T/tt-metal/ttnn:$T/tt-metal
export TT_METAL_OPERATION_TIMEOUT_SECONDS=${OP_TIMEOUT:-900}
export TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE="python3 $TT_METAL_HOME/tools/tt-triage.py --disable-progress > $TRI.rank\$OMPI_COMM_WORLD_RANK 2>&1"
rm -f $TRI.rank*; mkdir -p $OUT/$name
for h in ${HOSTS//,/ }; do ssh -n -o BatchMode=yes $h "mkdir -p \$HOME/k31pd_cache \$HOME/k31pd_tmp; rm -f $DECODE_DEVICE_MAP"; done

FWD="PYTHONPATH LD_LIBRARY_PATH TT_METAL_HOME TT_METAL_RUNTIME_ROOT TT_METAL_SLOW_DISPATCH_MODE TT_METAL_DISABLE_PRECOMPILED_FW TT_METAL_FW_SRC_BRISC TT_METAL_FW_HEADER_BRISC TT_METAL_ALLOCATOR_MODE_HYBRID TT_METAL_RISCV_DEBUG_INFO TT_METAL_FABRIC_ROUTER_SYNC_TIMEOUT_MS K3_WEIGHTS_TYPE HF_MODEL KIMI_K3_MODEL BLAZE_TENSORBIN_CACHE TT_METAL_CACHE TMPDIR PYTHONUNBUFFERED TT_K3_MAX_SEQ_LEN K31_PD_PROMPT_DIR K31_PD_PREFILL_LEN K31_PD_TOKENS K31_PD_MATCH_TOKENS K31_PD_DONE_FILE K31_PD_HANDOFF K31_PD_TABLE_PATH K31_PD_OUT TT_MIGRATION_EXPORT_TO_FILE TT_MIGRATION_DEVICE_MAP_PATH TT_METAL_OPERATION_TIMEOUT_SECONDS TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE"
MPI="--bind-to none --tag-output --wdir $T"; for v in $FWD; do MPI="$MPI -x $v"; done
# tt-run does not deliver --mpi-args -x values to remote hosts, so also set every variable on the command itself.
ENVSET=(env); for v in $FWD; do ENVSET+=("$v=${!v}"); done

echo "START $(date -u +%FT%TZ) $name decode P=$P N=$N_TOKENS table=$DECODE_TABLE" | tee $LOG
python tt-metal/ttnn/ttnn/distributed/ttrun.py --skip-executable-check --tcp-interface ens5f0np0 \
  --hosts $HOSTS ${REDISCOVER:+--force-rediscovery} \
  --mesh-graph-descriptor $T/tests/pipeline_builder/mesh_graph_descriptors/quad_bh_galaxy_4x2_mesh_graph_descriptor.textproto \
  --mpi-args "$MPI" -- "${ENVSET[@]}" python -u -m pytest -x -s -v -p no:benchmark -p no:cacheprovider \
  "tests/pipeline_builder/test_kimi_k31_pd_migration.py::test_k31_quad_decode_from_migrated" >> "$LOG" 2>&1
rc=$?
echo "weight providers per rank:"; sed "s/\x1b\[[0-9;]*m//g" "$LOG" | grep -a "weights = " | sed -E "s/^\[1,([0-9]+)\].*weights = ([A-Za-z0-9]+).*/rank \1 \2/" | sort -u | sort -t' ' -k2n | tr '\n' ' '; echo
if ls $TRI.rank* >/dev/null 2>&1; then echo "HANG rc=$rc: triage in $TRI.rank*"; exit 2; fi
echo "DONE rc=$rc $(date -u +%FT%TZ)" | tee -a $LOG
exit $rc
