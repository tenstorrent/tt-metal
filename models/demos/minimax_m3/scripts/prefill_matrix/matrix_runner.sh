#!/bin/bash
# MiniMax-M3 pipeline-prefill RUNNER for the prefill matrix (node side). Launched by matrix_row.sh ON rank 0's host:
#   env STAGES=16|12 CACHED=<C> USERS=<U> WORK=<dir> HOSTS=<A:4,B:4,...> matrix_runner.sh
# Writes the manifest the ranks read (PREFILL_MANIFEST is the only env remote ranks see), renders the binding for this
# row's KV capacity (CACHED + MAX_NEW) and exec's run_pipeline_prefill.sh. Every rank appends one timing row per chunk
# to $TIMING_DIR/rank<r>.csv (PREFILL_SYNC_PER_CHUNK=1) -- that is what matrix_producer.py measures from.
set -uo pipefail
PKG_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$PKG_DIR/matrix_common.sh"; eval "$MATRIX_ULIMITS"
TT_METAL_HOME=${TT_METAL_HOME:-$(cd "$PKG_DIR/../../../../.." && pwd)}
cd "$TT_METAL_HOME"
export TT_METAL_HOME PYTHONPATH=$TT_METAL_HOME
source python_env/bin/activate
export HF_MODEL=${HF_MODEL:-/mnt/weka/model-weights/llm/minimax/MiniMax-M3}
export TT_CACHE_PATH=${TT_CACHE_PATH:-/mnt/weka/model-cache/scratch/minimax/MiniMax-M3-cache/prefill}
STAGES=${STAGES:-16}; CACHED=${CACHED:?set CACHED=<cached tokens, chunk multiple>}
USERS=${USERS:-1}; [ "$USERS" -ge 1 ] || USERS=1   # USERS=0 means idle-only measurement; the runner still needs slot 0
MAX_NEW=${MAX_NEW:-$MATRIX_MAX_NEW}                 # matrix_row.sh passes the same value to the producer
WORK=${WORK:?set WORK=<shared work dir>}; HOSTS=${HOSTS:?set HOSTS=<host:ranks,...>}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)_s${STAGES}_c${CACHED}}
TIMING_DIR=$WORK/timing_$STAMP; mkdir -p "$TIMING_DIR"
MANIFEST=$WORK/manifest_$STAMP.json
cat > "$MANIFEST" <<JSON
{
  "env": {
    "PREFILL_MODEL": "minimax_m3",
    "PREFILL_NUM_LAYERS": "60",
    "M3_INDEX_CACHE_BF16": "1",
    "EXPERT_DTYPE": "${EXPERT_DTYPE:-bf4}",
    "HF_MODEL": "$HF_MODEL",
    "PREFILL_SYNC_PER_CHUNK": "1",
    "PREFILL_TIMING_DIR": "$TIMING_DIR",
    "PREFILL_NUM_USERS": "$USERS"
  }
}
JSON
case $STAGES in 16) TEMPLATE=$PKG_DIR/binding_16stage_quad.yaml.in;; 12) TEMPLATE=$PKG_DIR/binding_12stage_tri.yaml.in;; *) echo "STAGES must be 16 or 12"; exit 2;; esac
BINDING=$WORK/binding_$STAMP.yaml
esc() { printf '%s' "$1" | sed 's/[&#\\]/\\&/g'; }   # sed replacement-side escaping for paths
sed -e "s#@MAX_SEQ_LEN@#$((CACHED + MAX_NEW))#" -e "s#@PKG_DIR@#$(esc "$PKG_DIR")#g" -e "s#@MANIFEST@#$(esc "$MANIFEST")#" -e "s#@TT_CACHE_PATH@#$(esc "$TT_CACHE_PATH")#" "$TEMPLATE" > "$BINDING"
echo "[runner] $(date) launch_host=$(hostname -s) commit=$(git rev-parse --short HEAD) stages=$STAGES cached=$CACHED users=$USERS capacity=$((CACHED + MAX_NEW)) binding=$BINDING manifest=$MANIFEST hosts=$HOSTS timing_dir=$TIMING_DIR"
cat "$MANIFEST"
for v in "${!SLURM@}"; do unset "$v"; done   # mpirun launches the remote ranks over ssh, not through Slurm
export PRTE_MCA_ras="^slurm" PRTE_MCA_plm="^slurm"
exec ./models/demos/common/prefill/runners/run_pipeline_prefill.sh "$BINDING" "$HOSTS"
