#!/usr/bin/env bash
# Step 2: 1-rank smoke test of Mistral Small 4 through the REAL prefill runner + producer,
# with the per-slot KV PCC gate on, against a pre-built 36-layer golden trace.
set -euo pipefail
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$S/env.sh"
cd "$TT_METAL_HOME"

export PREFILL_MODEL=mistral_small_4
export PREFILL_GATE_FALLBACK_MODE=GPT_DEVICE
export PREFILL_NUM_LAYERS="${PREFILL_NUM_LAYERS:-36}"
export PREFILL_TTNN_CACHE=$CACHE_8x4
export PREFILL_SP=8
export PREFILL_TP=4
# 8x4 production profile: ring/ring needs the cabling-certified explicit descriptor.
export PREFILL_FABRIC_MODE=2d_torus_xy
export TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_torus_xy_graph_descriptor.textproto

# Drive the harness's opt-in prompt scenario, but REUSE the pre-built golden instead of
# regenerating it on the host (PREFILL_REUSE_TRACE_DIR short-circuits _generate_prompt_trace).
export PREFILL_PROMPT_FILE=$S/prompt.json
export PREFILL_REUSE_TRACE_DIR="${PREFILL_REUSE_TRACE_DIR:-$GOLDEN_5120}"
export PREFILL_PROMPT_CHUNKS="${PREFILL_PROMPT_CHUNKS:-1}"

export PREFILL_CI_RUNNER_READY_TIMEOUT_S=2400
export PREFILL_CI_PRODUCER_TIMEOUT_S=3600
export LOGURU_LEVEL=INFO

exec "$TT_METAL_HOME/python_env/bin/python" -m pytest -svv \
  models/demos/common/prefill/tests/test_producer_runner_e2e.py \
  -k prompt_single_user
