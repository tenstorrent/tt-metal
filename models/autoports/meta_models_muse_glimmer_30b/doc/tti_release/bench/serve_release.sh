#!/usr/bin/env bash
# TTI-release stage server launcher.
#
# Byte-identical serving configuration to the optimized-vLLM stage's shipped
# arm (doc/optimized_vllm/bench/serve.sh, mode `hold`, tracing OFF), so the
# implementation TTI evaluates is exactly the one that stage measured and
# committed.  The only differences are the output directory and the log path.
#
#   serve_release.sh            # launch and hold open
#
# The one addition on top of that arm is the model's reasoning parser
# (tt/reasoning_parser.py).  Muse Glimmer writes its analysis into a `to=self`
# channel ahead of the reply, and with no parser configured vLLM returns both
# concatenated in `content` -- which every eval harness then reads as the
# answer.  The parser is API-layer text routing only: same sampling, same
# generator, same tokens on device.  `smoke/reasoning_control_unparsed.json` vs
# `smoke/reasoning_parsed.json` is the control that shows the generation is
# unchanged.
#
# Env knobs: PORT (default 8000), OUT_DIR, REASONING_PARSER (empty disables).
set -euo pipefail

REPO=/home/ttuser/dev/muse-glimmer/tt-metal
PYENV=/home/ttuser/dev/muse-glimmer/muse-glimmer_pyenv
MODEL_DIR=models/autoports/meta_models_muse_glimmer_30b
HF_MODEL=meta-models/Muse-Glimmer-30B
MESH=P300x2
MAX_NUM_SEQS=${MAX_NUM_SEQS:-32}
# From doc/context_contract.json -- never lowered by this stage.
MAX_MODEL_LEN=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["current_supported_context"])' \
  "$REPO/$MODEL_DIR/doc/context_contract.json")
PORT=${PORT:-8000}
TT_CONFIG='{"trace_region_size": 400000000, "fabric_config": "FABRIC_1D_RING", "fabric_packet_payload_bytes": 8192, "l1_small_size": 6144, "trace_mode": "decode_only"}'
OUT_DIR=${OUT_DIR:-$REPO/$MODEL_DIR/doc/tti_release/server}

unset VIRTUAL_ENV VIRTUAL_ENV_PROMPT PYTHONPATH PYTHONHOME
# shellcheck disable=SC1091
source "$PYENV/bin/activate"
export TT_METAL_HOME="$REPO"
export PYTHONPATH="$REPO"
# tracing off: this is the optimized-vLLM stage's shipped default.  Set
# explicitly rather than left unset so the arm's identity is inside its own log.
export MUSE_GLIMMER_VLLM_PREFILL_TRACE=0

cd "$REPO"
mkdir -p "$OUT_DIR"

REASONING_PARSER=${REASONING_PARSER-muse_glimmer}
# One argv element: run_vllm_server shlex.splits the value, and an unquoted
# expansion here would hand it to argparse as several arguments instead.
EXTRA=()
if [ -n "$REASONING_PARSER" ]; then
  EXTRA=(--additional-server-args="--reasoning-parser-plugin $REPO/$MODEL_DIR/tt/reasoning_parser.py --reasoning-parser $REASONING_PARSER")
fi

echo "=== tti-release server: max_model_len=$MAX_MODEL_LEN max_num_seqs=$MAX_NUM_SEQS port=$PORT prefill_trace=$MUSE_GLIMMER_VLLM_PREFILL_TRACE reasoning_parser=${REASONING_PARSER:-none} ==="
exec python -m models.common.readiness_check.run_vllm_server \
  --stages serve \
  --model-dir "$MODEL_DIR" --hf-model "$HF_MODEL" --mesh-device "$MESH" \
  --max-num-seqs "$MAX_NUM_SEQS" --max-model-len "$MAX_MODEL_LEN" --port "$PORT" \
  --server-timeout 2400 --output-dir "$OUT_DIR" \
  "${EXTRA[@]}" \
  --tt-config "$TT_CONFIG"
