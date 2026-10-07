#!/bin/bash
set -euo pipefail
export PATH="/home/ttuser/kimi-prefill.Ubx2wY/runtime/qwen38-27b-20261006/python_env/bin:$PATH"
export TT_METAL_HOME="/home/ttuser/kimi-prefill.Ubx2wY/runtime/qwen38-27b-20261006/metal"
export PYTHONPATH="/home/ttuser/kimi-prefill.Ubx2wY/runtime/qwen38-27b-20261006/metal-galaxy:/home/ttuser/kimi-prefill.Ubx2wY/runtime/qwen38-27b-20261006/metal:/home/ttuser/kimi-prefill.Ubx2wY/runtime/qwen38-27b-20261006/metal/tools"
export LD_LIBRARY_PATH="/home/ttuser/kimi-prefill.Ubx2wY/runtime/qwen38-27b-20261006/metal-install/lib:/home/ttuser/kimi-prefill.Ubx2wY/runtime/qwen38-27b-20261006/metal-build/lib"
export ARCH_NAME=blackhole OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
export MODEL_WEIGHTS_DIR="/home/ttuser/qwen38-artifacts-20261007/checkpoint-pinned-1d4bf0f2"
export QWEN_PRECISION_CONFIG="/home/ttuser/kimi-prefill.Ubx2wY/runtime/qwen38-27b-20261006/metal-galaxy/models/demos/qwen38_27b_qb2/config/precision_accurate_decode.json"
export QWEN_DECODE_BUCKETS=1 QWEN_COMPACT_DECODE_RESIDUAL=1 QWEN_COMPACT_DECODE_MLP=1
export QWEN_BATCHED_DECODE_ROPE=1 QWEN_COMPACT_DECODE_ATTENTION=1
export QWEN_BATCHED_PREFILL=1 QWEN_PREFILL_RESIDUAL_LAYOUT=sharded_replicated_norm
export QWEN_PREFILL_BATCHED_HEAD=1 QWEN_PREFILL_SKIP_INTERMEDIATE_HEAD=1
export QWEN_PREFILL_STARTUP_WARMUP=1
cd "/home/ttuser/kimi-prefill.Ubx2wY/runtime/qwen38-27b-20261006/metal-galaxy"
python - <<'CHECK_SOURCE'
import json
from pathlib import Path
from models.demos.qwen38_27b_qb2.demo.galaxy_serving import model_source_hashes
source=Path('models/demos/qwen38_27b_qb2')
expected=json.loads(Path('/home/ttuser/qwen38-artifacts-20261007/qwen38-accurate-qualification-manifest.json').read_text())
assert model_source_hashes(source)==expected['source_sha256'], 'Published source manifest mismatch'
print('QUALIFICATION_COMMIT', expected['commit'], flush=True)
CHECK_SOURCE
echo "Starting fresh eight-replica G0 with accurate decode attention"
QWEN_GALAXY_REPLICAS=8 timeout --signal=TERM --kill-after=300s 6600s /bin/bash "/home/ttuser/kimi-prefill.Ubx2wY/runtime/qwen38-27b-20261006/metal-galaxy/models/demos/qwen38_27b_qb2/demo/run_galaxy_qualification.sh" "/home/ttuser/kimi-prefill.Ubx2wY/runtime/qwen38-27b-20261006" "/home/ttuser/qwen38-artifacts-20261007/accurate-decode-g0-v1" > "/home/ttuser/qwen38-artifacts-20261007/accurate-decode-g0-v1.log" 2>&1
echo "G0 passed; starting API qualification and full GPQA at unchanged 32K output budget"
exec /bin/bash "/home/ttuser/kimi-prefill.Ubx2wY/runtime/qwen38-27b-20261006/metal-galaxy/models/demos/qwen38_27b_qb2/demo/run_galaxy_serving.sh" "/home/ttuser/kimi-prefill.Ubx2wY/runtime/qwen38-27b-20261006" "/home/ttuser/qwen38-artifacts-20261007/accurate-decode-serving-v1" "/home/ttuser/qwen38-artifacts-20261007/accurate-decode-g0-v1/full-model.json"
