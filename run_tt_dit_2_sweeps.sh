#!/usr/bin/env bash
set -euo pipefail

source python_env/bin/activate

# CI env (galaxy 6U, model_traced run)
export TT_SMI_RESET_COMMAND="tt-smi -glx_reset_auto"
export TEST_GROUP_NAME="wormhole-galaxy-sweeps"
export ARCH_NAME="wormhole_b0"
export GITHUB_ACTIONS=true
export TRACY_NO_INVARIANT_CHECK=1
export LEAD_MODELS_RUN=0

# Clean state before starting
$TT_SMI_RESET_COMMAND
python -c "import ttnn; print('num_devices:', ttnn.get_num_devices())"

# 1) Reconstruct master JSON (CI uses scope-only; we add the tt_dit_2 filter
#    so we only validate trace 16 right now).
mkdir -p model_tracer/traced_operations
python tests/sweep_framework/load_ttnn_ops_data_v2.py reconstruct-manifest \
  model_tracer/trace_selection_registry.yaml \
  model_tracer/traced_operations/ttnn_operations_master.json \
  model_traced \
  --models-filter tt_dit_2

# 2) Generate vectors exactly like CI does for the model_traced run
python tests/sweep_framework/sweeps_parameter_generator.py \
  --model-traced all \
  --suite-name model_traced \
  --group-by hw \
  --master-trace model_tracer/traced_operations/ttnn_operations_master.json \
  --tag ci-main

NUM_VECTOR_FILES=$(python -c "
import json
m = json.load(open('tests/sweep_framework/vectors_export/generation_manifest.json'))
print(len(m.get('vector_files', [])))
")
if [[ "$NUM_VECTOR_FILES" -eq 0 ]]; then
  echo "ERROR: 0 vector files generated. Check that model_tracer/traced_operations/ttnn_operations_master.json exists and contains tt_dit_2 configs."
  exit 1
fi
echo "Generated $NUM_VECTOR_FILES grouped vector file(s)."

# 3) Run the generated modules as one CI-like batch. In CI, each matrix entry
# passes a comma-separated module selector into one sweeps_runner.py invocation.
MODULES=$(python -c "
import json
m = json.load(open('tests/sweep_framework/vectors_export/generation_manifest.json'))
print(','.join(sorted({f.rsplit('.hw_',1)[0] for f in m['vector_files']})))
")

mkdir -p sweep_debug_logs
log="sweep_debug_logs/tt_dit_2_model_traced_batch.log"

echo "Running module selector:"
echo "$MODULES"

# Reset once before the CI-like batch run.
$TT_SMI_RESET_COMMAND >/dev/null 2>&1 || true

python tests/sweep_framework/sweeps_runner.py \
  --module-name "$MODULES" \
  --suite-name model_traced \
  --vector-source vectors_export \
  --result-dest results_export \
  --tag ci-main \
  --summary \
  --skip-on-timeout 2>&1 | tee "$log"

rc=${PIPESTATUS[0]}

echo
echo "===== Summary ====="
echo "Results: tests/sweep_framework/results_export/"
if [[ $rc -ne 0 ]]; then
  echo "Sweep batch failed with exit code $rc. Log: $log"
  exit "$rc"
else
  echo "Sweep batch completed. Log: $log"
fi
