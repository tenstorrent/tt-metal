#!/usr/bin/env bash
set -euo pipefail

source python_env/bin/activate

export TT_SMI_RESET_COMMAND="tt-smi -glx_reset_auto"
export ARCH_NAME="wormhole_b0"
export TRACY_NO_INVARIANT_CHECK=1
export GITHUB_ACTIONS=true
export MESH_DEVICE_SHAPE=4x8
export SWEEPS_RUNNER_HARD_EXIT=1
export TEST_GROUP_NAME="wormhole-galaxy-sweeps"
export LEAD_MODELS_RUN=0

ts=$(date +%Y%m%d_%H%M%S)
log="sweep_debug_logs/tracer_validation_${ts}.log"
mkdir -p sweep_debug_logs
mkdir -p model_tracer/traced_operations

$TT_SMI_RESET_COMMAND >/dev/null 2>&1 || true

# 1) Master JSON is already reconstructed at
#    model_tracer/traced_operations/ttnn_operations_master.json (209 configs,
#    tt_dit_2 trace 16). Skip the reconstruct-manifest step (which requires
#    NEON_CONNECTION_STRING) and use the existing canonical master.
echo "Using existing master: model_tracer/traced_operations/ttnn_operations_master.json" | tee -a "$log"

# 2) Generate vectors for ci-validate-tt-dit-2
python tests/sweep_framework/sweeps_parameter_generator.py \
  --model-traced all \
  --suite-name model_traced \
  --group-by hw \
  --master-trace model_tracer/traced_operations/ttnn_operations_master.json \
  --tag ci-validate-tt-dit-2 2>&1 | tee -a "$log"

# Build module list
MODULES=$(python -c "
import json
m = json.load(open('tests/sweep_framework/vectors_export/generation_manifest.json'))
print(','.join(sorted({f.rsplit('.hw_',1)[0] for f in m['vector_files']})))
")

echo "Modules:" | tee -a "$log"
echo "$MODULES" | tr ',' '\n' | sed 's/^/  /' | tee -a "$log"
echo "log=$log" | tee -a "$log"

# 3) Run sweeps under operation tracer
$TT_SMI_RESET_COMMAND >/dev/null 2>&1 || true

# Remove any prior sweep trace so this run starts clean
rm -f model_tracer/traced_operations/sweep_trace_tt_dit_2_galaxy.json

python model_tracer/generic_ops_tracer.py \
  tests/sweep_framework/sweeps_runner.py \
  -o model_tracer/traced_operations/sweep_trace_tt_dit_2_galaxy.json \
  -- \
    --module-name "$MODULES" \
    --suite-name model_traced \
    --vector-source vectors_export \
    --result-dest results_export \
    --main-proc-verbose \
    --tag ci-validate-tt-dit-2 2>&1 | tee -a "$log"

# 4) Validate
python tests/sweep_framework/validate_sweep_trace.py \
  --master-trace model_tracer/traced_operations/ttnn_operations_master.json \
  --sweep-trace model_tracer/traced_operations/sweep_trace_tt_dit_2_galaxy.json \
  --output-report validation_summary_tt_dit_2.md 2>&1 | tee -a "$log"

rc=${PIPESTATUS[0]}
echo "Validator exit: $rc" | tee -a "$log"
echo "Log: $log"
exit "$rc"
