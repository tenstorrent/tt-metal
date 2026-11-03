#!/usr/bin/env bash
set -euo pipefail
source python_env/bin/activate
source /home/ttuser/k2-horizon-mova/lane-env.sh
unset TT_METAL_WATCHER
export TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0
export TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback":true}'
FUSED_TESTS=models.autoports.ifm_k2_horizon_mova_36b_a4b.tests
FUSED_DOC=models/autoports/ifm_k2_horizon_mova_36b_a4b/doc/fused_decoder
run_check() {
    local tag=$1
    shift
    echo "START $tag $(date -u +%FT%TZ)"
    "$@" > "$FUSED_DOC/logs/$tag.log" 2>&1
    echo "PASS $tag $(date -u +%FT%TZ)"
}
for layer in 0 3; do
    for capacity in 262112 262144; do
        run_check "capacity_${layer}_${capacity}" python -m "$FUSED_TESTS.fused_decode_capacity" --layer "$layer" --capacity "$capacity"
    done
    run_check "matrix_final_layer${layer}" python -m "$FUSED_TESTS.validate_fused_decoder" --layer "$layer" --batch 2 --page-size 64 --lengths 1 31 32 33 63 64 65 255 256 257 529 1023
    run_check "long_final_layer${layer}" python -m "$FUSED_TESTS.fused_long_context" --layer "$layer"
done
run_check batch32_sparse_final python -m "$FUSED_TESTS.validate_fused_decoder" --layer 3 --batch 32 --lengths 33
