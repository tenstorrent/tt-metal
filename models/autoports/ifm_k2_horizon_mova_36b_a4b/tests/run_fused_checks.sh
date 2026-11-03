#!/usr/bin/env bash
set -euo pipefail
source python_env/bin/activate
source /home/ttuser/k2-horizon-mova/lane-env.sh
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
run_check watcher_dense env TT_METAL_WATCHER=10 TT_METAL_LOGS_PATH="$PWD/$FUSED_DOC/watcher/dense" python -m "$FUSED_TESTS.validate_fused_decoder" --layer 0 --batch 32 --lengths 33
run_check watcher_sparse env TT_METAL_WATCHER=10 TT_METAL_LOGS_PATH="$PWD/$FUSED_DOC/watcher/mova_moe" python -m "$FUSED_TESTS.validate_fused_decoder" --layer 3 --batch 13 --lengths 31 33
run_check batch9_dense python -m "$FUSED_TESTS.validate_fused_decoder" --layer 0 --batch 9 --lengths 33
run_check page128_sparse python -m "$FUSED_TESTS.validate_fused_decoder" --layer 3 --batch 2 --page-size 128 --lengths 127 128 129
run_check page256_sparse python -m "$FUSED_TESTS.validate_fused_decoder" --layer 3 --batch 2 --page-size 256 --lengths 255 256 257
run_check pytest python -m pytest -q models/autoports/ifm_k2_horizon_mova_36b_a4b/tests/test_fused_decoder.py
run_check capacity_dense python -m "$FUSED_TESTS.fused_decode_capacity" --layer 0
run_check capacity_sparse python -m "$FUSED_TESTS.fused_decode_capacity" --layer 3
run_check audit python -m "$FUSED_TESTS.audit_fused_runtime"
