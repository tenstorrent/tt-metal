#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
SWEEP_SCRIPT="models/demos/glm4_moe_lite/scripts/run_sweep_isl_batch.py"
EXP_ROOT="models/demos/glm4_moe_lite/experiments"
cd "$REPO_ROOT"

TIMEOUT=900

run_experiment() {
    local name="$1"; shift
    local out_dir="$EXP_ROOT/$name"
    mkdir -p "$out_dir"
    echo "============================================================"
    echo "  EXPERIMENT: $name"
    echo "  Output:     $out_dir"
    echo "  Start:      $(date)"
    echo "============================================================"

    (
        export GLM4_MOE_LITE_SKIP_DEFENSIVE_CLONES=1
        export GLM4_MOE_LITE_FUSE_QKV_A=1
        export GLM4_MOE_LITE_FUSE_SHARED_GATE_UP=1
        export GLM4_MOE_LITE_BATCHED_PREFILL=1
        export GLM4_MOE_LITE_DECODE_L1_ACT=1
        export GLM4_MOE_LITE_EP_L1=1

        python "$SWEEP_SCRIPT" --out-dir "$out_dir" --timeout "$TIMEOUT" "$@"
    ) 2>&1 | tee "$out_dir/experiment.log"
    local rc=${PIPESTATUS[0]}
    echo "  Finished:   $(date)  exit=$rc"
    echo ""
    return $rc
}

echo ">>> Phase 0: Baseline"
run_experiment baseline --isl 128 512 1024 --batch 1 || echo "WARN: baseline failed with exit $?"

echo ">>> G5: Larger Batch Sizes"
run_experiment g5_batch --isl 128 512 --batch 1 2 4 8 16 32 || echo "WARN: g5_batch failed with exit $?"

echo ">>> G7: bf4 Expert Weights"
(
    export GLM4_MOE_LITE_EXPERTS_TT_DTYPE=bf4
    run_experiment g7_bf4 --isl 128 512 1024 --batch 1
) || echo "WARN: g7_bf4 failed with exit $?"

echo ">>> G8: Fused MOE Kernel"
(
    export GLM4_MOE_LITE_FUSED_MOE=1
    run_experiment g8_fused_moe --isl 128 512 1024 --batch 1
) || echo "WARN: g8_fused_moe failed with exit $?"

echo ">>> G6: DRAM Sharded Weights"
(
    export GLM4_MOE_LITE_DRAM_SHARDED_WEIGHTS=1
    run_experiment g6_dram_sharded --isl 128 512 1024 --batch 1
) || echo "WARN: g6_dram_sharded failed with exit $?"

echo "============================================================"
echo "  ALL PHASE 1 EXPERIMENTS COMPLETE"
echo "============================================================"
