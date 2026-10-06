#!/usr/bin/env bash
set -euo pipefail
export TT_MODEL_BRINGUP_ROOT=/home/vkovacevic/kolibri/codex_home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19
export TT_AUTODEBUG_ROOT=/home/vkovacevic/kolibri/codex_home/plugins/cache/tenstorrent-skills/tt-autodebug/0.1.8
BRINGUP_EXPORTS=$(python "$TT_MODEL_BRINGUP_ROOT/scripts/environment.py")
eval "$BRINGUP_EXPORTS"
export TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0
unset FUSIONS FUSION_IMPL FUSION_TAG TT_METAL_WATCHER TT_METAL_DEVICE_PROFILER
MODULE=models.autoports.aleph_alpha_kolibri_1_bf16.tests
EVIDENCE=models/autoports/aleph_alpha_kolibri_1_bf16/doc/fused_decoder
for layer in 0 4; do
  for batch in 3 17 32; do
    TT_METAL_TRACE_ALLOC_TRACKING=1 python -m "$MODULE.fused_decoder_checks" --layer "$layer" --length 65 --batch "$batch" --public --output "batch_${batch}_${layer}.json" > "$EVIDENCE/batch_${batch}_${layer}.log" 2>&1
    echo "batch=$batch layer=$layer passed"
  done
  TT_METAL_TRACE_ALLOC_TRACKING=1 python -m "$MODULE.fused_context_edges" --layer "$layer" > "$EVIDENCE/context_edges_${layer}.log" 2>&1
  echo "context edges layer=$layer passed"
  TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_WATCHER=10 TT_METAL_LOGS_PATH="$PWD/$EVIDENCE/watcher_${layer}" python -m "$MODULE.fused_coverage" --layer "$layer" --output "watcher_${layer}.json" > "$EVIDENCE/watcher_${layer}.log" 2>&1
  echo "watcher layer=$layer passed"
done
TT_METAL_TRACE_ALLOC_TRACKING=1 pytest -q models/autoports/aleph_alpha_kolibri_1_bf16/tests/test_fused_decoder.py > "$EVIDENCE/pytest.log" 2>&1
echo 'synthetic pytest passed'
bash models/autoports/aleph_alpha_kolibri_1_bf16/tests/run_fused_perf.sh
python -m "$MODULE.fused_report_performance" > "$EVIDENCE/report_generation.log"
python -m "$MODULE.fused_candidate_report" > "$EVIDENCE/candidate_report.log"
python -m "$MODULE.check_fused_evidence" > "$EVIDENCE/evidence_check.log"
