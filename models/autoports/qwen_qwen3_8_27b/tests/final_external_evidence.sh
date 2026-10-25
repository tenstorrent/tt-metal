#!/usr/bin/env bash
# Run only after final_full_model_evidence.sh releases the shared device mesh.
set -euo pipefail
base=models/autoports/qwen_qwen3_8_27b/doc/full_model
module=models.autoports.qwen_qwen3_8_27b.tests
python -m "$module.probe_external_prepare" > "$base/external_prepare.log" 2>&1
for kind in host device refreshed mutated; do
    python -m "$module.probe_owned_sparse_map" --map-kind "$kind" > "$base/owned_sparse_map_${kind}.log" 2>&1
done
python -m "$module.probe_mixed_slots" > "$base/probe_mixed_slots_06.log" 2>&1
python -m "$module.probe_trace_contract" > "$base/probe_contract_07.log" 2>&1
python -m "$module.probe_external_prepare" --full-stack --batch 32 --context 33024 --output "$base/external_prepare_b32.json" > "$base/external_prepare_b32.log" 2>&1
python -m "$module.run_full_validation" > "$base/full_validation_04.log" 2>&1
