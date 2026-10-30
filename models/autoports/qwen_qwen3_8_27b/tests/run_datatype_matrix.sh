#!/usr/bin/env bash
# Run inside qwen38-functional-build. Abort on runtime failure for investigation.
set -euo pipefail
root=models/autoports/qwen_qwen3_8_27b
out=$root/doc/datatype_sweep
control_args=()
if [[ -n ${QWEN38_CPU_CONTROL_RECEIPT:-} ]]; then
    control_args=(--wait-for-control "$QWEN38_CPU_CONTROL_RECEIPT")
fi
for config in "$@"; do
    bash "$root/tests/optimized_multichip_env.sh" python -m models.autoports.qwen_qwen3_8_27b.tests.run_datatype_candidate \
        --config "$out/configs/$config.json" --output "$out/results/$config.json" "${control_args[@]}" \
        > "$out/$config.log" 2>&1
    echo "CANDIDATE_COMPLETE $config"
done
