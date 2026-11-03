#!/usr/bin/env bash
set -euo pipefail
source python_env/bin/activate
source /home/ttuser/k2-horizon-mova/lane-env.sh
unset TT_METAL_WATCHER TTNN_CONFIG_OVERRIDES
export TT_METAL_TRACE_ALLOC_TRACKING=0
FUSED_TESTS=models.autoports.ifm_k2_horizon_mova_36b_a4b.tests
FUSED_DOC=models/autoports/ifm_k2_horizon_mova_36b_a4b/doc/fused_decoder
for tag in baseline final; do
    for layer in 0 3; do
        echo "START profile_${tag}_layer${layer} $(date -u +%FT%TZ)"
        fused_args=()
        if [ "$tag" = baseline ]; then fused_args+=(--baseline); fi
        python -m tracy -r -p -v --no-web-server -o "$FUSED_DOC/profile_raw/${tag}_layer${layer}" -m "$FUSED_TESTS.run_fused_decoder" --layer "$layer" --seq 256 --context 1024 --trace --profile --iterations 1 --tag "profile_${tag}" "${fused_args[@]}" > "$FUSED_DOC/logs/profile_${tag}_layer${layer}.log" 2>&1
        fused_csv=$(python - "$FUSED_DOC/profile_raw/${tag}_layer${layer}" <<'PY'
import pathlib, sys
paths=list(pathlib.Path(sys.argv[1]).rglob('ops_perf_results*.csv'))
assert len(paths)==1, paths
print(paths[0])
PY
)
        python -m "$FUSED_TESTS.fused_perf_report" "$layer" "$fused_csv" --tag "$tag" > "$FUSED_DOC/logs/report_${tag}_layer${layer}.log" 2>&1
        echo "PASS profile_${tag}_layer${layer} $(date -u +%FT%TZ)"
    done
done
