#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
# Run only after selection, with exclusive ownership of the QB2 mesh.
set -euo pipefail
source models/autoports/meta_models_muse_glimmer_30b/tests/full_model_env.sh
unset MUSE_PRECISION_CONFIG TT_METAL_WATCHER TT_METAL_DEVICE_PROFILER
unset TT_METAL_TRACE_ALLOC_TRACKING TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE
sweep_model_root=models/autoports/meta_models_muse_glimmer_30b
sweep_evidence="$sweep_model_root/doc/datatype_sweep"
run_selected_check() {
    local check_name="$1"
    shift
    echo "START $check_name"
    "$@" > "$sweep_evidence/$check_name.log" 2>&1
    echo "PASS $check_name"
}
run_selected_check post_selection python_env/bin/python "$sweep_model_root/tests/run_full_model.py" \
    --checks prefill teacher qualitative perf --evidence-dir "$sweep_evidence" \
    --output "$sweep_evidence/post_selection.json"
export TT_METAL_TRACE_ALLOC_TRACKING=1
export TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0
export TT_METAL_TRACE_ALLOC_TRACEBACKS=1
run_selected_check selected_tracking python_env/bin/python "$sweep_model_root/tests/run_datatype_candidate.py" \
    --config "$sweep_evidence/selected_precision_config.json" --output "$sweep_evidence/selected_tracking.json"
run_selected_check selected_eager_control python_env/bin/python "$sweep_model_root/tests/probe_full_model.py" \
    --output "$sweep_evidence/selected_eager_control.json"
run_selected_check selected_probe python_env/bin/python "$sweep_model_root/tests/probe_full_model.py" \
    --generator --audit --external-cache --pages 5000 --eager-control "$sweep_evidence/selected_eager_control.pt" \
    --output "$sweep_evidence/selected_probe.json"
run_selected_check selected_batch32 python_env/bin/python "$sweep_model_root/tests/check_full_batch.py" \
    --output "$sweep_evidence/selected_batch32.json"
run_selected_check selected_context python_env/bin/python "$sweep_model_root/tests/check_full_context.py" \
    --output "$sweep_evidence/selected_context.json"
unset TT_METAL_TRACE_ALLOC_TRACKING TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE TT_METAL_TRACE_ALLOC_TRACEBACKS
PROFILE_DESTINATION="$sweep_evidence/tracy_selected" bash "$sweep_model_root/tests/profile_full_model.sh"
echo "PASS selected_profile"
