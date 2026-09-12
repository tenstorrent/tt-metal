#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
# Exclusive device use required. Each child closes the mesh before the next starts.
set -euo pipefail
optimized_model=models/autoports/qwen_qwen3_8_27b
optimized_evidence=$optimized_model/doc/optimized_decoder
optimized_snapshot=/home/mvasiljevic/hf-cache/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
optimized_activations=/home/mvasiljevic/qwen38-full-rerun/tt-metal-cache/optimized_decoder_activations_long
export PYTHONPATH=.:${PYTHONPATH:-}
printf '%q\n' "$0" >> "$optimized_evidence/commands.log"
python_env/bin/python -m pytest -q "$optimized_model/tests/test_optimized_decoder.py" \
 --basetemp /tmp/qwen_optimized_selected_pytest \
 --junitxml "$optimized_evidence/selected_stress.xml" > "$optimized_evidence/selected_stress.log" 2>&1
mkdir -p "$optimized_evidence/selected_stress"
find /tmp/qwen_optimized_selected_pytest -name 'optimized_l*_b*.json' -exec cp '{}' "$optimized_evidence/selected_stress/" \;
for optimized_layer in 0 3; do
 for optimized_batch in 1 32; do
  optimized_name=watcher_l${optimized_layer}_b${optimized_batch}
  TT_METAL_WATCHER=10 TT_METAL_LOGS_PATH="$PWD/$optimized_evidence/$optimized_name" \
   bash "$optimized_model/tests/run_optimization_experiment.sh" "$optimized_name" \
   --layer "$optimized_layer" --batch "$optimized_batch" --length 257 --continuation --benchmark \
   --activations "$optimized_activations"
 done
done
for optimized_layer in 0 3; do
 bash "$optimized_model/tests/profile_optimized_decoder.sh" "final_l$optimized_layer" --layer "$optimized_layer" --length 128
 bash "$optimized_model/tests/profile_optimized_decoder.sh" "baseline_l$optimized_layer" --baseline --layer "$optimized_layer" --length 128
 bash "$optimized_model/tests/profile_optimized_decoder.sh" "final_long_l$optimized_layer" --layer "$optimized_layer" --length 2048 --activations "$optimized_activations"
done
TT_METAL_WATCHER=10 TT_METAL_LOGS_PATH="$PWD/$optimized_evidence/watcher_long_tails" \
 bash "$optimized_model/tests/run_optimization_experiment.sh" watcher_long_tails \
 --layer 3 --lengths 2049,4097,31 --continuation --benchmark --activations "$optimized_activations"
for optimized_layer in 0 3; do
 bash "$optimized_model/tests/run_optimization_experiment.sh" "final_benchmark_l$optimized_layer" \
 --layer "$optimized_layer" --length 128 --benchmark --trace-prefill --activations "$optimized_activations"
 bash "$optimized_model/tests/run_optimization_experiment.sh" "baseline_benchmark_l$optimized_layer" \
 --baseline --layer "$optimized_layer" --length 128 --benchmark --activations "$optimized_activations"
 bash "$optimized_model/tests/run_optimization_experiment.sh" "final_4097_l$optimized_layer" \
 --layer "$optimized_layer" --length 4097 --benchmark --activations "$optimized_activations"
done
for optimized_layer in 0 3; do
 timeout -k 10 7200 python_env/bin/python "$optimized_model/tests/run_optimized_context.py" \
  --snapshot "$optimized_snapshot" --layer "$optimized_layer" --activations "$optimized_activations" \
  --output "$optimized_evidence/context_l$optimized_layer.json" > "$optimized_evidence/context_l$optimized_layer.log" 2>&1
done

# Final default/control/default comparison and profile after grid selection.
python_env/bin/python "$optimized_model/tests/sweep_optimized_decoder.py" \
 --matrix "$optimized_evidence/verified_matrix.json" > "$optimized_evidence/verified_matrix.log" 2>&1
TT_METAL_WATCHER=10 TT_METAL_LOGS_PATH="$PWD/$optimized_evidence/watcher_selected_l3" \
 bash "$optimized_model/tests/run_optimization_experiment.sh" watcher_selected_l3 \
 --layer 3 --lengths 31,129,257,511 --continuation --benchmark --activations "$optimized_activations"
bash "$optimized_model/tests/profile_optimized_decoder.sh" selected_l3 --layer 3 --length 128
python_env/bin/python "$optimized_model/tests/summarize_optimization.py"
