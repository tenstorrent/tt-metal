#!/usr/bin/env bash
# Run only after the previous device process has exited. All device jobs serialize.
set -euo pipefail
MODEL_DIR=models/autoports/aleph_alpha_kolibri_1_bf16
EVIDENCE="$MODEL_DIR/doc/functional_decoder"
export TT_METAL_TRACE_ALLOC_TRACKING=1
unset TT_METAL_WATCHER TT_METAL_DEVICE_PROFILER TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE
for layer in 0 4; do
  python -m models.autoports.aleph_alpha_kolibri_1_bf16.tests.run_coverage --layer "$layer" > "$EVIDENCE/coverage_${layer}_final.log" 2>&1
done
for spec in '0 32' '4 32' '0 31' '4 13'; do
  read -r layer batch <<< "$spec"
  python -m models.autoports.aleph_alpha_kolibri_1_bf16.tests.run_decoder --layer "$layer" --length 65 --batch "$batch" --public --output "batch${batch}_${layer}_final.json" > "$EVIDENCE/batch${batch}_${layer}_final.log" 2>&1
done
# B2 uses thirteen attention cores/head on 110 cores: exercise the non-power-of-two tree.
for layer in 0 4; do
  python -m models.autoports.aleph_alpha_kolibri_1_bf16.tests.run_decoder --layer "$layer" --length 3329 --capacity 4096 --batch 2 --public --output "batch2_${layer}_final.json" > "$EVIDENCE/batch2_${layer}_final.log" 2>&1
done
HF_HOME="$PWD/$EVIDENCE/no_hf_cache" python -m pytest -q -s "$MODEL_DIR/tests/test_functional_decoder.py" > "$EVIDENCE/synthetic_pytest.log" 2>&1
for layer in 0 4; do
  TT_METAL_WATCHER=10 TT_METAL_LOGS_PATH="$PWD/$EVIDENCE/watcher/layer_${layer}" python -m models.autoports.aleph_alpha_kolibri_1_bf16.tests.run_coverage --layer "$layer" --output "watcher_${layer}.json" > "$EVIDENCE/watcher_${layer}.log" 2>&1
done
for layer in 0 4; do
  TT_METAL_TRACE_ALLOC_TRACKING=0 python -m tracy -r -p -v --no-web-server -o "$PWD/$EVIDENCE/tracy/layer_${layer}/raw" -m models.autoports.aleph_alpha_kolibri_1_bf16.tests.run_profile --layer "$layer" > "$EVIDENCE/profile_${layer}.log" 2>&1
done
