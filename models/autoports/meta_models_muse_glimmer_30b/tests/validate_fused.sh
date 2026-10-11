#!/usr/bin/env bash
# From checkout, after python_env and lane-env.sh. Device jobs remain serial.
set -euo pipefail
model_root=models/autoports/meta_models_muse_glimmer_30b
reference_python=${HF_REFERENCE_PYTHON:-/home/vkovacevic/muse-glimmer/hf-reference-env/bin/python}
phase=${1:-correctness}
export TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0
unset TT_METAL_DEVICE_PROFILER TT_METAL_WATCHER
for layer in 0 3; do
    destination="$model_root/doc/fused_decoder"
    case "$phase" in
        boundaries)
            "$reference_python" "$model_root/tests/run_fused.py" --real --layer "$layer" --lengths 1 31 32 33 1023 1024 1025 2047 2048 2049 3073 17 --output "$destination/boundaries_$layer.json" > "$destination/boundaries_$layer.log" 2>&1
            ;;
        correctness)
            "$reference_python" "$model_root/tests/run_fused.py" --real --layer "$layer" --batch 2 --lengths 33 --continuation 1057 --output "$destination/continuation_$layer.json" > "$destination/continuation_$layer.log" 2>&1
            "$reference_python" "$model_root/tests/run_fused.py" --real --layer "$layer" --lengths 3073 --control --output "$destination/control_$layer.json" > "$destination/control_$layer.log" 2>&1
            "$reference_python" "$model_root/tests/run_fused_trace_reuse.py" --layer "$layer" > "$destination/trace_reuse_$layer.log" 2>&1
            ;;
        context)
            "$reference_python" "$model_root/tests/run_fused.py" --real --layer "$layer" --lengths 4097 131071 131072 --output "$destination/long_context_$layer.json" > "$destination/long_context_$layer.log" 2>&1
            ;;
        partial)
            for batch in 9 17 25 31; do
                "$reference_python" "$model_root/tests/run_fused.py" --real --layer "$layer" --batch "$batch" --lengths 33 --output "$destination/partial_batch_${layer}_${batch}.json" > "$destination/partial_batch_${layer}_${batch}.log" 2>&1
            done
            ;;
        watcher)
            for batch in 9 32; do
                watcher_dir="$destination/watcher_${layer}_b${batch}"
                mkdir -p "$watcher_dir"
                TT_METAL_WATCHER=10 TT_METAL_LOGS_PATH="$PWD/$watcher_dir" "$reference_python" "$model_root/tests/run_fused.py" --real --layer "$layer" --batch "$batch" --lengths 33 --output "$watcher_dir/results.json" > "$watcher_dir/console.log" 2>&1
                cp "$watcher_dir/generated/watcher/watcher.log" "$watcher_dir/watcher.log"
            done
            ;;
        *) echo "Use boundaries, correctness, context, partial, or watcher" >&2; exit 2 ;;
    esac
done
