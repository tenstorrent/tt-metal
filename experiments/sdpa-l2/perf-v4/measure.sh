#!/usr/bin/env bash
set -euo pipefail
# Run from the configured/built repository. Use a fresh label to preserve results.
label=${1:?provide a fresh result label}
out=experiments/sdpa-l2/perf-v4/${label}
if [[ -e "${out}.jsonl" ]]; then
    echo "Refusing to overwrite ${out}.jsonl" >&2
    exit 1
fi
# tar/cp -p can restore an older timestamp and leave a newer host object linked.
# Force this translation unit to rebuild before measuring a restored snapshot.
touch ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_program_factory.cpp
CMAKE_BUILD_PARALLEL_LEVEL=12 cmake --build build_Release --target install > "${out}-build.log" 2>&1
python_env/bin/python tests/ttnn/unit_tests/operations/sdpa/repro_sdpa_l2.py \
    --kv-lens 262144 --full --heads 10 --q-chunk 128 --k-chunks 1024 \
    --variants fp32_hifi2 --q-round-bits 6 --q-bitceil --q-prescale 1.0027 \
    --seed 1236 --record-fp32-streaming --max-l2-pct 0.5 \
    --benchmark-warmup 40 --benchmark-iters 10 \
    --label "$label" --output "${out}.jsonl" > "${out}.log" 2>&1
