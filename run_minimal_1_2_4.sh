#!/bin/bash

for prompt in 16 32 64 128 256 512 1024; do
  for worker in 1 2 4; do
    echo "=== Running: prompt=${prompt}, worker=${worker} ==="
    TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_SYNC=1 \
    python3 -m tracy -v -r -p -n example -m \
      "pytest ./tests/nightly/tg/ccl/test_minimal_all_gather_async.py::test_all_gather_async \
      -k \"sd35_prompt_${prompt} and ${worker}worker and perf and normal\""
  done
done