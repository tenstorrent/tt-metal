#!/usr/bin/env bash
set -euo pipefail
# Run only after prior queue completes and the matching sources are rebuilt.
out=experiments/sdpa-l2/fp32-util-v1
python_env/bin/python "$out/run.py" --label fusion-v1 --batches 2 --fused --lengths 32768 131072
python_env/bin/python "$out/run.py" --label cache-v1 --batches 2 --cache-max --lengths 32768 131072
python_env/bin/python "$out/run.py" --label combined-v1 --batches 2 --cache-max --fused --lengths 32768 131072
