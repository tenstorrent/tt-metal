#!/usr/bin/env bash
# Run the LTX-2.3 distilled AV pipeline on the tt-device-mcp broker with kernel
# JIT warmed OFF-device first, so the reserved device window holds device work
# only (tt_metal/tools/kernel_prewarm). Defaults to the validated 720p config:
# 704x1280 (nearest %64 to 720; 1280 = 20*64) on a 4x8 Blackhole Galaxy, traced.
#
# Checkpoint/cache paths are read from an env yaml (ENV_YAML) so no secrets live
# in the tree. The yaml MUST define TT_METAL_CACHE (the cache the prewarm warms
# and the run consumes), TT_METAL_HOME, LTX_CHECKPOINT, GEMMA_PATH, TT_DIT_CACHE_DIR.
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

ENV_YAML=${ENV_YAML:?set ENV_YAML to a broker env yaml (TT_METAL_CACHE, TT_METAL_HOME, LTX_CHECKPOINT, GEMMA_PATH, TT_DIT_CACHE_DIR)}
MESH=${MESH:-bh_4x8sp1tp0_ring}
TIMEOUT=${TIMEOUT:-5400}

exec tt_metal/tools/kernel_prewarm/prewarm_and_submit.sh -e "$ENV_YAML" -w "$(dirname "$PWD")" -t "$TIMEOUT" -- \
  "cd $PWD && python_env/bin/python -m pytest \
     models/tt_dit/tests/models/ltx/test_pipeline_ltx_distilled.py::test_pipeline_distilled \
     -k $MESH -s --timeout $TIMEOUT"
