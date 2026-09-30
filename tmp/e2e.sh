#!/usr/bin/env bash
# One traced e2e LTX distilled run (seed 0, 1080p/145f, 4x8 ring). Usage: e2e.sh <tag> [ENV=VAL ...]
set -o pipefail
tag=$1; shift
mkdir -p tmp/e2e/$tag
export LTX_OUT_DIR=$PWD/tmp/e2e/$tag LTX_DUMP_LATENTS=$PWD/tmp/e2e/$tag/latents
for kv in "$@"; do export "$kv"; done
echo "E2E_TAG=$tag flags: $*"
source $PYTHON_ENV_DIR/bin/activate
pytest -sv --timeout=1500 'models/tt_dit/tests/models/ltx/test_pipeline_ltx_distilled.py::test_pipeline_distilled' -k bh_4x8sp1tp0_ring
