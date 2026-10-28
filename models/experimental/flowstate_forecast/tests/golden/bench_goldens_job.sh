#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0
#
# The stage-A golden jobs (CPU, reference venv). Submit from the worktree root:
#   ttq run    --model flowstate --kind cpu --quiet -- bash models/experimental/flowstate_forecast/tests/golden/bench_goldens_job.sh test
#   ttq submit --model flowstate --kind cpu --mem 32G -- bash models/experimental/flowstate_forecast/tests/golden/bench_goldens_job.sh bench
# Tiers: test (small goldens + weight facts), b0 (B0 SSM goldens), bench (realistic-mix bench goldens, run once).
set -euo pipefail
tier=${1:?tier: test | b0 | bench}
here=models/experimental/flowstate_forecast/tests/golden
py=/home/andy/bringup/venvs/flowstate/bin/python
out=${FLOWSTATE_GOLDEN_OUT:-/home/andy/bringup/goldens/flowstate}
# a re-make (D3 item 6) writes into its own directory and its own copy of the manifest
manifest=${FLOWSTATE_GOLDEN_MANIFEST:-$here/MANIFEST.json}
export HF_HUB_OFFLINE=1
mkdir -p "$out"
run() { env -u PYTHONPATH -u VIRTUAL_ENV "$py" "$@"; }
if [ "$tier" = test ]; then
    run "$here/weight_facts.py" --out "$out/weight_facts.json"
fi
run "$here/make_goldens.py" --tier "$tier" --out "$out" --manifest "$manifest"
