#!/usr/bin/env bash
# Compare a finished e2e run to t14's safe baseline. Usage: cmp.sh <tag> <job_id>
tag=$1; job=$2
tt-device-mcp logs -n 100000 $job | grep -E "denoise init|Stage [12] denoise|E2E_TAG|PASSED|FAILED|Error" | tail -40
source ${PYTHON_ENV_DIR:-/home/smarton/fasth3/tt-metal/python_env}/bin/activate
python - <<PY
import torch
for g in (0, 1, 2):
    a = torch.load(f"tmp/e2e/$tag/latents.gen{g}.pt"); b = torch.load(f"../t14/tmp/e2e/safe/latents.gen{g}.pt")
    if isinstance(a, dict):
        for k in a: print(g, k, torch.equal(a[k], b[k]))
    else: print(g, torch.equal(a, b))
PY
