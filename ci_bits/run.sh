#!/usr/bin/env bash
# #59149 headroom round, scratch: functional runs of the untilize callers the face-pair passes reach, LLK asserts on.
set +e +o pipefail
cd $(git rev-parse --show-toplevel)
echo "== head $(git rev-parse HEAD) $(hostname)"
export TT_METAL_LLK_ASSERTS=1
timeout -s INT -k 60 5400 python ci_bits/run_ids.py 2>&1 | grep -aE "^(PASSED|FAILED|ERROR)|passed|failed|error" | tail -80
exit 0
