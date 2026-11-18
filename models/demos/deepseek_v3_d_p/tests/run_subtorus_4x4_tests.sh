#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
# Run the 4x4 [RING, RING] test rows on a 16-chip sub-torus of one BH Galaxy, in CI.
#
# The rows skip on the unnarrowed 8x4 mesh. This script narrows the Galaxy with the AttnRes harness's
# SUBTORUS_4X4_ENV (TT_VISIBLE_DEVICES + the xy4 graph descriptor), runs pytest on the given paths with
# the rows' -k selector, and fails unless pytest ran at least one test (exactly N with --expect N) and
# reported no skipped, failed or errored test. A row that skips would otherwise turn a CI leg green
# without running anything.
#
# Usage, from TT_METAL_HOME:
#   models/demos/deepseek_v3_d_p/tests/run_subtorus_4x4_tests.sh [--expect N] [--timeout-s S] <pytest paths and options>
set -uo pipefail

# Row ids and test names of the 4x4 [RING, RING] rows: torus-xy-4x4-*, 4x4-torus-xy-*, SP4xTP4-torus-xy-4x4,
# fabric2d-torus-xy-4x4-2link, torusxy-mesh-4x4-sp1, the *xy4* rows and the ring_ring_4x4 test.
ROWS_SELECTOR="(4x4 and torus-xy) or torusxy or xy4 or ring_ring_4x4"

expect=""
timeout_s=1500
while [[ $# -gt 0 ]]; do
  case "$1" in
    --expect) expect="$2"; shift 2 ;;
    --timeout-s) timeout_s="$2"; shift 2 ;;
    *) break ;;
  esac
done

env_exports=$(python3 -c "
from models.demos.deepseek_v3_d_p.tests.attn_res.model.harness import SUBTORUS_4X4_ENV
print(*(f'export {k}={v}' for k, v in SUBTORUS_4X4_ENV.items()), sep=chr(10))") || exit 1
eval "$env_exports"

log=$(mktemp /tmp/subtorus_4x4.XXXXXX.log)
timeout -k 1m "${timeout_s}s" python3 -m pytest "$@" -k "$ROWS_SELECTOR" -vvv --tb=short -rs --timeout=$((timeout_s - 60)) 2>&1 | tee "$log"
rc=${PIPESTATUS[0]}
if [[ $rc -ne 0 ]]; then
  echo "run_subtorus_4x4_tests: pytest exited $rc" >&2
  exit "$rc"
fi

summary=$(grep -E "^=+ .*(passed|skipped|failed|error|no tests ran).* in [0-9.]+s" "$log" | tail -1)
passed=$(grep -oE "[0-9]+ passed" <<<"$summary" | grep -oE "[0-9]+" || true)
if [[ -z "$passed" ]] || ((passed < 1)) || [[ -n "$expect" && "$passed" != "$expect" ]] ||
  grep -qE "[0-9]+ (skipped|failed|errors?)" <<<"$summary"; then
  echo "run_subtorus_4x4_tests: expected ${expect:-at least 1} passed and no skipped/failed/error, got: ${summary:-no summary line}" >&2
  exit 1
fi
echo "run_subtorus_4x4_tests: $summary"
