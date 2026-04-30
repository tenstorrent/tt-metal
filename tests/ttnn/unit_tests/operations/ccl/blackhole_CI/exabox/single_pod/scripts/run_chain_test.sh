#!/bin/bash
# Run a CCL chain test from test_fake_moe_traffic.py on the single-pod cluster.
# Fast dispatch (sub_device_manager required by ttnn.broadcast / a2a / etc).
#
# Usage:
#   scripts/run_chain_test.sh <test_name>
# Example:
#   scripts/run_chain_test.sh test_fake_moe_chain_4x2_single_pod
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST="${1:-}" \
  TEST_FILE="test_fake_moe_traffic.py" \
  EXTRA_ENV="" \
  PYTEST_TIMEOUT="240" \
  "$SCRIPT_DIR/_run_common.sh"
