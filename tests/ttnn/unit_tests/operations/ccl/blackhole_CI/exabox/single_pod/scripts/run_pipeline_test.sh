#!/bin/bash
# Run a 16-stage pipeline test on the single-pod cluster.
# Slow dispatch (required by the blitz_decode pipeline framework).
#
# Usage:
#   scripts/run_pipeline_test.sh <test_name> [test_file.py]
# Examples:
#   scripts/run_pipeline_test.sh test_single_pod_pipeline_fake_moe
#   scripts/run_pipeline_test.sh test_single_pod_pipeline_setup_and_decode test_single_pod_pipeline.py
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST="${1:-}" \
  TEST_FILE="${2:-test_single_pod_pipeline_fake_moe.py}" \
  EXTRA_ENV="TT_METAL_SLOW_DISPATCH_MODE=1" \
  PYTEST_TIMEOUT="600" \
  "$SCRIPT_DIR/_run_common.sh"
