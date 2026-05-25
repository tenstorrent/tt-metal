#!/bin/bash
# Thin wrapper around scripts/run_op_sweep.sh for the SDPA sweep.
# See `scripts/run_op_sweep.sh --help` for the full option list.
exec "$(dirname "${BASH_SOURCE[0]}")/run_op_sweep.sh" \
    tests/ttnn/unit_tests/operations/sdpa/test_sdpa_sweep.py "$@"
