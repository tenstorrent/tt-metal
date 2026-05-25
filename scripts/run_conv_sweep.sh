#!/bin/bash
# Thin wrapper around scripts/run_op_sweep.sh for the conv2d sweep.
# See `scripts/run_op_sweep.sh --help` for the full option list.
exec "$(dirname "${BASH_SOURCE[0]}")/run_op_sweep.sh" \
    tests/ttnn/unit_tests/operations/conv/test_conv2d_sweep.py "$@"
