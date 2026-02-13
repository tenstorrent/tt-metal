#!/bin/bash
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Run unit tests for ttnn.pad sub_core_grids support
set -e

echo "================================================================"
echo "  Running ttnn.pad sub_core_grids unit tests"
echo "================================================================"

# Run the new sub_core_grids tests
pytest tests/ttnn/unit_tests/operations/data_movement/test_pad_subcoregrids.py -v

echo ""
echo "================================================================"
echo "  Running existing ttnn.pad tests (regression check)"
echo "================================================================"

# Run the existing pad tests to verify no regressions
pytest tests/ttnn/unit_tests/operations/data_movement/test_pad.py -v

echo ""
echo "================================================================"
echo "  All pad tests passed!"
echo "================================================================"
