#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Two CHIP_ARCH passes. Gathering is @blackhole_only. The out-of-tree contract
# inherits CHIP_ARCH into its compile child, so the blackhole pass runs it too.
# Both passes always run; the exit status is the last failure.
# -c and --confcutdir keep this suite's pytest.ini when a parent checkout has its own.
# tools/python is inserted by helpers.llk_pytest_plugin. Do not export PYTHONPATH:
# the out-of-tree child copies this environment and must attach like an external consumer.

set -euo pipefail

TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$TESTS_DIR/python_tests"
export LLK_HARDWARE_FREE=1

COMMON=(
    -c pytest.ini
    --rootdir=.
    --confcutdir=.
    -p helpers.llk_pytest_plugin
    --override-ini=log_cli=false
    -q
    --tb=short
    --timeout=120
)

pytest_exit=0

export CHIP_ARCH=wormhole
pytest "${COMMON[@]}" ../hardware_free \
    --ignore=../hardware_free/kernels/test_gathering_config.py \
    --junitxml=pytest-report-hardware-free-wormhole.xml \
    || pytest_exit=$?

export CHIP_ARCH=blackhole
pytest "${COMMON[@]}" \
    ../hardware_free/kernels/test_gathering_config.py \
    ../hardware_free/harness/test_out_of_tree_contract.py \
    --junitxml=pytest-report-hardware-free-blackhole.xml \
    || pytest_exit=$?

exit "$pytest_exit"
