#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Two CHIP_ARCH passes: gathering is @blackhole_only and that skip reads the variable.
# -c and --confcutdir keep this suite's pytest.ini when a parent checkout has its own.

set -euo pipefail

TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$TESTS_DIR/python_tests"
export PYTHONPATH="${TESTS_DIR}/../tools/python${PYTHONPATH:+:$PYTHONPATH}"
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

export CHIP_ARCH=wormhole
pytest "${COMMON[@]}" ../hardware_free \
    --ignore=../hardware_free/kernels/test_gathering_config.py \
    --junitxml=pytest-report-hardware-free-wormhole.xml

export CHIP_ARCH=blackhole
pytest "${COMMON[@]}" ../hardware_free/kernels/test_gathering_config.py \
    --junitxml=pytest-report-hardware-free-blackhole.xml
