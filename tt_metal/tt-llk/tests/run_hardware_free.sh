#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Two CHIP_ARCH passes: gathering is @blackhole_only, the out-of-tree contract inherits CHIP_ARCH so blackhole runs it too,
# and the accuracy suite judges the table of the CHIP_ARCH it runs under, so blackhole runs it over its own table.
# -c and --confcutdir keep this suite's pytest.ini when a parent checkout has its own.
# Do not export PYTHONPATH: the plugin inserts tools/python, and the out-of-tree child copies this environment.

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
# Cap at 8: more workers lose to startup, and the longest case is already under that.
XDIST=(-n auto --maxprocesses=8 --dist=worksteal)

pytest_exit=0

export CHIP_ARCH=wormhole
pytest "${COMMON[@]}" "${XDIST[@]}" \
    ../hardware_free \
    --ignore=../hardware_free/kernels/test_gathering_config.py \
    --junitxml=pytest-report-hardware-free-wormhole.xml \
    || pytest_exit=$?

export CHIP_ARCH=blackhole
# A directory plus -k, not two files: file args outside the rootdir leave classname empty in the junit report.
pytest "${COMMON[@]}" ../hardware_free \
    -k "test_gathering_config or test_out_of_tree_contract" \
    --junitxml=pytest-report-hardware-free-blackhole.xml \
    || pytest_exit=$?
# The accuracy suite by its directory, not by -k: a renamed directory fails here rather than selecting nothing.
pytest "${COMMON[@]}" "${XDIST[@]}" \
    ../hardware_free/accuracy \
    --junitxml=pytest-report-hardware-free-blackhole-accuracy.xml \
    || pytest_exit=$?

exit "$pytest_exit"
