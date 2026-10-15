#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Host-only LLK tests. python_tests stays the pytest root so its pytest.ini
# applies; the files themselves live in hardware_free/. CHIP_ARCH is set
# because helpers call get_chip_architecture() at import, and that probes a
# card when the variable is missing. Gathering is @blackhole_only, and that
# skip reads CHIP_ARCH, so it gets a second process. Neither process talks
# to a device.

set -euo pipefail

TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$TESTS_DIR/python_tests"
# The harness plugin also puts tools/python on sys.path. Set it here as well so
# `import tt_llk_perf` does not depend on plugin import order.
export PYTHONPATH="${TESTS_DIR}/../tools/python${PYTHONPATH:+:$PYTHONPATH}"
# Skip the plugin's ttexalens init. setup_build still runs, so CHIP_ARCH,
# LLK_ROOT and the artefact paths exist for tests that compile an ELF.
export LLK_HARDWARE_FREE=1

# -c keeps this suite's pytest.ini when a parent checkout (tt-metal) has its own.
# --confcutdir stops that parent's conftest.py from loading. pythonpath in the
# ini puts this directory on sys.path so `import helpers` works for tests that
# live outside it. -p loads the harness plugin: pytest only auto-loads a
# conftest.py that is an ancestor of the test path, and hardware_free/ is a
# sibling of python_tests/.
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
