#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Check that a harness wheel works for a suite that installs it: on its own, in a new environment.
#
#   verify_wheel.sh WHEEL
#
# Installs WHEEL into a new virtual environment with nothing but what it requires, at the versions of
# tests/requirements.txt, and runs the tests/out_of_tree fixture on the installed copy for Wormhole and Blackhole.
# The fixture compiles, so the SFPI toolchain must be installed (tests/install_sfpi.sh, or LLK_SFPI_ROOT) and this
# must be a tt-metal checkout. The release workflow runs this before it publishes.

set -euo pipefail

if [[ $# -ne 1 || ! -f "$1" ]]; then
    echo "Usage: $0 WHEEL" >&2
    exit 1
fi

WHEEL="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

python3 -m venv "$WORK/venv"
PYTHON="$WORK/venv/bin/python"

# As constraints, the file pins what the wheel pulls in and installs nothing else: a requirement the wheel does not
# declare is missing here, where the test environment of tt-llk would have had it anyway.
"$PYTHON" -m pip install --quiet --disable-pip-version-check -c "$TESTS_DIR/requirements.txt" "$WHEEL"
"$PYTHON" -m pip check

SITE="$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"

cd "$TESTS_DIR/out_of_tree"
for arch in wormhole blackhole; do
    artefacts="$WORK/$arch"
    mkdir "$artefacts"
    echo "== $arch"
    # PYTHONPATH and PYTEST_ADDOPTS of the caller would bring the checkout's harness or its options back in.
    env -u PYTHONPATH -u PYTEST_ADDOPTS \
        LLK_HARNESS_SITE="$SITE" CHIP_ARCH="$arch" LLK_HARDWARE_FREE=1 \
        RUNNER_TEMP="$artefacts" TMPDIR="$artefacts" \
        "$PYTHON" -m pytest --compile-producer -q --override-ini=addopts= -p no:cacheprovider .
    # The compile is the one test that skips without failing (no tt-metal hw/inc beside tt-llk).
    if [[ -z "$(find "$artefacts" -name 'unpack.elf' -print -quit)" ]]; then
        echo "ERROR: the fixture passed without compiling its driver for $arch" >&2
        exit 1
    fi
done

echo "Verified: $(basename "$WHEEL") installs on its own and builds the out-of-tree fixture."
