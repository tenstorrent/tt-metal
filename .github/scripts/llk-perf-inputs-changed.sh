#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Read changed paths on stdin and print true if any of them can move an LLK perf
# number on Wormhole or Blackhole, else false. The patterns follow the LLK filters
# of utils/find-changed-files.sh; Quasar-only paths do not count.
#
# Usage: git diff --name-only --diff-filter=ACMRT <base> <head> | llk-perf-inputs-changed.sh
set -euo pipefail

TESTS=tt_metal/tt-llk/tests

# Read from this script's own location, not the caller's directory, so the answer does
# not depend on where it was run from.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

# A perf_*.py may import from a functional test, and then editing that test does move a
# perf number. Ask the tree which ones, rather than keeping a list that goes stale: it
# already had, with eight of the eleven imported tests classified as harmless.
imported_by_a_perf_test() {
    local module
    module=$(basename "$1" .py)
    grep -qsE "^[[:space:]]*(from|import)[[:space:]]+${module}\b" "$ROOT/$TESTS"/python_tests/perf_*.py
}

CHANGED=false
while IFS= read -r FILE; do
    case "$FILE" in
        # Functional tests: only the ones a perf_*.py imports from can move a number.
        $TESTS/python_tests/test_*.py)
            if imported_by_a_perf_test "$FILE"; then CHANGED=true; fi
            ;;
        # Quasar, ttsim, accuracy sweeps (deselected by "not accuracy"), host-only tests,
        # and docs. None of these move a Wormhole or Blackhole perf number.
        tt_metal/tt-llk/tt_llk_quasar/**|tt_metal/hw/ckernels/quasar/**|\
        $TESTS/python_tests/accuracy/**|\
        $TESTS/hardware_free/**|$TESTS/run_hardware_free.sh|\
        $TESTS/run_quasar_regression.sh|$TESTS/run_ttsim_regression.sh|$TESTS/render_ttsim_report.py|\
        *.md)
            ;;
        # A "quasar" directory anywhere under tests/: sources/quasar, python_tests/quasar,
        # python_tests/fuser/quasar and python_tests/fuser/tests/quasar. A case pattern's "*"
        # matches "/" as well, so this one covers every depth.
        $TESTS/*/quasar/**)
            ;;
        # Quasar-only helpers, which only Quasar sources compile: sfpu_operations_quasar.h,
        # quasar_test_common.h, memory.quasar.ld. The shared helpers under python_tests/helpers
        # name them only in comments, so no Wormhole or Blackhole ELF changes with them.
        $TESTS/helpers/*/*quasar*)
            ;;
        tt_metal/sfpi-info.sh|tt_metal/sfpi-version|\
        tt_metal/tt-llk/tests/requirements.txt|\
        tt_metal/tt-llk/tt_llk_wormhole_b0/**|tt_metal/hw/ckernels/wormhole_b0/**|\
        tt_metal/tt-llk/tt_llk_blackhole/**|tt_metal/hw/ckernels/blackhole/**|\
        tt_metal/tt-llk/common/**|tt_metal/tt-llk/tests/**|\
        tt_metal/hw/inc/internal/tt-1xx/wormhole/**|tt_metal/hw/inc/internal/tt-1xx/blackhole/**|\
        tt_metal/hw/inc/internal/risc_attribs.h|\
        tt_metal/tt-llk/perf/**|\
        .github/workflows/llk-perf-impl.yaml|.github/workflows/llk-perf-gate-impl.yaml|.github/scripts/llk-perf-*.sh|\
        tests/pipeline_reorg/llk_perf_merge_gate_tests.yaml)
            CHANGED=true
            ;;
    esac
done
echo "$CHANGED"
