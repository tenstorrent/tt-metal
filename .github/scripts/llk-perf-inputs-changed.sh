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
CHANGED=false
while IFS= read -r FILE; do
    case "$FILE" in
        # Functional tests, except the ones a perf_*.py imports from.
        $TESTS/python_tests/test_pack.py|$TESTS/python_tests/test_eltwise_unary_datacopy.py)
            CHANGED=true
            ;;
        # Functional tests, Quasar, ttsim, accuracy sweeps (deselected by "not accuracy") and docs.
        $TESTS/python_tests/test_*.py|\
        tt_metal/tt-llk/tt_llk_quasar/**|$TESTS/sources/quasar/**|$TESTS/python_tests/quasar/**|tt_metal/hw/ckernels/quasar/**|\
        $TESTS/python_tests/accuracy/**|\
        $TESTS/run_quasar_regression.sh|$TESTS/run_ttsim_regression.sh|$TESTS/render_ttsim_report.py|\
        *.md)
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
