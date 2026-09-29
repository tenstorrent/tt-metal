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

CHANGED=false
while IFS= read -r FILE; do
    case "$FILE" in
        tt_metal/tt-llk/tt_llk_quasar/**|tt_metal/tt-llk/tests/sources/quasar/**|tt_metal/tt-llk/tests/python_tests/quasar/**|tt_metal/hw/ckernels/quasar/**)
            ;;
        tt_metal/sfpi-info.sh|tt_metal/sfpi-version|\
        tt_metal/tt-llk/.github/**|tt_metal/tt-llk/tests/requirements.txt|\
        tt_metal/tt-llk/tt_llk_wormhole_b0/**|tt_metal/hw/ckernels/wormhole_b0/**|\
        tt_metal/tt-llk/tt_llk_blackhole/**|tt_metal/hw/ckernels/blackhole/**|\
        tt_metal/tt-llk/common/**|tt_metal/tt-llk/tests/**|\
        .github/workflows/llk-*.yaml|.github/workflows/build-quasar-perf.yml|.github/scripts/llk-*.sh|\
        tests/pipeline_reorg/llk_*.yaml)
            CHANGED=true
            ;;
    esac
done
echo "$CHANGED"
