#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Run from the checkout used by the model-tier unit workflow.
cd "${TT_METAL_HOME:?Set TT_METAL_HOME to the built checkout}"
sweep=tests/ttnn/integration_tests/galaxy_transfer_corruption/run_sweep.py
output="generated/test_reports/galaxy-transfer-corruption/${GITHUB_RUN_ID:-local}-${GITHUB_RUN_ATTEMPT:-1}"

# Watcher changes the observed fault frequency. Match the recorded fast-dispatch
# configuration on an exclusively allocated Galaxy runner.
unset TT_METAL_WATCHER TT_METAL_WATCHER_APPEND TT_METAL_WATCHER_NOINLINE TT_METAL_WATCHER_DISABLE_ETH
unset TT_METAL_SLOW_DISPATCH_MODE TT_VISIBLE_DEVICES

# Sweep physical devices 0-31 sequentially, retaining a separate capture for each.
# Failures are accumulated so one bad device cannot skip the rest of the Galaxy.
# --dry-run verifies all 32 launch commands without importing Torch or TTNN.
exec python3 "$sweep" --tt-metal "$PWD" --iterations 100000 --output "$output" "$@"
