#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Run from the checkout used by the model-tier unit workflow.
cd "${TT_METAL_HOME:?Set TT_METAL_HOME to the built checkout}"
repro=tests/ttnn/integration_tests/galaxy_transfer_corruption/repro
output="generated/test_reports/galaxy-transfer-corruption/${GITHUB_RUN_ID:-local}-${GITHUB_RUN_ATTEMPT:-1}"

# Watcher changes the observed fault frequency. Match the recorded fast-dispatch
# configuration on an exclusively allocated Galaxy runner.
unset TT_METAL_WATCHER TT_METAL_WATCHER_APPEND TT_METAL_WATCHER_NOINLINE TT_METAL_WATCHER_DISABLE_ETH
unset TT_METAL_SLOW_DISPATCH_MODE TT_VISIBLE_DEVICES

# Preserve machine/run identity even when imports or device setup fail. RUNNER_NAME
# identifies the host; the hostname below may identify only the job container.
python3 - "$output" <<'PY'
import json
import os
import platform
import sys
from pathlib import Path

root = Path(sys.argv[1])
root.mkdir(parents=True, exist_ok=False)
identity = {
    "runner_name": os.environ.get("RUNNER_NAME"),
    "container_hostname": platform.node(),
    "platform": platform.platform(),
    "github_sha": os.environ.get("GITHUB_SHA"),
    "github_run_id": os.environ.get("GITHUB_RUN_ID"),
    "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
    "device_namespace": "UMD physical device ID",
    "device": 11,
    "requested_iterations": 100000,
}
(root / "runner.json").write_text(json.dumps(identity, indent=2) + "\n")
print(json.dumps(identity, indent=2), flush=True)
PY

# The launcher distinguishes numerical failures (1) from setup/runtime errors (2).
# Its first-fault captures and reports are uploaded by the existing CI workflow.
# --dry-run can be passed for a CPU-only integrity and command check.
exec python3 "$repro/run.py" --tt-metal "$PWD" \
    --device 11 --iterations 100000 --output "$output/capture" "$@"
