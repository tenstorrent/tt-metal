#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
QWEN_TASK_ROOT=${1:?Provide the isolated Qwen runtime directory}
QWEN_PROFILE_DIR=${2:?Provide a new profile directory}
if [[ -n "${QWEN_WAIT_FOR_UNIT:-}" ]]; then
    echo "Waiting for $QWEN_WAIT_FOR_UNIT before profiling"
    while true; do
        QWEN_PREVIOUS_STATE=$(systemctl --user show "$QWEN_WAIT_FOR_UNIT" -p ActiveState --value)
        case "$QWEN_PREVIOUS_STATE" in
            inactive|failed) break ;;
            active|activating|deactivating) sleep 5 ;;
            *) echo "Unrecognized predecessor state: $QWEN_PREVIOUS_STATE" >&2; exit 3 ;;
        esac
    done
fi
mkdir "$QWEN_PROFILE_DIR"
export PATH="$QWEN_TASK_ROOT/python_env/bin:$PATH"
export TT_METAL_HOME="$QWEN_TASK_ROOT/metal"
export PYTHONPATH="$QWEN_TASK_ROOT/metal-galaxy:$TT_METAL_HOME:$TT_METAL_HOME/tools"
export LD_LIBRARY_PATH="$QWEN_TASK_ROOT/metal-install/lib:$QWEN_TASK_ROOT/metal-build/lib:${LD_LIBRARY_PATH:-}"
export TT_METAL_CACHE="$QWEN_TASK_ROOT/jit-cache-metal-galaxy"
export MODEL_WEIGHTS_DIR=${MODEL_WEIGHTS_DIR:?Set the pinned checkpoint directory}
export ARCH_NAME=blackhole OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
export QWEN_GALAXY_LAYER_PROFILE=1 QWEN_PROFILE_RECEIPT="$QWEN_PROFILE_DIR/profile.json"
export TT_METAL_PROFILER_DIR="$QWEN_PROFILE_DIR/tracy" TRACY_NO_WEB_SERVER=1
export QWEN_COMPACT_DECODE_RESIDUAL=1
unset TT_METAL_SLOW_DISPATCH_MODE TT_METAL_ALLOCATOR_MODE_HYBRID
unset TT_METAL_DEVICE_PROFILER TT_METAL_PROFILER_MID_RUN_DUMP TT_METAL_PROFILER_CPP_POST_PROCESS
unset QWEN_DECODE_BUCKETS QWEN_COMPACT_DECODE_MLP QWEN_BATCHED_DECODE_ROPE QWEN_COMPACT_DECODE_ATTENTION
unset QWEN_BATCHED_PREFILL QWEN_PREFILL_RESIDUAL_LAYOUT QWEN_PREFILL_BATCHED_HEAD
unset QWEN_PREFILL_SKIP_INTERMEDIATE_HEAD QWEN_PREFILL_STARTUP_WARMUP
# The Metal wrapper supports Tracy and holds the same cooperative device lock.
/bin/bash "$QWEN_TASK_ROOT/metal-galaxy/scripts/run_safe_pytest.sh" --profile-ops \
    "$QWEN_TASK_ROOT/metal-galaxy/models/demos/qwen38_27b_qb2/tests/test_galaxy_layer_profile.py" \
    -vv -s --timeout=3600 --junitxml="$QWEN_PROFILE_DIR/hardware.xml"
# Tracy can mask pytest's exit status. Require the actual test receipt and JUnit
# success before reporting success for the persistent job.
python - "$QWEN_PROFILE_DIR" <<'PY'
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from models.demos.qwen38_27b_qb2.tests.layer_profile_report import PROFILE_CASES

root = Path(sys.argv[1])
receipt = json.loads((root / "profile.json").read_text())
assert receipt["passed"] is True and receipt["state"] == "completed", receipt
assert [(cell["input_tokens"], cell["batch"]) for cell in receipt["cells"]] == PROFILE_CASES
suites = ET.parse(root / "hardware.xml").getroot().findall(".//testsuite")
assert sum(int(suite.get("tests", 0)) for suite in suites) == 1
assert all(int(suite.get(field, 0)) == 0 for suite in suites for field in ("failures", "errors", "skipped"))
reports = list((root / "tracy").rglob("ops_perf_results*.csv"))
assert reports, "Tracy produced no per-op report"
assert len(reports) == 1, "Ambiguous profiler reports; select the diagnostic CSV explicitly"
print("PROFILE_REPORTS", *reports, sep="\n", flush=True)
subprocess.run([
    sys.executable, "-m", "models.demos.qwen38_27b_qb2.tests.layer_profile_report",
    "--csv", str(reports[0]), "--receipt", str(root / "profile.json"),
    "--output", str(root / "analysis"),
], check=True)
analysis = json.loads((root / "analysis/profile-summary.json").read_text())
assert analysis["measurements_complete"] is True, "Diagnostic windows lost device timings"
PY
