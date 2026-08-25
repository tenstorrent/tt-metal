#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../.." && pwd)"
test_path="tests/ttnn/unit_tests/operations/matmul/test_matmul_registry_canary.py"
output_root="${TTNN_MATMUL_REGISTRY_CANARY_OUTPUT_DIR:-${TMPDIR:-/tmp}/ttnn-matmul-registry-canary-${SLURM_JOB_ID:-$$}}"

if [[ "${1:-}" != "--inside-timeout" ]]; then
    if [[ -e "${output_root}" ]]; then
        echo "refusing to overwrite canary output: ${output_root}" >&2
        exit 2
    fi
    mkdir -p "${output_root}"
    export TTNN_MATMUL_REGISTRY_CANARY_OUTPUT_DIR="${output_root}"
    exec timeout --signal=TERM --kill-after=30s 13m "$0" --inside-timeout
fi

cd "${repo_root}"
export TTNN_MATMUL_REGISTRY_CANARY_REQUIRE_POPULATED=1

for mode_spec in off:0 shadow:1 on:2; do
    mode="${mode_spec%%:*}"
    mode_value="${mode_spec##*:}"
    overrides="$(${PYTHON:-python3} -c '
import json, os, sys
value = json.loads(os.environ.get("TTNN_CONFIG_OVERRIDES", "{}"))
value["matmul_registry_mode"] = int(sys.argv[1])
print(json.dumps(value, sort_keys=True, separators=(",", ":")))
' "${mode_value}")"
    echo "matmul registry silicon canary: mode=${mode} output=${output_root}"
    TTNN_CONFIG_OVERRIDES="${overrides}" \
    TTNN_MATMUL_REGISTRY_CANARY_MODE="${mode}" \
        "${PYTHON:-python3}" -m pytest -q "${test_path}" \
        --maxfail=1 \
        --junitxml="${output_root}/${mode}.xml"
done

"${PYTHON:-python3}" - "${output_root}" <<'PY'
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

output = Path(sys.argv[1])
lock = Path("ttnn/cpp/ttnn/operations/matmul/device/config/registry/matmul_registry.lock.json")
files = {}
for path in sorted(output.glob("*.xml")):
    files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
receipt = {
    "artifact_kind": "ttnn_matmul_registry_silicon_canary",
    "schema_version": 1,
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "mode_reports_sha256": files,
}
(output / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(receipt, sort_keys=True))
PY

echo "matmul registry silicon canary passed: ${output_root}/receipt.json"
