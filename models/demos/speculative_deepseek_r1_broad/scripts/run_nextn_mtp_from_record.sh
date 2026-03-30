#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Short launcher from repo root (avoids typing python_env/... path).
# Usage: ./models/demos/speculative_deepseek_r1_broad/scripts/run_nextn_mtp_from_record.sh [--quiet] [other args...]

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PY="${ROOT}/python_env/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "Expected venv at $PY — use: python /path/to/run_nextn_mtp_from_record_cpu.py" >&2
  exit 1
fi
exec "$PY" "${ROOT}/models/demos/speculative_deepseek_r1_broad/scripts/run_nextn_mtp_from_record_cpu.py" "$@"
