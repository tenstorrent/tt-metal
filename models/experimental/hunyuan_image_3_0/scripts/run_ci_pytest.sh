#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Robust CI pytest wrapper for HunyuanImage-3.0 pipeline entries.
#
#   run_ci_pytest.sh unit  --timeout N ... pytest args...
#   run_ci_pytest.sh e2e   --timeout N ... pytest args...
#
# - Forces offline / no-download mode (HF_HUB_OFFLINE + HY_SKIP_WEIGHT_DOWNLOAD)
# - Validates HUNYUAN_MODEL_DIR before pytest (hard-fail on wrong/missing checkpoint)
# - e2e mode adds --maxfail=1 (fail-fast)
# - Maps pytest exit 5 (no tests collected) to exit 1

set -euo pipefail

MODE="${1:?usage: run_ci_pytest.sh unit|e2e -- pytest args...}"
shift

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HY_SKIP_WEIGHT_DOWNLOAD="${HY_SKIP_WEIGHT_DOWNLOAD:-1}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT"

python3 - <<'PY'
from models.experimental.hunyuan_image_3_0.ref.weights import ENV_BASE, HF_REPO_BASE, validate_env_checkpoint_dir

validate_env_checkpoint_dir(ENV_BASE, HF_REPO_BASE)
print(f"[ci] checkpoint ok: {ENV_BASE} -> {__import__('os').environ[ENV_BASE]}", flush=True)
PY

EXTRA=()
case "$MODE" in
  unit) ;;
  e2e) EXTRA=(--maxfail=1) ;;
  *)
    echo "::error::run_ci_pytest.sh: unknown mode ${MODE} (expected unit or e2e)"
    exit 2
    ;;
esac

set +e
pytest "${EXTRA[@]}" "$@"
RC=$?
set -e

if [[ $RC -eq 5 ]]; then
  echo "::error::No tests collected (bad path, marker filter, or typo)"
  exit 1
fi
exit "$RC"
