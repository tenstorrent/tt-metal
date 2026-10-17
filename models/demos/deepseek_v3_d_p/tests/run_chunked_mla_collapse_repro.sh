#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
set -euo pipefail
REPO="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
cd "$REPO"
[[ -z "${VIRTUAL_ENV:-}" && -f python_env/bin/activate ]] && source python_env/bin/activate
export TT_METAL_HOME="${TT_METAL_HOME:-$REPO}" PYTHONPATH="${PYTHONPATH:-$REPO}"
pytest -svq models/demos/deepseek_v3_d_p/tests/test_chunked_mla_collapse.py "$@"
