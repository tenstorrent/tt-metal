#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
# Run only in a fresh remote CI task-image container, without device mounts.
set -euo pipefail
curl -LsSf https://astral.sh/uv/0.7.13/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.12
uv tool install --python 3.12 mini-swe-agent==2.2.8 \
    --with 'litellm[mcp]==1.106.0' --with mcp==2.2.0 --with orjson --with fastapi
"$HOME/.local/share/uv/tools/mini-swe-agent/bin/python" - <<'PY'
import importlib.metadata
import litellm
from litellm.responses.mcp.chat_completions_handler import acompletion_with_mcp
from mcp.types import CacheableResult
for name in ("mini-swe-agent", "litellm", "mcp", "fastapi", "orjson", "httpx2"):
    print(f"DEPENDENCY {name}=={importlib.metadata.version(name)}", flush=True)
print("DEPENDENCY_IMPORT_PASS", flush=True)
PY
