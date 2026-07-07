#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# Reproducible setup for the Qwen3-ASR TT server inside the dev container.
# The dev image's /opt/venv + a manually-copied qwen_asr processor are ephemeral
# (lost on container recreate), so this script re-establishes them.
#
# Usage (host):
#   docker exec qwen3asr-dev bash /work/models/demos/audio/qwen3_asr/server/setup_container.sh
set -e
source /opt/venv/bin/activate
SP=/opt/venv/lib/python3.10/site-packages

# 1) server + audio deps into the venv (NOTE: use uv / python -m pip, NOT the system `pip`,
#    which installs into /usr/local and is invisible to /opt/venv)
uv pip install --python /opt/venv/bin/python3 fastapi "uvicorn[standard]" python-multipart librosa requests

# 2) qwen_asr processor only (for prompt build + mel). Copy from the eval venv and
#    strip the package __init__ chain that imports the (newer-transformers) modeling code.
if [ ! -d "$SP/qwen_asr" ]; then
  cp -r /qwen3-asr-eval/venv/lib/python3.10/site-packages/qwen_asr "$SP/qwen_asr"
fi
: > "$SP/qwen_asr/__init__.py"
: > "$SP/qwen_asr/core/__init__.py"
cat > "$SP/qwen_asr/core/transformers_backend/__init__.py" <<'EOF'
from .configuration_qwen3_asr import Qwen3ASRConfig
from .processing_qwen3_asr import Qwen3ASRProcessor
EOF
echo "[setup] done"
