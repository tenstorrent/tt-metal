#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# Reproducible setup for the Qwen3-ASR TT server inside a stock tt-metal dev container.
# The dev image's /opt/venv is ephemeral (lost on container recreate), so this script
# re-establishes the extra runtime deps from the pinned requirement files in this repo.
#
# Usage (host):
#   docker exec qwen3asr-dev bash /work/models/demos/audio/qwen3_asr/server/setup_container.sh
set -e
source /opt/venv/bin/activate
REQ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Server + audio deps. NOTE: use uv / python -m pip, NOT the system `pip`, which installs
# into /usr/local and is invisible to /opt/venv.
uv pip install --python /opt/venv/bin/python3 -r "$REQ_DIR/requirements.txt"

# Qwen3-ASR processor (prompt build + mel) from PyPI, pinned, --no-deps so its own
# transformers pin cannot displace tt-metal's. reference/qwen_asr_processor.py imports the
# processing module without executing the package __init__ chain that would drag in the CPU
# modeling stack, so no site-packages surgery is needed (this used to copy the package out
# of a machine-local venv and truncate its __init__ files).
uv pip install --python /opt/venv/bin/python3 --no-deps -r "$REQ_DIR/requirements-processor.txt"

# Fail loudly here rather than at the first request.
python3 -c "
import sys
sys.path.insert(0, '$REQ_DIR/reference')
from qwen_asr_processor import load_processor_cls
load_processor_cls()
print('[setup] Qwen3ASRProcessor import OK')
"
echo "[setup] done"
