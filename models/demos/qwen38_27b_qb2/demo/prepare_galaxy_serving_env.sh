#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# This is host-only setup. It never opens or resets a device, and never installs
# into the environment used by an already running native qualification job.
QWEN_TASK_ROOT=${1:?Provide the isolated Qwen runtime directory}
QWEN_PLUGIN_ROOT="$QWEN_TASK_ROOT/vllm-plugin"
QWEN_PLUGIN_SHA=b7e4292e4193cba20abe9c7c68ce489201b2e36b
QWEN_BASE_ENV="$QWEN_TASK_ROOT/python_env"
QWEN_SERVING_ENV="$QWEN_TASK_ROOT/serving_env"
QWEN_EVAL_ENV="$QWEN_TASK_ROOT/eval_env"
test "$(git -C "$QWEN_PLUGIN_ROOT" rev-parse HEAD)" = "$QWEN_PLUGIN_SHA"
test ! -e "$QWEN_SERVING_ENV"
test ! -e "$QWEN_EVAL_ENV"
grep -qx 'relocatable = true' "$QWEN_BASE_ENV/pyvenv.cfg"
export UV_CACHE_DIR="$QWEN_TASK_ROOT/uv-cache"
export UV_PYTHON_INSTALL_DIR="$QWEN_TASK_ROOT/python-install"
export TMPDIR="$QWEN_TASK_ROOT/tmp"
mkdir -p "$TMPDIR"
export TT_METAL_HOME="$QWEN_TASK_ROOT/metal"
export PYTHONPATH="$QWEN_TASK_ROOT/metal-galaxy:$TT_METAL_HOME:$TT_METAL_HOME/tools"
export LD_LIBRARY_PATH="$QWEN_TASK_ROOT/metal-install/lib:$QWEN_TASK_ROOT/metal-build/lib:${LD_LIBRARY_PATH:-}"
cat > "$QWEN_TASK_ROOT/serving-constraints.txt" <<'EOF'
numpy==1.26.4
torch==2.11.0+cpu
transformers==5.12.1
EOF
export UV_CONSTRAINT="$QWEN_TASK_ROOT/serving-constraints.txt"

# uv marked the source environment relocatable; use separate inodes (reflinks
# where supported), never hardlinks that package installation could mutate.
cp -a --reflink=auto "$QWEN_BASE_ENV" "$QWEN_SERVING_ENV"
export VIRTUAL_ENV="$QWEN_SERVING_ENV"
export PATH="$VIRTUAL_ENV/bin:$PATH"
cd "$QWEN_PLUGIN_ROOT"
source docs/install-vllm-tt.sh
"$VIRTUAL_ENV/bin/python" -c 'import torch, transformers, ttnn, vllm; assert torch.__version__ == "2.11.0+cpu"; assert transformers.__version__ == "5.12.1"; assert vllm.__version__ == "0.26.0"; print("Serving environment imports and pinned versions passed")'
uv pip freeze --python "$QWEN_SERVING_ENV/bin/python" > "$QWEN_TASK_ROOT/serving-environment.txt"

uv venv --python "$QWEN_BASE_ENV/bin/python" "$QWEN_EVAL_ENV"
UV_TORCH_BACKEND=cpu uv pip install --python "$QWEN_EVAL_ENV/bin/python" \
    -r "$QWEN_TASK_ROOT/metal-galaxy/models/demos/qwen38_27b_qb2/tests/requirements-eval.txt"
"$QWEN_EVAL_ENV/bin/python" -c 'import httpx, datasets; from lm_eval.api.task import ConfigurableTask; print("Evaluation environment imports passed")'
uv pip freeze --python "$QWEN_EVAL_ENV/bin/python" > "$QWEN_TASK_ROOT/eval-environment.txt"
printf '%s\n' "$QWEN_PLUGIN_SHA" > "$QWEN_TASK_ROOT/serving-plugin-revision.txt"
echo "Isolated serving and evaluation environments prepared; no server started"
