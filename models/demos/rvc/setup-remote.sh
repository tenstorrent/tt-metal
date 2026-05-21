#!/usr/bin/env bash
# RVC Remote Environment Setup for Stateless N300 Instances
#
# Rebuilds a working RVC TTNN environment on a fresh Koyeb N300 container.
# Mirrors the validated bring-up flow: install deps into /opt/venv, push
# code + configs + weights from the local repo, then smoke-test imports.
#
# Usage (run from local machine, from anywhere in the repo):
#   bash models/demos/rvc/setup-remote.sh <ssh_port>
#
# The Koyeb TCP-proxy port changes on every redeploy, so it is a required
# argument — there is intentionally no hardcoded default to go stale.
# Optional overrides via env:
#   REMOTE   (default root@01.proxy.koyeb.app)
#   SSH_KEY  (default: ssh's default identity; matches the deployment's PUBLIC_KEY)
#
# Designed to be idempotent — safe to re-run after a container restart.

set -euo pipefail

PORT="${1:?Usage: bash setup-remote.sh <ssh_port>   (current Koyeb TCP-proxy port)}"
REMOTE="${REMOTE:-root@01.proxy.koyeb.app}"
KEY_OPT=""
if [ -n "${SSH_KEY:-}" ]; then KEY_OPT="-i $SSH_KEY"; fi

# Koyeb's TCP proxy presents a fresh host key on every redeploy, so strict
# host-key checking is off by default (otherwise each rebuild would fail with
# a "host key changed" error). In a trusted, stable environment, set
# SSH_STRICT=accept-new (or yes) to opt back into verification.
: "${SSH_STRICT:=no}"

SSH="ssh -o StrictHostKeyChecking=$SSH_STRICT $KEY_OPT -p $PORT $REMOTE"
SCP="scp -q -o StrictHostKeyChecking=$SSH_STRICT $KEY_OPT -P $PORT"
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
RVC="$REPO_ROOT/models/demos/rvc"
REMOTE_ROOT="/root/rvc_bringup"
RR="$REMOTE_ROOT/models/demos/rvc"
PY="/opt/venv/bin/python3"

echo "=== RVC Remote Setup ==="
echo "Local repo: $REPO_ROOT"
echo "Remote:     $REMOTE:$PORT -> $REMOTE_ROOT"
echo ""

# ---- Step 1: Python dependencies into /opt/venv ----
# The image's /opt/venv ships ttnn but no pip and no torch; bootstrap pip
# with ensurepip, then install the CPU torch stack + RVC runtime deps.
echo "[1/5] Installing Python dependencies into /opt/venv ..."
$SSH "
set -e
$PY -m pip --version >/dev/null 2>&1 || $PY -m ensurepip --upgrade
$PY -m pip install --quiet 'torch>=2.1,<2.7' --index-url https://download.pytorch.org/whl/cpu
$PY -m pip install --quiet safetensors numpy scipy soundfile pyworld av librosa faiss-cpu pytest
echo 'Deps installed OK'
"

# ---- Step 2: Workspace structure ----
echo "[2/5] Creating workspace structure ..."
$SSH "
mkdir -p $RR/ttnn/ops $RR/torch_impl/vc $RR/tests $RR/utils \
         $RR/data/assets/pretrained_v2 $RR/data/assets/rmvpe \
         $RR/data/configs $RR/data/speech
touch $REMOTE_ROOT/models/__init__.py \
      $REMOTE_ROOT/models/demos/__init__.py \
      $RR/__init__.py
"

# ---- Step 3: Deploy code ----
echo "[3/5] Deploying RVC code ..."
$SCP "$RVC/ttnn/runtime.py" "$RVC/ttnn/utils.py" "$RVC/ttnn/__init__.py" \
  "$REMOTE:$RR/ttnn/"
$SCP "$RVC/ttnn/ops/conv_transpose1d.py" "$RVC/ttnn/ops/__init__.py" \
  "$REMOTE:$RR/ttnn/ops/"
$SCP "$RVC/torch_impl/reference.py" "$RVC/torch_impl/rmvpe.py" \
  "$RVC/torch_impl/crepe.py" "$RVC/torch_impl/__init__.py" \
  "$REMOTE:$RR/torch_impl/"
$SCP "$RVC/torch_impl/vc/synthesizer.py" "$RVC/torch_impl/vc/pipeline.py" \
  "$RVC/torch_impl/vc/hubert.py" "$RVC/torch_impl/vc/__init__.py" \
  "$REMOTE:$RR/torch_impl/vc/"
$SCP "$RVC/tests/test_ttnn_ops.py" "$RVC/tests/test_runtime.py" \
  "$RVC/tests/test_production_shapes.py" "$RVC/tests/conftest.py" \
  "$RVC/tests/pcc_utils.py" "$RVC/tests/__init__.py" \
  "$REMOTE:$RR/tests/"
$SCP "$RVC/utils/audio.py" "$RVC/utils/config.py" "$RVC/utils/f0.py" \
  "$RVC/utils/__init__.py" \
  "$REMOTE:$RR/utils/"
$SCP "$RVC/demo.py" "$RVC/profile.py" "$RVC/evaluate.py" "$RVC/benchmark.py" \
  "$REMOTE:$RR/"

# ---- Step 4: Deploy configs + weights + a speech sample ----
# Weights are uploaded from the local repo (already converted to safetensors)
# rather than re-downloaded, so the layout exactly matches what the code reads.
echo "[4/5] Deploying configs, weights, and a speech sample ..."
$SCP -r "$RVC/data/configs/." "$REMOTE:$RR/data/configs/"
$SCP "$RVC/data/speech/sample-speech-0.wav" "$REMOTE:$RR/data/speech/"
$SCP "$RVC/data/assets/pretrained_v2/f0G48k.safetensors" \
  "$REMOTE:$RR/data/assets/pretrained_v2/"
$SCP "$RVC/data/assets/hubert.safetensors" "$REMOTE:$RR/data/assets/"
$SCP "$RVC/data/assets/rmvpe/rmvpe.safetensors" "$REMOTE:$RR/data/assets/rmvpe/"
# torch_impl/rmvpe.py loads the pitch model from data/rmvpe.safetensors
# (NOT data/assets/rmvpe/...), so place a copy at that path too.
$SSH "cp $RR/data/assets/rmvpe/rmvpe.safetensors $RR/data/rmvpe.safetensors"

# ---- Step 5: Smoke test ----
echo "[5/5] Smoke test ..."
$SSH "
export PYTHONPATH=$REMOTE_ROOT:\$PYTHONPATH
cd $REMOTE_ROOT
$PY -c '
import torch, ttnn
from models.demos.rvc.ttnn.runtime import TTNNFlowDecoder, TTNNGeneratorNSF
print(\"torch:\", torch.__version__)
print(\"imports OK\")
'
"

echo ""
echo "=== Setup complete ==="
echo "SSH:   ssh -p $PORT $REMOTE"
echo "Tests: PYTHONPATH=$REMOTE_ROOT $PY -m pytest $RR/tests/test_runtime.py"
echo "       (run each test file as a SEPARATE invocation — co-execution segfaults)"
echo "Bench: PYTHONPATH=$REMOTE_ROOT $PY -m models.demos.rvc.benchmark --max_secs 10.0"
