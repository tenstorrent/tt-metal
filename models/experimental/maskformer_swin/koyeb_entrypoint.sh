#!/bin/bash
set -euo pipefail

echo "[INFO] Starting MaskFormer Swin TTNN Koyeb entrypoint"

if [[ "${EUID:-0}" -ne 0 ]]; then
  echo "[ERROR] This script must be run as root (required by install_dependencies.sh and build_metal.sh)"
  exit 1
fi

export DEBIAN_FRONTEND=noninteractive

echo "[INFO] Updating base system packages..."
apt-get update
apt-get install -y --no-install-recommends \
  ca-certificates \
  curl \
  git \
  python3-pip \
  python3-venv \
  jq

REPO_URL="${REPO_URL:-https://github.com/bro4all/tt-metal.git}"
BRANCH_NAME="${BRANCH_NAME:-feature/maskformer-swin-ttnn-bounty}"

echo "[INFO] Cloning tt-metal fork and MaskFormer branch..."
mkdir -p /root
cd /root
rm -rf tt-metal
git clone --branch "${BRANCH_NAME}" "${REPO_URL}" tt-metal
cd tt-metal

echo "[INFO] Installing TT-Metalium dependencies via install_dependencies.sh --docker..."
chmod +x install_dependencies.sh
./install_dependencies.sh --docker || echo "[WARN] install_dependencies.sh exited with non-zero status"

echo "[INFO] Building TT-Metal with profiler enabled..."
./build_metal.sh || echo "[WARN] build_metal.sh exited with non-zero status"

echo "[INFO] Installing Python helper dependencies for MaskFormer demo..."
pip3 install --no-cache-dir \
  transformers \
  huggingface_hub \
  safetensors \
  pillow \
  pycocotools \
  >/tmp/maskformer_pip_install.log 2>&1 || echo "[WARN] MaskFormer pip install had non-zero exit code, see /tmp/maskformer_pip_install.log"

echo "[INFO] Running MaskFormer warmup TT run (no profiler)..."
python3 -m models.experimental.maskformer_swin.runner \
  --image models/sample_data/demo.jpeg \
  --weights facebook/maskformer-swin-base-coco \
  --device wormhole_n300 \
  --tt-run \
  --tt-repeats 1 \
  --dump-perf generated/maskformer_warmup_perf.json \
  || echo "[WARN] Warmup TT run failed"

echo "[INFO] Running TT-NN profiler to generate ops perf CSV..."
python3 tools/tracy/profile_this.py \
  -n maskformer_swin \
  -c "python -m models.experimental.maskformer_swin.runner --image models/sample_data/demo.jpeg --weights facebook/maskformer-swin-base-coco --device wormhole_n300 --tt-run --tt-repeats 2 --dump-perf generated/tt_perf.json" \
  || echo "[WARN] profile_this.py exited with non-zero status"

echo "[INFO] MaskFormer Swin TTNN Koyeb entrypoint completed; keeping worker alive for artifact inspection"

# Keep the worker alive so artifacts/logs can be inspected or copied
sleep 365d
