#!/usr/bin/env bash
# Create the shared CPU-only eval venv (~1.8 GB) the eval scripts run in. Idempotent.
# VBench model weights (~2.5 GB: DINO, CLIP ViT-L/14 + B/32, AMT-S, MUSIQ, RAFT, ViCLIP) download on
# first use into ~/.cache/vbench, ~/.cache/clip and ~/.cache/torch/hub.
set -euo pipefail
venv=/home/smarton/fasth3/tt-metal/.venv-eval
export UV_CACHE_DIR=${UV_CACHE_DIR:-/tmp/uv-cache-fasth3} UV_HTTP_TIMEOUT=300
[[ -x $venv/bin/python ]] || uv venv -q -p 3.10 "$venv"
# uv prefers --extra-index-url, so the CPU wheel index wins for torch and pypi serves the rest.
VIRTUAL_ENV=$venv uv pip install -q --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  "torch==2.4.1+cpu" "torchvision==0.19.1+cpu" "numpy<2" "timm<=1.0.12,>=0.9" opencv-python-headless \
  decord openai-clip easydict pyyaml scipy scikit-image omegaconf tqdm pillow matplotlib scikit-learn \
  imageio imageio-ffmpeg "pyiqa==0.1.10" pytest
VIRTUAL_ENV=$venv uv pip install -q --no-deps vbench==0.1.5
grep -qx "/.venv-eval/" /home/smarton/fasth3/tt-metal/.git/info/exclude || echo "/.venv-eval/" >> /home/smarton/fasth3/tt-metal/.git/info/exclude
rm -rf "$UV_CACHE_DIR"
"$venv/bin/python" -c "import torch, vbench, cv2; print('eval venv ok, torch', torch.__version__)"
