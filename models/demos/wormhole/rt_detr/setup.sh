# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Setting up RT-DETR environment..."

mkdir -p weights
mkdir -p data/coco

# Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Model weights
WEIGHTS_URL="https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_6x_coco_from_paddle.pth"
WEIGHTS_PATH="weights/rtdetr_r50vd.pth"

if [ ! -f "$WEIGHTS_PATH" ]; then
    echo "Downloading model weights..."
    wget -q --show-progress -O "$WEIGHTS_PATH" "$WEIGHTS_URL"
    echo "Weights saved to $WEIGHTS_PATH"
else
    echo "Weights already exist at $WEIGHTS_PATH, skipping."
fi

# lyuwenyu RT-DETR repo
if [ ! -d "RT-DETR" ]; then
    echo "Cloning RT-DETR repository..."
    git clone https://github.com/lyuwenyu/RT-DETR.git
    echo "RT-DETR cloned."
else
    echo "RT-DETR repository already exists, skipping clone."
fi

if [ ! -d "RT-DETR/rtdetr_pytorch/src" ]; then
    echo "ERROR: RT-DETR/rtdetr_pytorch/src not found after clone."
    echo "       The repo structure may have changed — check manually."
    exit 1
fi
echo "Adding RT-DETR src to Python path..."
RTDETR_SRC="$SCRIPT_DIR/RT-DETR/rtdetr_pytorch"
export PYTHONPATH="$RTDETR_SRC:$PYTHONPATH"
echo "  PYTHONPATH set to include $RTDETR_SRC"


echo ""
echo "Setup complete. Quick sanity check:"
$(which python) -c "
import torch
ckpt = torch.load('$WEIGHTS_PATH', map_location='cpu')
keys = list((ckpt.get('ema', {}).get('module', ckpt.get('model', ckpt))).keys())
print(f'  checkpoint keys: {len(keys)} tensors, first: {keys[0]}')
print('  weights look valid.')
"

echo ""
echo "Next steps:"
echo ""
echo "  Run reference inference:"
echo "    python reference/pytorch_inference.py"
echo ""
echo "  COCO evaluation (optional, needs ~1GB):"
echo "    cd data/coco"
echo "    wget http://images.cocodataset.org/zips/val2017.zip && unzip val2017.zip"
echo "    wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip && unzip annotations_trainval2017.zip"
echo "    cd ../.."
echo "    python tests/evaluate_coco.py"
echo ""
