# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
set -e

# make sure we always run from the directory this script lives in
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Setting up RT-DETR environment..."

mkdir -p weights
mkdir -p data/coco

echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "Downloading model weights..."
if [ ! -f "weights/rtdetr_r50vd.pth" ]; then
    wget -O weights/rtdetr_r50vd.pth \
        https://github.com/lyuwenyu/RT-DETR/releases/download/v1.0/rtdetr_r50vd_6x_coco.pth
    echo "Model weights downloaded."
else
    echo "Model weights already exist."
fi

echo "Cloning RT-DETR repository..."
if [ ! -d "RT-DETR" ]; then
    git clone https://github.com/lyuwenyu/RT-DETR.git
    cd RT-DETR
    git checkout main
    cd ..
    echo "RT-DETR repository cloned."
else
    echo "RT-DETR repository already exists."
fi

echo ""
echo "Setup complete!"
echo ""
echo "To download COCO dataset (required for evaluation):"
echo "  cd data/coco"
echo "  wget http://images.cocodataset.org/zips/val2017.zip"
echo "  unzip val2017.zip"
echo "  wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
echo "  unzip annotations_trainval2017.zip"
echo ""
echo "To run demo inference:"
echo "  python demo/demo_inference.py"
echo ""