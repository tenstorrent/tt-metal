# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
set -e

echo "Setting up RT-DETR environment..."

# Create directories
mkdir -p weights
mkdir -p data/coco

echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "Downloading model weights..."
cd weights
if [ ! -f "rtdetr_r50vd_6x_coco.pth" ]; then
    wget https://github.com/lyuwenyu/RT-DETR/releases/download/v1.0/rtdetr_r50vd_6x_coco.pth
    echo "Model weights downloaded."
else
    echo "Model weights already exist."
fi
cd ..

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
