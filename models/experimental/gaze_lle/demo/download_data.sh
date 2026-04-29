#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Fetch everything tt-Gaze-LLE's tests/eval need:
#   1. DINOv2 ViT-B/14 backbone weights (Meta public CDN, ~346 MB)
#   2. Gaze-LLE ViT-B/14 + inout decoder checkpoint (GitHub release, ~12 MB)
#   3. GazeFollow test-set parquet (vikhyatk/gazefollow on HF, ~239 MB)
#   4. Two sample PNGs from the gazelle repo (kB-scale)
#
# Overridable via env:
#   TT_GAZE_LLE_WEIGHTS  default ./weights
#   TT_GAZE_LLE_DATA     default ./data

set -euo pipefail

WEIGHTS_DIR="${TT_GAZE_LLE_WEIGHTS:-./weights}"
DATA_DIR="${TT_GAZE_LLE_DATA:-./data}"
mkdir -p "$WEIGHTS_DIR" "$DATA_DIR/gazefollow"

have() { [[ -s "$1" ]]; }

echo "Weights dir: $WEIGHTS_DIR"
echo "Data dir:    $DATA_DIR"

# --- DINOv2 backbone
if have "$WEIGHTS_DIR/dinov2_vitb14_pretrain.pth"; then
  echo "ok  dinov2_vitb14_pretrain.pth already present"
else
  echo "... dinov2_vitb14_pretrain.pth"
  curl -fL -o "$WEIGHTS_DIR/dinov2_vitb14_pretrain.pth" \
    "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth"
fi

# --- Gaze-LLE decoder checkpoint (with inout head)
if have "$WEIGHTS_DIR/gazelle_dinov2_vitb14_inout.pt"; then
  echo "ok  gazelle_dinov2_vitb14_inout.pt already present"
else
  echo "... gazelle_dinov2_vitb14_inout.pt"
  curl -fL -o "$WEIGHTS_DIR/gazelle_dinov2_vitb14_inout.pt" \
    "https://github.com/fkryan/gazelle/releases/download/v1.0.0/gazelle_dinov2_vitb14_inout.pt"
fi

# --- GazeFollow test parquet (4,782 images + head bboxes + multi-annotator gaze targets)
if have "$DATA_DIR/gazefollow/test.parquet"; then
  echo "ok  data/gazefollow/test.parquet already present"
else
  echo "... data/gazefollow/test.parquet"
  curl -fL -o "$DATA_DIR/gazefollow/test.parquet" \
    "https://huggingface.co/datasets/vikhyatk/gazefollow/resolve/main/data/test-00000-of-00001.parquet"
fi

# --- Sample images (for the small pretrained eval test)
for name in the_office.png succession.png; do
  if have "$DATA_DIR/$name"; then
    echo "ok  data/$name already present"
  else
    echo "... data/$name"
    curl -fL -o "$DATA_DIR/$name" \
      "https://raw.githubusercontent.com/fkryan/gazelle/main/assets/$name"
  fi
done

echo
echo "Done."
ls -lh "$WEIGHTS_DIR" "$DATA_DIR" "$DATA_DIR/gazefollow" 2>/dev/null || true
