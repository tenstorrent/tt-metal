# SPDX-License-Identifier: Apache-2.0
"""Standalone backbone PCC check vs the torch reference. Runs on device #1.

Usage:
  python3 models/experimental/rf_detr/tests/pcc/test_backbone_pcc.py [device_id]
"""
import sys

import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.rf_detr.reference.weights import load_rf_detr_base
from models.experimental.rf_detr.tt.ttnn_backbone import TtDinoBackbone

PCC_TARGET = 0.99


def main(device_id=1):
    torch.manual_seed(0)
    ref, _cfg = load_rf_detr_base()
    ref = ref.eval()
    wb = ref.backbone[0].encoder.encoder

    # Use the saved real-image pixel_values if present, else a fixed random input.
    try:
        saved = torch.load("/home/ttuser/experiments/rf-detr/reference_outputs.pt", map_location="cpu")
        pixel_values = saved["pixel_values"].float()
    except Exception:
        pixel_values = torch.randn(1, 3, 560, 560)

    # ---- golden: full reference backbone feature maps + per-layer hidden states ----
    with torch.no_grad():
        golden_feats = wb(pixel_values)  # 4 x [1,384,40,40]
        embed = wb.embeddings(pixel_values)
        golden_hidden = {}
        h = embed
        for i, layer in enumerate(wb.encoder.layer):
            h = layer(h)
            if i in (1, 4, 7, 10):
                golden_hidden[i] = h.clone()

    device = ttnn.open_device(device_id=device_id)
    try:
        tt = TtDinoBackbone(ref, device)
        tt_hidden = tt.run_layers(embed)  # dict idx -> torch [16,101,384]
        print("=== per-layer hidden-state PCC (windowed [16,101,384]) ===")
        worst = 1.0
        for i in (1, 4, 7, 10):
            ok, val = comp_pcc(golden_hidden[i], tt_hidden[i], PCC_TARGET)
            print(f"  after layer {i:2d}: PCC={val}  {'OK' if ok else 'FAIL'}")
            worst = min(worst, float(val.split('PCC: ')[-1]) if isinstance(val, str) and 'PCC:' in val else worst)

        tt_feats = tt.feature_maps(pixel_values)
        print("=== feature-map PCC ([1,384,40,40]) ===")
        all_ok = True
        for j, (g, t) in enumerate(zip(golden_feats, tt_feats)):
            ok, val = comp_pcc(g, t, PCC_TARGET)
            print(f"  feature map {j} (stage {(2,5,8,11)[j]}): PCC={val}  {'OK' if ok else 'FAIL'}")
            all_ok = all_ok and ok
        print("RESULT:", "BACKBONE PCC OK" if all_ok else "BACKBONE PCC FAIL")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
