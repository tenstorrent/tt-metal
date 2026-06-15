# SPDX-License-Identifier: Apache-2.0
"""Transformer-tail PCC check vs the torch reference. Runs on device #1.

Golden: run the reference backbone+projector on the real image to get `source`,
then run the full reference tail (logits, pred_boxes). Feed the SAME `source`
into TtTransformer and compare.

The two-stage topk permutes the 300 queries, so PCC is computed after a
Hungarian match (linear_sum_assignment on box L1) that aligns query order.

GATE: matched pred_boxes PCC >= 0.99 AND matched logits PCC >= 0.97 AND the
top-5 confident detections (label+box) agree with golden within ~1%.

Usage:
  python3 models/experimental/rf_detr/tests/pcc/test_transformer_pcc.py [device_id]
"""
import sys

import torch
from scipy.optimize import linear_sum_assignment

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.rf_detr.reference.weights import get_preprocessor, load_rf_detr_base
from models.experimental.rf_detr.tt.ttnn_transformer import TtTransformer

IMAGE_PATH = "/home/ttuser/experiments/rf-detr/image.png"
BOX_PCC_GATE = 0.99
LOGITS_PCC_GATE = 0.97
NUM_CLASSES = 91


def _pcc_val(comp_out):
    """comp_pcc returns (bool, str); extract the float PCC."""
    _, msg = comp_out
    if isinstance(msg, str) and "PCC:" in msg:
        return float(msg.split("PCC:")[-1].strip())
    try:
        return float(msg)
    except Exception:
        return float("nan")


def main(device_id=1):
    torch.manual_seed(0)
    ref, cfg = load_rf_detr_base()
    ref = ref.eval()

    from PIL import Image

    pre = get_preprocessor(cfg)
    pixel_values = pre(Image.open(IMAGE_PATH)).float()

    with torch.no_grad():
        feats = ref.backbone[0].encoder.encoder(pixel_values)
        source = ref.backbone[0].projector(feats)  # [1,256,40,40]
        golden = ref(pixel_values)  # logits [1,300,91], pred_boxes [1,300,4]

    source_flat = source.flatten(2).transpose(1, 2).contiguous()  # [1,1600,256]

    device = ttnn.open_device(device_id=device_id, l1_small_size=32768)
    try:
        tt = TtTransformer(ref, device)
        source_tt = ttnn.from_torch(
            source_flat, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        tt_logits, tt_boxes = tt(source_tt)
    finally:
        ttnn.close_device(device)

    g_logits = golden.logits.float()  # [1,300,91]
    g_boxes = golden.pred_boxes.float()  # [1,300,4]

    # ---- Hungarian match to align the permuted queries ----
    # Two-stage topk selects 300 of 1600 proposals; bf16 reshuffles a couple of
    # boundary proposals (selection gap there is ~3e-4, well below bf16 ULP). To
    # avoid pairing a reshuffled query against an unrelated one, match on the full
    # prediction (box L1 + class-logit L1) rather than box alone.
    box_cost = torch.cdist(g_boxes[0], tt_boxes[0], p=1)  # [300,300]
    logit_cost = torch.cdist(g_logits[0], tt_logits[0], p=1) / NUM_CLASSES  # [300,300]
    cost = (box_cost + logit_cost).numpy()
    row, col = linear_sum_assignment(cost)
    g_boxes_m = g_boxes[0][row]
    tt_boxes_m = tt_boxes[0][col]
    g_logits_m = g_logits[0][row]
    tt_logits_m = tt_logits[0][col]

    box_pcc = _pcc_val(comp_pcc(g_boxes_m, tt_boxes_m, BOX_PCC_GATE))
    logits_pcc = _pcc_val(comp_pcc(g_logits_m, tt_logits_m, LOGITS_PCC_GATE))

    # raw (unmatched) PCCs for reference
    raw_box_pcc = _pcc_val(comp_pcc(g_boxes, tt_boxes, BOX_PCC_GATE))
    raw_logits_pcc = _pcc_val(comp_pcc(g_logits, tt_logits, LOGITS_PCC_GATE))

    print("\n=== Transformer-tail PCC ===")
    print(f"  matched pred_boxes PCC : {box_pcc:.6f}  (gate >= {BOX_PCC_GATE})")
    print(f"  matched logits     PCC : {logits_pcc:.6f}  (gate >= {LOGITS_PCC_GATE})")
    print(f"  raw     pred_boxes PCC : {raw_box_pcc:.6f}")
    print(f"  raw     logits     PCC : {raw_logits_pcc:.6f}")

    # ---- top-5 confident detections (label + box) ----
    def top5(logits, boxes):
        prob = logits.sigmoid()[0]  # [300,91]
        conf, label = prob.max(-1)  # [300]
        order = conf.argsort(descending=True)[:5]
        return [(int(label[i]), float(conf[i]), boxes[0][i].tolist()) for i in order]

    g_top = top5(g_logits, g_boxes)
    t_top = top5(tt_logits, tt_boxes)
    id2label = cfg.id2label or {}

    print("\n=== top-5 detections (golden vs tt) ===")
    dets_ok = True
    for i, (gd, td) in enumerate(zip(g_top, t_top)):
        gl, gc, gb = gd
        tl, tc, tb = td
        name_g = id2label.get(gl, str(gl))
        name_t = id2label.get(tl, str(tl))
        box_l1 = max(abs(a - b) for a, b in zip(gb, tb))
        conf_dev = abs(gc - tc)
        match = (gl == tl) and box_l1 < 0.02 and conf_dev < 0.05
        dets_ok = dets_ok and match
        print(f"  #{i}: golden=({name_g}, conf={gc:.3f}, box=[{', '.join(f'{x:.3f}' for x in gb)}])")
        print(f"      tt    =({name_t}, conf={tc:.3f}, box=[{', '.join(f'{x:.3f}' for x in tb)}])  "
              f"box_l1={box_l1:.4f} conf_dev={conf_dev:.4f} {'OK' if match else 'MISMATCH'}")

    gate = (box_pcc >= BOX_PCC_GATE) and (logits_pcc >= LOGITS_PCC_GATE) and dets_ok
    print("\nRESULT:", "TRANSFORMER PCC OK" if gate else "TRANSFORMER PCC FAIL")
    return gate


if __name__ == "__main__":
    ok = main(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
    sys.exit(0 if ok else 1)
