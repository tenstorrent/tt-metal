# SPDX-License-Identifier: Apache-2.0
"""Sweep backbone precision configs: report detection accuracy + latency for each."""
import statistics
import time

import torch
import ttnn
from PIL import Image

from models.experimental.rf_detr.reference.weights import load_rf_detr_base, get_preprocessor
from models.experimental.rf_detr.tt.ttnn_backbone import TtDinoBackbone
from models.experimental.rf_detr.tt.ttnn_projector import TtProjector
from models.experimental.rf_detr.tt.ttnn_transformer import TtTransformer
from models.experimental.rf_detr.tests.perf.benchmark import detection_accuracy

IMG = "/home/ttuser/experiments/rf-detr/assets/cats_000000039769.jpg"
F = ttnn.MathFidelity

CONFIGS = [
    ("bf16 / default (current)", dict(weight_dtype=ttnn.bfloat16)),
    ("bf16 / HiFi4 + fp32_acc",  dict(weight_dtype=ttnn.bfloat16, math_fidelity=F.HiFi4, fp32_acc=True)),
    ("bf16 / HiFi2 + fp32_acc",  dict(weight_dtype=ttnn.bfloat16, math_fidelity=F.HiFi2, fp32_acc=True)),
    ("bf16 / HiFi4 (no fp32acc)", dict(weight_dtype=ttnn.bfloat16, math_fidelity=F.HiFi4, fp32_acc=False)),
]


def run():
    ref, cfg = load_rf_detr_base(); ref = ref.eval()
    pre = get_preprocessor(cfg)
    pv = pre(Image.open(IMG).convert("RGB"))
    pv = pv.unsqueeze(0).float() if pv.dim() == 3 else pv.float()
    with torch.no_grad():
        g = ref(pv)

    for name, opts in CONFIGS:
        d = ttnn.open_device(device_id=1, l1_small_size=32768)
        try:
            bb = TtDinoBackbone(ref, d, **opts)
            proj = TtProjector(ref, d)
            tf = TtTransformer(ref, d)

            def fwd():
                feats = bb.feature_maps(pv)
                fcl = [ttnn.from_torch(f.flatten(2).transpose(1, 2).contiguous(),
                                       dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=d) for f in feats]
                return tf(proj(fcl))

            logits, boxes = fwd()
            acc = detection_accuracy(g.logits, g.pred_boxes, logits, boxes)
            for _ in range(2):
                fwd(); ttnn.synchronize_device(d)
            ts = []
            for _ in range(8):
                t = time.perf_counter(); fwd(); ttnn.synchronize_device(d); ts.append(time.perf_counter() - t)
            print(f"  {name:28} acc={acc:.3f}  lat={statistics.median(ts)*1000:.1f}ms  fps={1/statistics.median(ts):.2f}")
        finally:
            ttnn.close_device(d)


if __name__ == "__main__":
    run()
