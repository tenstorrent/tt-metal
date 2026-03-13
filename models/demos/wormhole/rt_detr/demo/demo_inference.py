# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import sys
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw

import ttnn

sys.path.insert(0, str(Path(__file__).parent.parent))

repo_path = str(Path(__file__).parent.parent / "RT-DETR" / "rtdetr_pytorch")
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)

from src.core import YAMLConfig
from tt.rtdetr_encoder import run_encoder
from tt.weight_utils import get_tt_parameters

SCORE_THRESH = 0.3
TOP_K = 100


def load_image(path, size=(640, 640)):
    img = Image.open(path).convert("RGB")
    tf = T.Compose([T.Resize(size), T.ToTensor()])
    return tf(img).unsqueeze(0), img


def draw_boxes(img, scores, labels, boxes, thresh=SCORE_THRESH):
    draw = ImageDraw.Draw(img)
    count = 0
    for score, label, box in zip(scores, labels, boxes):
        if score < thresh:
            continue
        cx, cy, w, h = box
        x1 = (cx - w / 2) * img.width
        y1 = (cy - h / 2) * img.height
        x2 = (cx + w / 2) * img.width
        y2 = (cy + h / 2) * img.height
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, y1), f"cls {int(label)}: {score:.2f}", fill="red")
        count += 1
    return count


def main():
    config_path = "RT-DETR/rtdetr_pytorch/configs/rtdetr/rtdetr_r50vd_6x_coco.yml"
    ckpt_path = "weights/rtdetr_r50vd.pth"
    img_path = "demo/sample.jpg"
    out_path = "demo/output.jpg"

    device = ttnn.open_device(device_id=0, l1_small_size=16384)
    try:
        print("loading model...")
        cfg = YAMLConfig(config_path)
        model = cfg.model
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["ema"]["module"])
        model.eval()

        print("pushing encoder weights to device...")
        tt_params = get_tt_parameters(device, model)

        def ttnn_aifi_forward(src, pos_embed=None):
            res = run_encoder(src, tt_params.layers, device, pos_embed=pos_embed).float()
            return res.squeeze(1)

        model.encoder.encoder[0].forward = ttnn_aifi_forward

        model.encoder.encoder[0].forward = ttnn_aifi_forward

        model.encoder.encoder[0].forward = ttnn_aifi_forward

        img_tensor, img_pil = load_image(img_path)

        print("running inference...")
        with torch.no_grad():
            outputs = model(img_tensor)

        logits = outputs["pred_logits"][0]
        boxes = outputs["pred_boxes"][0]

        probs = logits.sigmoid()
        topk_scores, topk_idx = torch.topk(probs.flatten(), TOP_K)
        labels = topk_idx % logits.shape[-1]
        box_idx = topk_idx // logits.shape[-1]
        top_boxes = boxes[box_idx]

        n = draw_boxes(img_pil, topk_scores, labels, top_boxes)
        img_pil.save(out_path)
        print(f"found {n} detections above {SCORE_THRESH} — saved to {out_path}")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
