# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import json
import sys
from pathlib import Path

import torch
import torchvision.transforms as T
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.datasets import CocoDetection
from tqdm import tqdm

import ttnn

THIS_DIR = Path(__file__).parent.resolve()
PROJECT = THIS_DIR.parent
REPO_PATH = PROJECT / "RT-DETR" / "rtdetr_pytorch"

sys.path.insert(0, str(REPO_PATH))
sys.path.insert(0, str(PROJECT))

from src.core import YAMLConfig
from tt.rtdetr_encoder import run_encoder
from tt.weight_utils import get_tt_parameters

COCO_CLASS_IDS = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    27,
    28,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
    64,
    65,
    67,
    70,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    81,
    82,
    84,
    85,
    86,
    87,
    88,
    89,
    90,
]


def main():
    data_dir = PROJECT / "data/coco/val2017"
    ann_file = PROJECT / "data/coco/annotations/instances_val2017.json"
    config_path = PROJECT / "RT-DETR/rtdetr_pytorch/configs/rtdetr/rtdetr_r50vd_6x_coco.yml"
    ckpt_path = PROJECT / "weights/rtdetr_r50vd.pth"

    device = ttnn.open_device(device_id=0, l1_small_size=16384)
    try:
        print("loading model...")
        cfg = YAMLConfig(str(config_path))
        model = cfg.model
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        model.load_state_dict(ckpt["ema"]["module"])
        model.eval()

        print("pushing encoder weights to device...")
        tt_params = get_tt_parameters(device, model)

        def ttnn_aifi_forward(src, pos_embed=None):
            if pos_embed is not None:
                src = src + pos_embed
            # res is (Batch, 1, Seq, Hidden)
            res = run_encoder(src, tt_params.layers, device).float()
            # Squeeze to (Batch, Seq, Hidden)
            return res.squeeze(1)

        model.encoder.encoder[0].forward = ttnn_aifi_forward

        model.encoder.encoder[0].forward = ttnn_aifi_forward

        transforms = T.Compose([T.Resize((640, 640)), T.ToTensor()])
        dataset = CocoDetection(root=str(data_dir), annFile=str(ann_file), transform=transforms)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
        coco_gt = COCO(str(ann_file))

        results = []
        print(f"running eval on {len(dataset)} images...")
        with torch.no_grad():
            for i, (img, _) in enumerate(tqdm(loader)):
                img_id = dataset.ids[i]
                outputs = model(img)

                logits = outputs["pred_logits"][0].sigmoid()
                boxes = outputs["pred_boxes"][0]

                img_info = coco_gt.loadImgs(int(img_id))[0]
                orig_h, orig_w = img_info["height"], img_info["width"]

                scores, labels = torch.max(logits, dim=-1)
                keep = scores > 0.05

                for score, label, box in zip(scores[keep], labels[keep], boxes[keep]):
                    cx, cy, w, h = box
                    x1 = (cx - w / 2) * orig_w
                    y1 = (cy - h / 2) * orig_h
                    w_abs = w * orig_w
                    h_abs = h * orig_h
                    results.append(
                        {
                            "image_id": int(img_id),
                            "category_id": COCO_CLASS_IDS[int(label.item())],
                            "bbox": [float(x1), float(y1), float(w_abs), float(h_abs)],
                            "score": float(score.item()),
                        }
                    )

        res_file = PROJECT / "results.json"
        with open(res_file, "w") as f:
            json.dump(results, f)

        coco_dt = coco_gt.loadRes(str(res_file))
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
