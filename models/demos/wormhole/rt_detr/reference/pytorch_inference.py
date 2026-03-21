# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
pytorch reference inference for rt-detr — use this to validate outputs against the ttnn version
"""

import os
import sys

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "RT-DETR", "rtdetr_pytorch"))
from src.core import YAMLConfig


class RTDETRReference(nn.Module):
    def __init__(self, config_path, ckpt_path, device="cpu"):
        super().__init__()
        cfg = YAMLConfig(config_path)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt["ema"]["module"] if "ema" in ckpt else ckpt["model"]
        cfg.model.load_state_dict(state)
        self.model = cfg.model.deploy()
        self.postprocessor = cfg.postprocessor.deploy()
        self.model.to(device)
        self.device = device

    def forward(self, images, orig_sizes):
        out = self.model(images)
        return self.postprocessor(out, orig_sizes)


def load_image(path, size=(640, 640)):
    img = Image.open(path).convert("RGB")
    orig_size = torch.tensor([[img.size[0], img.size[1]]])  # [W, H]
    tf = T.Compose([T.Resize(size), T.ToTensor()])
    return tf(img).unsqueeze(0), orig_size, img


def main():
    config_path = "RT-DETR/rtdetr_pytorch/configs/rtdetr/rtdetr_r50vd_6x_coco.yml"
    ckpt_path = "weights/rtdetr_r50vd.pth"
    img_path = "demo/sample.jpg"

    model = RTDETRReference(config_path, ckpt_path).eval()
    img_tensor, orig_size, _ = load_image(img_path)

    with torch.no_grad():
        labels, boxes, scores = model(img_tensor, orig_size)

    print(f"detections: {len(labels[0])}")
    print(f"top score:  {scores[0].max().item():.4f}")


if __name__ == "__main__":
    main()
