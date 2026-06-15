# SPDX-License-Identifier: Apache-2.0
"""Standalone PCC check: ttnn DINOv2 ViT-L encoder vs torch reference encoder."""
import os, sys, glob
import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "reference"))

import ttnn
from moge.model.v2 import MoGeModel
from models.experimental.moge2.tt.ttnn_moge_encoder import TtMoGeEncoder

CKPT = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--Ruicheng--moge-2-vitl-normal/snapshots/*/model.pt"))[0]
IMAGE = os.environ.get("MOGE_IMAGE", "/home/ttuser/img.png")
NUM_TOKENS = int(os.environ.get("MOGE_NUM_TOKENS", "1800"))
DEV = int(os.environ.get("MOGE_DEVICE_ID", "0"))


def pcc(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def main():
    torch.manual_seed(0)
    model = MoGeModel.from_pretrained(CKPT).eval()
    im = Image.open(IMAGE).convert("RGB")
    arr = np.asarray(im, dtype=np.float32) / 255.0
    image = torch.from_numpy(arr).permute(2, 0, 1)[None]
    B, _, H, W = image.shape
    ar = W / H
    base_h = round((NUM_TOKENS / ar) ** 0.5)
    base_w = round((NUM_TOKENS * ar) ** 0.5)
    print(f"image {W}x{H} base_h={base_h} base_w={base_w} seq={base_h*base_w+1}")

    with torch.inference_mode():
        ref_x, ref_cls = model.encoder(image, base_h, base_w, return_class_token=True)
    print(f"ref_x {tuple(ref_x.shape)} ref_cls {tuple(ref_cls.shape)}")

    device = ttnn.CreateDevice(device_id=DEV, l1_small_size=32768)
    try:
        enc = TtMoGeEncoder(model, device)
        tt_x, tt_cls = enc(image, base_h, base_w)
        print(f"tt_x  {tuple(tt_x.shape)} tt_cls  {tuple(tt_cls.shape)}")
        print(f"PCC x   = {pcc(ref_x, tt_x):.6f}")
        print(f"PCC cls = {pcc(ref_cls, tt_cls):.6f}")
    finally:
        ttnn.CloseDevice(device)


if __name__ == "__main__":
    main()
