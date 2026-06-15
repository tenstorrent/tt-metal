# SPDX-License-Identifier: Apache-2.0
"""Standalone PCC check: ttnn TtConvStack neck (and points head) vs torch reference."""
import os, sys, glob
import numpy as np, torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "reference"))

import ttnn
from moge.model.v2 import MoGeModel
from moge.utils.geometry_torch import normalized_view_plane_uv
from models.experimental.moge2.tt.ttnn_moge_decoder import TtConvStack, _to_cl

CKPT = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--Ruicheng--moge-2-vitl-normal/snapshots/*/model.pt"))[0]
IMAGE = os.environ.get("MOGE_IMAGE", "/home/ttuser/img.png")
NUM_TOKENS = int(os.environ.get("MOGE_NUM_TOKENS", "1800"))
DEV = int(os.environ.get("MOGE_DEVICE_ID", "0"))


def pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def cl_to_nchw(t, H, W):
    x = ttnn.to_torch(t).float().reshape(1, H, W, -1).permute(0, 3, 1, 2)
    return x


def main():
    torch.manual_seed(0)
    model = MoGeModel.from_pretrained(CKPT).eval()
    arr = np.asarray(Image.open(IMAGE).convert("RGB"), np.float32) / 255.0
    image = torch.from_numpy(arr).permute(2, 0, 1)[None]
    B, _, H, W = image.shape
    ar = W / H
    bh, bw = round((NUM_TOKENS / ar) ** 0.5), round((NUM_TOKENS * ar) ** 0.5)

    with torch.inference_mode():
        x, cls = model.encoder(image, bh, bw, return_class_token=True)
        features = [x, None, None, None, None]
        for lv in range(5):
            uv = normalized_view_plane_uv(bw * 2 ** lv, bh * 2 ** lv, aspect_ratio=ar, dtype=image.dtype, device=x.device)
            uv = uv.permute(2, 0, 1).unsqueeze(0).expand(B, -1, -1, -1)
            features[lv] = uv if features[lv] is None else torch.cat([features[lv], uv], dim=1)
        ref_neck = model.neck(features)
        ref_points = model.points_head(ref_neck)

    device = ttnn.CreateDevice(device_id=DEV, l1_small_size=32768)
    try:
        cc = ttnn.init_device_compute_kernel_config(device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2,
                                                    fp32_dest_acc_en=False, packer_l1_acc=True)
        tt_neck = TtConvStack(model.neck, device, cc)
        tt_points = TtConvStack(model.points_head, device, cc)
        in_feats = [_to_cl(f, device) for f in features]
        neck_out = tt_neck(in_feats)
        print("=== NECK ===")
        for i, ((t, h, w), r) in enumerate(zip(neck_out, ref_neck)):
            print(f"  L{i} ref{tuple(r.shape)} tt({h}x{w}x{t.shape[-1]}) PCC={pcc(r, cl_to_nchw(t, h, w)):.5f}")
        pts_out = tt_points(neck_out)
        print("=== POINTS HEAD ===")
        for i, ((t, h, w), r) in enumerate(zip(pts_out, ref_points)):
            if r is None: continue
            print(f"  L{i} ref{tuple(r.shape)} tt({h}x{w}x{t.shape[-1]}) PCC={pcc(r, cl_to_nchw(t, h, w)):.5f}")
    finally:
        ttnn.CloseDevice(device)


if __name__ == "__main__":
    main()
