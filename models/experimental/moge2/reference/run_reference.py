"""Faithful MoGe-2 torch CPU reference runner.

Loads the official Ruicheng/moge-2-vitl-normal checkpoint via the vendored
upstream code and runs it on the canonical test image. Saves both the raw
forward() outputs (the direct model tensors used for PCC) and the infer()
post-processed geometry (depth/intrinsics) for sanity visualisation.

This is the ground-truth reference the ttnn port is graded against.
"""
import os, sys, glob, time
import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # make vendored `moge` and `utils3d` importable

from moge.model.v2 import MoGeModel  # noqa: E402

CKPT = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--Ruicheng--moge-2-vitl-normal/snapshots/*/model.pt"))[0]
IMAGE = os.environ.get("MOGE_IMAGE", "/home/ttuser/img.png")
NUM_TOKENS = int(os.environ.get("MOGE_NUM_TOKENS", "1800"))
OUT = os.environ.get("MOGE_REF_OUT",
                     os.path.join(os.path.dirname(HERE), "reference_outputs.pt"))


def load_image(path):
    im = Image.open(path).convert("RGB")
    arr = np.asarray(im, dtype=np.float32) / 255.0          # HWC [0,1]
    t = torch.from_numpy(arr).permute(2, 0, 1)[None]        # [1,3,H,W]
    return t


def main():
    torch.manual_seed(0)
    print(f"ckpt={CKPT}\nimage={IMAGE} num_tokens={NUM_TOKENS}")
    model = MoGeModel.from_pretrained(CKPT).eval()
    image = load_image(IMAGE)
    B, _, H, W = image.shape
    print(f"image {W}x{H}")

    with torch.inference_mode():
        # aspect-ratio derived token grid (must match forward())
        ar = W / H
        base_h = round((NUM_TOKENS / ar) ** 0.5)
        base_w = round((NUM_TOKENS * ar) ** 0.5)
        print(f"base_h={base_h} base_w={base_w} tokens={base_h*base_w} seq={base_h*base_w+1}")

        t0 = time.time()
        raw = model.forward(image, num_tokens=NUM_TOKENS)
        t1 = time.time()
        print(f"forward() wall {t1-t0:.3f}s; keys={list(raw.keys())}")
        for k, v in raw.items():
            print(f"  raw[{k}] {tuple(v.shape)} dtype={v.dtype} "
                  f"min={v.float().min():.4f} max={v.float().max():.4f} mean={v.float().mean():.4f}")

        out = model.infer(image, num_tokens=NUM_TOKENS, use_fp16=False)
        for k, v in out.items():
            if isinstance(v, torch.Tensor):
                finite = v[torch.isfinite(v)]
                print(f"  infer[{k}] {tuple(v.shape)} finite_mean={finite.float().mean():.4f}")

    save = {
        "config": {"num_tokens": NUM_TOKENS, "base_h": base_h, "base_w": base_w,
                   "H": H, "W": W, "image": IMAGE, "ckpt": CKPT},
        "raw": {k: v.float().cpu() for k, v in raw.items()},
        "infer": {k: v.float().cpu() for k, v in out.items() if isinstance(v, torch.Tensor)},
    }
    torch.save(save, OUT)
    print(f"SAVED reference -> {OUT}")


if __name__ == "__main__":
    main()
