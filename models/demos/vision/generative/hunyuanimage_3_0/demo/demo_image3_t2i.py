# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""HunyuanImage-3.0 hybrid text->image demo on Tenstorrent (Blackhole Galaxy).

Runs the 80B MoE transformer's `gen_image` diffusion forward on the TT mesh (the
32 decoder layers) with the FlowMatch scheduler + CFG + velocity head + VAE decode
on host, and saves a PNG.

  ./python_env/bin/python -m models.demos.vision.generative.hunyuanimage_3_0.demo.demo_image3_t2i \
      --prompt "a red panda astronaut, studio lighting" --steps 50 --out panda.png

Model load dominates cold-start (~15 min). Use a named tmux + tee a log.
"""

from __future__ import annotations

import argparse

import torch

from models.demos.vision.generative.hunyuanimage_3_0.tt import gen_image as gi
from models.demos.vision.generative.hunyuanimage_3_0.tt import pipeline as ttpipe


def main():
    ap = argparse.ArgumentParser(description="HunyuanImage-3.0 hybrid text->image on Tenstorrent")
    ap.add_argument("--prompt", default=gi.DEFAULT_PROMPT)
    ap.add_argument("--size", default="1024x1024", help="HxW; snapped to the nearest supported resolution")
    ap.add_argument("--steps", type=int, default=None, help="diffusion steps (default 50 from generation_config)")
    ap.add_argument("--guidance", type=float, default=None, help="CFG scale (default 5.0 from generation_config)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-layers", type=int, default=32, help="decoder layers (32=full; fewer for quick bring-up)")
    ap.add_argument("--out", default="hunyuan_t2i.png")
    args = ap.parse_args()
    H, W = (int(s) for s in args.size.lower().replace("x", ",").split(","))

    torch.manual_seed(args.seed)
    device = ttpipe._open_selftest_device()  # full mesh + FABRIC_1D
    try:
        model, tt_pipe, _uninstall = gi.build_tt_backed_model(device, num_layers=args.num_layers)
        _img, timing = gi.generate_image(
            model,
            tt_pipe,
            prompt=args.prompt,
            image_size=(H, W),
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            seed=args.seed,
            out_path=args.out,
        )
        print(f"DONE: {timing}")
    finally:
        ttpipe._close_device(device)


if __name__ == "__main__":
    main()
