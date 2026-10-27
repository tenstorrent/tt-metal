# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Text-to-image demo: Qwen-Image-2.1 on one Blackhole p150.

    python -m models.experimental.qwen_image_2_1.demo.demo \
        --prompt "White furry llama with black sunglasses, smiling and happy, jumping" --out llama.png
"""
import argparse
import json
import os
import time

import ttnn
from models.experimental.qwen_image_2_1.common.config import PROMPT_DEMO
from models.experimental.qwen_image_2_1.common.device import close_device, open_device
from models.experimental.qwen_image_2_1.tt.dit import DiTPrecision
from models.experimental.qwen_image_2_1.tt.pipeline import QwenImage21Pipeline
from models.experimental.qwen_image_2_1.tt.text_encoder import TEPrecision


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default=PROMPT_DEMO)
    ap.add_argument("--out", default="qwen_image_2_1_p150.png")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--repeat", type=int, default=1, help="generate this many times (warm timings)")
    ap.add_argument("--no-trace", action="store_true")
    ap.add_argument("--dit-bfp8", action="store_true", help="bfloat8_b DiT matmul weights")
    ap.add_argument("--worker-dispatch", action="store_true", help="Tensix dispatch (110 cores) instead of ETH (120)")
    ap.add_argument("--image", action="append", default=None, help="condition image(s) for editing (repeatable)")
    ap.add_argument(
        "--no-editing", action="store_true", help="do not load the vision/VAE encoders (text-to-image only)"
    )
    ap.add_argument(
        "--eth-dispatch", action="store_true", help="experimental ETH dispatch; requires additional runtime changes"
    )
    args = ap.parse_args(argv)
    if not 2 <= args.steps <= 100 or args.repeat < 1:
        ap.error("steps must be in 2..100 and repeat must be positive")
    if args.image and args.no_editing:
        ap.error("--image requires editing encoders")
    if args.eth_dispatch and not args.no_editing:
        ap.error("ETH dispatch with editing is unsupported")

    t0 = time.time()
    editing = not args.no_editing
    # ETH dispatch is validated for text-to-image only; with the editing encoders loaded it hangs (see README)
    eth = args.eth_dispatch and not args.worker_dispatch
    dev = open_device(eth_dispatch=eth)
    pipe = None
    try:
        grid = dev.compute_with_storage_grid_size()
        print(f"device open: {time.time()-t0:.1f}s, worker grid {grid.x}x{grid.y}")
        dit_prec = DiTPrecision()
        if args.dit_bfp8:
            dit_prec.weight_dtype = ttnn.bfloat8_b
        t0 = time.time()
        pipe = QwenImage21Pipeline(
            dev,
            dit_prec=dit_prec,
            te_prec=TEPrecision(),
            use_trace=not args.no_trace,
            height=args.size,
            width=args.size,
            load_editing=editing,
        )
        print(
            f"models loaded: {time.time()-t0:.1f}s (dit {pipe.load_dit_s:.1f}s, te {pipe.load_te_s:.1f}s, vae {pipe.load_vae_s:.1f}s)"
        )

        def prog(i, n):
            if (i + 1) % 10 == 0 or i + 1 == n:
                print(f"  step {i+1}/{n}", flush=True)

        images = None
        if args.image:
            from PIL import Image

            images = [Image.open(p).convert("RGBA") for p in args.image]
        for r in range(args.repeat):
            rgb, rgba, lat, tm = pipe.generate(
                args.prompt, seed=args.seed, num_steps=args.steps, progress=prog, images=images
            )
            print(f"run {r}: {json.dumps(tm.as_dict())}")
        base, ext = os.path.splitext(args.out)
        rgb.save(args.out)
        rgba.save(base + "_rgba.png")
        print(f"saved {args.out} and {base}_rgba.png")
    finally:
        try:
            if pipe is not None:
                pipe.release_traces()
        finally:
            close_device(dev)


if __name__ == "__main__":
    main()
