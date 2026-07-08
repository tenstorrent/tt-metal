# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared demo driver for the HunyuanVideo-1.5 TTNN pipeline.

Both ``demo_t2v`` and ``demo_i2v`` call :func:`run_demo`, which builds the ONE
shared ``tt.pipeline`` (identical code to the e2e test), runs one real diffusion
transformer forward for the requested conditioning regime, and prints the real
task output (the denoised velocity/flow prediction) alongside the PCC against the
HF golden.  A green e2e test therefore GUARANTEES a working demo — same pipeline.
"""

from __future__ import annotations

import argparse

import ttnn
from models.demos.hf_eager.hunyuanvideo_1_5.tt import pipeline as P


def build_argparser(task: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=f"HunyuanVideo-1.5 TTNN {task} denoise-step demo")
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--frames", type=int, default=2, help="latent temporal size (post-patch tokens = frames*h*w)")
    ap.add_argument("--height", type=int, default=4, help="latent height")
    ap.add_argument("--width", type=int, default=4, help="latent width")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--granularity", choices=P.GRANULARITIES, default="composite")
    return ap


def run_demo(task: str, args) -> float:
    model = P.load_reference_model()
    device = ttnn.open_device(device_id=args.device_id)
    try:
        pipe = P.build_pipeline(device, model)
        inputs = P.build_inputs(
            model.config,
            task=task,
            seed=args.seed,
            frames=args.frames,
            height=args.height,
            width=args.width,
        )
        golden = P.hf_reference(model, inputs)
        out = pipe.run(inputs, granularity=args.granularity)
        achieved = P.pcc(golden, out)

        print(f"\n=== HunyuanVideo-1.5 {task} denoise step (TTNN pipeline) ===")
        print(f"  latent (B,C,F,H,W) : {tuple(inputs['hidden_states'].shape)}")
        print(f"  mllm/qwen text     : {tuple(inputs['encoder_hidden_states'].shape)}")
        print(f"  byT5 text          : {tuple(inputs['encoder_hidden_states_2'].shape)}")
        print(
            f"  image conditioning : {tuple(inputs['image_embeds'].shape)}"
            + ("  (zeroed -> t2v)" if task == "t2v" else "  (active -> i2v)")
        )
        print(f"  granularity        : {args.granularity}")
        print(f"  output (velocity)  : {tuple(out.shape)}  mean={out.mean():.5f} std={out.std():.5f}")
        print(f"  stubs invoked      : {len(pipe.invoked)}")
        print(f"e2e PCC={achieved}")
        print(f"  result             : {'PASS' if achieved >= 0.95 else 'FAIL'} (>= 0.95)")
        return achieved
    finally:
        ttnn.close_device(device)
