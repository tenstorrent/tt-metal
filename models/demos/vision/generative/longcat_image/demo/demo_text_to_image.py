# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Runnable demo — LongCat-Image Call 1 (text -> image) on Tenstorrent.

Runs the SAME shared TTNN pipeline (tt/pipeline.py) that the e2e test asserts,
so a green test guarantees this demo works. Emits a real PNG and prints the
end-to-end PCC vs the HF golden.

Run:
    python -m models.demos.vision.generative.longcat_image.demo.demo_text_to_image \
        --prompt "a photograph of a cat on a red sofa" --size 512 --steps 24 --guidance 4.5
    # add --cq 2 to run the denoise loop under trace + two command queues
"""

from __future__ import annotations

import argparse

import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.vision.generative.longcat_image.tt import pipeline as P

HF_MODEL_ID = "meituan-longcat/LongCat-Image"


def _save_png(img_denorm, path):
    # img_denorm: [1,3,H,W] in [0,1]
    try:
        from PIL import Image

        arr = (img_denorm[0].permute(1, 2, 0).clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
        Image.fromarray(arr).save(path)
        print(f"[demo] wrote {path}")
    except Exception as exc:  # noqa: BLE001
        print(f"[demo] could not write PNG ({exc}); saving tensor instead")
        torch.save(img_denorm, path + ".pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="a photograph of a cat sitting on a red sofa")
    ap.add_argument("--negative_prompt", default="")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--guidance", type=float, default=4.5)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="longcat_image_t2i.png")
    ap.add_argument("--compare_golden", action="store_true", help="also run the HF golden and print e2e PCC")
    ap.add_argument(
        "--cq", type=int, default=1, choices=[1, 2],
        help="command queues; 2 enables the trace+2CQ denoise overlap (per-step input staging on CQ1)",
    )
    args = ap.parse_args()

    from diffusers import LongCatImagePipeline

    print(f"[demo] loading {HF_MODEL_ID} (bf16) ...", flush=True)
    pipe = LongCatImagePipeline.from_pretrained(HF_MODEL_ID, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    pipe.set_progress_bar_config(disable=True)
    pipe.tokenizer_max_length = args.max_length

    latents_packed = None
    open_kwargs = {"l1_small_size": 24576}
    if args.cq == 2:
        # trace+2CQ needs a second command queue AND an explicit trace region (the 1CQ path
        # rides Blackhole's default region; keep 2CQ robust by sizing it like the perf test).
        open_kwargs["num_command_queues"] = 2
        open_kwargs["trace_region_size"] = 209715200  # 200 MB
    device = ttnn.open_device(device_id=0, **open_kwargs)
    try:
        # deterministic seeded latents shared with the golden (if comparing)
        vsf = pipe.vae_scale_factor
        lh = 2 * (args.size // (vsf * 2))
        gen = torch.Generator("cpu").manual_seed(args.seed)
        raw = torch.randn(1, 16, lh, lh, generator=gen, dtype=torch.float32)
        latents_packed = P._pack_latents(raw, 1, 16, lh, lh)

        ttp = P.LongCatImagePipelineTT(device, pipe, num_cqs=args.cq)
        result = ttp.run_text_to_image(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            height=args.size,
            width=args.size,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            seed=args.seed,
            max_length=args.max_length,
            latents_packed=latents_packed,
        )
        print(f"[demo] invoked graduated stubs: {sorted(result['invoked'])}")
        _save_png(result["image_denorm"], args.out)

        if args.compare_golden:
            golden = P.hf_reference_text_to_image(
                pipe,
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                height=args.size,
                width=args.size,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                seed=args.seed,
                max_length=args.max_length,
                latents_packed=latents_packed,
            )
            _, pcc = comp_pcc(golden["image_denorm"], result["image_denorm"].float(), 0.0)
            print(f"e2e PCC={pcc}")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
