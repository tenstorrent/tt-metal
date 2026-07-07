# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Runnable demo — LongCat-Image Call 2 (image + text -> edited image) on Tenstorrent.

Runs the FULL edit pipeline on device via the shared tt/pipeline.py::run_image_edit:
VAE-encode the input image -> Qwen2.5-VL vision tower + multimodal (image-conditioned)
text-encode -> MMDiT edit-denoise (image latents concatenated onto the noise latents)
-> VAE-decode. Emits the edited PNG and, with --compare_golden, the e2e PCC vs the HF
LongCatImageEditPipeline golden. This fires the vision tower + VAE encoder that the
text->image path never touches.

Run:
    python -m models.demos.vision.generative.longcat_image.demo.demo_image_edit \
        --image path/to.jpg --prompt "change the cat to a dog" --steps 24
    # --image is optional (a synthetic test image is used if omitted)
    # --target_area sets the output size target (default ~1MP, like the reference)
"""

from __future__ import annotations

import argparse

import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.vision.generative.longcat_image.tt import pipeline as P

HF_MODEL_ID = "meituan-longcat/LongCat-Image"


def _test_image(size):
    from PIL import Image

    yy = torch.linspace(0, 1, size).view(-1, 1)
    xx = torch.linspace(0, 1, size).view(1, -1)
    base = torch.stack([xx.expand(size, size), yy.expand(size, size), 0.5 + 0.5 * torch.sin(6.28 * (xx + yy))])
    base[:, size // 4 : size // 2, size // 4 : size // 2] = 0.9
    return Image.fromarray((base.permute(1, 2, 0).clamp(0, 1) * 255).to(torch.uint8).numpy())


def _save_png(img_denorm, path):
    from PIL import Image

    arr = (img_denorm[0].permute(1, 2, 0).clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
    Image.fromarray(arr).save(path)
    print(f"[demo] wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None, help="input image path (a synthetic test image is used if omitted)")
    ap.add_argument("--prompt", default="change the square to a bright red circle")
    ap.add_argument("--negative_prompt", default="")
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--guidance", type=float, default=4.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target_area", type=int, default=1024 * 1024, help="output size target in px^2 (reference ~1MP)")
    ap.add_argument("--out", default="longcat_image_edit.png")
    ap.add_argument("--compare_golden", action="store_true", help="also run the HF golden edit + print e2e PCC (slow)")
    args = ap.parse_args()

    from diffusers import LongCatImagePipeline

    print(f"[demo] loading {HF_MODEL_ID} (bf16) ...", flush=True)
    pipe = LongCatImagePipeline.from_pretrained(HF_MODEL_ID, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    pipe.set_progress_bar_config(disable=True)

    if args.image:
        from PIL import Image

        image = Image.open(args.image).convert("RGB")
    else:
        image = _test_image(512)

    device = ttnn.open_device(device_id=0, l1_small_size=24576)
    try:
        ttp = P.LongCatImagePipelineTT(device, pipe)
        result = ttp.run_image_edit(
            image=image, prompt=args.prompt, negative_prompt=args.negative_prompt,
            num_inference_steps=args.steps, guidance_scale=args.guidance, seed=args.seed,
            target_area=args.target_area,
        )
        print(f"[demo] edited image {result['width']}x{result['height']}; invoked: {sorted(result['invoked'])}")
        _save_png(result["image_denorm"], args.out)

        if args.compare_golden:
            golden = P.hf_reference_image_edit(
                pipe, image=image, prompt=args.prompt, negative_prompt=args.negative_prompt,
                num_inference_steps=args.steps, guidance_scale=args.guidance, seed=args.seed,
                target_area=args.target_area,
            )
            _, pcc = comp_pcc(golden["image_denorm"], result["image_denorm"].float(), 0.0)
            print(f"e2e PCC={pcc}")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
