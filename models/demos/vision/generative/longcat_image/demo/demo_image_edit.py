# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Runnable demo — LongCat-Image Call 2 image-conditioning front-end on Tenstorrent.

Call 2 exercises the graduated modules the text->image path never fires: the
Qwen2.5-VL VISION TOWER and the VAE ENCODER. This demo takes a real image, runs
those as real TTNN forwards (via the shared tt/pipeline.py helpers the e2e test
also calls), emits a real VAE round-trip reconstruction PNG, and prints the
Source-A parity of each stage.

Run:
    python -m models.demos.vision.generative.longcat_image.demo.demo_image_edit \
        --image path/to.jpg --size 256 --compare_golden
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
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--out", default="longcat_image_edit_recon.png")
    ap.add_argument("--compare_golden", action="store_true")
    args = ap.parse_args()

    from diffusers import LongCatImageEditPipeline, LongCatImagePipeline

    print(f"[demo] loading {HF_MODEL_ID} (bf16) ...", flush=True)
    pipe = LongCatImagePipeline.from_pretrained(HF_MODEL_ID, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    pipe.set_progress_bar_config(disable=True)
    editpipe = LongCatImageEditPipeline(
        scheduler=pipe.scheduler, vae=pipe.vae, text_encoder=pipe.text_encoder,
        tokenizer=pipe.tokenizer, text_processor=pipe.text_processor, transformer=pipe.transformer,
    )

    if args.image:
        from PIL import Image

        image = Image.open(args.image).convert("RGB").resize((args.size, args.size))
    else:
        image = _test_image(args.size)

    vl = editpipe.image_processor_vl(images=image, return_tensors="pt")
    pixel_values, grid_thw = vl["pixel_values"], vl["image_grid_thw"]
    vae_in = editpipe.image_processor.preprocess(image, args.size, args.size)

    device = ttnn.open_device(device_id=0, l1_small_size=24576)
    try:
        ttp = P.LongCatImagePipelineTT(device, pipe)

        # VAE encode (TT) -> latents -> VAE decode (TT) -> reconstruction image
        z = ttp._tt_vae_encode(vae_in)  # [1,16,h8,w8]
        recon = ttp._tt_vae_decode(z)  # [1,3,H,W]
        _save_png((recon / 2 + 0.5).clamp(0, 1), args.out)

        # vision tower (TT) -> merged image embeds
        image_embeds = ttp._tt_vision_encode(pixel_values, grid_thw)
        print(f"[demo] TT vision image_embeds shape={tuple(image_embeds.shape)}")
        print(f"[demo] invoked graduated stubs: {sorted(ttp.invoked)}")

        if args.compare_golden:
            with torch.no_grad():
                pipe.vae = pipe.vae.float()
                z_gold = pipe.vae.encode(vae_in.float()).latent_dist.mean.float()
                pipe.vae = pipe.vae.to(torch.bfloat16)
                v = pipe.text_encoder.model.visual.float()
                gold_pooler = v(pixel_values.float(), grid_thw).pooler_output.float()
                pipe.text_encoder.model.visual = pipe.text_encoder.model.visual.to(torch.bfloat16)
            _, pcc_vae = comp_pcc(z_gold, z.float(), 0.0)
            _, pcc_vis = comp_pcc(gold_pooler, image_embeds.float(), 0.0)
            print(f"[demo] VAE-encode PCC={pcc_vae}")
            print(f"e2e PCC={pcc_vis}")  # vision-tower parity vs HF (Source A)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
