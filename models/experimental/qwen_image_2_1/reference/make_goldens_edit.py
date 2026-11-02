# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Goldens for the image-conditioned (editing) path of Qwen-Image-2.1 with the pinned diffusers pipeline.

Saves: processor outputs (input_ids, pixel_values, image_grid_thw), per-layer text-encoder hidden states with
the vision tokens injected, the vision tower outputs (merged tokens + deepstack features), the VAE encoder
latents of the condition image, the DiT's joint layout and per-step inputs/outputs with the prefix KV cache,
and the final image. Run in the separate reference environment.
"""
import argparse
import json
import os
import time

import torch
from PIL import Image

PROMPT = "Change the background to a sunset beach"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="generated/qwen_image_2_1/goldens/edit")
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    ap.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    ap.add_argument("--vae-memory-format", choices=("contiguous", "channels_last"), default="contiguous")
    ap.add_argument(
        "--image",
        action="append",
        required=True,
        help="condition image(s), repeatable",
    )
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default="Qwen/Qwen-Image-2.1")
    ap.add_argument("--revision", default="790c92633540aa0cb11d9abf19eb46d861714758")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from diffusers import QwenImage21Pipeline
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_dimensions

    dtype = getattr(torch, args.dtype)
    pipe = QwenImage21Pipeline.from_pretrained(args.model, revision=args.revision, dtype=dtype)
    if args.vae_memory_format == "channels_last":
        torch.nn.utils.convert_conv2d_weight_memory_format(pipe.vae, torch.channels_last)
    if args.device == "cuda":
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cpu")
    dev = pipe._execution_device
    tf = pipe.transformer

    conds = [Image.open(p).convert("RGBA") for p in args.image]
    conds_resized, vae_imgs, sizes = [], [], []
    for cond in conds:
        w0, h0 = cond.size
        in_w, in_h, _ = calculate_dimensions(1024 * 1024, w0 / h0)
        conds_resized.append(pipe.image_processor.resize(cond, width=in_w, height=in_h))
        vae_imgs.append(pipe.image_processor.preprocess(cond, width=in_w, height=in_h).unsqueeze(2))
        sizes.append((in_w, in_h))
        print("condition image", cond.size, "->", (in_w, in_h))
    cond, cond_resized, vae_img, in_w, in_h = conds[0], conds_resized[0], vae_imgs[0], sizes[0][0], sizes[0][1]

    # ---------------- text encoder with the image ----------------
    replace = "<image1><|vision_start|><|image_pad|><|vision_end|>"
    for i in range(2, len(conds) + 1):
        replace += f" <image{i}><|vision_start|><|image_pad|><|vision_end|>"
    template = pipe.prompt_template_ti2i.replace("<image1><|vision_start|><|image_pad|><|vision_end|>", replace)
    text = template.format(args.prompt)
    whites = []
    for cr in conds_resized:
        wh = Image.new("RGB", cr.size, (255, 255, 255))
        wh.paste(cr, mask=cr.getchannel("A"))
        whites.append(wh)
    white = whites[0]
    model_inputs = pipe.processor(text=[text], images=whites, padding=True, padding_side="left", return_tensors="pt")
    print(
        "input_ids",
        tuple(model_inputs.input_ids.shape),
        "pixel_values",
        tuple(model_inputs.pixel_values.shape),
        "grid_thw",
        model_inputs.image_grid_thw.tolist(),
    )
    vision_taps = {}
    text_model = pipe.text_encoder.model.language_model
    visual = pipe.text_encoder.model.visual
    hooks = [
        text_model.norm.register_forward_hook(lambda m, a, o: a[0]),
        visual.register_forward_hook(lambda m, a, o: vision_taps.__setitem__("visual_out", o)),
    ]
    with torch.no_grad():
        fw = {
            "input_ids": model_inputs.input_ids.to(dev),
            "attention_mask": model_inputs.attention_mask.to(dev),
            "pixel_values": model_inputs.pixel_values.to(dev),
            "image_grid_thw": model_inputs.image_grid_thw.to(dev),
            "output_hidden_states": True,
            "mm_token_type_ids": model_inputs.mm_token_type_ids.to(dev),
        }
        te_out = pipe.text_encoder(**fw)
    for h in hooks:
        h.remove()
    hs = [h.cpu() for h in te_out.hidden_states]
    vis = vision_taps["visual_out"]

    def _cpu(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu()
        if isinstance(x, (list, tuple)):
            return [_cpu(v) for v in x]
        if hasattr(x, "__dict__"):
            return {k: _cpu(v) for k, v in vars(x).items() if isinstance(v, (torch.Tensor, list, tuple))}
        return x

    # position ids (mrope) as the model computes them
    pos_ids, _ = pipe.text_encoder.model.get_rope_index(
        input_ids=model_inputs.input_ids.to(dev),
        mm_token_type_ids=model_inputs.mm_token_type_ids.to(dev),
        image_grid_thw=model_inputs.image_grid_thw.to(dev),
        attention_mask=model_inputs.attention_mask.to(dev),
    )
    pos_ids = pos_ids.cpu()
    with torch.no_grad():
        prompt_embeds, prompt_embeds_mask, image_pad_mask = pipe.encode_prompt(
            prompt=args.prompt, image=conds_resized, device=dev
        )
    print("prompt_embeds", tuple(prompt_embeds.shape), "image tokens", int(image_pad_mask.sum()))
    torch.save(
        {
            "prompt": args.prompt,
            "template_text": text,
            "cond_size": (in_w, in_h),
            "input_ids": model_inputs.input_ids.cpu(),
            "attention_mask": model_inputs.attention_mask.cpu(),
            "pixel_values": model_inputs.pixel_values.cpu(),
            "image_grid_thw": model_inputs.image_grid_thw.cpu(),
            "mm_token_type_ids": model_inputs.mm_token_type_ids.cpu(),
            "hidden_states": hs,
            "visual_out": _cpu(vis),
            "position_ids": pos_ids,
            "drop_idx": pipe._drop_idx,
            "img_token_id": pipe._img_token_id,
            "prompt_embeds": prompt_embeds.cpu(),
            "prompt_embeds_mask": None if prompt_embeds_mask is None else prompt_embeds_mask.cpu(),
            "image_pad_mask": image_pad_mask.cpu(),
            "cond_rgb_white": torch.from_numpy(__import__("numpy").asarray(white).copy()),
        },
        os.path.join(args.out, "text_encoder_edit.pt"),
    )

    # ---------------- VAE encoder ----------------
    enc_all = []
    with torch.no_grad():
        pipe.vae.to(dev)
        for vi in vae_imgs:
            post = pipe.vae.encode(vi.to(dev, dtype)).latent_dist
            mode = post.mode()
            mean = torch.tensor(pipe.vae.config.latents_mean).view(1, 64, 1, 1, 1).to(mode.device, mode.dtype)
            std = torch.tensor(pipe.vae.config.latents_std).view(1, 64, 1, 1, 1).to(mode.device, mode.dtype)
            enc_all.append(
                {
                    "vae_in": vi.cpu(),
                    "latent_mode": mode.cpu(),
                    "latent_logvar": post.logvar.cpu(),
                    "latent_normalized": ((mode - mean) / std).cpu(),
                }
            )
    torch.save(dict(enc_all[0], all_images=enc_all), os.path.join(args.out, "vae_encode.pt"))
    print("vae encode", [tuple(e["latent_mode"].shape) for e in enc_all])

    # ---------------- denoising with taps ----------------
    steps, kv_dump = {}, {}
    orig_forward = tf.forward

    def tapped_forward(*a, **kw):
        i = len(steps)
        rec = {
            k: (kw[k].detach().cpu() if isinstance(kw.get(k), torch.Tensor) else kw.get(k))
            for k in (
                "hidden_states",
                "timestep",
                "kv_cache_mode",
                "img_shapes",
                "img_mask",
                "encoder_hidden_states_mask",
            )
        }
        out = orig_forward(*a, **kw)
        rec["noise_pred"] = out[0].detach().cpu()
        steps[i] = rec
        if i == 0 and kw.get("kv_cache") is not None:
            for li, lc in enumerate(kw["kv_cache"].layer_caches):
                kv_dump[li] = (lc.k.detach().cpu(), lc.v.detach().cpu())
        return out

    tf.forward = tapped_forward
    latents_log = []

    def cb(p, i, t, kwargs):
        latents_log.append((i, float(t), kwargs["latents"].detach().cpu()))
        return {}

    gen = torch.Generator("cpu").manual_seed(args.seed)
    target_w, target_h, _ = calculate_dimensions(1024 * 1024, conds[-1].width / conds[-1].height)
    latent_h, latent_w = 2 * (target_h // 32), 2 * (target_w // 32)
    noise = torch.randn((1, 1, 64, latent_h, latent_w), generator=gen, dtype=torch.bfloat16)
    initial_latents = noise.view(1, 64, -1).transpose(1, 2).contiguous()
    t0 = time.time()
    with torch.no_grad():
        image = pipe(
            prompt=args.prompt,
            image=conds,
            num_inference_steps=args.steps,
            generator=gen,
            latents=initial_latents,
            callback_on_step_end=cb,
            callback_on_step_end_tensor_inputs=["latents"],
        ).images[0]
    print(f"pipeline {time.time()-t0:.1f}s, output {image.size}")
    tf.forward = orig_forward
    image.save(os.path.join(args.out, "image_edit.png"))
    torch.save(
        {
            "steps": steps,
            "kv_cache_step0": kv_dump,
            "latents_after_step": latents_log,
            "timesteps": pipe.scheduler.timesteps.cpu(),
            "sigmas": pipe.scheduler.sigmas.cpu(),
            "seed": args.seed,
            "num_steps": args.steps,
            "output_size": image.size,
        },
        os.path.join(args.out, "denoise_edit.pt"),
    )
    json.dump(
        {
            "prompt": args.prompt,
            "image": args.image,
            "model": args.model,
            "revision": args.revision,
            "diffusers": __import__("diffusers").__version__,
            "transformers": __import__("transformers").__version__,
            "torch": torch.__version__,
            "device": args.device,
            "dtype": args.dtype,
            "initial_noise_dtype": "bfloat16",
            "vae_memory_format": args.vae_memory_format,
            "cond_size": [in_w, in_h],
            "output_size": list(image.size),
            "steps": args.steps,
            "seed": args.seed,
            "text_tokens_total": int(prompt_embeds.shape[1]),
            "image_tokens": int(image_pad_mask.sum()),
        },
        open(os.path.join(args.out, "meta.json"), "w"),
        indent=2,
    )
    print("done ->", args.out)


if __name__ == "__main__":
    main()
