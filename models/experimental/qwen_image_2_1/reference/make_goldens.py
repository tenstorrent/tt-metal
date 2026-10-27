# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Generate fp32/bf16 reference goldens for Qwen-Image-2.1 with the pinned diffusers pipeline.

Saves everything the TT port needs to be verified component by component:
  text encoder inputs/outputs, per-step DiT inputs/outputs (with the prefix KV cache), the
  final latents, the VAE decode and the PNG.  Run in the separate reference environment.
"""
import argparse
import json
import os
import time

import torch

PROMPT = "White furry llama with black sunglasses, smiling and happy, jumping"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="generated/qwen_image_2_1/goldens")
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    ap.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default="Qwen/Qwen-Image-2.1")
    ap.add_argument("--revision", default="790c92633540aa0cb11d9abf19eb46d861714758")
    ap.add_argument("--save-block-taps", action="store_true", help="save per-block hidden states of step 0")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from diffusers import QwenImage21Pipeline

    dtype = getattr(torch, args.dtype)
    pipe = QwenImage21Pipeline.from_pretrained(args.model, revision=args.revision, dtype=dtype)
    if args.device == "cuda":
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cpu")
    dev = pipe._execution_device
    tf = pipe.transformer

    # ---------------- text encoder ----------------
    t0 = time.time()
    with torch.no_grad():
        prompt_embeds, prompt_embeds_mask, image_pad_mask = pipe.encode_prompt(prompt=args.prompt, device=dev)
    print(f"encode_prompt {time.time()-t0:.2f}s  prompt_embeds {tuple(prompt_embeds.shape)} mask={prompt_embeds_mask}")
    # raw tokenizer view for the TT side
    prompts = [pipe.prompt_template_t2i.format(args.prompt)]
    model_inputs = pipe.processor(text=prompts, padding=True, padding_side="left", return_tensors="pt")
    torch.save(
        {
            "prompt": args.prompt,
            "template_text": prompts[0],
            "input_ids": model_inputs.input_ids.cpu(),
            "attention_mask": model_inputs.attention_mask.cpu(),
            "drop_idx": pipe._drop_idx,
            "prompt_embeds": prompt_embeds.cpu(),
            "prompt_embeds_mask": None if prompt_embeds_mask is None else prompt_embeds_mask.cpu(),
            "image_pad_mask": image_pad_mask.cpu(),
        },
        os.path.join(args.out, "text_encoder.pt"),
    )
    # per-layer hidden states of the text encoder (pre-final-norm last layer, like the pipeline)
    with torch.no_grad():
        text_model = pipe.text_encoder.model.language_model
        handle = text_model.norm.register_forward_hook(lambda module, a, out: a[0])
        try:
            te_out = pipe.text_encoder(
                input_ids=model_inputs.input_ids.to(dev),
                attention_mask=model_inputs.attention_mask.to(dev),
                output_hidden_states=True,
            )
        finally:
            handle.remove()
        hs = [h.cpu() for h in te_out.hidden_states]
    torch.save({"hidden_states": hs}, os.path.join(args.out, "text_encoder_hidden_states.pt"))
    print(f"text encoder hidden states: {len(hs)} x {tuple(hs[0].shape)}")

    # ---------------- denoising with taps ----------------
    steps = {}
    kv_dump = {}

    orig_forward = tf.forward

    def tapped_forward(*a, **kw):
        i = len(steps)
        rec = {
            "hidden_states": kw["hidden_states"].detach().cpu(),
            "timestep": kw["timestep"].detach().cpu(),
            "kv_cache_mode": kw.get("kv_cache_mode"),
            "img_shapes": kw["img_shapes"],
            "img_mask": kw["img_mask"].detach().cpu(),
        }
        out = orig_forward(*a, **kw)
        rec["noise_pred"] = out[0].detach().cpu()
        steps[i] = rec
        if i == 0 and kw.get("kv_cache") is not None:
            for li, lc in enumerate(kw["kv_cache"].layer_caches):
                kv_dump[li] = (lc.k.detach().cpu(), lc.v.detach().cpu())
        return out

    tf.forward = tapped_forward

    block_taps = {}
    hooks = []
    if args.save_block_taps:
        for bi in (0, 1, 2, 15, 31):

            def mk(bi):
                def hook(mod, inp, out):
                    if 0 not in block_taps or bi not in block_taps[0]:
                        block_taps.setdefault(0, {})[bi] = (inp[0].detach().cpu() if inp else None, out.detach().cpu())

                return hook

            hooks.append(tf.transformer_blocks[bi].register_forward_hook(mk(bi)))

    latents_log = []

    def cb(p, i, t, kwargs):
        latents_log.append((i, float(t), kwargs["latents"].detach().cpu()))
        return {}

    gen = torch.Generator("cpu").manual_seed(args.seed)
    latent_h = 2 * (args.size // 32)
    noise = torch.randn((1, 1, 64, latent_h, latent_h), generator=gen, dtype=torch.bfloat16)
    initial_latents = noise.view(1, 64, -1).transpose(1, 2).contiguous()
    t0 = time.time()
    with torch.no_grad():
        image = pipe(
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            width=args.size,
            height=args.size,
            num_inference_steps=args.steps,
            generator=gen,
            latents=initial_latents,
            callback_on_step_end=cb,
            callback_on_step_end_tensor_inputs=["latents"],
            output_type="pil",
        ).images[0]
    print(f"pipeline {time.time()-t0:.1f}s")
    for h in hooks:
        h.remove()
    tf.forward = orig_forward
    image.save(os.path.join(args.out, "image.png"))

    # initial noise as the pipeline drew it (same generator state)
    gen2 = torch.Generator("cpu").manual_seed(args.seed)
    lat_h = 2 * (args.size // 32)
    lat_w = 2 * (args.size // 32)
    noise = torch.randn((1, 1, 64, lat_h, lat_w), generator=gen2, dtype=torch.bfloat16)
    noise_packed = noise.view(1, 64, lat_h * lat_w).transpose(1, 2)
    assert torch.equal(noise_packed, steps[0]["hidden_states"]), "initial latents mismatch"

    sched = pipe.scheduler
    torch.save(
        {
            "steps": steps,
            "kv_cache_step0": kv_dump,
            "latents_after_step": latents_log,
            "timesteps": sched.timesteps.cpu(),
            "sigmas": sched.sigmas.cpu(),
            "seed": args.seed,
            "size": args.size,
            "num_steps": args.steps,
            "block_taps": block_taps,
        },
        os.path.join(args.out, "denoise.pt"),
    )
    # VAE decode golden: the pipeline's own final latents -> image tensor
    final_latents = latents_log[-1][2].to(dev)
    with torch.no_grad():
        lat = pipe._unpack_latents(final_latents, args.size, args.size, pipe.vae_scale_factor).to(pipe.vae.dtype)
        mean = torch.tensor(pipe.vae.config.latents_mean).view(1, 64, 1, 1, 1).to(lat.device, lat.dtype)
        std = torch.tensor(pipe.vae.config.latents_std).view(1, 64, 1, 1, 1).to(lat.device, lat.dtype)
        vae_in = lat * std + mean
        pipe.vae.to(dev)
        dec = pipe.vae.decode(vae_in, return_dict=False)[0][:, :, 0]
    torch.save({"vae_in": vae_in.cpu(), "vae_out": dec.cpu()}, os.path.join(args.out, "vae.pt"))
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(
            {
                "prompt": args.prompt,
                "steps": args.steps,
                "size": args.size,
                "seed": args.seed,
                "model": args.model,
                "revision": args.revision,
                "diffusers": __import__("diffusers").__version__,
                "transformers": __import__("transformers").__version__,
                "torch": torch.__version__,
                "device": args.device,
                "dtype": args.dtype,
                "initial_noise_dtype": "bfloat16",
                "text_tokens": int(prompt_embeds.shape[1]),
                "mu": None,
            },
            f,
            indent=2,
        )
    print("done ->", args.out)


if __name__ == "__main__":
    main()
