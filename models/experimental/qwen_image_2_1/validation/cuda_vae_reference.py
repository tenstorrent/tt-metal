# SPDX-FileCopyrightText: © 2026 Qwen Image 2.1 contributors
# SPDX-License-Identifier: Apache-2.0

"""Capture an independent CUDA VAE decode of a saved Qwen-Image 2.1 latent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from diffusers import AutoencoderKLQwenImage21, QwenImage21Pipeline
from diffusers.image_processor import VaeImageProcessor
from huggingface_hub import snapshot_download

from models.experimental.qwen_image_2_1.checkpoint import MODEL_ID, MODEL_REVISION


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latents", type=Path, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(args.output_dir)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the VAE oracle")
    args.output_dir.mkdir(parents=True)
    checkpoint = snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    vae = (
        AutoencoderKLQwenImage21.from_pretrained(
            checkpoint, subfolder="vae", dtype=torch.bfloat16, local_files_only=True
        )
        .eval()
        .to("cuda")
    )
    packed = torch.load(args.latents, map_location="cpu", weights_only=True)
    latents = QwenImage21Pipeline._unpack_latents(packed, args.height, args.width, 16).to("cuda")
    mean = torch.tensor(vae.config.latents_mean, device="cuda", dtype=vae.dtype).view(1, 64, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device="cuda", dtype=vae.dtype).view(1, 64, 1, 1, 1)
    normalized = latents * std + mean
    torch.save(packed, args.output_dir / "packed_latents.pt")
    torch.save(normalized.cpu(), args.output_dir / "vae_input.pt")
    selected = {
        "post_quant_conv",
        "decoder.conv_in",
        "decoder.mid_block",
        "decoder.norm_out",
        "decoder.conv_out",
        "decoder.mid_block.resnets.0",
        "decoder.mid_block.resnets.0.norm1",
        "decoder.mid_block.resnets.0.conv1",
        "decoder.mid_block.resnets.0.norm2",
        "decoder.mid_block.resnets.0.conv2",
        "decoder.mid_block.attentions.0",
        "decoder.mid_block.attentions.0.norm",
        "decoder.mid_block.attentions.0.to_qkv",
        "decoder.mid_block.attentions.0.proj",
        *(f"decoder.up_blocks.{index}" for index in range(5)),
        *(f"decoder.up_blocks.{index}.upsampler.resample.0" for index in range(4)),
        *(f"decoder.up_blocks.{index}.avg_shortcut" for index in range(4)),
    }
    records = []
    handles = []

    def hook(name):
        def save(_module, inputs, output):
            target = args.output_dir / name
            target.mkdir()
            torch.save(inputs[0].detach().cpu(), target / "input.pt")
            torch.save(output.detach().cpu(), target / "output.pt")
            records.append({"name": name, "shape": list(output.shape)})

        return save

    for name, module in vae.named_modules():
        if name in selected:
            handles.append(module.register_forward_hook(hook(name)))
    with torch.inference_mode():
        output = vae.decode(normalized, return_dict=False)[0]
    for handle in handles:
        handle.remove()
    torch.save(output.cpu(), args.output_dir / "output.pt")
    processor = VaeImageProcessor(vae_scale_factor=16, vae_latent_channels=64)
    processor.postprocess(output[:, :, 0], output_type="pil")[0].save(args.output_dir / "cuda_vae.png")
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "model_revision": MODEL_REVISION,
                "height": args.height,
                "width": args.width,
                "latent_source": str(args.latents),
                "stages": records,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"output_dir": str(args.output_dir), "shape": list(output.shape), "stages": len(records)}))


if __name__ == "__main__":
    main()
