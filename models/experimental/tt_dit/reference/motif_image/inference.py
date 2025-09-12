import sys

sys.path.append(".")
import argparse
import os
import pickle
import random

import numpy as np
import torch
from loguru import logger
from models.experimental.tt_dit.reference.motif_image.configuration_motifimage import MotifImageConfig
from models.experimental.tt_dit.reference.motif_image.modeling_motifimage import MotifImage
from safetensors.torch import load_file


def main(args: argparse.Namespace) -> None:
    """Generate images for prompts specified in a file.

    Args:
        args: Parsed CLI arguments controlling model paths, generation settings, and output.
    """
    if not os.path.isfile(args.prompt_file):
        print(f"Error: The prompt file '{args.prompt_file}' does not exist.")
        sys.exit(1)

    with open(args.prompt_file) as f:
        prompts = [prompt.rstrip() for prompt in f.readlines()]

    config = MotifImageConfig.from_json_file(args.model_config)
    config.vae_type = args.vae_type  # VAE overriding
    config.height = args.resolution
    config.width = args.resolution

    model = MotifImage(config)

    try:
        ema_instance = torch.load(args.model_ckpt, weights_only=False)
        ema_instance = {k: v for k, v in ema_instance.items() if "dit" in k}
    except pickle.UnpicklingError as e:
        logger.warning(f"Error loading checkpoint, trying to load via safetensors for given checkpoint: {e}")
        ema_instance = load_file(args.model_ckpt)
        ema_instance = {k: v for k, v in ema_instance.items() if "dit" in k}

    if args.ema:
        # EMA checkpoint consists of shadow_params, having the list of parameters
        for param, ema_param in zip(model.parameters(), ema_instance["shadow_params"]):
            param.data.copy_(ema_param.data)
    else:
        model.load_state_dict(ema_instance)

    model = model.cuda()
    model = model.to(dtype=torch.bfloat16)
    model.eval()

    guidance_scales = args.guidance_scales if args.guidance_scales else [5.0]

    if isinstance(args.seed, int):
        seeds = [args.seed]
    else:
        seeds = args.seed

    for seed in seeds:
        for guidance_scale in guidance_scales:
            output_dir = os.path.join(args.output_dir, f"seed_{seed}", f"scale_{guidance_scale}")
            os.makedirs(output_dir, exist_ok=True)

            for i in range(0, len(prompts), args.batch_size):
                torch.manual_seed(seed)
                random.seed(seed)
                np.random.seed(seed)

                batch_prompts = prompts[i : i + args.batch_size]
                imgs = model.sample(
                    batch_prompts,
                    args.steps,
                    resolution=(args.resolution, args.resolution),
                    guidance_scale=guidance_scale,
                    use_linear_quadratic_schedule=args.use_linear_quadratic_schedule,
                    linear_quadratic_emulating_steps=args.linear_quadratic_emulating_steps,
                    get_intermediate_steps=args.streaming,
                    zero_masking=args.zero_masking,
                    zero_embedding_for_cfg=args.zero_embedding_for_cfg,
                    negative_prompt=args.negative_prompt,
                    clip_t=[0.0, 1.0],
                    get_rare_negative_token=args.get_rare_negative_token,
                    negative_strategy_switch_t=args.negative_strategy_switch_t,
                )
                if args.streaming:
                    imgs, intermediate_imgs = imgs
                    for j, intermediate_img in enumerate(intermediate_imgs):
                        for k, img in enumerate(intermediate_img):
                            img.save(os.path.join(output_dir, f"{i + k:03d}_{j:03d}_intermediate.png"))
                for j, img in enumerate(imgs):
                    img.save(os.path.join(output_dir, f"{i + j:03d}_check.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images with MotifImage")
    parser.add_argument("--model-config", type=str, required=True, help="Path to model configuration JSON")
    parser.add_argument("--model-ckpt", type=str, required=True, help="Path to model checkpoint (.bin or .safetensors)")
    parser.add_argument(
        "--seed",
        type=int,
        nargs="*",
        default=[7777],
        help="Random seed(s). Provide multiple values to sweep across seeds.",
    )
    parser.add_argument("--steps", type=int, default=50, help="Number of sampling steps")
    parser.add_argument("--resolution", type=int, default=1024, help="Output image resolution (square)")
    parser.add_argument("--batch-size", type=int, default=32, help="Number of prompts to process per batch")
    parser.add_argument("--streaming", action="store_true", help="If set, save intermediate images for each step")
    parser.add_argument("--zero-masking", action="store_true", help="Zero-out embeddings at padding tokens")
    parser.add_argument("--vae-type", type=str, default="SD3", help="VAE backend: 'SD3' or 'SDXL'")
    parser.add_argument(
        "--prompt-file", type=str, default="sample_prompts.txt", help="Path to a text file, one prompt per line"
    )
    parser.add_argument(
        "--guidance-scales",
        type=float,
        nargs="*",
        default=None,
        help="List of CFG scales to evaluate. If omitted, uses [5.0].",
    )
    parser.add_argument("--output-dir", type=str, default="output", help="Directory to save generated images")
    parser.add_argument("--ema", action="store_true", help="Load EMA 'shadow_params' from the checkpoint")
    parser.add_argument(
        "--use-linear-quadratic-schedule",
        action="store_true",
        help="Use linear-quadratic time schedule (MovieGen-inspired).",
    )
    parser.add_argument(
        "--linear-quadratic-emulating-steps",
        type=int,
        default=100,
        help="N for linear-quadratic schedule emulation (see Figure 10 from the MovieGen technical report).",
    )
    parser.add_argument(
        "--zero-embedding-for-cfg",
        action="store_true",
        help="Use zero embeddings for the unconditional CFG branch when no negative prompt is given.",
    )
    parser.add_argument("--negative-prompt", type=str, default=None, help="Negative prompt text for CFG")
    parser.add_argument("--get-rare-negative-token", action="store_true", help="Get rare negative token")
    parser.add_argument(
        "--negative-strategy-switch-t", type=float, default=0.15, help="Negative strategy switch threshold"
    )
    args = parser.parse_args()

    main(args)
