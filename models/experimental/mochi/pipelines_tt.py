from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from genmo.lib.progress import get_new_progress_bar
from genmo.lib.utils import Timer
from genmo.mochi_preview.vae.vae_stats import dit_latents_to_vae_latents
import numpy as np
import random

from genmo.mochi_preview.pipelines import (
    MochiSingleGPUPipeline,
    move_to_device,
    t5_tokenizer,
    get_conditioning,
    decode_latents,
    decode_latents_tiled_full,
    decode_latents_tiled_spatial,
)


def sample_model_tt(device, dit, conditioning, **args):
    random.seed(args["seed"])
    np.random.seed(args["seed"])
    torch.manual_seed(args["seed"])

    generator = torch.Generator(device=device)
    generator.manual_seed(args["seed"])

    w, h, t = args["width"], args["height"], args["num_frames"]
    sample_steps = args["num_inference_steps"]
    cfg_schedule = args["cfg_schedule"]
    sigma_schedule = args["sigma_schedule"]

    assert len(cfg_schedule) == sample_steps, "cfg_schedule must have length sample_steps"
    assert (t - 1) % 6 == 0, "t - 1 must be divisible by 6"
    assert len(sigma_schedule) == sample_steps + 1, "sigma_schedule must have length sample_steps + 1"

    B = 1
    SPATIAL_DOWNSAMPLE = 8
    TEMPORAL_DOWNSAMPLE = 6
    IN_CHANNELS = 12
    latent_t = ((t - 1) // TEMPORAL_DOWNSAMPLE) + 1
    latent_w, latent_h = w // SPATIAL_DOWNSAMPLE, h // SPATIAL_DOWNSAMPLE

    z_BCTHW = torch.randn(
        (B, IN_CHANNELS, latent_t, latent_h, latent_w),
        device=device,
        dtype=torch.float32,
    )

    cond_text = cond_null = None
    if "cond" in conditioning:
        cond_text = conditioning["cond"]
        cond_null = conditioning["null"]
    else:
        assert False, "Batched mode not supported"

    def model_fn(*, z_1BNI, sigma_B, cfg_scale):
        cond_z_1BNI = dit.forward_inner(
            x_1BNI=z_1BNI,
            sigma=sigma_B,
            y_feat_1BLY=cond_y_feat_1BLY,
            y_pool_11BX=cond_y_pool_11BX,
            rope_cos_1HND=rope_cos_1HND,
            rope_sin_1HND=rope_sin_1HND,
            trans_mat=trans_mat,
            uncond=False,
        )

        uncond_z_1BNI = dit.forward_inner(
            x_1BNI=z_1BNI,
            sigma=sigma_B,
            y_feat_1BLY=uncond_y_feat_1BLY,
            y_pool_11BX=uncond_y_pool_11BX,
            rope_cos_1HND=rope_cos_1HND,
            rope_sin_1HND=rope_sin_1HND,
            trans_mat=trans_mat,
            uncond=True,
        )

        assert cond_z_1BNI.shape == uncond_z_1BNI.shape
        return uncond_z_1BNI + cfg_scale * (cond_z_1BNI - uncond_z_1BNI)

    # Preparation before first iteration
    rope_cos_1HND, rope_sin_1HND, trans_mat = dit.prepare_rope_features(T=latent_t, H=latent_h, W=latent_w)
    # Note that conditioning contains list of len 1 to index into
    cond_y_feat_1BLY, cond_y_pool_11BX = dit.prepare_text_features(
        t5_feat=cond_text["y_feat"][0], t5_mask=cond_text["y_mask"][0]
    )
    uncond_y_feat_1BLY, uncond_y_pool_11BX = dit.prepare_text_features(
        t5_feat=cond_null["y_feat"][0], t5_mask=cond_null["y_mask"][0]
    )
    z_1BNI = dit.preprocess_input(z_BCTHW)

    for i in get_new_progress_bar(range(0, sample_steps), desc="Sampling"):
        sigma = sigma_schedule[i]
        dsigma = sigma - sigma_schedule[i + 1]

        sigma_B = torch.full([B], sigma, device=device)
        pred_1BNI = model_fn(z_1BNI=z_1BNI, sigma_B=sigma_B, cfg_scale=cfg_schedule[i])
        # assert pred_BCTHW.dtype == torch.float32
        z_1BNI = z_1BNI + dsigma * pred_1BNI

    # Postprocess z
    z_BCTHW = dit.reverse_preprocess(z_1BNI, latent_t, latent_h, latent_w).float()
    return dit_latents_to_vae_latents(z_BCTHW)


class TTPipeline(MochiSingleGPUPipeline):
    """TensorTorch-specific version of MochiSingleGPUPipeline."""

    def __call__(self, batch_cfg, prompt, negative_prompt, **kwargs):
        with torch.inference_mode():
            print_max_memory = lambda: print(
                f"Max memory reserved: {torch.cuda.max_memory_reserved() / 1024**3:.2f} GB"
            )
            print_max_memory()

            print("get_conditioning")
            with move_to_device(self.text_encoder, self.device):
                conditioning = get_conditioning(
                    tokenizer=self.tokenizer,
                    encoder=self.text_encoder,
                    device=self.device,
                    batch_inputs=batch_cfg,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                )
            print_max_memory()

            print("sample_model")
            latents = sample_model_tt(self.device, self.dit, conditioning, **kwargs)
            print_max_memory()

            with move_to_device(self.decoder, self.device):
                if self.decode_type == "tiled_full":
                    frames = decode_latents_tiled_full(self.decoder, latents, **self.decode_args)
                elif self.decode_type == "tiled_spatial":
                    frames = decode_latents_tiled_spatial(
                        self.decoder, latents, **self.decode_args, num_tiles_w=4, num_tiles_h=2
                    )
                else:
                    frames = decode_latents(self.decoder, latents)
            print_max_memory()
            return frames.cpu().numpy()
