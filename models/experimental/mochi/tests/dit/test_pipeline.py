import torch
import time
import pytest
from loguru import logger
import os
import ttnn
import numpy as np
from models.utility_functions import skip_for_grayskull
from models.experimental.mochi.tt.common import compute_metrics
from genmo.mochi_preview.pipelines import sample_model as reference_sample_model
from models.experimental.mochi.pipelines import sample_model as sample_model_tt
from genmo.mochi_preview.pipelines import (
    linear_quadratic_schedule,
    get_conditioning,
    t5_tokenizer,
    T5_MODEL,
)
from transformers import T5EncoderModel
from models.experimental.mochi.tests.dit.test_model import create_models


@torch.no_grad()
@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("FAKE_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("n_layers", [1, 2, 4, 48], ids=["L1", "L2", "L4", "L48"])
def test_sample_model(mesh_device, use_program_cache, reset_seeds, n_layers):
    """Test TensorTorch sample_model against reference implementation."""
    mesh_device.enable_async(True)
    device = torch.device("cpu")

    # Create models
    ref_dit, tt_dit, _ = create_models(mesh_device, n_layers)

    # Test parameters
    width = 848
    height = 480
    num_frames = 163
    num_steps = 1
    cfg_scale = 6.0
    prompt = """A hand with delicate fingers picks up a bright yellow lemon from a wooden bowl
    filled with lemons and sprigs of mint against a peach-colored background."""
    negative_prompt = ""

    # Create schedules
    # sigma_schedule = linear_quadratic_schedule(num_steps, 0.025)
    sigma_schedule = linear_quadratic_schedule(num_steps + 1, 0.025)[: num_steps + 1]
    cfg_schedule = [cfg_scale] * num_steps

    # Get text conditioning
    tokenizer = t5_tokenizer(T5_MODEL)
    text_encoder = T5EncoderModel.from_pretrained(T5_MODEL)
    text_encoder.eval()
    conditioning = get_conditioning(
        tokenizer=tokenizer,
        encoder=text_encoder,
        device=torch.device("cpu"),  # Keep on CPU as in pipeline
        batch_inputs=False,
        prompt=prompt,
        negative_prompt=negative_prompt,
    )

    # Common arguments for both sample_model functions
    sample_args = {
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "sigma_schedule": sigma_schedule,
        "cfg_schedule": cfg_schedule,
        "num_inference_steps": num_steps,
        "seed": 1234,
    }

    # Run both implementations
    logger.info("Running reference sample_model")
    reference_output = reference_sample_model(device=device, dit=ref_dit, conditioning=conditioning, **sample_args)

    logger.info("Running TT sample_model")
    tt_output = sample_model_tt(device=device, dit=tt_dit, conditioning=conditioning, **sample_args)

    # Compute metrics
    pcc_required = 0.985
    pcc, mse, mae = compute_metrics(reference_output, tt_output)

    logger.info(f"Sample Model Output Metrics:")
    logger.info(f"PCC: {pcc}, MSE: {mse}, MAE: {mae}")

    passing = pcc >= pcc_required

    if passing:
        logger.info("Sample model test Passed!")
    else:
        logger.warning("Sample model test Failed!")
        logger.warning(f"PCC {pcc} below required {pcc_required}")

    assert passing, f"Sample model output does not meet PCC requirement {pcc_required}"


@torch.no_grad()
@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("FAKE_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("n_layers", [1, 2, 4, 48], ids=["L1", "L2", "L4", "L48"])
def test_sample_model_perf(mesh_device, use_program_cache, reset_seeds, n_layers):
    from genmo.lib.progress import get_new_progress_bar
    from genmo.mochi_preview.vae.vae_stats import dit_latents_to_vae_latents

    """Test TensorTorch sample_model against reference implementation."""
    mesh_device.enable_async(True)
    device = torch.device("cpu")

    # Create models
    ref_dit, tt_dit, _ = create_models(mesh_device, n_layers)
    del ref_dit

    # Test parameters
    width = 848
    height = 480
    num_frames = 163
    cfg_scale = 6.0
    prompt = """A hand with delicate fingers picks up a bright yellow lemon from a wooden bowl
    filled with lemons and sprigs of mint against a peach-colored background."""
    negative_prompt = ""

    # Create schedules
    # sigma_schedule = linear_quadratic_schedule(num_steps, 0.025)
    def get_schedule(num_steps):
        sigma_schedule = linear_quadratic_schedule(num_steps, 0.025)
        cfg_schedule = [cfg_scale] * num_steps
        return sigma_schedule, cfg_schedule

    # Get text conditioning
    tokenizer = t5_tokenizer(T5_MODEL)
    text_encoder = T5EncoderModel.from_pretrained(T5_MODEL)
    text_encoder.eval()
    conditioning = get_conditioning(
        tokenizer=tokenizer,
        encoder=text_encoder,
        device=torch.device("cpu"),  # Keep on CPU as in pipeline
        batch_inputs=False,
        prompt=prompt,
        negative_prompt=negative_prompt,
    )

    num_warmup_steps = 2
    warmup_sigma, warmup_cfg = get_schedule(num_warmup_steps)
    sample_args = {
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "sigma_schedule": warmup_sigma,
        "cfg_schedule": warmup_cfg,
        "num_inference_steps": num_warmup_steps,
        "seed": 1234,
    }

    def sample_model_profile(device, dit, conditioning, **args):
        start = time.perf_counter()
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

        host_prepare = time.perf_counter() - start

        def model_fn(*, z_1BNI, sigma_B, cfg_scale):
            print("in model_fn")
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

        start = time.perf_counter()
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
        ttnn.synchronize_device(mesh_device)
        device_prepare = time.perf_counter() - start

        start = time.perf_counter()
        for i in get_new_progress_bar(range(0, sample_steps), desc="Sampling"):
            sigma = sigma_schedule[i]
            dsigma = sigma - sigma_schedule[i + 1]

            sigma_B = torch.full([B], sigma, device=device)
            pred_1BNI = model_fn(z_1BNI=z_1BNI, sigma_B=sigma_B, cfg_scale=cfg_schedule[i])
            # assert pred_BCTHW.dtype == torch.float32
            z_1BNI = z_1BNI + dsigma * pred_1BNI
        ttnn.synchronize_device(mesh_device)
        device_steps = time.perf_counter() - start
        # Postprocess z
        start = time.perf_counter()
        z_BCTHW = dit.reverse_preprocess(z_1BNI, latent_t, latent_h, latent_w).float()
        device_postprocess = time.perf_counter() - start
        ttnn.synchronize_device(mesh_device)
        start = time.perf_counter()
        latents = dit_latents_to_vae_latents(z_BCTHW)
        host_postprocess = time.perf_counter() - start
        return {
            "host_prepare": host_prepare,
            "device_prepare": device_prepare,
            "device_steps": device_steps,
            "device_postprocess": device_postprocess,
            "host_postprocess": host_postprocess,
        }

    logger.info("Running warmup")
    warmup_output = sample_model_profile(device=device, dit=tt_dit, conditioning=conditioning, **sample_args)

    num_bench_steps = 10
    bench_sigma, bench_cfg = get_schedule(num_bench_steps)
    bench_args = {
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "sigma_schedule": bench_sigma,
        "cfg_schedule": bench_cfg,
        "num_inference_steps": num_bench_steps,
        "seed": 1234,
    }
    logger.info(f"Running benchmark for {num_bench_steps} steps")
    bench_output = sample_model_profile(device=device, dit=tt_dit, conditioning=conditioning, **bench_args)

    logger.info(f"Host prepare: {bench_output['host_prepare']} s")
    logger.info(f"Device prepare: {bench_output['device_prepare']} s")
    logger.info(f"Device steps: {bench_output['device_steps']} s")
    logger.info(f"Device per-step: {bench_output['device_steps'] / num_bench_steps} s")
    logger.info(f"Device postprocess: {bench_output['device_postprocess']} s")
    logger.info(f"Host postprocess: {bench_output['host_postprocess']} s")
