# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
UNet Deep Dive Investigation Test

This test module implements comprehensive parallel PyTorch + TT UNet execution
with detailed per-step instrumentation to identify the root cause of SSIM
degradation from 0.95 to 0.6879.

Key Features:
- Parallel PyTorch and TT execution with identical inputs
- Per-timestep capture of: raw_unet_output, post_guidance_output, scheduler_step_output
- JSON file-based output for reliability
- Memory-efficient: tensors converted to CPU after each step
- Single-step test capability for validation before full run

Usage:
    # Single-step validation
    pytest test_unet_deep_dive.py::test_unet_deep_dive_single_step -v
    
    # Full 50-step analysis
    pytest test_unet_deep_dive.py::test_unet_deep_dive_full -v
"""

import gc
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import torch
from diffusers import DiffusionPipeline
from loguru import logger
from tqdm import tqdm

import ttnn
from models.common.utility_functions import is_wormhole_b0
from models.experimental.stable_diffusion_xl_base.tests.pcc.unet_analysis_utils import (
    compute_tensor_stats,
    compute_comparison_metrics,
    save_step_data,
    aggregate_run_data,
    generate_analysis_csv,
    analyze_spatial_error,
    identify_component_contribution,
)
from models.experimental.stable_diffusion_xl_base.tests.test_common import (
    SDXL_L1_SMALL_SIZE,
    SDXL_TRACE_REGION_SIZE,
    retrieve_timesteps,
    run_tt_iteration,
    prepare_input_tensors,
    allocate_input_tensors,
    create_user_tensors,
)
from models.experimental.stable_diffusion_xl_base.tt.model_configs import ModelOptimisations
from models.experimental.stable_diffusion_xl_base.tt.tt_euler_discrete_scheduler import TtEulerDiscreteScheduler
from models.experimental.stable_diffusion_xl_base.tt.tt_unet import TtUNet2DConditionModel
from tests.ttnn.utils_for_testing import comp_pcc


# Fixed test configuration for reproducibility
DEFAULT_CONFIG = {
    "prompt": "A cat sitting on a windowsill",
    "seed": 42,
    "guidance_scale": 5.0,
    "height": 1024,
    "width": 1024,
}


def create_output_directory() -> str:
    """Create timestamped output directory for this run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"/tmp/unet_deep_dive/run_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def run_pytorch_unet_step(
    pipeline,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    prompt_embeds: torch.Tensor,
    added_cond_kwargs: dict,
    guidance_scale: float,
    extra_step_kwargs: dict,
) -> dict:
    """
    Execute a single PyTorch UNet denoising step with intermediate captures.
    
    Returns:
        Dictionary containing:
        - raw_unet_output: UNet output before guidance
        - noise_pred_uncond: Unconditional prediction
        - noise_pred_text: Text-conditional prediction
        - post_guidance_output: After guidance computation
        - scheduler_output: After scheduler step
        - latents: Final latents for next step
    """
    # Prepare input (duplicate for classifier-free guidance)
    latent_model_input = torch.cat([latents] * 2)
    latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, timestep)
    
    # Run UNet
    with torch.no_grad():
        noise_pred = pipeline.unet(
            latent_model_input,
            timestep,
            encoder_hidden_states=prompt_embeds,
            timestep_cond=None,
            cross_attention_kwargs=None,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]
    
    # Split for guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    raw_unet_output = noise_pred_text.clone()  # Store before guidance
    
    # Apply classifier-free guidance
    post_guidance = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    
    # Scheduler step
    scheduler_output = pipeline.scheduler.step(
        post_guidance, timestep, latents, **extra_step_kwargs, return_dict=False
    )[0]
    
    return {
        "raw_unet_output": raw_unet_output,
        "noise_pred_uncond": noise_pred_uncond,
        "noise_pred_text": noise_pred_text,
        "post_guidance_output": post_guidance,
        "scheduler_output": scheduler_output,
        "latents": scheduler_output,
    }


def run_tt_unet_step(
    ttnn_device,
    tt_latents_device,
    tt_unet,
    tt_scheduler,
    input_shape: list,
    tt_prompt_embeds_device: list,
    tt_text_embeds_device: list,
    tt_time_ids_device: list,
    guidance_scale: float,
    extra_step_kwargs: dict,
) -> dict:
    """
    Execute a single TT UNet denoising step with intermediate captures.
    
    Returns:
        Dictionary containing:
        - raw_unet_output: UNet output (text prediction)
        - noise_pred_uncond: Unconditional prediction
        - post_guidance_output: After guidance computation
        - scheduler_output: After scheduler step
        - tt_latents_device: Device tensor for next step
    """
    B, C, H, W = input_shape
    unet_outputs = []
    
    # Run UNet for both unconditional and conditional
    for unet_slice in range(len(tt_prompt_embeds_device)):
        tt_latent_model_input = tt_latents_device
        noise_pred, noise_shape = run_tt_iteration(
            tt_unet,
            tt_scheduler,
            tt_latent_model_input,
            [B, C, H, W],
            tt_prompt_embeds_device[unet_slice],
            tt_time_ids_device[unet_slice],
            tt_text_embeds_device[unet_slice],
        )
        C, H, W = noise_shape
        unet_outputs.append(noise_pred)
    
    noise_pred_uncond, noise_pred_text = unet_outputs
    
    # Capture raw UNet output before guidance
    raw_unet_torch = ttnn.to_torch(noise_pred_text).cpu()
    uncond_torch = ttnn.to_torch(noise_pred_uncond).cpu()
    
    # Apply guidance (in-place operations)
    noise_pred_text = ttnn.sub_(noise_pred_text, noise_pred_uncond)
    noise_pred_text = ttnn.mul_(noise_pred_text, guidance_scale)
    noise_pred = ttnn.add_(noise_pred_uncond, noise_pred_text)
    
    # Capture post-guidance output
    post_guidance_torch = ttnn.to_torch(noise_pred).cpu()
    
    # Scheduler step
    tt_latents = tt_scheduler.step(
        noise_pred, None, tt_latents_device, **extra_step_kwargs, return_dict=False
    )[0]
    
    # Capture scheduler output
    scheduler_torch = ttnn.to_torch(tt_latents).cpu()
    
    # Cleanup
    ttnn.deallocate(noise_pred_text)
    
    return {
        "raw_unet_output": raw_unet_torch,
        "noise_pred_uncond": uncond_torch,
        "post_guidance_output": post_guidance_torch,
        "scheduler_output": scheduler_torch,
        "tt_latents_device": tt_latents,
        "output_shape": [C, H, W],
    }


def reshape_tt_to_pytorch(tt_tensor: torch.Tensor, B: int, C: int, H: int, W: int) -> torch.Tensor:
    """Reshape TT tensor from (B, 1, H*W, C) to PyTorch format (B, C, H, W)."""
    tt_reshaped = tt_tensor.reshape(B, H, W, C)
    tt_reshaped = torch.permute(tt_reshaped, (0, 3, 1, 2))
    return tt_reshaped


@torch.no_grad()
def run_comparison_analysis(
    ttnn_device,
    num_steps: int = 50,
    config: dict = None,
) -> str:
    """
    Run parallel PyTorch + TT UNet execution with comprehensive instrumentation.
    
    Args:
        ttnn_device: TT device handle
        num_steps: Number of denoising steps (default 50)
        config: Override default configuration
        
    Returns:
        Path to output directory containing all analysis files
    """
    config = config or DEFAULT_CONFIG
    output_dir = create_output_directory()
    
    logger.info(f"Starting UNet deep dive analysis")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Configuration: {config}")
    
    # Save configuration
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    # Set seed for reproducibility
    torch.manual_seed(config["seed"])
    
    # Load pipeline
    logger.info("Loading diffusion pipeline...")
    pipeline = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float32,
        use_safetensors=True,
    )
    
    # Load TT models
    logger.info("Loading TT UNet and scheduler...")
    tt_model_config = ModelOptimisations()
    tt_unet = TtUNet2DConditionModel(
        ttnn_device,
        pipeline.unet.state_dict(),
        "unet",
        model_config=tt_model_config,
    )
    tt_scheduler = TtEulerDiscreteScheduler(
        ttnn_device,
        pipeline.scheduler.config.num_train_timesteps,
        pipeline.scheduler.config.beta_start,
        pipeline.scheduler.config.beta_end,
        pipeline.scheduler.config.beta_schedule,
        pipeline.scheduler.config.trained_betas,
        pipeline.scheduler.config.prediction_type,
        pipeline.scheduler.config.interpolation_type,
        pipeline.scheduler.config.use_karras_sigmas,
        pipeline.scheduler.config.use_exponential_sigmas,
        pipeline.scheduler.config.use_beta_sigmas,
        pipeline.scheduler.config.sigma_min,
        pipeline.scheduler.config.sigma_max,
        pipeline.scheduler.config.timestep_spacing,
        pipeline.scheduler.config.timestep_type,
        pipeline.scheduler.config.steps_offset,
        pipeline.scheduler.config.rescale_betas_zero_snr,
        pipeline.scheduler.config.final_sigmas_type,
    )
    
    # Encode prompt
    logger.info("Encoding prompt...")
    prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = (
        pipeline.encode_prompt(
            prompt=config["prompt"],
            prompt_2=None,
            device="cpu",
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=None,
        )
    )
    
    # Prepare timesteps
    timesteps, num_inference_steps = retrieve_timesteps(pipeline.scheduler, num_steps, "cpu", None, None)
    ttnn_timesteps, tt_num_inference_steps = retrieve_timesteps(tt_scheduler, num_steps, "cpu", None, None)
    
    # Prepare latents
    num_channels_latents = pipeline.unet.config.in_channels
    latents = pipeline.prepare_latents(
        1,
        num_channels_latents,
        config["height"],
        config["width"],
        prompt_embeds.dtype,
        "cpu",
        None,
        None,
    )
    B, C, H, W = latents.shape
    
    extra_step_kwargs = pipeline.prepare_extra_step_kwargs(None, 0.0)
    
    # Prepare added conditions
    text_encoder_projection_dim = pipeline.text_encoder_2.config.projection_dim
    original_size = (config["height"], config["width"])
    target_size = (config["height"], config["width"])
    crops_coords_top_left = (0, 0)
    add_time_ids = pipeline._get_add_time_ids(
        original_size,
        crops_coords_top_left,
        target_size,
        dtype=prompt_embeds.dtype,
        text_encoder_projection_dim=text_encoder_projection_dim,
    )
    negative_add_time_ids = add_time_ids
    
    # Prepare TT tensors
    tt_latents_cpu = torch.permute(latents, (0, 2, 3, 1))
    tt_latents_cpu = tt_latents_cpu.reshape(1, 1, B * H * W, C)
    
    tt_latents, tt_prompt_embeds, tt_add_text_embeds = create_user_tensors(
        ttnn_device=ttnn_device,
        latents=tt_latents_cpu,
        negative_prompt_embeds=[negative_prompt_embeds],
        prompt_embeds=[prompt_embeds],
        negative_pooled_prompt_embeds=[negative_pooled_prompt_embeds],
        add_text_embeds=[pooled_prompt_embeds],
    )
    
    tt_latents_device, tt_prompt_embeds_device, tt_text_embeds_device, tt_time_ids_device = allocate_input_tensors(
        ttnn_device=ttnn_device,
        tt_latents=tt_latents,
        tt_prompt_embeds=tt_prompt_embeds,
        tt_text_embeds=tt_add_text_embeds,
        tt_time_ids=[negative_add_time_ids, add_time_ids],
    )
    
    # Prepare PyTorch inputs
    pt_prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
    pt_add_text_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
    pt_add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)
    pt_added_cond_kwargs = {"text_embeds": pt_add_text_embeds, "time_ids": pt_add_time_ids}
    
    # Warmup run
    logger.info("Performing warmup run...")
    prepare_input_tensors(
        [tt_latents, *tt_prompt_embeds[0], tt_add_text_embeds[0][0], tt_add_text_embeds[0][1]],
        [tt_latents_device, *tt_prompt_embeds_device, *tt_text_embeds_device],
    )
    _ = run_tt_unet_step(
        ttnn_device,
        tt_latents_device,
        tt_unet,
        tt_scheduler,
        [B, C, H, W],
        tt_prompt_embeds_device,
        tt_text_embeds_device,
        tt_time_ids_device,
        config["guidance_scale"],
        extra_step_kwargs,
    )
    ttnn.synchronize_device(ttnn_device)
    
    # Reset for actual run
    tt_scheduler.set_step_index(0)
    prepare_input_tensors(
        [tt_latents, *tt_prompt_embeds[0], tt_add_text_embeds[0][0], tt_add_text_embeds[0][1]],
        [tt_latents_device, *tt_prompt_embeds_device, *tt_text_embeds_device],
    )
    
    # Main denoising loop with instrumentation
    logger.info(f"Starting {num_steps}-step denoising with instrumentation...")
    pcc_values = []
    pt_latents = latents.clone()
    
    for step_idx, (t, tt_t) in tqdm(enumerate(zip(timesteps, ttnn_timesteps)), total=num_steps):
        timestep_value = int(t.item())
        logger.info(f"Step {step_idx}: timestep={timestep_value}")
        
        # Run PyTorch step
        pt_results = run_pytorch_unet_step(
            pipeline,
            pt_latents,
            t,
            pt_prompt_embeds,
            pt_added_cond_kwargs,
            config["guidance_scale"],
            extra_step_kwargs,
        )
        
        # Run TT step
        tt_results = run_tt_unet_step(
            ttnn_device,
            tt_latents_device,
            tt_unet,
            tt_scheduler,
            [B, C, H, W],
            tt_prompt_embeds_device,
            tt_text_embeds_device,
            tt_time_ids_device,
            config["guidance_scale"],
            extra_step_kwargs,
        )
        
        ttnn.synchronize_device(ttnn_device)
        
        # Reshape TT outputs for comparison
        tt_raw_unet = reshape_tt_to_pytorch(tt_results["raw_unet_output"], B, C, H, W)
        tt_post_guidance = reshape_tt_to_pytorch(tt_results["post_guidance_output"], B, C, H, W)
        tt_scheduler_out = reshape_tt_to_pytorch(tt_results["scheduler_output"], B, C, H, W)
        
        # Compute statistics and comparisons
        pt_stats = compute_tensor_stats(pt_results["scheduler_output"], "pytorch_latents")
        tt_stats = compute_tensor_stats(tt_scheduler_out, "tt_latents")
        
        # Compare at each stage
        raw_unet_comparison = compute_comparison_metrics(
            pt_results["raw_unet_output"], tt_raw_unet, "raw_unet"
        )
        guidance_comparison = compute_comparison_metrics(
            pt_results["post_guidance_output"], tt_post_guidance, "post_guidance"
        )
        scheduler_comparison = compute_comparison_metrics(
            pt_results["scheduler_output"], tt_scheduler_out, "scheduler_output"
        )
        
        # Spatial error analysis on scheduler output
        spatial_analysis = analyze_spatial_error(
            pt_results["scheduler_output"],
            tt_scheduler_out,
            expected_shape=(B, C, H, W)
        )
        
        # Component contribution analysis
        component_analysis = identify_component_contribution(
            raw_unet_comparison,
            guidance_comparison,
            scheduler_comparison
        )
        
        # Track PCC progression
        latent_pcc = scheduler_comparison.get("pcc", float("nan"))
        pcc_values.append(latent_pcc)
        
        # Save step data
        metadata = {
            "timestep": timestep_value,
            "step_index": step_idx,
            "guidance_scale": config["guidance_scale"],
        }
        
        comparison_data = {
            "latent_comparison": scheduler_comparison,
            "raw_unet_comparison": raw_unet_comparison,
            "guidance_comparison": guidance_comparison,
            "spatial_analysis": spatial_analysis,
            "component_analysis": component_analysis,
        }
        
        save_step_data(
            step_index=step_idx,
            output_dir=output_dir,
            pytorch_stats=pt_stats,
            tt_stats=tt_stats,
            comparison=comparison_data,
            metadata=metadata,
        )
        
        logger.info(f"  PCC: {latent_pcc:.6f}, UNet PCC: {raw_unet_comparison.get('pcc', 'N/A'):.6f}")
        
        # Update for next iteration
        pt_latents = pt_results["latents"]
        tt_latents_device = tt_results["tt_latents_device"]
        C, H, W = tt_results["output_shape"]
        
        if step_idx < num_steps - 1:
            tt_scheduler.inc_step_index()
        
        # Memory cleanup
        gc.collect()
    
    # Aggregate results
    logger.info("Aggregating results...")
    summary = aggregate_run_data(output_dir)
    csv_path = generate_analysis_csv(output_dir)
    
    logger.info(f"Analysis complete!")
    logger.info(f"Summary: {summary.get('summary_path', 'N/A')}")
    logger.info(f"CSV: {csv_path}")
    logger.info(f"Initial PCC: {pcc_values[0]:.6f}")
    logger.info(f"Final PCC: {pcc_values[-1]:.6f}")
    
    # Print divergence analysis
    if "divergence_analysis" in summary:
        div = summary["divergence_analysis"]
        logger.info(f"Divergence pattern: {div.get('degradation_pattern', 'unknown')}")
        logger.info(f"Divergence step: {div.get('divergence_step', 'N/A')}")
    
    return output_dir


# =============================================================================
# PYTEST TEST FUNCTIONS
# =============================================================================

@pytest.mark.skipif(not is_wormhole_b0(), reason="SDXL supported on WH only")
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": SDXL_L1_SMALL_SIZE, "trace_region_size": SDXL_TRACE_REGION_SIZE}],
    indirect=True,
)
@pytest.mark.timeout(600)
def test_unet_deep_dive_single_step(device):
    """
    Single-step validation test.
    
    Validates that the instrumentation framework works correctly
    before running the full 50-step analysis.
    """
    output_dir = run_comparison_analysis(
        ttnn_device=device,
        num_steps=1,
        config=DEFAULT_CONFIG,
    )
    
    # Validate output
    step_file = os.path.join(output_dir, "step_000.json")
    assert os.path.exists(step_file), f"Step file not created: {step_file}"
    
    with open(step_file, "r") as f:
        data = json.load(f)
    
    # Validate structure
    assert "step_index" in data
    assert "pytorch" in data
    assert "tt" in data
    assert "comparison" in data
    assert "latent_comparison" in data["comparison"]
    
    # Validate PCC is computed
    pcc = data["comparison"]["latent_comparison"].get("pcc")
    assert pcc is not None
    assert not np.isnan(pcc), "PCC is NaN"
    
    logger.info(f"Single-step validation passed. PCC: {pcc:.6f}")
    logger.info(f"Output: {output_dir}")


@pytest.mark.skipif(not is_wormhole_b0(), reason="SDXL supported on WH only")
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": SDXL_L1_SMALL_SIZE, "trace_region_size": SDXL_TRACE_REGION_SIZE}],
    indirect=True,
)
@pytest.mark.timeout(3600)
def test_unet_deep_dive_full(device):
    """
    Full 50-step analysis test.
    
    Runs complete denoising loop with instrumentation and generates
    comprehensive analysis report.
    """
    output_dir = run_comparison_analysis(
        ttnn_device=device,
        num_steps=50,
        config=DEFAULT_CONFIG,
    )
    
    # Validate complete dataset
    for i in range(50):
        step_file = os.path.join(output_dir, f"step_{i:03d}.json")
        assert os.path.exists(step_file), f"Missing step file: {step_file}"
    
    # Validate summary
    summary_file = os.path.join(output_dir, "summary.json")
    assert os.path.exists(summary_file), "Summary file not created"
    
    with open(summary_file, "r") as f:
        summary = json.load(f)
    
    assert summary["num_steps"] == 50
    assert "pcc_progression" in summary
    assert "divergence_analysis" in summary
    
    logger.info(f"Full analysis complete: {output_dir}")
    logger.info(f"Final PCC: {summary['pcc_stats']['final']:.6f}")
    logger.info(f"Divergence pattern: {summary['divergence_analysis']['degradation_pattern']}")


@pytest.mark.skipif(not is_wormhole_b0(), reason="SDXL supported on WH only")
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": SDXL_L1_SMALL_SIZE, "trace_region_size": SDXL_TRACE_REGION_SIZE}],
    indirect=True,
)
@pytest.mark.parametrize("num_steps", [10, 25])
@pytest.mark.timeout(1800)
def test_unet_deep_dive_checkpoints(device, num_steps):
    """
    Checkpoint analysis at key step counts.
    
    Useful for understanding how error evolves at different inference lengths.
    """
    output_dir = run_comparison_analysis(
        ttnn_device=device,
        num_steps=num_steps,
        config=DEFAULT_CONFIG,
    )
    
    # Validate output
    summary_file = os.path.join(output_dir, "summary.json")
    assert os.path.exists(summary_file)
    
    with open(summary_file, "r") as f:
        summary = json.load(f)
    
    assert summary["num_steps"] == num_steps
    logger.info(f"{num_steps}-step analysis complete")
    logger.info(f"Final PCC: {summary['pcc_stats']['final']:.6f}")


if __name__ == "__main__":
    # Allow running directly for debugging
    import argparse
    
    parser = argparse.ArgumentParser(description="UNet Deep Dive Analysis")
    parser.add_argument("--steps", type=int, default=50, help="Number of denoising steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--prompt", type=str, default=DEFAULT_CONFIG["prompt"], help="Prompt")
    args = parser.parse_args()
    
    config = {
        **DEFAULT_CONFIG,
        "seed": args.seed,
        "prompt": args.prompt,
    }
    
    # Manual device setup for direct execution
    print("This script should be run via pytest to properly initialize the device.")
    print("Example: pytest test_unet_deep_dive.py::test_unet_deep_dive_single_step -v")
