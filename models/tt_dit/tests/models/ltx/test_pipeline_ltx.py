# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
LTX-2 pipeline tests:
- Scheduler primitives (no device needed): sigma schedule, Euler step
- End-to-end one-stage AV generation (LTXPipeline) on a multi-chip mesh
"""

import itertools
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.pipelines.ltx.pipeline_ltx import LTXPipeline, compute_sigmas, euler_step
from models.tt_dit.utils.test import line_params, ring_params


def test_sigma_schedule():
    """Test sigma schedule produces valid monotonically-decreasing values ending at 0."""
    steps = 30
    num_tokens = 2048

    sigmas = compute_sigmas(steps=steps, num_tokens=num_tokens)

    assert sigmas.shape == (steps + 1,), f"Expected {steps+1} sigma values, got {sigmas.shape}"
    assert sigmas[0] > 0, "First sigma must be positive"
    assert sigmas[-1] == 0.0, "Last sigma must be 0"
    # Monotonically decreasing
    for i in range(len(sigmas) - 1):
        assert sigmas[i] >= sigmas[i + 1], f"Sigma not decreasing at index {i}: {sigmas[i]} < {sigmas[i+1]}"
    logger.info(f"Sigma schedule: {sigmas[0]:.4f} -> {sigmas[-1]:.4f} ({len(sigmas)} values)")


def test_euler_step():
    """Test Euler step: x_{t+1} = x_t + velocity * dt."""
    torch.manual_seed(42)
    sample = torch.randn(1, 256, 128)
    denoised = torch.randn(1, 256, 128)
    sigma, sigma_next = 0.8, 0.6

    result = euler_step(sample, denoised, sigma=sigma, sigma_next=sigma_next)

    # Manual computation
    velocity = (sample.float() - denoised.float()) / sigma
    dt = sigma_next - sigma
    expected = (sample.float() + velocity * dt).to(sample.dtype)

    assert torch.allclose(
        result, expected, atol=1e-6
    ), f"Euler step mismatch: max diff {(result - expected).abs().max():.2e}"
    logger.info("Euler step matches manual computation")


# ---------------------------------------------------------------------------
# End-to-end one-stage AV generation (requires device + checkpoint)
# ---------------------------------------------------------------------------


def _default_checkpoint() -> str:
    """Resolve LTX checkpoint: env var > local file > HF repo string default."""
    explicit = os.environ.get("LTX_CHECKPOINT")
    if explicit:
        return explicit
    local = os.path.expanduser("~/.cache/ltx-checkpoints/ltx-2.3-22b-dev.safetensors")
    if os.path.exists(local):
        return local
    return "Lightricks/LTX-2.3:ltx-2.3-22b-dev.safetensors"


def _default_gemma() -> str:
    """Resolve Gemma path: env var > local HF snapshot > HF repo string default."""
    explicit = os.environ.get("GEMMA_PATH")
    if explicit:
        return explicit
    import glob

    candidates = glob.glob(
        os.path.expanduser("~/.cache/huggingface/hub/models--google--gemma-3-12b-it-qat-q4_0-unquantized/snapshots/*/")
    )
    if candidates:
        return candidates[0].rstrip("/")
    return "google/gemma-3-12b-it-qat-q4_0-unquantized"


@pytest.mark.parametrize(
    "no_prompt",
    [{"1": True, "0": False}.get(os.environ.get("NO_PROMPT"), True)],
)
@pytest.mark.parametrize(
    "mesh_device, mesh_shape, sp_axis, tp_axis, num_links, dynamic_load, device_params, topology, is_fsdp",
    [
        [(2, 2), (2, 2), 0, 1, 2, False, line_params, ttnn.Topology.Linear, True],
        [(2, 4), (2, 4), 0, 1, 1, True, line_params, ttnn.Topology.Linear, True],
        # BH on 2x4
        [(2, 4), (2, 4), 1, 0, 2, True, line_params, ttnn.Topology.Linear, False],
        # WH (ring) on 4x8
        [(4, 8), (4, 8), 1, 0, 4, False, ring_params, ttnn.Topology.Ring, True],
        # BH (linear) on 4x8
        [(4, 8), (4, 8), 1, 0, 2, False, line_params, ttnn.Topology.Linear, False],
        # BH (ring) on 4x8
        [(4, 8), (4, 8), 1, 0, 2, False, ring_params, ttnn.Topology.Ring, False],
        [(4, 32), (4, 32), 1, 0, 2, False, ring_params, ttnn.Topology.Ring, False],
    ],
    ids=[
        "2x2sp0tp1",
        "2x4sp0tp1",
        "bh_2x4sp1tp0",
        "wh_4x8sp1tp0",
        "bh_4x8sp1tp0_linear",
        "bh_4x8sp1tp0_ring",
        "bh_4x32sp1tp0",
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    "width, height",
    [
        (768, 512),
    ],
    ids=[
        "resolution_512p",
    ],
)
def test_pipeline_inference(
    mesh_device,
    mesh_shape,
    sp_axis,
    tp_axis,
    num_links,
    dynamic_load,
    topology,
    width,
    height,
    is_fsdp,
    no_prompt,
):
    ckpt = _default_checkpoint()
    gemma = _default_gemma()
    # ckpt / gemma always resolve (env var → local → HF repo string fallback). The pipeline's
    # resolver downloads from HF if needed.

    parent_mesh = mesh_device
    mesh_device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))

    num_frames = int(os.environ.get("NUM_FRAMES", "121"))
    num_inference_steps = int(os.environ.get("NUM_STEPS", "30"))
    width = int(os.environ.get("WIDTH", width))
    height = int(os.environ.get("HEIGHT", height))

    run_warmup = os.environ.get("RUN_WARMUP", "0") in ("1", "true", "True")

    pipeline = LTXPipeline.create_pipeline(
        mesh_device=mesh_device,
        checkpoint_name=ckpt,
        gemma_path=gemma,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        num_links=num_links,
        dynamic_load=dynamic_load,
        topology=topology,
        is_fsdp=is_fsdp,
        run_warmup=run_warmup,
        num_frames=num_frames,
        height=height,
        width=width,
    )

    prompt = os.environ.get(
        "PROMPT",
        "a cat playing piano",
    )

    def run(*, prompt, number, seed):
        logger.info(f"Running inference with prompt: '{prompt}'")
        logger.info(f"Parameters: {height}x{width}, {num_frames} frames, {num_inference_steps} steps")

        output_filename = os.environ.get("OUTPUT_PATH", f"ltx_av_pro_{width}x{height}_{number}.mp4")

        pipeline.generate(
            prompt,
            output_path=output_filename,
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            seed=seed,
        )

        if int(ttnn.distributed_context_get_rank()) == 0:
            logger.info(f"Saved video to: {output_filename}")
        else:
            logger.info(f"Skipping video export on rank {ttnn.distributed_context_get_rank()}")

    if no_prompt:
        run(prompt=prompt, number=0, seed=42)
    else:
        for i in itertools.count():
            new_prompt = input("Enter the input prompt, or q to exit: ")
            if new_prompt:
                prompt = new_prompt
            if prompt[0] == "q":
                break
            run(prompt=prompt, number=i, seed=i)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
