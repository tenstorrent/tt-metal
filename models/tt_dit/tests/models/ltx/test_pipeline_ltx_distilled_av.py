# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import itertools
import os

import pytest
from loguru import logger

import ttnn
from models.tt_dit.pipelines.ltx.pipeline_ltx_distilled import LTXDistilledPipeline
from models.tt_dit.utils.test import line_params, ring_params

# Trace region for LTX_TRACED=1. Holds both stage traces' command streams (s1 + larger-seq
# s2); measured need is ~236 MB at 1080p (get_trace_buffers_size), so 300 MB gives headroom.
ring_trace_params = {**ring_params, "trace_region_size": 300_000_000}


def _default_checkpoint() -> str:
    """Resolve distilled checkpoint: env var > local file > HF repo string default."""
    explicit = os.environ.get("LTX_CHECKPOINT")
    if explicit:
        return explicit
    local = os.path.expanduser("~/.cache/ltx-checkpoints/ltx-2.3-22b-distilled-1.1.safetensors")
    if os.path.exists(local):
        return local
    return "Lightricks/LTX-2.3:ltx-2.3-22b-distilled-1.1.safetensors"


def _default_gemma() -> str:
    """Resolve Gemma: env var > HF repo id (the pipeline resolves + caches it). Passing the
    repo id (not a pre-resolved snapshot dir) keeps the on-device cache under a readable name."""
    return os.environ.get("GEMMA_PATH") or "google/gemma-3-12b-it-qat-q4_0-unquantized"


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
        # WH (ring) on 4x8. Requires increased worker_l1_size to avoid code-size error in RingAttention.
        [(4, 8), (4, 8), 1, 0, 4, True, {"worker_l1_size": 1344544, **ring_params}, ttnn.Topology.Ring, True],
        # BH (linear) on 4x8
        [(4, 8), (4, 8), 1, 0, 2, False, line_params, ttnn.Topology.Linear, False],
        # BH (ring) on 4x8
        [(4, 8), (4, 8), 1, 0, 2, False, ring_trace_params, ttnn.Topology.Ring, False],
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
def test_pipeline_av_fast(
    mesh_device,
    mesh_shape,
    sp_axis,
    tp_axis,
    num_links,
    dynamic_load,
    topology,
    is_fsdp,
    no_prompt,
):
    """LTX-2.3 distilled 2-stage AV pipeline."""
    ckpt = _default_checkpoint()
    gemma = _default_gemma()

    parent_mesh = mesh_device
    mesh_device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))

    # Stage-2 output target: 1080p @ 24fps, ~6s. Constraints from the pipeline:
    #   - height/width must be divisible by 64 (asserted at the top of generate(),
    #     so half-res stage-1 is divisible by 32 for the VAE).
    #     1920 is already % 64; 1080 is not → round up to 1088 (next multiple of 64).
    #     If you need exactly 1080 lines, post-crop the 8 extra rows.
    #   - num_frames must satisfy (num_frames - 1) % 8 == 0 so the VAE decoder maps
    #     latent_frames → num_frames frames exactly. 144 (== 6s × 24fps) doesn't
    #     satisfy this; 145 does (19 latent frames → 145 decoded frames ≈ 6.04s).
    num_frames = int(os.environ.get("NUM_FRAMES", "145"))
    height = int(os.environ.get("HEIGHT", "1088"))
    width = int(os.environ.get("WIDTH", "1920"))

    run_warmup = os.environ.get("RUN_WARMUP", "0") in ("1", "true", "True")
    traced = os.environ.get("LTX_TRACED", "0") in ("1", "true", "True")

    pipeline = LTXDistilledPipeline.create_pipeline(
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
        traced=traced,
        num_frames=num_frames,
        height=height,
        width=width,
    )

    prompt = os.environ.get(
        "PROMPT",
        (
            "A young woman with shoulder-length wavy brown hair sits on a wooden stool, "
            "cradling an acoustic guitar. The camera holds a steady medium close-up, "
            "framing her face and guitar neck. Warm key light illuminates her left side "
            "while soft fill light prevents harsh shadows. She strums gently, looking "
            "directly at camera with genuine warmth. Her mouth opens clearly as she sings "
            '"Doo-be-doo, doo-be-day, oh what a sunny day" with precise lip sync and '
            "natural facial expressions. Her head moves subtly with the rhythm. Simple "
            "chord progression underlies her melodic voice. Shot with 50mm lens at f/2.0, "
            "shallow depth of field, warm color grade emphasizing skin tones."
        ),
    )

    def run(*, prompt, number, seed):
        output_filename = os.environ.get("OUTPUT_PATH", f"ltx_av_fast_{width}x{height}_{number}.mp4")
        logger.info(f"Running LTX AV Fast: '{prompt[:80]}...'")
        logger.info(f"Config: {height}x{width}, {num_frames} frames")

        if int(ttnn.distributed_context_get_rank()) != 0:
            logger.info(f"Skipping generation on rank {ttnn.distributed_context_get_rank()}")
            return

        pipeline.generate(
            prompt,
            output_path=output_filename,
            num_frames=num_frames,
            height=height,
            width=width,
            seed=seed,
        )
        logger.info(f"Saved video to: {output_filename}")

    if no_prompt:
        seed = int(os.environ.get("SEED", "10"))
        run(prompt=prompt, number=0, seed=seed)
        # Traced: gen #0 captures (lazily, on first step of each stage); gen #1 is pure
        # replay — its Stage 1/2 denoise times are the steady-state measurement.
        if traced:
            logger.info("=== traced steady-state pass (gen #1, pure replay) ===")
            run(prompt=prompt, number=1, seed=seed)
    else:
        for i in itertools.count():
            new_prompt = input("Enter the input prompt, or q to exit: ")
            if new_prompt:
                prompt = new_prompt
            if prompt[0] == "q":
                break
            run(prompt=prompt, number=i, seed=i)

    if traced:
        pipeline.release_traces()
