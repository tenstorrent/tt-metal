# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""LTX-2.5 distilled-fast pipeline bring-up test.

Mirrors ``test_pipeline_distilled`` for the T2V entrypoint only. VBench/CLIP default off
for bring-up (override with ``RUN_VBENCH`` / ``RUN_CLIP``).
"""

import os

import pytest
from loguru import logger

import ttnn
from models.tt_dit.pipelines.ltx.pipeline_ltx25_distilled import LTX25DistilledPipeline
from models.tt_dit.utils.ltx import (
    DEFAULT_LTX_PROMPT,
    LTX25_DISTILLED_TRANSFORMER,
    LTX25_TEXT_ENCODER,
    STEADY_STATE_LTX_PROMPT,
    STEADY_STATE_REPLAY_LTX_PROMPT,
    default_ltx25_path,
    print_ltx_timing_table,
)
from models.tt_dit.utils.test import skip_if_unsupported_num_links

from .ltx_mesh_params import LTX_DISTILLED_MESH_PARAMS_DL


def _ltx25_distilled_ready() -> bool:
    return bool(default_ltx25_path(LTX25_TEXT_ENCODER) and default_ltx25_path(LTX25_DISTILLED_TRANSFORMER))


@pytest.mark.skipif(not _ltx25_distilled_ready(), reason="needs LTX-2.5 text encoder + distilled transformer")
@pytest.mark.parametrize(
    "no_prompt",
    [{"1": True, "0": False}.get(os.environ.get("NO_PROMPT"), True)],
)
@pytest.mark.parametrize(
    "mesh_device, sp_axis, tp_axis, num_links, device_params, topology, is_fsdp, dynamic_load",
    LTX_DISTILLED_MESH_PARAMS_DL,
    indirect=["mesh_device", "device_params"],
)
def test_pipeline_ltx25_distilled(
    mesh_device,
    sp_axis,
    tp_axis,
    num_links,
    dynamic_load,
    topology,
    is_fsdp,
    no_prompt,
):
    """LTX-2.5 distilled 2-stage AV pipeline (Gemma-4 + split checkpoints)."""
    skip_if_unsupported_num_links(mesh_device, num_links)

    parent_mesh = mesh_device
    mesh_shape = tuple(parent_mesh.shape)
    mesh_device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))

    num_frames = int(os.environ.get("NUM_FRAMES", "145"))
    height = int(os.environ.get("HEIGHT", "1088"))
    width = int(os.environ.get("WIDTH", "1920"))

    run_warmup = os.environ.get("RUN_WARMUP", "0") in ("1", "true", "True")
    traced = os.environ.get("LTX_TRACED", "0") in ("1", "true", "True")

    pipeline = LTX25DistilledPipeline.create_pipeline(
        mesh_device=mesh_device,
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

    prompt = os.environ.get("PROMPT", DEFAULT_LTX_PROMPT)
    # Same 3-pass traced structure as 2.3 distilled (#52968): gen #0 captures denoise/VAE/audio,
    # gen #1 captures encode (gate opens after gen #0), gen #2 is pure replay + measured Encoder.
    steady_state_prompt = prompt if dynamic_load else os.environ.get("PROMPT_STEADY_STATE", STEADY_STATE_LTX_PROMPT)
    replay_prompt = prompt if dynamic_load else STEADY_STATE_REPLAY_LTX_PROMPT

    def run(*, prompt, number, seed):
        output_filename = os.environ.get("OUTPUT_PATH", f"ltx25_av_fast_{width}x{height}_{number}.mp4")
        # Per-gen filenames so gen #0/#1/#2 don't clobber when OUTPUT_PATH is set.
        if os.environ.get("OUTPUT_PATH") and number != 0:
            root, ext = os.path.splitext(output_filename)
            output_filename = f"{root}_{number}{ext}"
        logger.info(f"Running LTX-2.5 AV Fast: '{prompt[:80]}...'")
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
        print_ltx_timing_table(
            pipeline,
            label="LTX-2.5 DISTILLED",
            num_frames=num_frames,
            height=height,
            width=width,
            mesh_shape=mesh_shape,
            sp_axis=sp_axis,
            tp_axis=tp_axis,
            topology=topology,
            output_path=output_filename,
            prompt=prompt,
        )

    # Quality gates (RUN_VBENCH / RUN_CLIP) stay off for bring-up; re-enable once generate is green.

    if no_prompt:
        seed = int(os.environ.get("SEED", "10"))
        run(prompt=prompt, number=0, seed=seed)
        if traced:
            logger.info("=== gen #1: encode trace capture ===")
            run(prompt=steady_state_prompt, number=1, seed=seed)
            logger.info("=== traced steady-state pass (gen #2, pure replay) ===")
            run(prompt=replay_prompt, number=2, seed=seed)
    else:
        while True:
            user_prompt = input("Enter a prompt (or 'q' to quit): ")
            if user_prompt.strip().lower() == "q":
                break
            seed = int(os.environ.get("SEED", "10"))
            run(prompt=user_prompt or prompt, number=0, seed=seed)

    if traced:
        pipeline.release_traces()
