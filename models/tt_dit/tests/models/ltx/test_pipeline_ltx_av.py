# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import itertools
import os

import pytest
from loguru import logger

import ttnn
from models.tt_dit.pipelines.ltx.pipeline_ltx_av import LTXAVPipeline
from models.tt_dit.utils.test import (
    bh_lb_2x4_id,
    bh_lb_2x4_params,
    ring_params,
    skip_ltx_mesh_config_unless_matching_arch,
    wh_lb_2x4_id,
    wh_lb_2x4_params,
)


def _default_checkpoint() -> str | None:
    ckpt = os.environ.get("LTX_CHECKPOINT", os.path.expanduser("~/.cache/ltx-checkpoints/ltx-2.3-22b-dev.safetensors"))
    return ckpt if os.path.exists(ckpt) else None


def _default_gemma() -> str | None:
    gemma = os.environ.get("GEMMA_PATH", "")
    if gemma and os.path.isdir(gemma):
        return gemma
    import glob

    candidates = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--google--gemma-3-12b-it/snapshots/*/"))
    return candidates[0].rstrip("/") if candidates else None


def _no_prompt_param() -> bool:
    """One-shot generation by default so pytest does not block on ``input()`` or hit 300s timeout.

    - ``NO_PROMPT=1`` / ``0``: force non-interactive / interactive.
    - ``INTERACTIVE_LTX_AV=1``: prompt loop (TTY); use when running this test manually.
    """
    env = os.environ.get("NO_PROMPT")
    if env is not None:
        return {"1": True, "0": False}.get(env, False)
    if os.environ.get("INTERACTIVE_LTX_AV", "").lower() in ("1", "true", "yes"):
        return False
    return True


@pytest.mark.timeout(7200)
@pytest.mark.parametrize("no_prompt", [_no_prompt_param()])
@pytest.mark.parametrize(
    "mesh_device, mesh_shape, sp_axis, tp_axis, num_links, dynamic_load, device_params, topology, is_fsdp, mesh_config_id",
    [
        [*wh_lb_2x4_params, wh_lb_2x4_id],
        [*bh_lb_2x4_params, bh_lb_2x4_id],
        [(4, 8), (4, 8), 1, 0, 2, False, ring_params, ttnn.Topology.Ring, False, "bh_glx_4x8sp1tp0"],
    ],
    ids=[wh_lb_2x4_id, bh_lb_2x4_id, "bh_glx_4x8sp1tp0"],
    indirect=["mesh_device", "device_params"],
)
def test_pipeline_av_pro(
    mesh_device,
    mesh_shape,
    sp_axis,
    tp_axis,
    num_links,
    dynamic_load,
    topology,
    is_fsdp,
    mesh_config_id,
    no_prompt,
):
    """Full LTX-2.3 Pro AV pipeline (one-stage, CFG+STG+modality guidance)."""
    skip_ltx_mesh_config_unless_matching_arch(mesh_config_id)
    ckpt = _default_checkpoint()
    gemma = _default_gemma()
    if ckpt is None:
        pytest.skip("22B checkpoint not found (set LTX_CHECKPOINT)")
    if gemma is None:
        pytest.skip("Gemma model not found (set GEMMA_PATH)")

    parent_mesh = mesh_device
    mesh_device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))

    pipeline = LTXAVPipeline.create_pipeline(
        mesh_device,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        num_links=num_links,
        dynamic_load=dynamic_load,
        topology=topology,
        is_fsdp=is_fsdp,
    )

    prompt = os.environ.get(
        "PROMPT",
        ("a cat playing piano"),
    )
    num_frames = int(os.environ.get("NUM_FRAMES", "121"))
    height = int(os.environ.get("HEIGHT", "512"))
    width = int(os.environ.get("WIDTH", "768"))
    num_steps = int(os.environ.get("NUM_STEPS", "30"))

    def run(*, prompt, number, seed):
        output_filename = os.environ.get("OUTPUT_PATH", f"ltx_av_pro_{width}x{height}_{number}.mp4")
        logger.info(f"Running LTX AV Pro: '{prompt[:80]}...'")
        logger.info(f"Config: {height}x{width}, {num_frames} frames, {num_steps} steps")

        if int(ttnn.distributed_context_get_rank()) != 0:
            logger.info(f"Skipping generation on rank {ttnn.distributed_context_get_rank()}")
            return

        gen_kwargs = {
            "output_path": output_filename,
            "checkpoint_path": ckpt,
            "gemma_path": gemma,
            "num_frames": num_frames,
            "height": height,
            "width": width,
            "num_inference_steps": num_steps,
            "seed": seed,
        }
        for env_name, kw in (
            ("VIDEO_CFG_SCALE", "video_cfg_scale"),
            ("AUDIO_CFG_SCALE", "audio_cfg_scale"),
            ("VIDEO_STG_SCALE", "video_stg_scale"),
            ("AUDIO_STG_SCALE", "audio_stg_scale"),
            ("VIDEO_MODALITY_SCALE", "video_modality_scale"),
            ("AUDIO_MODALITY_SCALE", "audio_modality_scale"),
            ("RESCALE_SCALE", "rescale_scale"),
            ("GE_GAMMA", "ge_gamma"),
        ):
            if os.environ.get(env_name) is not None:
                gen_kwargs[kw] = float(os.environ[env_name])

        pipeline.generate(prompt, **gen_kwargs)
        logger.info(f"Saved video to: {output_filename}")

    if no_prompt:
        run(prompt=prompt, number=0, seed=int(os.environ.get("SEED", "10")))
    else:
        for i in itertools.count():
            new_prompt = input("Enter the input prompt, or q to exit: ")
            if new_prompt:
                prompt = new_prompt
            if prompt[0] == "q":
                break
            run(prompt=prompt, number=i, seed=i)
