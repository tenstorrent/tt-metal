# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import itertools
import os

import pytest
from loguru import logger

import ttnn
from models.tt_dit.pipelines.ltx.pipeline_ltx_fast import LTXFastPipeline
from models.tt_dit.utils.test import line_params

wh_lb_params = {**line_params, "l1_small_size": 16384}


def _default_checkpoint() -> str | None:
    ckpt = os.environ.get(
        "LTX_CHECKPOINT",
        os.path.expanduser("~/.cache/ltx-checkpoints/ltx-2-19b-distilled.safetensors"),
    )
    return ckpt if os.path.exists(ckpt) else None


def _default_upsampler() -> str | None:
    path = os.environ.get(
        "LTX_UPSAMPLER",
        os.path.expanduser("~/.cache/ltx-checkpoints/ltx-2-spatial-upscaler-x2-1.0.safetensors"),
    )
    return path if os.path.exists(path) else None


def _default_gemma() -> str | None:
    gemma = os.environ.get("GEMMA_PATH", "")
    if gemma and os.path.isdir(gemma):
        return gemma
    import glob

    candidates = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--google--gemma-3-12b-it/snapshots/*/"))
    return candidates[0].rstrip("/") if candidates else None


@pytest.mark.parametrize(
    "no_prompt",
    [{"1": True, "0": False}.get(os.environ.get("NO_PROMPT"), False)],
)
@pytest.mark.parametrize(
    "mesh_device, mesh_shape, sp_axis, tp_axis, num_links, device_params, topology",
    [
        [(2, 4), (2, 4), 0, 1, 2, wh_lb_params, ttnn.Topology.Linear],
        [(4, 8), (4, 8), 1, 0, 2, line_params, ttnn.Topology.Ring],
    ],
    ids=["wh_lb_2x4sp0tp1", "bh_glx_4x8sp1tp0"],
    indirect=["mesh_device", "device_params"],
)
def test_pipeline_av_fast(
    mesh_device,
    mesh_shape,
    sp_axis,
    tp_axis,
    num_links,
    topology,
    no_prompt,
):
    """LTX-2.3 Fast distilled 2-stage AV pipeline."""
    ckpt = _default_checkpoint()
    upsampler = _default_upsampler()
    gemma = _default_gemma()
    if ckpt is None:
        pytest.skip("Distilled checkpoint not found (set LTX_CHECKPOINT)")
    if upsampler is None:
        pytest.skip("Spatial upsampler not found (set LTX_UPSAMPLER)")
    if gemma is None:
        pytest.skip("Gemma model not found (set GEMMA_PATH)")

    parent_mesh = mesh_device
    mesh_device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))

    pipeline = LTXFastPipeline.create_pipeline(
        mesh_device,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        num_links=num_links,
        topology=topology,
    )

    prompt = os.environ.get(
        "PROMPT",
        (
            "Medium shot in a quiet sunlit parlor: an orange tabby cat sits on a piano bench with "
            "its front paws on the white keys, pressing them in a slow, uneven rhythm while dust "
            "motes hang in a bright shaft of late-morning sunlight. Warm natural light rakes across "
            "honey-toned floorboards and the lacquered piano lid; shallow depth of field keeps the "
            "cat's whiskers and fur sharp as the far wall softens into a gentle blur. The camera "
            "slowly dollies in toward the keyboard as the cat shifts its weight and lifts one paw "
            "to swipe another cluster of keys, ears swiveling toward the bright plink of each note. "
            "Audio: close, dry piano taps and short resonant chords, a faint bench creak, and soft "
            "room tone with no speech—just the small acoustic of a single sunlit room for a brief "
            "five-second moment."
        ),
    )
    num_frames = int(os.environ.get("NUM_FRAMES", "121"))
    height = int(os.environ.get("HEIGHT", "512"))
    width = int(os.environ.get("WIDTH", "768"))

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
            checkpoint_path=ckpt,
            upsampler_path=upsampler,
            gemma_path=gemma,
            num_frames=num_frames,
            height=height,
            width=width,
            seed=seed,
        )
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
