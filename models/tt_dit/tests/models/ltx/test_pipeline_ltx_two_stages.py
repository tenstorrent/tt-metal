# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import itertools
import os

import pytest
from loguru import logger

import ttnn
from models.tt_dit.pipelines.ltx.pipeline_ltx_two_stages import LTXTwoStagesPipeline
from models.tt_dit.utils.test import line_params, ring_params


def _default_checkpoint() -> str:
    """Resolve base 22B checkpoint: env var > local file > HF repo string default."""
    explicit = os.environ.get("LTX_CHECKPOINT")
    if explicit:
        return explicit
    local = os.path.expanduser("~/.cache/ltx-checkpoints/ltx-2.3-22b-dev.safetensors")
    if os.path.exists(local):
        return local
    return "Lightricks/LTX-2.3:ltx-2.3-22b-dev.safetensors"


def _default_distilled_lora() -> str:
    """Resolve distilled stage-2 LoRA: env var > local file > HF (auto-downloaded here)."""
    explicit = os.environ.get("DISTILLED_LORA_PATH")
    if explicit:
        return explicit
    local = os.path.expanduser("~/.cache/ltx-checkpoints/ltx-2.3-22b-distilled-lora-384-1.1.safetensors")
    if os.path.exists(local):
        return local
    from huggingface_hub import hf_hub_download

    logger.info(
        "Resolving HuggingFace distilled LoRA Lightricks/LTX-2.3:ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
    )
    return hf_hub_download(repo_id="Lightricks/LTX-2.3", filename="ltx-2.3-22b-distilled-lora-384-1.1.safetensors")


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
def test_pipeline_av_two_stages(
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
    """LTX-2.3 22B 2-stage AV pipeline: full-guidance s1 + distilled-LoRA s2 refine."""
    ckpt = _default_checkpoint()
    distilled_lora = _default_distilled_lora()
    gemma = _default_gemma()

    parent_mesh = mesh_device
    mesh_device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))

    num_frames = int(os.environ.get("NUM_FRAMES", "121"))
    height = int(os.environ.get("HEIGHT", "512"))
    width = int(os.environ.get("WIDTH", "768"))
    num_inference_steps = int(os.environ.get("NUM_INFERENCE_STEPS", "30"))

    run_warmup = os.environ.get("RUN_WARMUP", "0") in ("1", "true", "True")
    pipeline = LTXTwoStagesPipeline.create_pipeline(
        mesh_device=mesh_device,
        checkpoint_name=ckpt,
        gemma_path=gemma,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        num_links=num_links,
        dynamic_load=dynamic_load,
        topology=topology,
        is_fsdp=is_fsdp,
        distilled_lora_path=distilled_lora,
        run_warmup=run_warmup,
        num_frames=num_frames,
        height=height,
        width=width,
    )

    prompt = os.environ.get(
        "PROMPT",
        ("a cat playing piano"),
    )

    def run(*, prompt, number, seed):
        output_filename = os.environ.get("OUTPUT_PATH", f"ltx_av_two_stages_{width}x{height}_{number}.mp4")
        logger.info(f"Running LTX AV Two-Stages: '{prompt[:80]}...'")
        logger.info(f"Config: {height}x{width}, {num_frames} frames, {num_inference_steps} stage-1 steps")

        if int(ttnn.distributed_context_get_rank()) != 0:
            logger.info(f"Skipping generation on rank {ttnn.distributed_context_get_rank()}")
            return

        pipeline.generate(
            prompt,
            output_path=output_filename,
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
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
