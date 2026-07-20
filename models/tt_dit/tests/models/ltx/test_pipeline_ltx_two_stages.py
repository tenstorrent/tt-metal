# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import itertools
import os

import pytest
from loguru import logger

import ttnn
from models.tt_dit.pipelines.ltx.pipeline_ltx_two_stages import LTXTwoStagesPipeline
from models.tt_dit.utils.ltx import (
    DEFAULT_LTX_PROMPT,
    default_ltx_checkpoint,
    default_ltx_gemma,
    print_ltx_timing_table,
)
from models.tt_dit.utils.test import line_params, ring_params


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
def test_pipeline_two_stages(
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
    ckpt = default_ltx_checkpoint("ltx-2.3-22b-dev.safetensors")
    distilled_lora = _default_distilled_lora()
    gemma = default_ltx_gemma()

    parent_mesh = mesh_device
    mesh_device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))

    num_frames = int(os.environ.get("NUM_FRAMES", "145"))
    height = int(os.environ.get("HEIGHT", "1088"))
    width = int(os.environ.get("WIDTH", "1920"))
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

    prompt = os.environ.get("PROMPT", DEFAULT_LTX_PROMPT)

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
        print_ltx_timing_table(
            pipeline,
            label="LTX TWO-STAGES",
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


# ---------------------------------------------------------------------------
# I2V smoke test (image + text -> video)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "mesh_device, mesh_shape, sp_axis, tp_axis, num_links, dynamic_load, device_params, topology, is_fsdp",
    [
        [(2, 4), (2, 4), 1, 0, 2, True, line_params, ttnn.Topology.Linear, False],
    ],
    ids=["bh_2x4sp1tp0"],
    indirect=["mesh_device", "device_params"],
)
def test_pipeline_two_stages_i2v_smoke(
    mesh_device,
    mesh_shape,
    sp_axis,
    tp_axis,
    num_links,
    dynamic_load,
    topology,
    is_fsdp,
):
    """End-to-end I2V smoke: generate(images=[(img, 0, 1.0)]) produces a non-empty MP4 whose
    first frame correlates with the conditioning image.

    Gated behind LTX_I2V_IMAGE (path to a conditioning image); skips otherwise so it never runs
    in the default CI sweep without an input image / checkpoint present.
    """
    image_path = os.environ.get("LTX_I2V_IMAGE")
    if not image_path or not os.path.exists(image_path):
        pytest.skip("set LTX_I2V_IMAGE to a conditioning image path to run the I2V smoke test")

    ckpt = default_ltx_checkpoint("ltx-2.3-22b-dev.safetensors")
    distilled_lora = _default_distilled_lora()
    gemma = default_ltx_gemma()

    parent_mesh = mesh_device
    mesh_device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))

    num_frames = int(os.environ.get("NUM_FRAMES", "25"))
    height = int(os.environ.get("HEIGHT", "512"))
    width = int(os.environ.get("WIDTH", "768"))
    num_inference_steps = int(os.environ.get("NUM_INFERENCE_STEPS", "8"))
    strength = float(os.environ.get("LTX_I2V_STRENGTH", "1.0"))

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
        run_warmup=False,
        num_frames=num_frames,
        height=height,
        width=width,
    )

    if int(ttnn.distributed_context_get_rank()) != 0:
        logger.info(f"Skipping generation on rank {ttnn.distributed_context_get_rank()}")
        return

    output_filename = os.environ.get("OUTPUT_PATH", f"ltx_i2v_two_stages_{width}x{height}.mp4")
    prompt = os.environ.get("PROMPT", DEFAULT_LTX_PROMPT)
    pipeline.generate(
        prompt,
        output_path=output_filename,
        images=[(image_path, 0, strength)],
        num_frames=num_frames,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        seed=int(os.environ.get("SEED", "10")),
    )

    assert os.path.exists(output_filename), f"no MP4 produced at {output_filename}"
    assert os.path.getsize(output_filename) > 0, "produced MP4 is empty"
    logger.info(f"I2V smoke: saved {output_filename} ({os.path.getsize(output_filename)} bytes)")

    # Optional first-frame correlation check (only if imageio + PIL are available).
    try:
        import imageio.v3 as iio
        import numpy as np
        from PIL import Image

        frames = iio.imread(output_filename, index=None)  # (T, H, W, 3)
        first = frames[0].astype(np.float32) / 255.0
        cond = Image.open(image_path).convert("RGB").resize((width, height))
        cond_np = np.asarray(cond, dtype=np.float32) / 255.0
        a = first.reshape(-1) - first.mean()
        b = cond_np.reshape(-1) - cond_np.mean()
        pcc = float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
        logger.info(f"I2V smoke: first-frame vs conditioning-image PCC = {pcc:.3f}")
        if strength >= 1.0:
            assert pcc > 0.3, f"first frame does not resemble the conditioning image (PCC={pcc:.3f})"
    except ImportError:
        logger.info("imageio/PIL unavailable — skipping first-frame correlation check")
