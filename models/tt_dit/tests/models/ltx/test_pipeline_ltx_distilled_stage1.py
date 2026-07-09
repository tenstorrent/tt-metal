# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Stage-1 only fast AV pipeline (no upsampler / stage 2) for quicker audio iteration."""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.pipelines.ltx.pipeline_ltx_distilled import LTXDistilledPipeline
from models.tt_dit.tests.models.ltx.test_pipeline_ltx_distilled_av import _default_checkpoint, _default_gemma
from models.tt_dit.utils.test import line_params

_DEFAULT_PROMPT = (
    "A young woman with long, wavy brown hair and a bright smile is playing an acoustic guitar. "
    "She is wearing a light-colored, off-the-shoulder top and is seated in a cozy, warmly lit room."
)


def _load_prompt() -> str:
    path = os.environ.get("LTX_PROMPT_FILE")
    if path and os.path.isfile(path):
        with open(path) as f:
            return f.read().strip()
    explicit = os.environ.get("PROMPT")
    if explicit:
        return explicit
    return _DEFAULT_PROMPT


@pytest.mark.parametrize(
    "mesh_device, mesh_shape, sp_axis, tp_axis, num_links, dynamic_load, device_params, topology, is_fsdp",
    [
        [(2, 4), (2, 4), 1, 0, 2, True, line_params, ttnn.Topology.Linear, False],
    ],
    ids=["bh_2x4sp1tp0"],
    indirect=["mesh_device", "device_params"],
)
def test_pipeline_av_fast_stage1_only(
    mesh_device,
    mesh_shape,
    sp_axis,
    tp_axis,
    num_links,
    dynamic_load,
    topology,
    is_fsdp,
):
    if int(ttnn.distributed_context_get_rank()) != 0:
        pytest.skip("generation only on rank 0")

    parent_mesh = mesh_device
    mesh_device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))

    num_frames = int(os.environ.get("NUM_FRAMES", "121"))
    height = int(os.environ.get("HEIGHT", "512"))
    width = int(os.environ.get("WIDTH", "768"))

    # In-init warmup off: stage-1-only test never runs s2, so skip the
    # full-res s2 compile. We warm s1 explicitly below.
    pipeline = LTXDistilledPipeline.create_pipeline(
        mesh_device=mesh_device,
        checkpoint_name=_default_checkpoint(),
        gemma_path=_default_gemma(),
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        num_links=num_links,
        dynamic_load=dynamic_load,
        topology=topology,
        is_fsdp=is_fsdp,
        run_warmup=False,
    )

    if os.environ.get("RUN_WARMUP", "0") in ("1", "true", "True"):
        pipeline.warmup_buffers(
            num_frames=num_frames,
            height=height,
            width=width,
            stages=("s1",),
        )

    os.environ.setdefault("LTX_DUMP_STAGE1_AUDIO", "1")
    output_path = os.environ.get("OUTPUT_PATH", "ltx_fast_stage1_only.mp4")

    s1_video, s1_audio = pipeline.generate_stage1_only(
        _load_prompt(),
        output_path=output_path,
        num_frames=num_frames,
        height=height,
        width=width,
        seed=int(os.environ.get("SEED", "10")),
    )

    assert s1_video.shape[0] == 1
    assert s1_audio.ndim in (2, 3)
    assert torch.isfinite(s1_audio).all()
    logger.info(f"Stage-1 only done: video={tuple(s1_video.shape)} audio={tuple(s1_audio.shape)}")
