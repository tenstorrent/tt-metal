# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3-Image DiT gate: one PDD step through the image pipeline's own device path vs the diffusers reference.

Runs `MiniMaxH3ImagePipeline._prepare_request` + `_step` (source table with the pinned reference anchors, assembly /
out indices, RoPE and AdaLN tables, the two timestep slots `[1 - sigma_k, 0.999]`, PDD head k) on a few-layer
checkpoint, and the pinned diffusers `MiniMaxH3Transformer3DModel` with the same packed sequence, audio-free (zero
audio rows) and its `proj_out` replaced by the fp32 PDD head k, the way the release swaps heads per step.

Env: MINIMAX_H3_IMAGE_MODEL_PATH -> a converted snapshot (`convert_h3_image.py ... transformer-layers=2` is enough);
its `text_encoder/` must exist (the pipeline builds, but does not load, the conditioner).
"""

import os

import pytest
import torch
from loguru import logger

import ttnn

from ....pipelines.minimax_h3 import image_host_minimax_h3 as host
from ....pipelines.minimax_h3.pipeline_minimax_h3_image import MiniMaxH3ImagePipeline
from ....utils.check import assert_quality
from .common import MESH_4X8_RING

GALAXY = [MESH_4X8_RING]


def _weights_dir() -> str:
    path = os.environ.get("MINIMAX_H3_IMAGE_MODEL_PATH")
    if not path:
        pytest.skip("set MINIMAX_H3_IMAGE_MODEL_PATH to a converted H3-Image snapshot")
    return path


def _reference(weights_dir: str, step: int):
    import json

    from diffusers import MiniMaxH3Transformer3DModel
    from safetensors.torch import load_file

    cfg = json.load(open(os.path.join(weights_dir, "transformer", "config.json")))
    kwargs = {k: v for k, v in cfg.items() if not k.startswith("_") and k != "audio_enabled"}
    kwargs["patch_size"] = tuple(kwargs["patch_size"])
    model = MiniMaxH3Transformer3DModel(**kwargs).eval()
    state = {}
    tdir = os.path.join(weights_dir, "transformer")
    index = json.load(open(os.path.join(tdir, "diffusion_pytorch_model.safetensors.index.json")))["weight_map"]
    for shard in sorted(set(index.values())):
        state.update(load_file(os.path.join(tdir, shard)))
    heads = load_file(os.path.join(tdir, "pdd_heads.safetensors"))
    state["proj_out.weight"] = heads["pdd_heads.weight"][step]
    state["proj_out.bias"] = heads["pdd_heads.bias"][step]
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected, unexpected
    assert all(k.startswith("audio_proj") for k in missing), missing  # audio-free checkpoint
    return model.float()


def _one_step_pcc(pipe, mesh_device, reference_for_step, with_reference: bool, step: int) -> tuple[float, float]:
    """Run one PDD step through the pipeline's device path and the torch reference; return (pcc, relative_rmse)."""
    torch.manual_seed(step + 10 * with_reference)
    text_len = 200
    tags = torch.full((text_len,), host.TEXT_TAG, dtype=torch.long)
    refs = [host.ReferenceGeometry(32, 32)] if with_reference else []
    if with_reference:
        tags[150:190] = host.IMAGE_TAG
    layout = host.build_image_packed_sequence(tags, refs, (64, 64))
    prompt = torch.randn(1, text_len, 5120) * 0.5
    ref_rows = torch.randn(layout.num_ref_rows, 96)
    target = torch.randn(layout.target_rows, 96)
    pad = (-text_len) % (pipe.sp_factor * ttnn.TILE_SIZE)
    tt_prompt = ttnn.from_torch(
        torch.nn.functional.pad(prompt, (0, 0, 0, pad)),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    request = pipe._prepare_request(tt_prompt, layout, ref_rows)
    with pipe.ccl_manager.transient_ping_pong_buffers():
        actual = pipe._step(request, target, step)
    reference = reference_for_step(step)
    L, R = layout.text_len, layout.num_ref_rows
    image_rows = torch.cat([ref_rows, target])
    timestep = torch.tensor([pipe.schedule.timestep(step), host.REFERENCE_TIMESTEP], dtype=torch.float32)
    with torch.no_grad():
        out = reference(
            hidden_states=image_rows[None],
            audio_hidden_states=torch.zeros(1, 0, 32),
            encoder_hidden_states=prompt,
            timestep=timestep,
            timestep_indices=layout.row_slot,
            token_tags=layout.token_tags,
            position_ids=layout.position_ids.float(),
            video_indices=torch.arange(L, L + image_rows.shape[0]),
            audio_indices=torch.zeros(0, dtype=torch.long),
            text_indices=torch.arange(L),
            return_dict=False,
        )[0][0]
    e, a = out[R:].float().flatten(), actual.float().flatten()
    pcc = torch.corrcoef(torch.stack([e, a]))[0, 1].item()
    rel = ((e - a).pow(2).mean().sqrt() / e.pow(2).mean().sqrt()).item()
    return pcc, rel


@pytest.mark.timeout(7200)
@pytest.mark.parametrize(("mesh_device", "device_params"), GALAXY, indirect=["mesh_device", "device_params"])
def test_image_dit_full_depth(mesh_device):
    """Every layer of the real checkpoint: one fp32 torch reference (built once, PDD head swapped per step) against
    the device path, t2i and edit at PDD steps 0, 3, 5. Quantifies how much bf16 error 50 layers accumulate."""
    import time

    weights_dir = _weights_dir()
    torch.set_num_threads(max(1, len(os.sched_getaffinity(0)) // 2))
    pipe = MiniMaxH3ImagePipeline(mesh_device=mesh_device, weights_dir=weights_dir)
    mark = time.perf_counter()
    model = _reference(weights_dir, 0)
    from safetensors.torch import load_file

    heads = load_file(os.path.join(weights_dir, "transformer", "pdd_heads.safetensors"))
    logger.info(
        f"RESULT reference built ({pipe.transformer_config['num_layers']} layers fp32) in {time.perf_counter() - mark:.0f}s"
    )

    def reference_for_step(step: int):
        with torch.no_grad():
            model.proj_out.weight.copy_(heads["pdd_heads.weight"][step].float())
            model.proj_out.bias.copy_(heads["pdd_heads.bias"][step].float())
        return model

    for with_reference in (False, True):
        for step in (0, 3, 5):
            mark = time.perf_counter()
            pcc, rel = _one_step_pcc(pipe, mesh_device, reference_for_step, with_reference, step)
            tag = "edit" if with_reference else "t2i"
            logger.info(
                f"RESULT full_depth step{step}_{tag}: pcc={pcc:.6f} relative_rmse={rel:.5f} ({time.perf_counter() - mark:.0f}s)"
            )


@pytest.mark.parametrize(("mesh_device", "device_params"), GALAXY, indirect=["mesh_device", "device_params"])
@pytest.mark.parametrize("with_reference", [pytest.param(False, id="t2i"), pytest.param(True, id="edit")])
@pytest.mark.parametrize("step", [0, 3, 5])
def test_image_dit_step(mesh_device, with_reference, step):
    weights_dir = _weights_dir()
    torch.manual_seed(step + 10 * with_reference)
    pipe = MiniMaxH3ImagePipeline(mesh_device=mesh_device, weights_dir=weights_dir)

    # A small request: 200 text rows (the last 40 tagged as a Qwen vision block), one 512x512-class reference
    # (32x32 latent -> 256 rows) for edit, and a 1024x1024 canvas (64x64 latent -> 1024 target rows).
    text_len = 200
    tags = torch.full((text_len,), host.TEXT_TAG, dtype=torch.long)
    refs = [host.ReferenceGeometry(32, 32)] if with_reference else []
    if with_reference:
        tags[150:190] = host.IMAGE_TAG
    layout = host.build_image_packed_sequence(tags, refs, (64, 64))
    prompt = torch.randn(1, text_len, 5120) * 0.5
    ref_rows = torch.randn(layout.num_ref_rows, 96)
    target = torch.randn(layout.target_rows, 96)

    # tt: pad the prompt to SP alignment the way encode_prompt returns it, replicated.
    pad = (-text_len) % (pipe.sp_factor * ttnn.TILE_SIZE)
    tt_prompt = ttnn.from_torch(
        torch.nn.functional.pad(prompt, (0, 0, 0, pad)),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    request = pipe._prepare_request(tt_prompt, layout, ref_rows)
    with pipe.ccl_manager.transient_ping_pong_buffers():
        actual = pipe._step(request, target, step)

    reference = _reference(weights_dir, step)
    L, R = layout.text_len, layout.num_ref_rows
    image_rows = torch.cat([ref_rows, target])
    timestep = torch.tensor([pipe.schedule.timestep(step), host.REFERENCE_TIMESTEP], dtype=torch.float32)
    with torch.no_grad():
        out = reference(
            hidden_states=image_rows[None],
            audio_hidden_states=torch.zeros(1, 0, 32),
            encoder_hidden_states=prompt,
            timestep=timestep,
            timestep_indices=layout.row_slot,
            token_tags=layout.token_tags,
            position_ids=layout.position_ids.float(),
            video_indices=torch.arange(L, L + image_rows.shape[0]),
            audio_indices=torch.zeros(0, dtype=torch.long),
            text_indices=torch.arange(L),
            return_dict=False,
        )[0][0]
    expected = out[R:]
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    e, a = expected.float().flatten(), actual.float().flatten()
    pcc = torch.corrcoef(torch.stack([e, a]))[0, 1].item()
    rel = ((e - a).pow(2).mean().sqrt() / e.pow(2).mean().sqrt()).item()
    tag = "edit" if with_reference else "t2i"
    logger.info(f"RESULT dit_step{step}_{tag}: pcc={pcc:.6f} relative_rmse={rel:.5f}")
    assert_quality(expected, actual, pcc=0.995, relative_rmse=0.1)
