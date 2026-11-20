# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch
from loguru import logger

import ttnn

from ....models.transformers.longcat_avatar.transformer_longcat_avatar import LongCatAvatarTransformer
from ....parallel.config import DiTParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....reference.longcat_avatar import modeling_longcat_avatar as ref
from ....utils.check import assert_quality
from ....utils.test import (
    line_params,
    line_params_req_exact_devices,
    ring_params_req_exact_devices,
    skip_if_unsupported_num_links,
)

CHECKPOINT_DIR = os.environ.get("LONGCAT_AVATAR_DIR", "")
NUM_LAYERS = int(os.environ.get("LONGCAT_AVATAR_TEST_LAYERS", "2"))


@pytest.mark.parametrize(
    ("mesh_device", "sp_axis", "tp_axis", "num_links", "device_params", "topology"),
    [
        pytest.param((4, 8), 1, 0, 2, ring_params_req_exact_devices, ttnn.Topology.Ring, id="4x8sp1tp0_ring"),
        pytest.param((4, 8), 1, 0, 2, line_params_req_exact_devices, ttnn.Topology.Linear, id="4x8sp1tp0_line"),
        pytest.param((2, 4), 1, 0, 1, line_params, ttnn.Topology.Linear, id="2x4sp1tp0_line"),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    ("T", "H", "W"),
    [
        pytest.param(5, 60, 104, id="480p_17f"),
        pytest.param(24, 60, 104, id="480p_93f"),
    ],
)
@pytest.mark.parametrize(
    "mode",
    ["at2v", "ai2v", "avc"],
)
@pytest.mark.parametrize(
    "lora",
    # v1.6 distilled sampling runs loraA (DMD) for the first steps and loraB (OPSD) for the rest
    ["base", "loraA", "loraB"],
)
def test_longcat_avatar_transformer(
    mesh_device: ttnn.MeshDevice,
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    topology: ttnn.Topology,
    T: int,
    H: int,
    W: int,
    mode: str,
    lora: str,
) -> None:
    if not CHECKPOINT_DIR:
        pytest.skip("set LONGCAT_AVATAR_DIR to a LongCat-Video-Avatar-1.6 checkpoint")
    skip_if_unsupported_num_links(mesh_device, num_links)

    torch.manual_seed(0)
    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(mesh_axis=0, factor=1),
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tuple(mesh_device.shape)[tp_axis]),
        sequence_parallel=ParallelFactor(mesh_axis=sp_axis, factor=tuple(mesh_device.shape)[sp_axis]),
    )
    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)

    lora_path = None if lora == "base" else os.path.join(CHECKPOINT_DIR, lora, "lora.safetensors")
    torch_model = ref.build_model(CHECKPOINT_DIR, num_blocks=NUM_LAYERS, lora_path=lora_path)
    config = ref.load_config(CHECKPOINT_DIR)
    tt_model = LongCatAvatarTransformer(
        config,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        num_layers=NUM_LAYERS,
    )
    tt_model.load_torch_state_dict(ref.load_state_dict(CHECKPOINT_DIR, num_blocks=NUM_LAYERS, lora_path=lora_path))

    num_frames = 1 + (T - 1) * config["vae_scale"]
    latents = torch.randn(1, config["in_channels"], T, H, W)
    text = torch.randn(1, 1, config["model_max_length"], config["caption_channels"]) * 0.1
    text_mask = torch.zeros(1, config["model_max_length"], dtype=torch.long)
    text_mask[:, :120] = 1
    audio = torch.randn(1, num_frames, config["audio_window"], config["audio_block"], config["audio_channel"])

    kwargs = {
        "at2v": dict(num_cond_latents=0),
        "ai2v": dict(num_cond_latents=1),
        "avc": dict(num_cond_latents=min(4, T - 1), num_ref_latents=1, ref_img_index=10, mask_frame_range=0),
    }[mode]
    timestep = 700.0
    t = torch.full((1, T), timestep)
    t[:, : kwargs["num_cond_latents"]] = 0

    with torch.no_grad():
        expected = torch_model(latents, t, text, text_mask, audio_embs=audio, **kwargs)
    actual = tt_model(latents, timestep, text, text_mask, audio, **kwargs)

    n_cond = kwargs["num_cond_latents"]
    logger.info(f"{mode} {lora} {NUM_LAYERS} layers, latent {T}x{H}x{W}")
    # conditioning frames are discarded by the sampler, so check the noise frames on their own as well
    assert_quality(expected[:, :, n_cond:], actual[:, :, n_cond:], pcc=0.995)
    assert_quality(expected, actual, pcc=0.99)
