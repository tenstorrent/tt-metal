# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Bit-exactness gate for the CFG hoist in `WanTransformer3DModel.combined_step`.

Under classifier-free guidance `combined_step` runs `inner_step` twice with the same timestep
and spatial input, so the timestep embedding and the patch embedding are now computed once and
handed to both passes. That is pure reuse of identical device results, so the gate is exact
equality, not a PCC floor: the hoisted `combined_step` must match the pre-hoist computation
(two independent `inner_step` calls and a `ttnn.lerp`) to the bit, and each pass must match
its standalone `inner_step`.

Real 720p latent geometry at 2 transformer blocks, so the patch-embed and modulation shapes
are the production ones while the run stays short.

    pytest models/tt_dit/tests/models/wan2_2/test_cfg_hoist_ti2v_5b.py -sv --timeout=0
"""

import os
import sys

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils.tensor import bf16_tensor, float32_tensor, from_torch, local_device_to_torch
from models.tt_dit.utils.test import skip_if_unsupported_num_links

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # tests dir is not a package
from test_transformer_wan_ti2v_5b import (  # noqa: E402
    LATENT_H,
    LATENT_T,
    LATENT_W,
    MESH_PARAMS,
    PROMPT_SEQ_LEN,
    TIMESTEP,
    _load_5b_config_and_state,
    _make_tt_transformer,
    _parallel_config,
)

NUM_LAYERS = 2
GUIDANCE_SCALE = 4.0


@pytest.mark.parametrize(
    ("mesh_device", "sp_axis", "tp_axis", "num_links", "device_params", "topology", "is_fsdp"),
    MESH_PARAMS,
    ids=["bh_4x8"],
    indirect=["mesh_device", "device_params"],
)
def test_cfg_hoist_bit_exact(mesh_device, sp_axis, tp_axis, num_links, device_params, topology, is_fsdp):
    if not ttnn.device.is_blackhole():
        pytest.skip("TI2V-5B targets BH Galaxy")
    skip_if_unsupported_num_links(mesh_device, num_links)

    parallel_config = _parallel_config(mesh_device, sp_axis, tp_axis)
    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)

    cfg, torch_model = _load_5b_config_and_state(num_layers=NUM_LAYERS)
    state_dict = torch_model.state_dict()
    del torch_model

    tt_model = _make_tt_transformer(
        cfg, mesh_device=mesh_device, ccl_manager=ccl_manager, parallel_config=parallel_config, num_layers=NUM_LAYERS
    )
    tt_model.load_torch_state_dict(state_dict)

    torch.manual_seed(0)
    spatial = torch.randn((1, cfg.in_channels, LATENT_T, LATENT_H, LATENT_W), dtype=torch.float32)
    prompt = torch.randn((1, PROMPT_SEQ_LEN, cfg.text_dim), dtype=torch.float32)
    negative = torch.randn((1, PROMPT_SEQ_LEN, cfg.text_dim), dtype=torch.float32)

    spatial_host, n = tt_model.preprocess_spatial_input_host(spatial)
    rope_cos, rope_sin, trans_mat = tt_model.prepare_rope_features(spatial)
    prompt_1BLP = tt_model.prepare_text_conditioning(bf16_tensor(prompt.unsqueeze(0), device=mesh_device))
    negative_1BLP = tt_model.prepare_text_conditioning(bf16_tensor(negative.unsqueeze(0), device=mesh_device))
    sp_axis_ = parallel_config.sequence_parallel.mesh_axis
    spatial_device = from_torch(spatial_host, device=mesh_device, mesh_axes=[None, None, sp_axis_, None])
    logger.info(f"5B transformer: N={n} (padded {spatial_host.shape[2]}), {NUM_LAYERS} blocks")

    timestep = float32_tensor(
        torch.full((1,), TIMESTEP, dtype=torch.float32).unsqueeze(1).unsqueeze(1).unsqueeze(1), device=mesh_device
    )
    guidance = float32_tensor(torch.tensor(GUIDANCE_SCALE, dtype=torch.float32).reshape(1, 1, 1, 1), device=mesh_device)

    common = {
        "spatial_1BNI": spatial_device,
        "rope_cos_1HND": rope_cos,
        "rope_sin_1HND": rope_sin,
        "trans_mat": trans_mat,
        "N": n,
        "timestep": timestep,
        "gather_output": False,
    }

    # Pre-hoist computation: each pass embeds the timestep and the patches itself.
    ref_cond = local_device_to_torch(tt_model.inner_step(prompt_1BLP=prompt_1BLP, **common))
    ref_uncond_tt = tt_model.inner_step(prompt_1BLP=negative_1BLP, **common)
    ref_uncond = local_device_to_torch(ref_uncond_tt)
    ref_combined = local_device_to_torch(
        ttnn.lerp(ref_uncond_tt, tt_model.inner_step(prompt_1BLP=prompt_1BLP, **common), guidance)
    )

    # Hoisted path, untraced (the traced_function wrapper passes straight through).
    new_combined = local_device_to_torch(
        tt_model.combined_step(
            do_classifier_free_guidance=True,
            spatial_1BNI=spatial_device,
            prompt_1BLP=prompt_1BLP,
            negative_prompt_1BLP=negative_1BLP,
            N=n,
            rope_cos_1HND=rope_cos,
            rope_sin_1HND=rope_sin,
            trans_mat=trans_mat,
            timestep=timestep,
            guidance_scale=guidance,
            gather_output=False,
        )
    )

    # Each pass with the shared embeddings handed in explicitly, against its standalone self.
    shared = {
        "timestep_conditioning": tt_model.prepare_timestep_conditioning(timestep),
        "spatial_1BND": tt_model.patch_embedding(spatial_device),
    }
    new_cond = local_device_to_torch(tt_model.inner_step(prompt_1BLP=prompt_1BLP, **common, **shared))
    new_uncond = local_device_to_torch(tt_model.inner_step(prompt_1BLP=negative_1BLP, **common, **shared))

    checks = {
        "cond pass": (ref_cond, new_cond),
        "uncond pass": (ref_uncond, new_uncond),
        "combined_step": (ref_combined, new_combined),
    }
    failures = []
    for name, (ref, new) in checks.items():
        diff = (ref.float() - new.float()).abs().max().item()
        logger.info(f"CFG hoist {name}: max_abs_diff = {diff}")
        print(f"CFGHOIST {name}: max_abs_diff = {diff}")
        if diff != 0.0:
            failures.append(f"{name}: max_abs_diff {diff} != 0")
    assert not failures, "; ".join(failures)
