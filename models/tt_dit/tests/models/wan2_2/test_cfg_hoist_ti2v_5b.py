# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Bit-exactness gate for the CFG hoist in `WanTransformer3DModel.combined_step`.

Under classifier-free guidance `combined_step` runs `inner_step` twice with the same timestep
and spatial input, so the timestep embedding, the patch embedding and every block's AdaLN
modulation (plus the `norm_out` pair) are computed once and handed to both passes. Since sprint 6
the hoisted path also builds the modulation in a different layout: the flat timestep projection
is sliced at tile-aligned group boundaries (`split_timestep_proj`) and added to pre-split table
rows (`WanTransformerBlock.prepare_modulation_split`, `prepare_norm_out_modulation_split`),
while the inline `inner_step` keeps the original add-then-`ttnn.chunk` form. Both are pure data
movement plus the same elementwise ops, so the gate is exact equality, not a PCC floor: the
hoisted `combined_step` must match the pre-hoist computation (two independent `inner_step` calls
and a `ttnn.lerp`) to the bit, each pass must match its standalone `inner_step`, and every one
of the six modulation tensors per block (and the two `norm_out` ones) must match its legacy
counterpart, so a mismatch is attributed to one op.

Both timestep layouts are covered: the scalar (T2V) one and the two-row per-token (I2V) one,
the latter with all-ones masks so it reduces to the scalar values while exercising the
per-token arm's shapes.

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
MODULATION_NAMES = ("shift", "1+scale", "gate", "c_shift", "1+c_scale", "c_gate")


def _bit_exact_checks(tt_model, *, timestep, guidance, spatial_device, prompt_1BLP, negative_1BLP, n, rope):
    """Every (reference, new) pair the gate compares, for one timestep layout."""
    common = {
        "spatial_1BNI": spatial_device,
        "N": n,
        "timestep": timestep,
        "gather_output": False,
        **rope,
    }

    checks = {}

    # Pre-hoist computation: each pass embeds the timestep and the patches itself, and each
    # block builds its modulation inline with the legacy add-then-chunk layout.
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
            rope_cos_1HND=common["rope_cos_1HND"],
            rope_sin_1HND=common["rope_sin_1HND"],
            trans_mat=common["trans_mat"],
            timestep=timestep,
            guidance_scale=guidance,
            gather_output=False,
        )
    )

    # Each pass with the shared embeddings and the split-layout modulations handed in
    # explicitly (exactly what combined_step builds), against its standalone self.
    temb_11BD, proj_flat = tt_model.prepare_timestep_conditioning(timestep, flat_proj=True)
    block_modulations, norm_out_modulation = tt_model.prepare_hoisted_modulation(temb_11BD, proj_flat)
    shared = {
        "timestep_conditioning": (temb_11BD, proj_flat),
        "spatial_1BND": tt_model.patch_embedding(spatial_device),
        "block_modulations": block_modulations,
        "norm_out_modulation": norm_out_modulation,
    }
    new_cond = local_device_to_torch(tt_model.inner_step(prompt_1BLP=prompt_1BLP, **common, **shared))
    new_uncond = local_device_to_torch(tt_model.inner_step(prompt_1BLP=negative_1BLP, **common, **shared))

    checks["cond pass"] = (ref_cond, new_cond)
    checks["uncond pass"] = (ref_uncond, new_uncond)
    checks["combined_step"] = (ref_combined, new_combined)

    # Per-tensor: legacy layout (add the 6-row / flat table, ttnn.chunk, typecast) against the
    # split layout (tile-aligned slices of the projection, pre-split table rows, bf16 adds).
    _, proj_legacy = tt_model.prepare_timestep_conditioning(timestep)
    for b, (block, split) in enumerate(zip(tt_model.blocks, block_modulations)):
        legacy = block.prepare_modulation(proj_legacy)
        assert len(legacy) == len(split) == 6
        for name, l_t, s_t in zip(MODULATION_NAMES, legacy, split):
            assert l_t.dtype == s_t.dtype, f"block {b} {name}: dtype {l_t.dtype} vs {s_t.dtype}"
            assert tuple(l_t.shape) == tuple(s_t.shape), f"block {b} {name}: shape {l_t.shape} vs {s_t.shape}"
            checks[f"block {b} {name}"] = (local_device_to_torch(l_t), local_device_to_torch(s_t))
    legacy_norm_out = tt_model.prepare_norm_out_modulation(temb_11BD)
    for name, l_t, s_t in zip(("norm_out shift", "norm_out 1+scale"), legacy_norm_out, norm_out_modulation):
        assert tuple(l_t.shape) == tuple(s_t.shape), f"{name}: shape {l_t.shape} vs {s_t.shape}"
        checks[name] = (local_device_to_torch(l_t), local_device_to_torch(s_t))
    return checks


def _assert_bit_exact(checks, label):
    failures = []
    for name, (ref, new) in checks.items():
        diff = (ref.float() - new.float()).abs().max().item()
        logger.info(f"CFG hoist [{label}] {name}: max_abs_diff = {diff}")
        print(f"CFGHOIST [{label}] {name}: max_abs_diff = {diff}")
        if diff != 0.0:
            failures.append(f"[{label}] {name}: max_abs_diff {diff} != 0")
    assert not failures, "; ".join(failures)


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

    guidance = float32_tensor(torch.tensor(GUIDANCE_SCALE, dtype=torch.float32).reshape(1, 1, 1, 1), device=mesh_device)
    inputs = {
        "guidance": guidance,
        "spatial_device": spatial_device,
        "prompt_1BLP": prompt_1BLP,
        "negative_1BLP": negative_1BLP,
        "n": n,
        "rope": {"rope_cos_1HND": rope_cos, "rope_sin_1HND": rope_sin, "trans_mat": trans_mat},
    }

    # Scalar timestep (T2V, 14B): projection (1, 1, 1, 6*D/tp) -> legacy (1, 1, 6, D/tp).
    scalar_ts = float32_tensor(
        torch.full((1,), TIMESTEP, dtype=torch.float32).unsqueeze(1).unsqueeze(1).unsqueeze(1), device=mesh_device
    )
    _assert_bit_exact(_bit_exact_checks(tt_model, timestep=scalar_ts, **inputs), "scalar")

    # Two-row per-token timestep (TI2V-5B I2V): projection (1, 1, N, 6*D/tp). All-ones masks
    # select row 1 (TIMESTEP) everywhere, so the values equal the scalar case while every
    # modulation tensor takes the per-token shape.
    padded_n = spatial_host.shape[2]
    dim_tp = (cfg.num_attention_heads * cfg.attention_head_dim) // tuple(mesh_device.shape)[tp_axis]

    def ones_mask(width):
        m = torch.ones(1, 1, padded_n, width, dtype=torch.float32)
        return from_torch(m, device=mesh_device, mesh_axes=[None, None, sp_axis_, None], dtype=ttnn.float32)

    tt_model.set_per_token_timestep_masks(ones_mask(dim_tp), ones_mask(6 * dim_tp))
    two_row_ts = float32_tensor(
        torch.tensor([0.0, TIMESTEP], dtype=torch.float32).reshape(1, 1, 2, 1), device=mesh_device
    )
    _assert_bit_exact(_bit_exact_checks(tt_model, timestep=two_row_ts, **inputs), "two-row")
