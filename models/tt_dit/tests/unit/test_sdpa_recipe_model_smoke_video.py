# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device smoke tests: SDPA recipes wired into the Mochi, Wan2.2 and LTX-2 video attentions (LTX-2 audio:
test_sdpa_recipe_model_smoke_ltx_audio.py), variants and gates from sdpa_smoke.py (gate_vs_torch).

Random weights (no checkpoints), the torch reference (the diffusers attention the model tests use) in fp32 on
the bf16-rounded weights and inputs the device sees.
"""

from __future__ import annotations

import pytest
import torch

import ttnn

from ...models.transformers.attention_mochi import MochiAttention
from ...models.transformers.ltx.attention_ltx import LTXAttention
from ...models.transformers.wan2_2.attention_wan import WanAttention
from ...utils.mochi import get_rot_transformation_mat, stack_cos_sin
from ...utils.padding import pad_vision_seq_parallel
from ...utils.tensor import bf16_tensor, bf16_tensor_2dshard, from_torch
from .sdpa_smoke import (
    MESHES,
    bf16_weights,
    convert_ltx_state,
    gate_vs_torch,
    gather,
    ltx_video_rope,
    parallel,
    randn_bf16,
    run_variants,
)

# variant -> (absolute L2 bound %, margin over legacy L2 vs torch in % points); bfp8 K/V storage adds its own
# quantization error on top of the recipe, so FAST gets 1 point over legacy.
BOUNDS = {"default": (3.0, 0.5), "FAST_bfp8": (3.0, 1.0)}


def rope_cos_sin(*shape) -> tuple[torch.Tensor, torch.Tensor]:
    """Unit-magnitude RoPE factors (cos/sin of random angles), like real rotary embeddings."""
    theta = torch.rand(*shape) * 2 * torch.pi
    return theta.cos(), theta.sin()


# ---------------------------------------------------------------------------------------------- Mochi


@MESHES
@pytest.mark.parametrize(("spatial_seq_len", "prompt_seq_len"), [(1024, 118)], ids=["N1024L118"])
def test_mochi_joint_attention_recipes(
    mesh_device, sp_axis, tp_axis, spatial_seq_len, prompt_seq_len, record_property
) -> None:
    from diffusers.models.transformers.transformer_mochi import MochiAttention as TorchMochiAttention
    from diffusers.models.transformers.transformer_mochi import MochiAttnProcessor2_0

    heads, head_dim = 4, 128
    query_dim = heads * head_dim
    added_kv_proj_dim = 256
    B = 1

    torch.manual_seed(0)
    torch_model = TorchMochiAttention(
        query_dim=query_dim,
        added_kv_proj_dim=added_kv_proj_dim,
        processor=MochiAttnProcessor2_0(),
        heads=heads,
        dim_head=head_dim,
        bias=False,
        added_proj_bias=False,
        out_dim=None,
        out_context_dim=added_kv_proj_dim,
        out_bias=True,
        context_pre_only=False,
        eps=1e-5,
    )
    state = bf16_weights(torch_model).state_dict()

    spatial = randn_bf16(B, spatial_seq_len, query_dim)
    prompt = randn_bf16(B, prompt_seq_len, added_kv_proj_dim)
    rope_cos, rope_sin = rope_cos_sin(spatial_seq_len, heads, head_dim // 2)
    with torch.no_grad():
        torch_spatial, torch_prompt = torch_model(
            spatial,
            prompt,
            attention_mask=torch.ones(B, prompt_seq_len),
            image_rotary_emb=[rope_cos, rope_sin],
        )

    ccl_manager, parallel_config, sp_factor = parallel(mesh_device, sp_axis, tp_axis)
    cos_stack, sin_stack = stack_cos_sin(
        rope_cos.unsqueeze(0).permute(0, 2, 1, 3), rope_sin.unsqueeze(0).permute(0, 2, 1, 3)
    )
    tt_spatial = bf16_tensor(
        pad_vision_seq_parallel(spatial.unsqueeze(0), num_devices=sp_factor),
        device=mesh_device,
        mesh_axis=sp_axis,
        shard_dim=-2,
    )
    tt_prompt = bf16_tensor(prompt.unsqueeze(0), device=mesh_device)
    shard = {sp_axis: 2, tp_axis: 1}
    tt_cos = bf16_tensor_2dshard(pad_vision_seq_parallel(cos_stack, num_devices=sp_factor), mesh_device, shard)
    tt_sin = bf16_tensor_2dshard(pad_vision_seq_parallel(sin_stack, num_devices=sp_factor), mesh_device, shard)
    tt_trans_mat = bf16_tensor(get_rot_transformation_mat(), device=mesh_device)

    def run_tt(precision, kv_dtype):
        tt_model = MochiAttention(
            query_dim=query_dim,
            added_kv_proj_dim=added_kv_proj_dim,
            heads=heads,
            head_dim=head_dim,
            bias=False,
            added_proj_bias=False,
            out_bias=True,
            out_context_dim=added_kv_proj_dim,
            context_pre_only=False,
            eps=1e-5,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            sdpa_precision=precision,
            sdpa_kv_dtype=kv_dtype,
        )
        tt_model.load_torch_state_dict(dict(state))
        out_spatial, out_prompt = tt_model(
            tt_spatial, tt_prompt, N=spatial_seq_len, rope_cos=tt_cos, rope_sin=tt_sin, trans_mat=tt_trans_mat
        )
        out_spatial = gather(mesh_device, out_spatial, sp_axis, tp_axis, sp_dim=2, tp_dim=0)
        out_prompt = gather(mesh_device, out_prompt, sp_axis, tp_axis, sp_dim=0, tp_dim=1)
        out_prompt = out_prompt[:, :, :prompt_seq_len, :].reshape(-1, 1, prompt_seq_len, added_kv_proj_dim)
        # The replicated prompt output is checked on every SP device (ring order differs per device).
        outs = {"spatial": out_spatial[0][:, :spatial_seq_len, :]}
        outs.update({f"prompt_dev{i}": out_prompt[i] for i in range(out_prompt.shape[0])})
        return outs

    gate_vs_torch(
        record_property,
        f"mochi.{tuple(mesh_device.shape)}.N{spatial_seq_len}L{prompt_seq_len}",
        {"spatial": torch_spatial, **{f"prompt_dev{i}": torch_prompt for i in range(sp_factor)}},
        run_variants(run_tt),
        BOUNDS,
    )


# ------------------------------------------------------------------------------------------------ Wan


def _wan_torch(dim, heads, head_dim, is_self):
    from diffusers.models.transformers.transformer_wan import WanAttention as TorchWanAttention
    from diffusers.models.transformers.transformer_wan import WanAttnProcessor

    return TorchWanAttention(
        dim=dim,
        heads=heads,
        dim_head=head_dim,
        eps=1e-6,
        cross_attention_dim_head=None if is_self else head_dim,
        processor=WanAttnProcessor(),
    )


@MESHES
@pytest.mark.parametrize(("spatial_seq_len", "prompt_seq_len"), [(1024, 77)], ids=["N1024L77"])
def test_wan_cross_attention_recipes(
    mesh_device, sp_axis, tp_axis, spatial_seq_len, prompt_seq_len, record_property
) -> None:
    heads, head_dim = 4, 128
    dim = heads * head_dim
    B = 1

    torch.manual_seed(0)
    torch_model = bf16_weights(_wan_torch(dim, heads, head_dim, is_self=False))
    state = torch_model.state_dict()
    spatial = randn_bf16(B, spatial_seq_len, dim)
    prompt = randn_bf16(B, prompt_seq_len, dim)
    with torch.no_grad():
        torch_out = torch_model(hidden_states=spatial, encoder_hidden_states=prompt, rotary_emb=None)

    ccl_manager, parallel_config, sp_factor = parallel(mesh_device, sp_axis, tp_axis)
    tt_spatial = bf16_tensor_2dshard(
        pad_vision_seq_parallel(spatial.unsqueeze(0), num_devices=sp_factor),
        device=mesh_device,
        shard_mapping={sp_axis: 2, tp_axis: 3},
    )
    tt_prompt = bf16_tensor(prompt.unsqueeze(0), device=mesh_device)

    def run_tt(precision, kv_dtype):
        tt_model = WanAttention(
            dim=dim,
            num_heads=heads,
            qk_norm=True,
            eps=1e-6,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            is_self=False,
            sdpa_precision=precision,
            sdpa_kv_dtype=kv_dtype,
        )
        tt_model.load_torch_state_dict(dict(state))
        out = tt_model(tt_spatial, N=spatial_seq_len, prompt_1BLP=tt_prompt)
        return {"spatial": gather(mesh_device, out, sp_axis, tp_axis)[0][:, :spatial_seq_len, :]}

    gate_vs_torch(
        record_property,
        f"wan_cross.{tuple(mesh_device.shape)}.N{spatial_seq_len}L{prompt_seq_len}",
        {"spatial": torch_out},
        run_variants(run_tt),
        BOUNDS,
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{}], indirect=True)
def test_wan_self_attention_recipes_1x1(mesh_device, record_property) -> None:
    heads, head_dim = 4, 128
    dim = heads * head_dim
    B, seq_len = 1, 1024
    sp_axis, tp_axis = 0, 1

    torch.manual_seed(0)
    torch_model = bf16_weights(_wan_torch(dim, heads, head_dim, is_self=True))
    state = torch_model.state_dict()
    spatial = randn_bf16(B, seq_len, dim)
    rope_cos, rope_sin = rope_cos_sin(B, seq_len, 1, head_dim // 2)
    torch_cos, torch_sin = stack_cos_sin(rope_cos, rope_sin)
    with torch.no_grad():
        torch_out = torch_model(hidden_states=spatial, encoder_hidden_states=None, rotary_emb=[torch_cos, torch_sin])

    ccl_manager, parallel_config, _ = parallel(mesh_device, sp_axis, tp_axis)
    tt_spatial = bf16_tensor(spatial.unsqueeze(0), device=mesh_device)
    tt_cos = from_torch(torch_cos.permute(0, 2, 1, 3), device=mesh_device, dtype=ttnn.float32)
    tt_sin = from_torch(torch_sin.permute(0, 2, 1, 3), device=mesh_device, dtype=ttnn.float32)
    tt_trans_mat = bf16_tensor(get_rot_transformation_mat(), device=mesh_device)

    def run_tt(precision, kv_dtype):
        tt_model = WanAttention(
            dim=dim,
            num_heads=heads,
            qk_norm=True,
            eps=1e-6,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            is_self=True,
            sdpa_precision=precision,
            sdpa_kv_dtype=kv_dtype,
        )
        tt_model.load_torch_state_dict(dict(state))
        out = tt_model(tt_spatial, N=seq_len, rope_cos=tt_cos, rope_sin=tt_sin, trans_mat=tt_trans_mat)
        return {"spatial": gather(mesh_device, out, sp_axis, tp_axis)[0][:, :seq_len, :]}

    gate_vs_torch(record_property, f"wan_self.(1, 1).N{seq_len}", {"spatial": torch_out}, run_variants(run_tt), BOUNDS)


# ------------------------------------------------------------------------------------------------ LTX


def _ltx_model(mesh_device, ccl_manager, parallel_config, dim, heads, is_self, precision=None, kv_dtype=None):
    return LTXAttention(
        dim=dim,
        num_heads=heads,
        eps=1e-6,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_self=is_self,
        context_dim=None if is_self else dim,
        sdpa_precision=precision,
        sdpa_kv_dtype=kv_dtype,
    )


@MESHES
def test_ltx_video_self_attention_recipes(mesh_device, sp_axis, tp_axis, record_property) -> None:
    from diffusers.models.transformers.transformer_ltx2 import LTX2Attention

    heads, head_dim = 4, 128
    dim = heads * head_dim
    B, (F, H, W) = 1, (4, 16, 16)
    seq_len = F * H * W  # 1024

    torch.manual_seed(42)
    torch_model = LTX2Attention(
        query_dim=dim, heads=heads, kv_heads=heads, dim_head=head_dim, norm_eps=1e-6, rope_type="interleaved"
    )
    state = convert_ltx_state(bf16_weights(torch_model).state_dict(), heads, head_dim)
    x = randn_bf16(B, seq_len, dim)
    cos_freq, sin_freq = ltx_video_rope(dim, heads, head_dim, F, H, W)
    with torch.no_grad():
        torch_out = torch_model(x, query_rotary_emb=(cos_freq, sin_freq))

    ccl_manager, parallel_config, _ = parallel(mesh_device, sp_axis, tp_axis)
    tt_spatial = bf16_tensor_2dshard(x.unsqueeze(0), device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 3})
    cos_bhnd = cos_freq.reshape(B, seq_len, heads, head_dim).permute(0, 2, 1, 3)
    sin_bhnd = sin_freq.reshape(B, seq_len, heads, head_dim).permute(0, 2, 1, 3)
    tt_cos = bf16_tensor_2dshard(cos_bhnd, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 1})
    tt_sin = bf16_tensor_2dshard(sin_bhnd, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 1})
    tt_trans_mat = bf16_tensor(get_rot_transformation_mat(), device=mesh_device)

    def run_tt(precision, kv_dtype):
        tt_model = _ltx_model(mesh_device, ccl_manager, parallel_config, dim, heads, True, precision, kv_dtype)
        tt_model.load_torch_state_dict(dict(state))
        out = tt_model(spatial_1BND=tt_spatial, N=seq_len, rope_cos=tt_cos, rope_sin=tt_sin, trans_mat=tt_trans_mat)
        return {"spatial": gather(mesh_device, out, sp_axis, tp_axis).squeeze(0)}

    gate_vs_torch(
        record_property,
        f"ltx_self.{tuple(mesh_device.shape)}.N{seq_len}",
        {"spatial": torch_out},
        run_variants(run_tt),
        BOUNDS,
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("prompt_seq_len", [32], ids=["L32"])
@pytest.mark.parametrize("device_params", [{}], indirect=True)
def test_ltx_video_text_cross_attention_recipes_1x1(mesh_device, prompt_seq_len, record_property) -> None:
    from diffusers.models.transformers.transformer_ltx2 import LTX2Attention

    heads, head_dim = 4, 128
    dim = heads * head_dim
    B, seq_len = 1, 1024
    sp_axis, tp_axis = 0, 1

    torch.manual_seed(42)
    torch_model = LTX2Attention(
        query_dim=dim, cross_attention_dim=dim, heads=heads, kv_heads=heads, dim_head=head_dim, norm_eps=1e-6
    )
    # No RoPE on cross-attention: the state loads as-is (like test_ltx_cross_attention).
    state = bf16_weights(torch_model).state_dict()
    x = randn_bf16(B, seq_len, dim)
    context = randn_bf16(B, prompt_seq_len, dim)
    with torch.no_grad():
        torch_out = torch_model(x, encoder_hidden_states=context)

    ccl_manager, parallel_config, _ = parallel(mesh_device, sp_axis, tp_axis)
    tt_spatial = bf16_tensor(x.unsqueeze(0), device=mesh_device)
    tt_prompt = bf16_tensor(context.unsqueeze(0), device=mesh_device)

    def run_tt(precision, kv_dtype):
        tt_model = _ltx_model(mesh_device, ccl_manager, parallel_config, dim, heads, False, precision, kv_dtype)
        tt_model.load_torch_state_dict(dict(state))
        out = tt_model(spatial_1BND=tt_spatial, N=seq_len, prompt_1BLP=tt_prompt)
        return {"spatial": gather(mesh_device, out, sp_axis, tp_axis).squeeze(0)}

    gate_vs_torch(
        record_property,
        f"ltx_cross.(1, 1).N{seq_len}L{prompt_seq_len}",
        {"spatial": torch_out},
        run_variants(run_tt),
        BOUNDS,
    )
