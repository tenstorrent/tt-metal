# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device smoke tests: opt-in SDPA recipes on the LTX-2 D64 audio attentions.

Covers audio self-attn (padded: legacy key-column mask vs recipe K/V slice to the real length or the recipe
attn_mask itself; and unpadded),
audio<->text cross-attn, A2V (video Q, audio K/V) and V2A (audio Q, SP-sharded video K/V; ring is_cross on 1x2).
Random weights; per case a fresh tt module per variant (legacy, FAST, ACCURATE, LOW_PRECISION bfp8) from the SAME
torch state dict and inputs, gated vs the torch reference and the legacy tt output like
test_sdpa_recipe_model_smoke_video.py (run_variants). For padded audio the torch reference runs on the real tokens
only, and the comparison covers the real query rows (padded rows are zeroed downstream by the model).
"""

from __future__ import annotations

import pytest
import torch

import ttnn

from ...models.transformers.ltx.attention_ltx import LTXAttention
from ...models.transformers.ltx.transformer_ltx import build_audio_masks
from ...utils.mochi import get_rot_transformation_mat
from ...utils.tensor import bf16_tensor, bf16_tensor_2dshard
from .test_sdpa_recipe_model_smoke_video import (
    MESH_ARGS,
    MESH_INDIRECT,
    MESHES,
    _convert_ltx_state,
    _gather,
    _ltx_video_rope,
    _parallel,
    bf16_weights,
    randn_bf16,
    run_variants,
)

HEADS, HEAD_DIM = 4, 64
AUDIO_DIM = HEADS * HEAD_DIM  # 256: D64 like LTX-2 audio (2048 / 32)
VIDEO_DIM = 512


def _rope(inner_dim, grid):
    """(B, N, inner) interleaved cos/sin for the positions of a small (F, H, W) grid."""
    return _ltx_video_rope(inner_dim, HEADS, HEAD_DIM, *grid)


def _bhnd(t, n):
    return t.reshape(1, n, HEADS, HEAD_DIM).permute(0, 2, 1, 3)


def _tt_rope(cos, sin, n, mesh_device, sp_axis, tp_axis):
    shard = {sp_axis: 2, tp_axis: 1}
    return (
        bf16_tensor_2dshard(_bhnd(cos, n), device=mesh_device, shard_mapping=shard),
        bf16_tensor_2dshard(_bhnd(sin, n), device=mesh_device, shard_mapping=shard),
    )


def _audio_model(mesh_device, ccl_manager, parallel_config, precision, kv_dtype, **kwargs):
    return LTXAttention(
        dim=AUDIO_DIM,
        num_heads=HEADS,
        eps=1e-6,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        sdpa_precision=precision,
        sdpa_kv_dtype=kv_dtype,
        **kwargs,
    )


@pytest.mark.parametrize(MESH_ARGS, MESHES, indirect=MESH_INDIRECT)
@pytest.mark.parametrize("audio_n_real", [200, 224, 256], ids=["real200", "real224", "unpadded256"])
@pytest.mark.parametrize("recipe_mask", ["slice", "mask"])
def test_ltx_audio_self_attention_recipes(
    mesh_device, sp_axis, tp_axis, audio_n_real, recipe_mask, record_property
) -> None:
    """Padded audio self-attn: legacy passes the key-column mask; a recipe slices K/V to audio_n_real
    ("slice", attn_kv_len given) or receives the mask itself ("mask", no attn_kv_len) (1x1 dense;
    1x2 gathered K/V). Unpadded 256 on 1x2 is the D64 ring joint path."""
    if recipe_mask == "mask" and audio_n_real == 256:
        pytest.skip("unpadded: no mask")
    from diffusers.models.transformers.transformer_ltx2 import LTX2Attention

    audio_n, grid = 256, (1, 16, 16)
    torch.manual_seed(11)
    torch_model = LTX2Attention(
        query_dim=AUDIO_DIM, heads=HEADS, kv_heads=HEADS, dim_head=HEAD_DIM, norm_eps=1e-6, rope_type="interleaved"
    )
    state = _convert_ltx_state(bf16_weights(torch_model).state_dict(), HEADS, HEAD_DIM)
    x = randn_bf16(1, audio_n, AUDIO_DIM)
    cos, sin = _rope(AUDIO_DIM, grid)
    with torch.no_grad():
        # Reference: attention over the real tokens only (== key-column masking for the real query rows).
        torch_out = torch_model(
            x[:, :audio_n_real], query_rotary_emb=(cos[:, :audio_n_real], sin[:, :audio_n_real])
        )

    ccl_manager, parallel_config, _ = _parallel(mesh_device, sp_axis, tp_axis)
    tt_x = bf16_tensor_2dshard(x.unsqueeze(0), device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 3})
    tt_cos, tt_sin = _tt_rope(cos, sin, audio_n, mesh_device, sp_axis, tp_axis)
    tt_trans_mat = bf16_tensor(get_rot_transformation_mat(), device=mesh_device)
    tt_mask, _, _ = build_audio_masks(audio_n, audio_n_real, mesh_device=mesh_device, sp_axis=sp_axis)
    assert (tt_mask is None) == (audio_n_real == audio_n)

    def run_tt(precision, kv_dtype):
        tt_model = _audio_model(mesh_device, ccl_manager, parallel_config, precision, kv_dtype, is_self=True)
        tt_model.load_torch_state_dict(dict(state))
        out = tt_model(
            spatial_1BND=tt_x,
            N=audio_n,
            rope_cos=tt_cos,
            rope_sin=tt_sin,
            trans_mat=tt_trans_mat,
            attn_mask=tt_mask,
            attn_kv_len=audio_n_real if recipe_mask == "slice" else None,
        )
        out = _gather(mesh_device, out, sp_axis, tp_axis).squeeze(0)
        return {"audio": out[:, :audio_n_real]}

    label = f"ltx_audio_self.{tuple(mesh_device.shape)}.N{audio_n}real{audio_n_real}.{recipe_mask}"
    run_variants(record_property, label, {"audio": torch_out}, run_tt)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{}], indirect=True)
@pytest.mark.parametrize("prompt_seq_len", [32, 128], ids=["L32", "L128"])
def test_ltx_audio_text_cross_attention_recipes_1x1(mesh_device, prompt_seq_len, record_property) -> None:
    from diffusers.models.transformers.transformer_ltx2 import LTX2Attention

    audio_n = 256
    torch.manual_seed(12)
    torch_model = LTX2Attention(
        query_dim=AUDIO_DIM, cross_attention_dim=AUDIO_DIM, heads=HEADS, kv_heads=HEADS, dim_head=HEAD_DIM
    )
    state = bf16_weights(torch_model).state_dict()  # no RoPE: loads as-is
    x = randn_bf16(1, audio_n, AUDIO_DIM)
    context = randn_bf16(1, prompt_seq_len, AUDIO_DIM)
    with torch.no_grad():
        torch_out = torch_model(x, encoder_hidden_states=context)

    ccl_manager, parallel_config, _ = _parallel(mesh_device, 0, 1)
    tt_x = bf16_tensor(x.unsqueeze(0), device=mesh_device)
    tt_prompt = bf16_tensor(context.unsqueeze(0), device=mesh_device)

    def run_tt(precision, kv_dtype):
        tt_model = _audio_model(
            mesh_device, ccl_manager, parallel_config, precision, kv_dtype, is_self=False, context_dim=AUDIO_DIM
        )
        tt_model.load_torch_state_dict(dict(state))
        out = tt_model(spatial_1BND=tt_x, N=audio_n, prompt_1BLP=tt_prompt, kv_replicated=True)
        return {"audio": _gather(mesh_device, out, 0, 1).squeeze(0)}

    run_variants(
        record_property, f"ltx_audio_text.(1, 1).N{audio_n}L{prompt_seq_len}", {"audio": torch_out}, run_tt
    )


def _av_torch(query_dim, context_dim, seed):
    from diffusers.models.transformers.transformer_ltx2 import LTX2Attention

    torch.manual_seed(seed)
    torch_model = LTX2Attention(
        query_dim=query_dim,
        cross_attention_dim=context_dim,
        heads=HEADS,
        kv_heads=HEADS,
        dim_head=HEAD_DIM,
        norm_eps=1e-6,
        rope_type="interleaved",
    )
    return torch_model, _convert_ltx_state(bf16_weights(torch_model).state_dict(), HEADS, HEAD_DIM)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{}], indirect=True)
def test_ltx_a2v_cross_attention_recipes_1x1(mesh_device, record_property) -> None:
    """A2V: video Q (N=1024, video dim) attends audio K/V (N=256), D64 dense cross with Q and K RoPE."""
    video_n, audio_n = 1024, 256
    torch_model, state = _av_torch(VIDEO_DIM, AUDIO_DIM, seed=13)
    x = randn_bf16(1, video_n, VIDEO_DIM)
    context = randn_bf16(1, audio_n, AUDIO_DIM)
    q_cos, q_sin = _rope(AUDIO_DIM, (4, 16, 16))
    k_cos, k_sin = _rope(AUDIO_DIM, (1, 16, 16))
    with torch.no_grad():
        torch_out = torch_model(
            x, encoder_hidden_states=context, query_rotary_emb=(q_cos, q_sin), key_rotary_emb=(k_cos, k_sin)
        )

    ccl_manager, parallel_config, _ = _parallel(mesh_device, 0, 1)
    tt_x = bf16_tensor(x.unsqueeze(0), device=mesh_device)
    tt_prompt = bf16_tensor(context.unsqueeze(0), device=mesh_device)
    tt_q_cos, tt_q_sin = _tt_rope(q_cos, q_sin, video_n, mesh_device, 0, 1)
    tt_k_cos, tt_k_sin = _tt_rope(k_cos, k_sin, audio_n, mesh_device, 0, 1)
    tt_trans_mat = bf16_tensor(get_rot_transformation_mat(), device=mesh_device)

    def run_tt(precision, kv_dtype):
        tt_model = _audio_model(
            mesh_device,
            ccl_manager,
            parallel_config,
            precision,
            kv_dtype,
            is_self=False,
            context_dim=AUDIO_DIM,
            query_input_dim=VIDEO_DIM,
            output_dim=VIDEO_DIM,
        )
        tt_model.load_torch_state_dict(dict(state))
        out = tt_model(
            spatial_1BND=tt_x,
            N=video_n,
            prompt_1BLP=tt_prompt,
            rope_cos=tt_q_cos,
            rope_sin=tt_q_sin,
            k_rope_cos=tt_k_cos,
            k_rope_sin=tt_k_sin,
            trans_mat=tt_trans_mat,
        )
        return {"video": _gather(mesh_device, out, 0, 1).squeeze(0)}

    run_variants(record_property, f"ltx_a2v.(1, 1).Nq{video_n}Nk{audio_n}", {"video": torch_out}, run_tt)


@pytest.mark.parametrize(MESH_ARGS, MESHES, indirect=MESH_INDIRECT)
def test_ltx_v2a_cross_attention_recipes(mesh_device, sp_axis, tp_axis, record_property) -> None:
    """V2A: audio Q (N=128; 64 rows per device on 1x2) attends video K/V (N=1024). On 1x2 the video K/V stay
    SP-sharded and ring joint SDPA (is_cross=True) gathers them; on 1x1 it is a dense cross."""
    audio_n, video_n = 128, 1024
    torch_model, state = _av_torch(AUDIO_DIM, VIDEO_DIM, seed=14)
    x = randn_bf16(1, audio_n, AUDIO_DIM)
    context = randn_bf16(1, video_n, VIDEO_DIM)
    q_cos, q_sin = _rope(AUDIO_DIM, (1, 8, 16))
    k_cos, k_sin = _rope(AUDIO_DIM, (4, 16, 16))
    with torch.no_grad():
        torch_out = torch_model(
            x, encoder_hidden_states=context, query_rotary_emb=(q_cos, q_sin), key_rotary_emb=(k_cos, k_sin)
        )

    ccl_manager, parallel_config, sp_factor = _parallel(mesh_device, sp_axis, tp_axis)
    shard = {sp_axis: 2, tp_axis: 3}
    tt_x = bf16_tensor_2dshard(x.unsqueeze(0), device=mesh_device, shard_mapping=shard)
    tt_prompt = bf16_tensor(context.unsqueeze(0), device=mesh_device, mesh_axis=sp_axis, shard_dim=2)
    tt_q_cos, tt_q_sin = _tt_rope(q_cos, q_sin, audio_n, mesh_device, sp_axis, tp_axis)
    tt_k_cos, tt_k_sin = _tt_rope(k_cos, k_sin, video_n, mesh_device, sp_axis, tp_axis)
    tt_trans_mat = bf16_tensor(get_rot_transformation_mat(), device=mesh_device)

    def run_tt(precision, kv_dtype):
        tt_model = _audio_model(
            mesh_device, ccl_manager, parallel_config, precision, kv_dtype, is_self=False, context_dim=VIDEO_DIM
        )
        tt_model.load_torch_state_dict(dict(state))
        out = tt_model(
            spatial_1BND=tt_x,
            N=audio_n,
            prompt_1BLP=tt_prompt,
            rope_cos=tt_q_cos,
            rope_sin=tt_q_sin,
            k_rope_cos=tt_k_cos,
            k_rope_sin=tt_k_sin,
            kv_logical_n=video_n,
            trans_mat=tt_trans_mat,
        )
        return {"audio": _gather(mesh_device, out, sp_axis, tp_axis).squeeze(0)}

    label = f"ltx_v2a.{tuple(mesh_device.shape)}.Nq{audio_n}Nk{video_n}.sp{sp_factor}"
    run_variants(record_property, label, {"audio": torch_out}, run_tt)
