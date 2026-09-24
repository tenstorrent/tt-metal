# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device smoke tests: opt-in SDPA recipes wired into the Mochi, Wan2.2 and LTX-2 video attentions
(LTX-2 audio: test_sdpa_recipe_model_smoke_ltx_audio.py).

Random weights (no checkpoints). Each case builds a fresh tt module per variant (legacy, FAST, ACCURATE,
LOW_PRECISION with bfp8 K/V) from the SAME torch state dict and inputs and compares every output with the
torch reference module (the diffusers attention the model tests use) and with the legacy tt output.

The torch reference runs in fp32 on the bf16-rounded weights and inputs the device sees. Gate per recipe
variant: L2 vs torch <= legacy L2 vs torch + margin (0.5 points, 1.0 for bfp8 LOW_PRECISION), and L2 vs torch
<= the absolute bound (3% FAST / LOW_PRECISION, 1% ACCURATE). If the legacy tt output itself misses the bound
vs torch, the absolute bound may be met vs the legacy tt output instead. L2 is 100 * ||a - b|| / ||b||.
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn

from ...models.transformers.attention_mochi import MochiAttention
from ...models.transformers.ltx.attention_ltx import LTXAttention
from ...models.transformers.wan2_2.attention_wan import WanAttention
from ...parallel.config import DiTParallelConfig, ParallelFactor
from ...parallel.manager import CCLManager
from ...utils.mochi import get_rot_transformation_mat, stack_cos_sin
from ...utils.padding import pad_vision_seq_parallel
from ...utils.tensor import bf16_tensor, bf16_tensor_2dshard, from_torch

P = ttnn.SDPAPrecision
LINE_1D = {"fabric_config": ttnn.FabricConfig.FABRIC_1D}

# name -> (sdpa_precision, sdpa_kv_dtype, absolute L2 bound %, margin over legacy L2 vs torch in % points)
VARIANTS = {
    "legacy": (None, None, None, None),
    "FAST": (P.FAST, None, 3.0, 0.5),
    "ACCURATE": (P.ACCURATE, None, 1.0, 0.5),
    # bfp8 K/V storage adds its own quantization error on top of the recipe; allow 1 point over legacy.
    "LOW_PRECISION_bfp8": (P.LOW_PRECISION, ttnn.bfloat8_b, 3.0, 1.0),
}

# 1x1 runs without fabric (a 1x1 submesh with FABRIC_1D fails the router handshake on a 2-chip host).
MESHES = [
    pytest.param((1, 1), 0, 1, {}, id="1x1"),
    pytest.param((1, 2), 1, 0, LINE_1D, id="1x2sp1"),
]
MESH_ARGS = ("mesh_device", "sp_axis", "tp_axis", "device_params")
MESH_INDIRECT = ["mesh_device", "device_params"]


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double().flatten(), b.double().flatten()
    return (100.0 * torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b)).item()


def randn_bf16(*shape) -> torch.Tensor:
    """Random input already representable in bf16, so the torch reference sees the device input."""
    return torch.randn(*shape).bfloat16().float()


def bf16_weights(module: torch.nn.Module) -> torch.nn.Module:
    """Round the random weights to bf16 in place: torch reference and tt module share identical weights."""
    with torch.no_grad():
        for param in module.parameters():
            param.copy_(param.bfloat16().float())
    return module.eval()


def rope_cos_sin(*shape) -> tuple[torch.Tensor, torch.Tensor]:
    """Unit-magnitude RoPE factors (cos/sin of random angles), like real rotary embeddings."""
    theta = torch.rand(*shape) * 2 * torch.pi
    return theta.cos(), theta.sin()


def _parallel(mesh_device, sp_axis, tp_axis):
    shape = tuple(mesh_device.shape)
    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=1, topology=ttnn.Topology.Linear)
    parallel_config = DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=shape[tp_axis]),
        sequence_parallel=ParallelFactor(mesh_axis=sp_axis, factor=shape[sp_axis]),
        cfg_parallel=ParallelFactor(factor=1, mesh_axis=0),
    )
    return ccl_manager, parallel_config, shape[sp_axis]


def _gather(mesh_device, tt_out, sp_axis, tp_axis, sp_dim=2, tp_dim=3):
    dims = [None, None]
    dims[sp_axis] = sp_dim
    dims[tp_axis] = tp_dim
    return ttnn.to_torch(
        tt_out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=dims, mesh_shape=tuple(mesh_device.shape))
    )


def run_variants(record_property, label: str, torch_outs: dict, run_tt) -> None:
    """run_tt(precision, kv_dtype) -> dict name->torch tensor (same keys as torch_outs)."""
    tt_outs = {}
    for variant, (precision, kv_dtype, _, _) in VARIANTS.items():
        tt_outs[variant] = run_tt(precision, kv_dtype)

    failures = []
    for key, ref in torch_outs.items():
        legacy = tt_outs["legacy"][key]
        legacy_l2 = rel_l2(legacy, ref)
        record_property(f"{label}.{key}.legacy.l2_vs_torch", round(legacy_l2, 4))
        logger.info(f"{label}.{key} legacy: L2 vs torch {legacy_l2:.4f}%")
        for variant, (_, _, bound, margin) in VARIANTS.items():
            if bound is None:
                continue
            out = tt_outs[variant][key]
            l2_torch = rel_l2(out, ref)
            l2_legacy = rel_l2(out, legacy)
            record_property(f"{label}.{key}.{variant}.l2_vs_torch", round(l2_torch, 4))
            record_property(f"{label}.{key}.{variant}.l2_vs_legacy", round(l2_legacy, 4))
            ok_rel = l2_torch <= legacy_l2 + margin
            if variant.startswith("LOW_PRECISION"):
                # LOW_PRECISION also rounds its inputs (RNE7 Q, RNE5+BFP8 KV), so its error may scale with
                # the legacy error (short-KV cross attention has a small reference norm): allow 1.3x.
                bound = max(bound, 1.3 * legacy_l2)
            if legacy_l2 <= bound:
                ok_abs = l2_torch <= bound
                gate = f"vs torch <= legacy {legacy_l2:.3f} + {margin} and <= {bound}"
            else:
                # The legacy tt output already misses the bound vs torch: bound the recipe vs legacy tt.
                ok_abs = l2_torch <= bound or l2_legacy <= bound
                gate = f"vs torch <= legacy {legacy_l2:.3f} + {margin}; vs torch or vs legacy tt <= {bound}"
            ok = ok_rel and ok_abs
            logger.info(
                f"{label}.{key} {variant}: L2 vs torch {l2_torch:.4f}%, vs legacy {l2_legacy:.4f}% "
                f"[{gate}] {'OK' if ok else 'FAIL'}"
            )
            if not ok:
                failures.append(f"{key}/{variant}: vs torch {l2_torch:.4f}%, vs legacy {l2_legacy:.4f}%, gate {gate}")
    assert not failures, f"{label}: " + "; ".join(failures)


# ---------------------------------------------------------------------------------------------- Mochi


@pytest.mark.parametrize(MESH_ARGS, MESHES, indirect=MESH_INDIRECT)
@pytest.mark.parametrize(("spatial_seq_len", "prompt_seq_len"), [(1024, 128), (1024, 118)], ids=["N1024L128", "N1024L118"])
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

    ccl_manager, parallel_config, sp_factor = _parallel(mesh_device, sp_axis, tp_axis)
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
        out_spatial = _gather(mesh_device, out_spatial, sp_axis, tp_axis, sp_dim=2, tp_dim=0)
        out_prompt = _gather(mesh_device, out_prompt, sp_axis, tp_axis, sp_dim=0, tp_dim=1)
        out_prompt = out_prompt[:, :, :prompt_seq_len, :].reshape(-1, 1, prompt_seq_len, added_kv_proj_dim)
        # The replicated prompt output is checked on every SP device (ring order differs per device).
        outs = {"spatial": out_spatial[0][:, :spatial_seq_len, :]}
        outs.update({f"prompt_dev{i}": out_prompt[i] for i in range(out_prompt.shape[0])})
        return outs

    run_variants(
        record_property,
        f"mochi.{tuple(mesh_device.shape)}.N{spatial_seq_len}L{prompt_seq_len}",
        {"spatial": torch_spatial, **{f"prompt_dev{i}": torch_prompt for i in range(sp_factor)}},
        run_tt,
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


@pytest.mark.parametrize(MESH_ARGS, MESHES, indirect=MESH_INDIRECT)
@pytest.mark.parametrize(("spatial_seq_len", "prompt_seq_len"), [(1024, 128), (1024, 77)], ids=["N1024L128", "N1024L77"])
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

    ccl_manager, parallel_config, sp_factor = _parallel(mesh_device, sp_axis, tp_axis)
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
        return {"spatial": _gather(mesh_device, out, sp_axis, tp_axis)[0][:, :spatial_seq_len, :]}

    run_variants(
        record_property,
        f"wan_cross.{tuple(mesh_device.shape)}.N{spatial_seq_len}L{prompt_seq_len}",
        {"spatial": torch_out},
        run_tt,
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

    ccl_manager, parallel_config, _ = _parallel(mesh_device, sp_axis, tp_axis)
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
        return {"spatial": _gather(mesh_device, out, sp_axis, tp_axis)[0][:, :seq_len, :]}

    run_variants(record_property, f"wan_self.(1, 1).N{seq_len}", {"spatial": torch_out}, run_tt)


# ------------------------------------------------------------------------------------------------ LTX


def _diffusers_qk_to_split(t: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    """diffusers (interleaved-rotation) Q/K -> Lightricks SPLIT convention the TT loader expects."""
    inv = torch.empty(head_dim, dtype=torch.long)
    inv[: head_dim // 2] = torch.arange(0, head_dim, 2)
    inv[head_dim // 2 :] = torch.arange(1, head_dim, 2)
    rest = t.shape[1:]
    return t.reshape(num_heads, head_dim, *rest).index_select(1, inv).reshape(num_heads * head_dim, *rest)


def _convert_ltx_state(state: dict, num_heads: int, head_dim: int) -> dict:
    out = dict(state)
    for k in ("to_q.weight", "to_q.bias", "to_k.weight", "to_k.bias", "norm_q.weight", "norm_k.weight"):
        if k in out:
            out[k] = _diffusers_qk_to_split(out[k], num_heads, head_dim)
    return out


def _ltx_video_rope(dim, heads, head_dim, F, H, W):
    from diffusers.models.transformers.transformer_ltx2 import LTX2AudioVideoRotaryPosEmbed

    grid = torch.meshgrid(torch.arange(F), torch.arange(H), torch.arange(W), indexing="ij")
    indices = torch.stack([g.flatten() for g in grid], dim=0).float().unsqueeze(0)
    rope = LTX2AudioVideoRotaryPosEmbed(
        dim=dim,
        base_num_frames=20,
        base_height=2048,
        base_width=2048,
        theta=10000.0,
        modality="video",
        double_precision=False,
        rope_type="interleaved",
        num_attention_heads=heads,
    )
    return rope(indices)  # (B, N, dim) interleaved cos/sin


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


@pytest.mark.parametrize(MESH_ARGS, MESHES, indirect=MESH_INDIRECT)
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
    state = _convert_ltx_state(bf16_weights(torch_model).state_dict(), heads, head_dim)
    x = randn_bf16(B, seq_len, dim)
    cos_freq, sin_freq = _ltx_video_rope(dim, heads, head_dim, F, H, W)
    with torch.no_grad():
        torch_out = torch_model(x, query_rotary_emb=(cos_freq, sin_freq))

    ccl_manager, parallel_config, _ = _parallel(mesh_device, sp_axis, tp_axis)
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
        return {"spatial": _gather(mesh_device, out, sp_axis, tp_axis).squeeze(0)}

    run_variants(record_property, f"ltx_self.{tuple(mesh_device.shape)}.N{seq_len}", {"spatial": torch_out}, run_tt)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("prompt_seq_len", [32, 128], ids=["L32", "L128"])
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

    ccl_manager, parallel_config, _ = _parallel(mesh_device, sp_axis, tp_axis)
    tt_spatial = bf16_tensor(x.unsqueeze(0), device=mesh_device)
    tt_prompt = bf16_tensor(context.unsqueeze(0), device=mesh_device)

    def run_tt(precision, kv_dtype):
        tt_model = _ltx_model(mesh_device, ccl_manager, parallel_config, dim, heads, False, precision, kv_dtype)
        tt_model.load_torch_state_dict(dict(state))
        out = tt_model(spatial_1BND=tt_spatial, N=seq_len, prompt_1BLP=tt_prompt)
        return {"spatial": _gather(mesh_device, out, sp_axis, tp_axis).squeeze(0)}

    run_variants(
        record_property, f"ltx_cross.(1, 1).N{seq_len}L{prompt_seq_len}", {"spatial": torch_out}, run_tt
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{}], indirect=True)
def test_ltx_recipe_accepts_general_mask(mesh_device) -> None:
    """Recipes accept masks they cannot turn into a K/V slice (no attn_kv_len; any cross mask).
    An all-zero additive mask adds exactly 0 to every score, so masked outputs match the
    unmasked recipe run. The D64 audio attentions are covered by
    test_sdpa_recipe_model_smoke_ltx_audio.py."""
    from diffusers.models.transformers.transformer_ltx2 import LTX2Attention

    ccl_manager, parallel_config, _ = _parallel(mesh_device, 0, 1)
    heads, head_dim = 2, 128
    dim = heads * head_dim
    B, (F, H, W) = 1, (1, 16, 16)
    seq_len = F * H * W  # 256
    torch.manual_seed(7)
    state = _convert_ltx_state(
        bf16_weights(
            LTX2Attention(query_dim=dim, heads=heads, kv_heads=heads, dim_head=head_dim, rope_type="interleaved")
        ).state_dict(),
        heads,
        head_dim,
    )
    x = randn_bf16(B, seq_len, dim)
    cos_freq, sin_freq = _ltx_video_rope(dim, heads, head_dim, F, H, W)
    tt_x = bf16_tensor(x.unsqueeze(0), device=mesh_device)
    tt_cos = bf16_tensor(cos_freq.reshape(B, seq_len, heads, head_dim).permute(0, 2, 1, 3), device=mesh_device)
    tt_sin = bf16_tensor(sin_freq.reshape(B, seq_len, heads, head_dim).permute(0, 2, 1, 3), device=mesh_device)
    tt_trans_mat = bf16_tensor(get_rot_transformation_mat(), device=mesh_device)
    tt_mask = bf16_tensor(torch.zeros(1, 1, seq_len, seq_len), device=mesh_device)

    tt_model = _ltx_model(mesh_device, ccl_manager, parallel_config, dim, heads, True, P.FAST)
    tt_model.load_torch_state_dict(dict(state))
    rope = dict(rope_cos=tt_cos, rope_sin=tt_sin, trans_mat=tt_trans_mat)
    masked = _gather(mesh_device, tt_model(spatial_1BND=tt_x, N=seq_len, attn_mask=tt_mask, **rope), 0, 1)
    unmasked = _gather(mesh_device, tt_model(spatial_1BND=tt_x, N=seq_len, **rope), 0, 1)
    assert torch.isfinite(masked.float()).all()
    assert rel_l2(masked.float(), unmasked.float()) < 1e-3

    tt_cross = _ltx_model(mesh_device, ccl_manager, parallel_config, dim, heads, False, P.FAST)
    tt_cross.load_torch_state_dict(dict(state))
    out = tt_cross(spatial_1BND=tt_x, N=seq_len, prompt_1BLP=tt_x, attn_mask=tt_mask, attn_kv_len=128)
    assert torch.isfinite(_gather(mesh_device, out, 0, 1).float()).all()
