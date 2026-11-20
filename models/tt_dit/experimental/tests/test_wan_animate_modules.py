# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Device-vs-reference (PCC) tests for the Wan2.2-Animate modules added on top of tt_dit Wan 2.2.

Goldens are generated at test time on the host CPU by the Diffusers reference
(``diffusers.models.transformers.transformer_wan_animate``), so nothing large is committed:

* ``tiny``  - scaled-down random-init config (dim 1024, 8 heads of 128: 2 heads per device at TP=4, which the per-head QK norm kernel needs). Fast; run first.
* ``full``  - the real 14B dimensions (dim 5120, 40 heads), random init. Exercises real shapes/sharding.
* ``real``  - real checkpoint weights for block 0 / face adapter 0 / face encoder. Opt-in with
              ``WAN_ANIMATE_REAL_WEIGHTS=1`` (needs the Diffusers checkpoint in the HF cache).

Run order and expected PCCs: ``models/tt_dit/experimental/models/Wan2_2_Animate_RUNBOOK.md``.
"""

from __future__ import annotations

import os

import pytest
import torch
from diffusers.models.transformers.transformer_wan_animate import (
    WanAnimateFaceBlockCrossAttention,
    WanAnimateFaceEncoder,
    WanAnimateTransformer3DModel,
)
from diffusers.models.transformers.transformer_wan_animate import WanTransformerBlock as TorchAnimateBlock
from loguru import logger

import ttnn

from models.tt_dit.layers.embeddings import WanPatchEmbed
from models.tt_dit.models.transformers.wan2_2.transformer_wan import WanTransformerBlock
from models.tt_dit.parallel.config import DiTParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils.check import assert_quality
from models.tt_dit.utils.mochi import get_rot_transformation_mat, stack_cos_sin
from models.tt_dit.utils.padding import pad_vision_seq_parallel
from models.tt_dit.utils.tensor import bf16_tensor, bf16_tensor_2dshard, float32_tensor, from_torch
from models.tt_dit.utils.test import line_params_req_exact_devices, ring_params_req_exact_devices, skip_if_unsupported_num_links
from models.tt_dit.experimental.wan_animate import reference_math as rm
from models.tt_dit.experimental.wan_animate.transformer_wan_animate import WanAnimateFaceAdapter
from models.tt_dit.experimental.wan_animate.transformer_wan_animate import WanAnimateFaceEncoder as TTFaceEncoder
from models.tt_dit.experimental.wan_animate.transformer_wan_animate import build_from_diffusers_config

CHECKPOINT = os.environ.get("WAN_ANIMATE_CHECKPOINT", "Wan-AI/Wan2.2-Animate-14B-Diffusers")
REAL = os.environ.get("WAN_ANIMATE_REAL_WEIGHTS") == "1"

TINY = dict(
    patch_size=(1, 2, 2),
    num_attention_heads=8,
    attention_head_dim=128,
    in_channels=36,
    latent_channels=16,
    out_channels=16,
    text_dim=256,
    freq_dim=256,
    ffn_dim=1024,
    num_layers=5,
    cross_attn_norm=True,
    qk_norm="rms_norm_across_heads",
    eps=1e-6,
    image_dim=128,
    added_kv_proj_dim=1024,
    rope_max_seq_len=1024,
    motion_encoder_size=64,
    motion_style_dim=128,
    motion_dim=20,
    motion_encoder_dim=128,
    face_encoder_hidden_dim=256,
    face_encoder_num_heads=4,
    inject_face_latents_blocks=5,
    motion_encoder_batch_size=8,
)
# Real Animate-14B transformer config (transformer/config.json of the Diffusers checkpoint).
FULL = dict(
    TINY,
    num_attention_heads=40,
    text_dim=4096,
    ffn_dim=13824,
    num_layers=40,
    image_dim=1280,
    added_kv_proj_dim=5120,
    motion_encoder_size=512,
    motion_style_dim=512,
    motion_encoder_dim=512,
    face_encoder_hidden_dim=1024,
)

MESH_PARAMS = [
    # Blackhole Galaxy (the POC target) and Wormhole Galaxy, same axes as the Wan 2.2 presets.
    pytest.param((4, 8), 1, 0, 2, line_params_req_exact_devices, ttnn.Topology.Linear, False, id="bh_4x8sp1tp0_linear"),
    pytest.param((4, 8), 1, 0, 2, ring_params_req_exact_devices, ttnn.Topology.Ring, False, id="bh_4x8sp1tp0_ring"),
    pytest.param((4, 8), 1, 0, 4, ring_params_req_exact_devices, ttnn.Topology.Ring, True, id="wh_4x8sp1tp0"),
    # Smaller boxes, for bring-up away from the Galaxy.
    pytest.param((2, 4), 1, 0, 2, line_params_req_exact_devices, ttnn.Topology.Linear, False, id="2x4sp1tp0"),
    pytest.param((2, 2), 0, 1, 2, line_params_req_exact_devices, ttnn.Topology.Linear, False, id="2x2sp0tp1"),
]
MESH_ARGS = ("mesh_device", "sp_axis", "tp_axis", "num_links", "device_params", "topology", "is_fsdp")


def _setup(mesh_device, sp_axis, tp_axis, num_links, topology):
    skip_if_unsupported_num_links(mesh_device, num_links)
    shape = tuple(mesh_device.shape)
    parallel_config = DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=shape[tp_axis]),
        sequence_parallel=ParallelFactor(mesh_axis=sp_axis, factor=shape[sp_axis]),
        cfg_parallel=None,
    )
    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
    return parallel_config, ccl_manager


def _gather_spatial(t, mesh_device, sp_axis, tp_axis, n):
    dims = [None, None]
    dims[sp_axis] = 2
    dims[tp_axis] = 3
    out = ttnn.to_torch(
        t, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=dims, mesh_shape=tuple(mesh_device.shape))
    )
    return out[:, :, :n, :]


def _config(name):
    return {"tiny": TINY, "full": FULL}[name]


def _seed_nontrivial(module):
    """Diffusers zero-inits some Animate parameters; make them non-zero so the test exercises them."""
    with torch.no_grad():
        for name, p in module.named_parameters():
            if "padding_tokens" in name or name.endswith("pose_patch_embedding.bias"):
                p.normal_()


def _real_state(prefix: str, keep=None) -> dict[str, torch.Tensor]:
    """Load only the tensors under ``prefix`` (and passing ``keep``) from the Diffusers checkpoint,
    shard by shard, so the 14B model is never materialized in fp32 on the host."""
    import json

    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    index_path = hf_hub_download(CHECKPOINT, "transformer/diffusion_pytorch_model.safetensors.index.json")
    weight_map = json.load(open(index_path))["weight_map"]
    state = {}
    for key, shard in weight_map.items():
        if key.startswith(prefix) and (keep is None or keep(key)):
            with safe_open(hf_hub_download(CHECKPOINT, f"transformer/{shard}"), "pt") as f:
                state[key.removeprefix(prefix)] = f.get_tensor(key).float()
    assert state, f"no keys under {prefix!r} in {CHECKPOINT}"
    return state


# ------------------------------------------------------------------------------------------------
# 1. Pose patch embedding fused into the patch embedding
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(MESH_ARGS, MESH_PARAMS, indirect=["mesh_device", "device_params"])
@pytest.mark.parametrize("cfg", ["tiny", "full"])
def test_animate_patch_embedding(mesh_device, sp_axis, tp_axis, num_links, device_params, topology, is_fsdp, cfg):
    MIN_PCC = 0.9999
    parallel_config, _ = _setup(mesh_device, sp_axis, tp_axis, num_links, topology)
    c = _config(cfg)
    dim = c["num_attention_heads"] * c["attention_head_dim"]
    torch.manual_seed(0)
    patch = torch.nn.Conv3d(36, dim, (1, 2, 2), (1, 2, 2))
    pose = torch.nn.Conv3d(16, dim, (1, 2, 2), (1, 2, 2))
    torch.nn.init.normal_(pose.bias)
    t, h, w = 5, 16, 24
    x = torch.randn(1, 36, t + 1, h, w)
    p = torch.randn(1, 16, t, h, w)
    with torch.no_grad():
        ref = patch(x)
        ref[:, :, 1:] += pose(p)
    ref = ref.flatten(2).transpose(1, 2)  # [1, N, dim]

    tt = WanPatchEmbed(
        patch_size=(1, 2, 2),
        in_channels=36 + rm.POSE_EXTRA_CHANNELS,
        embed_dim=dim,
        mesh_device=mesh_device,
        tp_mesh_axis=tp_axis,
    )
    w_f, b_f = rm.fuse_pose_patch_embedding(patch.weight, patch.bias, pose.weight, pose.bias)
    tt.load_torch_state_dict({"weight": w_f, "bias": b_f})

    x_ext = torch.cat([x, rm.pose_conditioning(p)], dim=1)
    b_, c_, f_, h_, w_ = x_ext.shape
    patches = (
        x_ext.reshape(1, c_, f_, 1, h_ // 2, 2, w_ // 2, 2).permute(0, 2, 4, 6, 3, 5, 7, 1).reshape(1, 1, -1, 4 * c_)
    )
    n = patches.shape[2]
    sp = tuple(mesh_device.shape)[sp_axis]
    patches = pad_vision_seq_parallel(patches, num_devices=sp)
    tt_in = bf16_tensor(patches, device=mesh_device, mesh_axis=sp_axis, shard_dim=-2)
    out = _gather_spatial(tt(tt_in), mesh_device, sp_axis, tp_axis, n)
    assert_quality(ref, out[0], pcc=MIN_PCC)


# ------------------------------------------------------------------------------------------------
# 2. Face encoder (temporal causal convs, once per segment)
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(MESH_ARGS, MESH_PARAMS, indirect=["mesh_device", "device_params"])
@pytest.mark.parametrize("cfg", ["tiny", "full"] + (["real"] if REAL else []))
@pytest.mark.parametrize("face_frames", [17, 77])
def test_animate_face_encoder(
    mesh_device, sp_axis, tp_axis, num_links, device_params, topology, is_fsdp, cfg, face_frames
):
    MIN_PCC = 0.999
    _setup(mesh_device, sp_axis, tp_axis, num_links, topology)
    c = FULL if cfg == "real" else _config(cfg)
    dim = c["num_attention_heads"] * c["attention_head_dim"]
    torch.manual_seed(0)
    ref_mod = WanAnimateFaceEncoder(
        in_dim=c["motion_encoder_dim"],
        out_dim=dim,
        hidden_dim=c["face_encoder_hidden_dim"],
        num_heads=c["face_encoder_num_heads"],
    ).eval()
    if cfg == "real":
        ref_mod.load_state_dict(_real_state("face_encoder."))
    else:
        _seed_nontrivial(ref_mod)
    motion = torch.randn(1, face_frames, c["motion_encoder_dim"])
    with torch.no_grad():
        ref = ref_mod(motion)

    tt = TTFaceEncoder(
        in_dim=c["motion_encoder_dim"],
        out_dim=dim,
        hidden_dim=c["face_encoder_hidden_dim"],
        num_heads=c["face_encoder_num_heads"],
        mesh_device=mesh_device,
    )
    tt.load_torch_state_dict(ref_mod.state_dict())
    out = tt(motion)
    assert out.shape == ref.shape, (out.shape, ref.shape)
    assert_quality(ref, out, pcc=MIN_PCC)


# ------------------------------------------------------------------------------------------------
# 3. Face adapter block (masked per-frame cross-attention to the motion tokens)
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(MESH_ARGS, MESH_PARAMS, indirect=["mesh_device", "device_params"])
@pytest.mark.parametrize("cfg", ["tiny", "full"] + (["real"] if REAL else []))
@pytest.mark.parametrize(
    ("latent_frames", "h", "w"),
    [pytest.param(6, 16, 24, id="small"), pytest.param(21, 60, 104, id="480p_segment")],
)
@pytest.mark.parametrize("impl", ["sdpa", "matmul"])
def test_animate_face_adapter(
    mesh_device, sp_axis, tp_axis, num_links, device_params, topology, is_fsdp, cfg, latent_frames, h, w, impl
):
    MIN_PCC = 0.999
    parallel_config, ccl_manager = _setup(mesh_device, sp_axis, tp_axis, num_links, topology)
    c = FULL if cfg == "real" else _config(cfg)
    heads = c["num_attention_heads"]
    dim = heads * c["attention_head_dim"]
    torch.manual_seed(0)
    ref_mod = WanAnimateFaceBlockCrossAttention(
        dim=dim, heads=heads, dim_head=dim // heads, eps=1e-6, cross_attention_dim_head=dim // heads
    ).eval()
    if cfg == "real":
        ref_mod.load_state_dict(_real_state("face_adapter.0."))
    tpf = (h // 2) * (w // 2)
    n = latent_frames * tpf
    hidden = torch.randn(1, n, dim)
    motion = torch.randn(1, latent_frames, rm.FACE_TOKENS_PER_FRAME, dim)
    motion[:, 0] = 0
    with torch.no_grad():
        ref = hidden + ref_mod(hidden, motion)

    tt = WanAnimateFaceAdapter(
        dim=dim,
        num_heads=heads,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_fsdp=is_fsdp,
    )
    tt.attn.cross_attn_impl = impl
    tt.load_torch_state_dict(ref_mod.state_dict())

    sp = tuple(mesh_device.shape)[sp_axis]
    hidden_pad = pad_vision_seq_parallel(hidden.unsqueeze(0), num_devices=sp)
    padded_n = hidden_pad.shape[2]
    tt_hidden = bf16_tensor_2dshard(hidden_pad, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 3})
    tt_motion = bf16_tensor(rm.face_motion_tokens(motion), device=mesh_device)
    mask = rm.face_attention_mask(num_frames=latent_frames, tokens_per_frame=tpf, padded_seq_len=padded_n)
    tt_mask = from_torch(mask, device=mesh_device, dtype=ttnn.bfloat16, mesh_axes=[None, None, sp_axis, None])

    out = tt(tt_hidden, tt_motion, N=n, face_mask=tt_mask)
    out = _gather_spatial(out, mesh_device, sp_axis, tp_axis, n)
    assert_quality(ref, out[0], pcc=MIN_PCC)
    # The adapter's own contribution must also match. Real weights make it ~1% of the residual, and rounding
    # the sum to bf16 alone caps this PCC at ~0.985 (measured on CPU), so the real-weights bar is lower.
    assert_quality(ref - hidden, out[0] - hidden, pcc=0.97 if cfg == "real" else 0.995)


# ------------------------------------------------------------------------------------------------
# 4. One transformer block with the CLIP image K/V branch (the block every Animate layer runs)
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(MESH_ARGS, MESH_PARAMS, indirect=["mesh_device", "device_params"])
@pytest.mark.parametrize("cfg", ["tiny", "full"] + (["real"] if REAL else []))
@pytest.mark.parametrize(("T", "H", "W"), [pytest.param(6, 16, 24, id="small"), pytest.param(21, 60, 104, id="480p")])
def test_animate_block_with_image_kv(
    mesh_device, sp_axis, tp_axis, num_links, device_params, topology, is_fsdp, cfg, T, H, W
):
    MIN_PCC = 0.9995
    MAX_RMSE = 0.04
    parallel_config, ccl_manager = _setup(mesh_device, sp_axis, tp_axis, num_links, topology)
    c = FULL if cfg == "real" else _config(cfg)
    heads = c["num_attention_heads"]
    dim = heads * c["attention_head_dim"]
    head_dim = dim // heads
    torch.manual_seed(0)
    ref_mod = TorchAnimateBlock(
        dim=dim, ffn_dim=c["ffn_dim"], num_heads=heads, cross_attn_norm=True, eps=1e-6, added_kv_proj_dim=dim
    ).eval()
    if cfg == "real":
        ref_mod.load_state_dict(_real_state("blocks.0."))
    n = T * (H // 2) * (W // 2)
    spatial = torch.randn(1, n, dim)
    text = torch.randn(1, 512, dim)
    image = torch.randn(1, rm.IMAGE_CONTEXT_TOKENS, dim)
    temb = torch.randn(1, 6, dim)
    rope_cos = torch.randn(1, n, 1, head_dim // 2)
    rope_sin = torch.randn(1, n, 1, head_dim // 2)
    torch_cos, torch_sin = stack_cos_sin(rope_cos, rope_sin)
    with torch.no_grad():
        ref = ref_mod(spatial, torch.cat([image, text], dim=1), temb, [torch_cos, torch_sin])

    tt = WanTransformerBlock(
        dim=dim,
        ffn_dim=c["ffn_dim"],
        num_heads=heads,
        cross_attention_norm=True,
        eps=1e-6,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_fsdp=is_fsdp,
        image_kv=True,
    )
    tt.load_torch_state_dict(ref_mod.state_dict())

    sp = tuple(mesh_device.shape)[sp_axis]
    spatial_pad = pad_vision_seq_parallel(spatial.unsqueeze(0), num_devices=sp)
    padded_n = spatial_pad.shape[2]
    cos_pad = pad_vision_seq_parallel(torch_cos.permute(0, 2, 1, 3), num_devices=sp)
    sin_pad = pad_vision_seq_parallel(torch_sin.permute(0, 2, 1, 3), num_devices=sp)
    tt_out = tt(
        spatial_1BND=bf16_tensor_2dshard(spatial_pad, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 3}),
        prompt_1BLP=bf16_tensor(text.unsqueeze(0), device=mesh_device),
        temb_1BTD=from_torch(temb.unsqueeze(0), device=mesh_device, dtype=ttnn.float32, mesh_axes=[..., tp_axis]),
        N=n,
        rope_cos=from_torch(cos_pad, device=mesh_device, dtype=ttnn.float32, mesh_axes=[..., sp_axis, None]),
        rope_sin=from_torch(sin_pad, device=mesh_device, dtype=ttnn.float32, mesh_axes=[..., sp_axis, None]),
        trans_mat=bf16_tensor(get_rot_transformation_mat(), device=mesh_device),
        image_prompt_1BLP=bf16_tensor(rm.pad_image_context(image), device=mesh_device),
        image_attn_mask=from_torch(
            rm.image_attention_mask(padded_seq_len=padded_n),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            mesh_axes=[None, None, sp_axis, None],
        ),
    )
    out = _gather_spatial(tt_out, mesh_device, sp_axis, tp_axis, n)
    assert_quality(ref, out[0], pcc=MIN_PCC, relative_rmse=MAX_RMSE)


# ------------------------------------------------------------------------------------------------
# 5. Whole Animate transformer (tiny, 5 layers = 5 blocks + 1 face adapter + all embeddings)
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(MESH_ARGS, MESH_PARAMS, indirect=["mesh_device", "device_params"])
@pytest.mark.parametrize("cfg", ["tiny"] + (["real"] if REAL else []))
def test_animate_transformer_model(mesh_device, sp_axis, tp_axis, num_links, device_params, topology, is_fsdp, cfg):
    MIN_PCC = 0.99
    MAX_RMSE = 0.15
    parallel_config, ccl_manager = _setup(mesh_device, sp_axis, tp_axis, num_links, topology)
    torch.manual_seed(0)
    if cfg == "real":
        # Real weights, truncated to block 0 + face adapter 0 (+ all embeddings and encoders).
        # 5 layers is the smallest config Diffusers builds with one face adapter; keep block 0 only.
        config = WanAnimateTransformer3DModel.load_config(CHECKPOINT, subfolder="transformer")
        ref_model = WanAnimateTransformer3DModel.from_config(config, num_layers=5)
        ref_model.blocks = torch.nn.ModuleList([ref_model.blocks[0]])
        keep = ("blocks.0.", "face_adapter.0.")
        state = _real_state("", keep=lambda k: not k.startswith(("blocks.", "face_adapter.")) or k.startswith(keep))
        ref_model.load_state_dict(state)
        num_layers = 1
    else:
        ref_model = WanAnimateTransformer3DModel(**TINY)
        _seed_nontrivial(ref_model)
        num_layers = TINY["num_layers"]
    ref_model.eval()
    cfgd = ref_model.config
    t, h, w = 5, 16, 24
    face_frames = (t - 1) * 4 + 1
    s = cfgd.motion_encoder_size
    hidden = torch.randn(1, 36, t + 1, h, w)
    timestep = torch.tensor([700.0])
    text = torch.randn(1, 512, cfgd.text_dim)
    clip = torch.randn(1, rm.IMAGE_CONTEXT_TOKENS, cfgd.image_dim)
    pose = torch.randn(1, 16, t, h, w)
    face = torch.rand(1, 3, face_frames, s, s) * 2 - 1
    with torch.no_grad():
        ref = ref_model(
            hidden_states=hidden,
            timestep=timestep,
            encoder_hidden_states=text,
            encoder_hidden_states_image=clip,
            pose_hidden_states=pose,
            face_pixel_values=face,
            return_dict=False,
        )[0]

    tt = build_from_diffusers_config(
        cfgd,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_fsdp=is_fsdp,
        num_layers=num_layers,
    )
    tt.load_torch_state_dict(ref_model.state_dict())
    out = tt(
        hidden[:, :16],
        bf16_tensor(text.unsqueeze(0), device=mesh_device),
        float32_tensor(timestep.view(1, 1, 1, 1), device=mesh_device),
        hidden[:, 16:],
        clip_hidden_states=clip,
        pose_latents=pose,
        face_pixel_values=face,
    )
    logger.info(f"output shape {tuple(out.shape)}")
    assert_quality(ref, out, pcc=MIN_PCC, relative_rmse=MAX_RMSE)
