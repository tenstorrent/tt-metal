# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch
import transformers
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLMLP,
    Qwen2_5_VLPatchMerger,
    Qwen2_5_VLVisionBlock,
)

import ttnn

from ....encoders.qwen25vl.model_qwen25vl_vision import (
    Qwen25VlPatchMerger,
    Qwen25VlVisionBlock,
    Qwen25VlVisionContext,
    Qwen25VlVisionEncoder,
    Qwen25VlVisionMLP,
    Qwen25VlVisionPatchEmbed,
    build_vision_rope_tensors,
)
from ....parallel.manager import CCLManager
from ....utils import tensor
from ....utils.check import assert_quality


def _load_vision_tower():
    model = transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen-Image-Edit-2511", subfolder="text_encoder"
    )
    return model.model.visual


def _vision_ctx(mesh_device: ttnn.MeshDevice, tp_axis: int | None) -> Qwen25VlVisionContext:
    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear) if tp_axis is not None else None
    return Qwen25VlVisionContext(device=mesh_device, tp_axis=tp_axis, ccl_manager=ccl_manager)


def test_vision_rope_tables() -> None:
    torch.manual_seed(0)

    vision = _load_vision_tower()
    grid_thw = [(1, 28, 28)]
    head_dim = vision.config.hidden_size // vision.config.num_heads
    spatial_merge_size = vision.config.spatial_merge_size

    cos_ours, sin_ours = build_vision_rope_tensors(
        grid_thw,
        head_dim=head_dim,
        spatial_merge_size=spatial_merge_size,
        theta=10000.0,
    )

    grid_tensor = torch.tensor([list(g) for g in grid_thw], dtype=torch.int64)
    rotary_pos_emb = vision.rot_pos_emb(grid_tensor)
    emb_ref = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    cos_ref = emb_ref.cos().to(torch.float32)
    sin_ref = emb_ref.sin().to(torch.float32)

    assert cos_ours.shape == cos_ref.shape, f"cos shape mismatch: {cos_ours.shape} vs {cos_ref.shape}"
    assert sin_ours.shape == sin_ref.shape, f"sin shape mismatch: {sin_ours.shape} vs {sin_ref.shape}"
    torch.testing.assert_close(cos_ours, cos_ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(sin_ours, sin_ref, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "mesh_device",
    [pytest.param((4, 8), id="4x8")],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
def test_patch_embed_parity(*, mesh_device: ttnn.MeshDevice) -> None:
    torch.manual_seed(0)
    vision = _load_vision_tower()
    cfg = vision.config
    tp_axis = None
    ctx = _vision_ctx(mesh_device, tp_axis)

    torch_pe = vision.patch_embed.eval()

    model = Qwen25VlVisionPatchEmbed(
        patch_size=cfg.patch_size,
        temporal_patch_size=cfg.temporal_patch_size,
        in_channels=cfg.in_channels,
        embed_dim=cfg.hidden_size,
        ctx=ctx,
    )
    model.load_torch_state_dict(torch_pe.state_dict())

    num_patches = 784
    in_features = cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size
    x = torch.randn(num_patches, in_features)

    with torch.no_grad():
        ref = torch_pe(x)

    tt_x = tensor.from_torch(x.unsqueeze(0), device=mesh_device)
    tt_y = model.forward(tt_x)
    got = ttnn.to_torch(ttnn.get_device_tensors(tt_y)[0]).squeeze(0)

    assert_quality(ref, got, pcc=0.99)


@pytest.mark.parametrize(
    "mesh_device",
    [pytest.param((4, 8), id="4x8")],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
def test_vision_mlp_parity(*, mesh_device: ttnn.MeshDevice) -> None:
    torch.manual_seed(0)
    vision = _load_vision_tower()

    cfg = vision.config
    tp_axis = None
    ctx = _vision_ctx(mesh_device, tp_axis)

    torch_mlp = Qwen2_5_VLMLP(cfg, bias=True).eval()

    model = Qwen25VlVisionMLP(
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        ctx=ctx,
    )
    model.load_torch_state_dict(torch_mlp.state_dict())

    seq_len = 784
    x = torch.randn(1, seq_len, cfg.hidden_size)
    with torch.no_grad():
        ref = torch_mlp(x)

    tt_x = tensor.from_torch(x, device=mesh_device)
    tt_y = model.forward(tt_x)
    got = ttnn.to_torch(ttnn.get_device_tensors(tt_y)[0])

    assert_quality(ref, got, pcc=0.98)


@pytest.mark.parametrize(
    "mesh_device",
    [pytest.param((4, 8), id="4x8")],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
def test_vision_block_parity(*, mesh_device: ttnn.MeshDevice) -> None:
    torch.manual_seed(0)
    vision = _load_vision_tower()
    cfg = vision.config
    tp_axis = None
    ctx = _vision_ctx(mesh_device, tp_axis)

    torch_block: Qwen2_5_VLVisionBlock = vision.blocks[7].eval()

    model = Qwen25VlVisionBlock(
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        num_heads=cfg.num_heads,
        rms_norm_eps=1e-6,
        ctx=ctx,
    )
    model.load_torch_state_dict(torch_block.state_dict())

    grid_thw = [(1, 28, 28)]
    seq_len = 28 * 28
    head_dim = cfg.hidden_size // cfg.num_heads

    x = torch.randn(seq_len, cfg.hidden_size)
    cos_t_ref, sin_t_ref = build_vision_rope_tensors(
        grid_thw, head_dim=head_dim, spatial_merge_size=cfg.spatial_merge_size
    )
    padded_head_dim = ((head_dim + 31) // 32) * 32
    cos_t, sin_t = build_vision_rope_tensors(
        grid_thw,
        head_dim=head_dim,
        spatial_merge_size=cfg.spatial_merge_size,
        pad_to=padded_head_dim,
    )

    with torch.no_grad():
        cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32)
        ref = torch_block(
            x,
            cu_seqlens=cu_seqlens,
            position_embeddings=(cos_t_ref.to(x.dtype), sin_t_ref.to(x.dtype)),
        )

    x_in = x.unsqueeze(0)
    tt_x = tensor.from_torch(x_in, device=mesh_device)
    tt_cos = tensor.from_torch(cos_t.unsqueeze(0), device=mesh_device)
    tt_sin = tensor.from_torch(sin_t.unsqueeze(0), device=mesh_device)

    tt_y = model.forward(tt_x, pos_embeds=(tt_cos, tt_sin))
    got = ttnn.to_torch(ttnn.get_device_tensors(tt_y)[0]).squeeze(0)

    pytest.xfail()
    assert_quality(ref, got, pcc=0.97)


@pytest.mark.parametrize(
    "mesh_device",
    [pytest.param((4, 8), id="4x8")],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
def test_patch_merger_parity(*, mesh_device: ttnn.MeshDevice) -> None:
    torch.manual_seed(0)
    vision = _load_vision_tower()
    cfg = vision.config
    tp_axis = None
    ctx = _vision_ctx(mesh_device, tp_axis)

    torch_merger: Qwen2_5_VLPatchMerger = vision.merger.eval()

    model = Qwen25VlPatchMerger(
        context_dim=cfg.hidden_size,
        out_dim=cfg.out_hidden_size,
        spatial_merge_size=cfg.spatial_merge_size,
        rms_norm_eps=1e-6,
        ctx=ctx,
    )
    model.load_torch_state_dict(torch_merger.state_dict())

    seq_len = 784
    x = torch.randn(seq_len, cfg.hidden_size)

    with torch.no_grad():
        ref = torch_merger(x)

    tt_x = tensor.from_torch(x.unsqueeze(0), device=mesh_device)
    tt_y = model.forward(tt_x)
    got = ttnn.to_torch(ttnn.get_device_tensors(tt_y)[0]).reshape(-1, cfg.out_hidden_size)

    assert_quality(ref, got, pcc=0.97)


@pytest.mark.parametrize(
    "mesh_device",
    [pytest.param((4, 8), id="4x8")],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
def test_full_encoder_smoke(*, mesh_device: ttnn.MeshDevice) -> None:
    torch.manual_seed(0)
    vision = _load_vision_tower()
    cfg = vision.config

    model = Qwen25VlVisionEncoder(
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        num_heads=cfg.num_heads,
        depth=cfg.depth,
        patch_size=cfg.patch_size,
        temporal_patch_size=cfg.temporal_patch_size,
        in_channels=cfg.in_channels,
        out_hidden_size=cfg.out_hidden_size,
        spatial_merge_size=cfg.spatial_merge_size,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        device=mesh_device,
        parallel_config=None,
        ccl_manager=None,
    )
    model.load_torch_state_dict(vision.state_dict())

    num_patches = 784
    in_features = cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size
    pixel_values = torch.randn(num_patches, in_features)

    grid_thw = [(1, 28, 28)]
    cos, sin = model.build_pos_embeds(grid_thw)

    tt_pixel = tensor.from_torch(pixel_values.unsqueeze(0), device=mesh_device)
    tt_cos = tensor.from_torch(cos.unsqueeze(0), device=mesh_device)
    tt_sin = tensor.from_torch(sin.unsqueeze(0), device=mesh_device)

    tt_out = model.forward(tt_pixel, pos_embeds=(tt_cos, tt_sin))
    out = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0]).squeeze(0)

    expected_merged_seq = num_patches // (cfg.spatial_merge_size * cfg.spatial_merge_size)
    assert out.shape == (
        expected_merged_seq,
        cfg.out_hidden_size,
    ), f"unexpected output shape {out.shape}, expected {(expected_merged_seq, cfg.out_hidden_size)}"
    assert not torch.isnan(out).any(), "output contains NaNs"
