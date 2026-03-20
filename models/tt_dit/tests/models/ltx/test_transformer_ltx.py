# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import sys

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.models.transformers.ltx.rope_ltx import precompute_freqs_cis
from models.tt_dit.models.transformers.ltx.transformer_ltx import LTXTransformerBlock
from models.tt_dit.parallel.config import DiTParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils.check import assert_quality
from models.tt_dit.utils.mochi import get_rot_transformation_mat
from models.tt_dit.utils.tensor import bf16_tensor, bf16_tensor_2dshard

sys.path.insert(0, "LTX-2/packages/ltx-core/src")

from models.tt_dit.models.transformers.ltx.transformer_ltx import LTXTransformerModel


def _make_parallel_config(mesh_device, sp_axis, tp_axis):
    return DiTParallelConfig(
        cfg_parallel=ParallelFactor(factor=1, mesh_axis=0),
        sequence_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[sp_axis], mesh_axis=sp_axis),
        tensor_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[tp_axis], mesh_axis=tp_axis),
    )


@pytest.mark.parametrize(
    "mesh_device, sp_axis, tp_axis",
    [
        [(1, 1), 0, 1],
        [(2, 4), 0, 1],
    ],
    ids=["1x1sp0tp1", "2x4sp0tp1"],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_ltx_transformer_block(mesh_device: ttnn.MeshDevice, sp_axis: int, tp_axis: int):
    """
    Test LTXTransformerBlock: compare TT vs LTX-2 PyTorch BasicAVTransformerBlock.
    """
    from ltx_core.model.transformer.transformer import BasicAVTransformerBlock, TransformerConfig

    dim = 4096
    num_heads = 32
    head_dim = dim // num_heads
    context_dim = 4096
    B = 1
    F, H, W = 4, 8, 8  # seq_len = 256
    seq_len = F * H * W
    prompt_len = 32

    # Create PyTorch reference
    video_cfg = TransformerConfig(dim=dim, heads=num_heads, d_head=head_dim, context_dim=context_dim)
    torch_block = BasicAVTransformerBlock(idx=0, video=video_cfg, audio=None)
    torch_block.eval()
    torch_state = torch_block.state_dict()

    # Create TT model
    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)
    parallel_config = _make_parallel_config(mesh_device, sp_axis, tp_axis)

    tt_block = LTXTransformerBlock(
        dim=dim,
        ffn_dim=dim * 4,
        num_heads=num_heads,
        cross_attention_dim=context_dim,
        eps=1e-6,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
    )
    tt_block.load_torch_state_dict(torch_state)

    # Create inputs
    torch.manual_seed(42)
    x = torch.randn(B, seq_len, dim, dtype=torch.float32)
    context = torch.randn(B, prompt_len, context_dim, dtype=torch.float32)

    # Timestep embedding: 6 modulation params (from AdaLayerNormSingle)
    # Shape (B, 1, 6*dim) — the middle dim is the "time" index
    temb = torch.randn(B, 1, 6 * dim, dtype=torch.float32)

    # RoPE
    t_ids = torch.arange(F)
    h_ids = torch.arange(H)
    w_ids = torch.arange(W)
    grid_t, grid_h, grid_w = torch.meshgrid(t_ids, h_ids, w_ids, indexing="ij")
    indices_grid = torch.stack([grid_t.flatten(), grid_h.flatten(), grid_w.flatten()], dim=-1).float().unsqueeze(0)
    cos_freq, sin_freq = precompute_freqs_cis(
        indices_grid, dim=dim, out_dtype=torch.float32, max_pos=[20, 2048, 2048], num_attention_heads=num_heads
    )

    # PyTorch forward
    from ltx_core.model.transformer.transformer import TransformerArgs

    # Build TransformerArgs for the PyTorch block
    cos_flat = cos_freq  # (B, seq_len, dim) — interleaved RoPE
    sin_flat = sin_freq
    # embedded_timestep: just the base timestep embedding (B, dim) — not used in basic block
    embedded_timestep = torch.randn(B, dim, dtype=torch.float32)
    with torch.no_grad():
        torch_args = TransformerArgs(
            x=x,
            context=context,
            context_mask=None,
            timesteps=temb,  # (B, 6*dim) modulation params
            embedded_timestep=embedded_timestep,
            positional_embeddings=(cos_flat, sin_flat),
            cross_positional_embeddings=None,
            cross_scale_shift_timestep=None,
            cross_gate_timestep=None,
            enabled=True,
        )
        torch_out_args, _ = torch_block(video=torch_args, audio=None)
        torch_out = torch_out_args.x

    logger.info(f"PyTorch block output shape: {torch_out.shape}")

    # Prepare TT tensors
    spatial = x.unsqueeze(0)  # (1, B, N, D)
    tt_spatial = bf16_tensor_2dshard(spatial, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 3})

    prompt = context.unsqueeze(0)  # (1, B, L, D)
    tt_prompt = bf16_tensor(prompt, device=mesh_device)

    # temb: reshape from (B, 1, 6*dim) to (1, B, 6, dim) for the TT block
    temb_reshaped = temb.reshape(B, 6, dim).unsqueeze(0)  # (1, B, 6, D)
    tt_temb = bf16_tensor(temb_reshaped, device=mesh_device, mesh_axis=tp_axis, shard_dim=3)

    # RoPE: (B, H, N, head_dim) for per-head application
    cos_heads = cos_freq.reshape(B, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
    sin_heads = sin_freq.reshape(B, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
    tt_cos = bf16_tensor_2dshard(cos_heads, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 1})
    tt_sin = bf16_tensor_2dshard(sin_heads, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 1})
    tt_trans_mat = bf16_tensor(get_rot_transformation_mat(), device=mesh_device)

    # TT forward
    tt_out = tt_block(
        spatial_1BND=tt_spatial,
        prompt_1BLP=tt_prompt,
        temb_1BTD=tt_temb,
        N=seq_len,
        rope_cos=tt_cos,
        rope_sin=tt_sin,
        trans_mat=tt_trans_mat,
    )

    # Gather and compare
    concat_dims = [None, None]
    concat_dims[sp_axis] = 2
    concat_dims[tp_axis] = 3
    tt_out_torch = ttnn.to_torch(
        tt_out,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=concat_dims, mesh_shape=tuple(mesh_device.shape)),
    ).squeeze(0)

    assert_quality(torch_out, tt_out_torch, pcc=0.999, relative_rmse=0.032)
    logger.info("PASSED: LTX transformer block matches PyTorch reference")


@pytest.mark.parametrize(
    "mesh_device, sp_axis, tp_axis",
    [
        [(1, 1), 0, 1],
        [(2, 4), 0, 1],
    ],
    ids=["1x1sp0tp1", "2x4sp0tp1"],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_ltx_transformer_model(mesh_device: ttnn.MeshDevice, sp_axis: int, tp_axis: int):
    """
    Test LTXTransformerModel: compare 1-layer TT model vs LTX-2 PyTorch LTXModel.
    """
    from ltx_core.guidance.perturbations import BatchedPerturbationConfig, PerturbationConfig
    from ltx_core.model.transformer.model import LTXModel, LTXModelType, Modality

    dim = 4096
    num_heads = 32
    head_dim = dim // num_heads
    in_channels = 128
    out_channels = 128
    cross_attention_dim = 4096
    num_layers = 1  # Single layer for fast test
    B = 1
    F, H, W = 4, 8, 8
    seq_len = F * H * W  # 256
    prompt_len = 32

    # Create PyTorch reference (1 layer, video only)
    torch_model = LTXModel(
        model_type=LTXModelType.VideoOnly,
        num_layers=num_layers,
        num_attention_heads=num_heads,
        attention_head_dim=head_dim,
        in_channels=in_channels,
        out_channels=out_channels,
        cross_attention_dim=cross_attention_dim,
        use_middle_indices_grid=True,
    )
    torch_model.eval()
    torch_state = torch_model.state_dict()
    logger.info(f"PyTorch model: {len(torch_state)} keys, {sum(p.numel() for p in torch_model.parameters()):,} params")

    # Create TT model
    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)
    parallel_config = _make_parallel_config(mesh_device, sp_axis, tp_axis)

    tt_model = LTXTransformerModel(
        num_attention_heads=num_heads,
        attention_head_dim=head_dim,
        in_channels=in_channels,
        out_channels=out_channels,
        num_layers=num_layers,
        cross_attention_dim=cross_attention_dim,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
    )
    tt_model.load_torch_state_dict(torch_state)

    # Create inputs
    torch.manual_seed(42)
    latent = torch.randn(B, seq_len, in_channels, dtype=torch.float32)
    context = torch.randn(B, prompt_len, cross_attention_dim, dtype=torch.float32)

    # Positions: (B, 3, T, 2) for use_middle_indices_grid=True
    t_ids = torch.arange(F)
    h_ids = torch.arange(H)
    w_ids = torch.arange(W)
    grid_t, grid_h, grid_w = torch.meshgrid(t_ids, h_ids, w_ids, indexing="ij")
    indices = torch.stack([grid_t.flatten(), grid_h.flatten(), grid_w.flatten()], dim=0).float()
    positions = torch.stack([indices, indices], dim=-1).unsqueeze(0)  # (1, 3, T, 2)

    # Timestep
    timestep_val = 500.0

    # === PyTorch reference forward ===
    video = Modality(
        latent=latent,
        sigma=torch.tensor([0.5]),
        timesteps=torch.ones(B, seq_len) * timestep_val,
        positions=positions,
        context=context,
        enabled=True,
        context_mask=None,
        attention_mask=None,
    )
    perturbations = BatchedPerturbationConfig(perturbations=[PerturbationConfig(perturbations=None) for _ in range(B)])
    with torch.no_grad():
        torch_out, _ = torch_model(video=video, audio=None, perturbations=perturbations)
    logger.info(f"PyTorch model output shape: {torch_out.shape}")

    # === TT forward ===
    # Prepare spatial: SP-sharded only (patchify_proj ColParallelLinear handles TP sharding)
    spatial = latent.unsqueeze(0)  # (1, B, N, in_channels)
    tt_spatial = bf16_tensor(spatial, device=mesh_device, mesh_axis=sp_axis, shard_dim=2)

    # Prompt
    prompt = context.unsqueeze(0)  # (1, B, L, D)
    tt_prompt = bf16_tensor(prompt, device=mesh_device)

    # Timestep: the model's adaln_single handles the embedding
    # Need to pass the scalar timestep * timestep_scale_multiplier
    # LTXModel internally does: timestep * timestep_scale_multiplier (default 1000)
    # Then passes to adaln_single
    tt_timestep = ttnn.from_torch(
        torch.tensor([[[[timestep_val * 1000.0]]]], dtype=torch.float32).expand(1, 1, B, 1),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.float32,
    )

    # RoPE: compute from positions, same as what the model does internally
    from models.tt_dit.models.transformers.ltx.rope_ltx import precompute_freqs_cis

    indices_grid_for_rope = torch.stack([grid_t.flatten(), grid_h.flatten(), grid_w.flatten()], dim=-1).float()
    # For use_middle_indices_grid=True, average start/end (same here)
    indices_grid_for_rope = indices_grid_for_rope.unsqueeze(0)  # (1, T, 3)

    cos_freq, sin_freq = precompute_freqs_cis(
        indices_grid_for_rope,
        dim=dim,
        out_dtype=torch.float32,
        theta=10000.0,
        max_pos=[20, 2048, 2048],
        num_attention_heads=num_heads,
    )
    # Shape: (1, T, dim) -> reshape to (B, H, N, head_dim) for per-head RoPE
    cos_heads = cos_freq.reshape(B, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
    sin_heads = sin_freq.reshape(B, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
    tt_cos = bf16_tensor_2dshard(cos_heads, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 1})
    tt_sin = bf16_tensor_2dshard(sin_heads, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 1})
    tt_trans_mat = bf16_tensor(get_rot_transformation_mat(), device=mesh_device)

    tt_out = tt_model(
        spatial_1BND=tt_spatial,
        temb=tt_timestep,
        prompt_1BLP=tt_prompt,
        N=seq_len,
        rope_cos=tt_cos,
        rope_sin=tt_sin,
        trans_mat=tt_trans_mat,
    )

    # Gather SP shards, then take device 0's copy (output is replicated on TP after all_gather + proj_out)
    if parallel_config.sequence_parallel.factor > 1:
        tt_out = ccl_manager.all_gather_persistent_buffer(
            tt_out, dim=2, mesh_axis=parallel_config.sequence_parallel.mesh_axis
        )
    tt_out_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0]).squeeze(0)

    logger.info(f"TT model output shape: {tt_out_torch.shape}")
    assert_quality(torch_out, tt_out_torch, pcc=0.992, relative_rmse=0.15)
    logger.info("PASSED: LTX transformer model matches PyTorch reference")


@pytest.mark.parametrize(
    "mesh_device, sp_axis, tp_axis",
    [
        [(1, 1), 0, 1],
        [(2, 4), 0, 1],
    ],
    ids=["1x1sp0tp1", "2x4sp0tp1"],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_ltx_transformer_inner_step(mesh_device: ttnn.MeshDevice, sp_axis: int, tp_axis: int):
    """
    Test LTXTransformerModel.inner_step: validates the pipeline denoising loop path.
    Caches prompt/RoPE on device, then calls inner_step with torch spatial input.
    """
    from ltx_core.guidance.perturbations import BatchedPerturbationConfig, PerturbationConfig
    from ltx_core.model.transformer.model import LTXModel, LTXModelType, Modality

    dim = 4096
    num_heads = 32
    head_dim = dim // num_heads
    in_channels = 128
    out_channels = 128
    cross_attention_dim = 4096
    num_layers = 1
    B = 1
    F, H, W = 4, 8, 8
    seq_len = F * H * W
    prompt_len = 32

    # PyTorch reference
    torch_model = LTXModel(
        model_type=LTXModelType.VideoOnly,
        num_layers=num_layers,
        num_attention_heads=num_heads,
        attention_head_dim=head_dim,
        in_channels=in_channels,
        out_channels=out_channels,
        cross_attention_dim=cross_attention_dim,
        use_middle_indices_grid=True,
    )
    torch_model.eval()
    torch_state = torch_model.state_dict()

    # TT model
    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)
    parallel_config = _make_parallel_config(mesh_device, sp_axis, tp_axis)

    tt_model = LTXTransformerModel(
        num_attention_heads=num_heads,
        attention_head_dim=head_dim,
        in_channels=in_channels,
        out_channels=out_channels,
        num_layers=num_layers,
        cross_attention_dim=cross_attention_dim,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
    )
    tt_model.load_torch_state_dict(torch_state)

    # Inputs
    torch.manual_seed(42)
    latent = torch.randn(B, seq_len, in_channels, dtype=torch.float32)
    context = torch.randn(B, prompt_len, cross_attention_dim, dtype=torch.float32)
    timestep_val = 500.0

    # Positions
    t_ids = torch.arange(F)
    h_ids = torch.arange(H)
    w_ids = torch.arange(W)
    grid_t, grid_h, grid_w = torch.meshgrid(t_ids, h_ids, w_ids, indexing="ij")
    indices = torch.stack([grid_t.flatten(), grid_h.flatten(), grid_w.flatten()], dim=0).float()
    positions = torch.stack([indices, indices], dim=-1).unsqueeze(0)

    # PyTorch reference forward
    video = Modality(
        latent=latent,
        sigma=torch.tensor([0.5]),
        timesteps=torch.ones(B, seq_len) * timestep_val,
        positions=positions,
        context=context,
        enabled=True,
        context_mask=None,
        attention_mask=None,
    )
    perturbations = BatchedPerturbationConfig(perturbations=[PerturbationConfig(perturbations=None) for _ in range(B)])
    with torch.no_grad():
        torch_out, _ = torch_model(video=video, audio=None, perturbations=perturbations)

    # Cache prompt and RoPE on device (step-independent)
    prompt = context.unsqueeze(0)
    tt_prompt = bf16_tensor(prompt, device=mesh_device)

    from models.tt_dit.models.transformers.ltx.rope_ltx import precompute_freqs_cis

    indices_grid = torch.stack([grid_t.flatten(), grid_h.flatten(), grid_w.flatten()], dim=-1).float().unsqueeze(0)
    cos_freq, sin_freq = precompute_freqs_cis(
        indices_grid,
        dim=dim,
        out_dtype=torch.float32,
        theta=10000.0,
        max_pos=[20, 2048, 2048],
        num_attention_heads=num_heads,
    )
    cos_heads = cos_freq.reshape(B, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
    sin_heads = sin_freq.reshape(B, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
    tt_cos = bf16_tensor_2dshard(cos_heads, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 1})
    tt_sin = bf16_tensor_2dshard(sin_heads, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 1})
    tt_trans_mat = bf16_tensor(get_rot_transformation_mat(), device=mesh_device)

    # Call inner_step (torch spatial input, cached device tensors)
    spatial_torch = latent.unsqueeze(0)  # (1, B, N, in_channels)
    timestep_torch = torch.tensor([timestep_val])

    tt_out = tt_model.inner_step(
        spatial_1BNI_torch=spatial_torch,
        prompt_1BLP=tt_prompt,
        rope_cos=tt_cos,
        rope_sin=tt_sin,
        trans_mat=tt_trans_mat,
        N=seq_len,
        timestep_torch=timestep_torch,
    )

    tt_out_torch = LTXTransformerModel.device_to_host(tt_out).squeeze(0)
    logger.info(f"inner_step output shape: {tt_out_torch.shape}")

    assert_quality(torch_out, tt_out_torch, pcc=0.992, relative_rmse=0.15)
    logger.info("PASSED: LTX inner_step matches PyTorch reference")
