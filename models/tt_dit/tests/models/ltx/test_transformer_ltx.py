# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
LTX transformer unit tests. Mesh / fabric / topology rows mirror
``test_transformer_wan.py`` (Wan2_2) block parametrization: ``line_params`` /
``ring_params``, ``num_links``, ``Topology.Linear`` vs ``Ring``, and ``is_fsdp``.

Omitted vs Wan: ``2x2`` and ``4x32`` submesh cases (LTX coverage uses ``1x1``
through ``4x8`` only).
"""

import sys

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.models.transformers.ltx.ltx_transformer import LTXTransformerBlock
from models.tt_dit.models.transformers.ltx.rope_ltx import LTXRopeType, precompute_freqs_cis
from models.tt_dit.parallel.config import DiTParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils.check import assert_quality
from models.tt_dit.utils.tensor import bf16_tensor, bf16_tensor_2dshard
from models.tt_dit.utils.test import line_params, ring_params

sys.path.insert(0, "LTX-2/packages/ltx-core/src")

from models.tt_dit.models.transformers.ltx.ltx_transformer import LTXTransformerModel

# Subset of Wan ``test_wan_transformer_block`` mesh rows + LTX ``1x1``.
_LTX_TRANSFORMER_MESH_PARAMS = [
    pytest.param((1, 1), 0, 1, 1, {}, ttnn.Topology.Linear, False, id="1x1sp0tp1"),
    pytest.param((2, 4), 0, 1, 1, line_params, ttnn.Topology.Linear, True, id="2x4sp0tp1"),
    pytest.param((2, 4), 1, 0, 1, line_params, ttnn.Topology.Linear, True, id="2x4sp1tp0"),
    pytest.param((4, 8), 1, 0, 4, ring_params, ttnn.Topology.Ring, True, id="wh_4x8sp1tp0"),
    pytest.param((4, 8), 1, 0, 2, ring_params, ttnn.Topology.Ring, False, id="ring_bh_4x8sp1tp0"),
    pytest.param((4, 8), 1, 0, 2, line_params, ttnn.Topology.Linear, False, id="line_bh_4x8sp1tp0"),
]


def _make_parallel_config(mesh_device, sp_axis, tp_axis):
    return DiTParallelConfig(
        cfg_parallel=ParallelFactor(factor=1, mesh_axis=0),
        sequence_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[sp_axis], mesh_axis=sp_axis),
        tensor_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[tp_axis], mesh_axis=tp_axis),
    )


@pytest.mark.parametrize(
    ("mesh_device", "sp_axis", "tp_axis", "num_links", "device_params", "topology", "is_fsdp"),
    _LTX_TRANSFORMER_MESH_PARAMS,
    indirect=["mesh_device", "device_params"],
)
def test_ltx_transformer_block(
    mesh_device: ttnn.MeshDevice,
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    topology: ttnn.Topology,
    is_fsdp: bool,
    reset_seeds,
) -> None:
    """
    Test LTXTransformerBlock: compare TT vs LTX-2 PyTorch BasicAVTransformerBlock.
    """
    from ltx_core.model.transformer.rope import LTXRopeType as RefRopeType
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
    video_cfg = TransformerConfig(
        dim=dim, heads=num_heads, d_head=head_dim, context_dim=context_dim, cross_attention_adaln=True
    )
    # SPLIT matches production pipeline (rope_type: split in checkpoint)
    torch_block = BasicAVTransformerBlock(idx=0, video=video_cfg, audio=None, rope_type=RefRopeType.SPLIT)
    torch_block.eval()
    torch_state = torch_block.state_dict()

    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
    parallel_config = _make_parallel_config(mesh_device, sp_axis, tp_axis)

    tt_block = LTXTransformerBlock(
        video_dim=dim,
        video_ffn_dim=dim * 4,
        video_num_heads=num_heads,
        video_cross_attention_dim=context_dim,
        eps=1e-6,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_fsdp=is_fsdp,
        has_audio=False,
    )
    tt_block.load_torch_state_dict(torch_state)

    # Create inputs
    torch.manual_seed(42)
    x = torch.randn(B, seq_len, dim, dtype=torch.float32)
    context = torch.randn(B, prompt_len, context_dim, dtype=torch.float32)

    # Timestep embedding: 9 modulation params (from AdaLayerNormSingle with cross_attention_adaln=True)
    # Shape (B, 1, 9*dim) — the middle dim is the "time" index
    temb = torch.randn(B, 1, 9 * dim, dtype=torch.float32)
    # Prompt modulation: 2 params (shift, scale for prompt cross-attention KV)
    prompt_temb = torch.randn(B, 1, 2 * dim, dtype=torch.float32)

    # RoPE — SPLIT format: (B, H, N, D_half) = (1, 32, 256, 64)
    t_ids = torch.arange(F)
    h_ids = torch.arange(H)
    w_ids = torch.arange(W)
    grid_t, grid_h, grid_w = torch.meshgrid(t_ids, h_ids, w_ids, indexing="ij")
    indices_grid = torch.stack([grid_t.flatten(), grid_h.flatten(), grid_w.flatten()], dim=-1).float().unsqueeze(0)
    cos_freq, sin_freq = precompute_freqs_cis(
        indices_grid,
        dim=dim,
        out_dtype=torch.float32,
        max_pos=[20, 2048, 2048],
        num_attention_heads=num_heads,
        rope_type=LTXRopeType.SPLIT,
    )
    # cos_freq shape: (1, 32, 256, 64) = (B, H, N, D_half)

    # PyTorch forward
    from ltx_core.model.transformer.transformer import TransformerArgs

    embedded_timestep = torch.randn(B, dim, dtype=torch.float32)
    with torch.no_grad():
        torch_args = TransformerArgs(
            x=x,
            context=context,
            context_mask=None,
            timesteps=temb,
            embedded_timestep=embedded_timestep,
            positional_embeddings=(cos_freq, sin_freq),  # SPLIT: (B, H, N, D_half)
            cross_positional_embeddings=None,
            cross_scale_shift_timestep=None,
            cross_gate_timestep=None,
            enabled=True,
            prompt_timestep=prompt_temb,
        )
        torch_out_args, _ = torch_block(video=torch_args, audio=None)
        torch_out = torch_out_args.x

    logger.info(f"PyTorch block output shape: {torch_out.shape}")

    # Prepare TT tensors
    spatial = x.unsqueeze(0)  # (1, B, N, D)
    tt_spatial = bf16_tensor_2dshard(spatial, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 3})

    prompt = context.unsqueeze(0)  # (1, B, L, D)
    tt_prompt = bf16_tensor(prompt, device=mesh_device)

    # temb: reshape from (B, 1, 9*dim) to (1, B, 9, dim) for the TT block
    temb_reshaped = temb.reshape(B, 9, dim).unsqueeze(0)  # (1, B, 9, D)
    tt_temb = bf16_tensor(temb_reshaped, device=mesh_device, mesh_axis=tp_axis, shard_dim=3)

    # prompt_temb: reshape from (B, 1, 2*dim) to (1, B, 2, dim)
    prompt_temb_reshaped = prompt_temb.reshape(B, 2, dim).unsqueeze(0)  # (1, B, 2, D)
    tt_prompt_temb = bf16_tensor(prompt_temb_reshaped, device=mesh_device)

    # RoPE for TT: SPLIT format (B, H, N, D_half) — no trans_mat needed
    tt_cos = bf16_tensor_2dshard(cos_freq, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 1})
    tt_sin = bf16_tensor_2dshard(sin_freq, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 1})

    # TT forward — trans_mat=None uses SPLIT rope path
    tt_out = tt_block(
        video_1BND=tt_spatial,
        video_prompt=tt_prompt,
        video_temb=tt_temb,
        video_N=seq_len,
        video_rope_cos=tt_cos,
        video_rope_sin=tt_sin,
        trans_mat=None,
        video_prompt_temb=tt_prompt_temb,
    )

    # Gather and compare
    concat_dims = [None, None]
    concat_dims[sp_axis] = 2
    concat_dims[tp_axis] = 3
    tt_out_torch = ttnn.to_torch(
        tt_out,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=concat_dims, mesh_shape=tuple(mesh_device.shape)),
    ).squeeze(0)

    # 4x8 mesh has 8-way SP ring all-gathers that accumulate more BF16 rounding
    pcc_threshold = 0.988 if mesh_device.get_num_devices() > 8 else 0.999
    rmse_threshold = 0.10 if mesh_device.get_num_devices() > 8 else 0.032
    assert_quality(torch_out, tt_out_torch, pcc=pcc_threshold, relative_rmse=rmse_threshold)
    logger.info("PASSED: LTX transformer block matches PyTorch reference")


@pytest.mark.parametrize(
    ("mesh_device", "sp_axis", "tp_axis", "num_links", "device_params", "topology", "is_fsdp"),
    _LTX_TRANSFORMER_MESH_PARAMS,
    indirect=["mesh_device", "device_params"],
)
def test_ltx_transformer_model(
    mesh_device: ttnn.MeshDevice,
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    topology: ttnn.Topology,
    is_fsdp: bool,
    reset_seeds,
) -> None:
    """
    Test LTXTransformerModel: compare 1-layer TT model vs LTX-2 PyTorch LTXModel.
    """
    from ltx_core.guidance.perturbations import BatchedPerturbationConfig, PerturbationConfig
    from ltx_core.model.transformer.model import LTXModel, LTXModelType, Modality
    from ltx_core.model.transformer.rope import LTXRopeType as RefRopeType

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

    # Create PyTorch reference (1 layer, video only).
    # SPLIT rope_type matches the production pipeline and the block test.
    # LTX-2's default rope_type changed from INTERLEAVED to SPLIT in a vendor
    # update; pinning explicitly here prevents the same drift from biting again.
    # Deterministic, scaled-down random weights keep both reference and TT
    # forwards numerically well-behaved (full default init can explode through
    # adaln_single + (1+scale)*x chains in fp32 with no real training signal).
    torch_model = LTXModel(
        model_type=LTXModelType.VideoOnly,
        num_layers=num_layers,
        num_attention_heads=num_heads,
        attention_head_dim=head_dim,
        in_channels=in_channels,
        out_channels=out_channels,
        cross_attention_dim=cross_attention_dim,
        use_middle_indices_grid=True,
        cross_attention_adaln=True,
        rope_type=RefRopeType.SPLIT,
    )
    torch_model.eval()
    WEIGHT_SEED = 1234
    torch.manual_seed(WEIGHT_SEED)
    with torch.no_grad():
        for p in torch_model.parameters():
            p.copy_(torch.randn(p.shape, dtype=p.dtype) * 0.1)
    logger.info(f"PyTorch model: {sum(p.numel() for p in torch_model.parameters()):,} params (overwritten N(0, 0.1²))")

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

    # Timestep — small enough that adaln_single (input timestep * 1000) stays
    # comfortably in float32 range even with the scaled-down random init.
    timestep_val = 0.01

    # === PyTorch reference forward (compute BEFORE any TT operations so that
    # ttnn weight conversions cannot perturb torch_model parameters and push
    # the random-init model over its narrow stability cliff). ===
    # Match sigma to timestep_val: TT's inner_step uses a single `timestep_torch`
    # for BOTH adaln_single (driven by timesteps in torch) AND prompt_adaln_single
    # (driven by sigma in torch). Production calls inner_step with timestep=sigma,
    # so we mirror that here. Without this, torch's prompt_adaln sees a different
    # input than TT's, yielding a systematically diverged cross-attention path
    # and ~10% PCC loss masquerading as a precision issue.
    video = Modality(
        latent=latent,
        sigma=torch.tensor([timestep_val]),
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
    logger.info(f"PyTorch model output shape: {torch_out.shape}, range=[{torch_out.min():.4f}, {torch_out.max():.4f}]")

    # Snapshot state dict with cloned tensors so TT-side _prepare_torch_state
    # views/permutations cannot share storage with torch_model parameters.
    torch_state = {k: v.detach().clone() for k, v in torch_model.state_dict().items()}
    logger.info(f"State dict: {len(torch_state)} keys")

    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
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
        is_fsdp=is_fsdp,
    )
    tt_model.load_torch_state_dict(torch_state)

    # === TT forward ===
    # Prompt
    prompt = context.unsqueeze(0)  # (1, B, L, D)
    tt_prompt = bf16_tensor(prompt, device=mesh_device)

    # RoPE: compute from positions, same as what the model does internally
    from models.tt_dit.models.transformers.ltx.rope_ltx import precompute_freqs_cis

    indices_grid_for_rope = torch.stack([grid_t.flatten(), grid_h.flatten(), grid_w.flatten()], dim=-1).float()
    # For use_middle_indices_grid=True, average start/end (same here)
    indices_grid_for_rope = indices_grid_for_rope.unsqueeze(0)  # (1, T, 3)

    # SPLIT-format RoPE returns (B, H, N, head_dim/2) directly — no reshape needed.
    # Must match the rope_type used by torch_model (now SPLIT) and TT attention
    # (selected by passing trans_mat=None below).
    cos_freq, sin_freq = precompute_freqs_cis(
        indices_grid_for_rope,
        dim=dim,
        out_dtype=torch.float32,
        theta=10000.0,
        max_pos=[20, 2048, 2048],
        num_attention_heads=num_heads,
        rope_type=LTXRopeType.SPLIT,
    )
    tt_cos = bf16_tensor_2dshard(cos_freq, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 1})
    tt_sin = bf16_tensor_2dshard(sin_freq, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 1})

    # Use inner_step: takes timestep value, multiplies by 1000 internally for adaln.
    # trans_mat=None selects the SPLIT RoPE path in TT attention to match torch.
    spatial_torch = latent.unsqueeze(0)  # (1, B, N, in_channels)
    timestep_torch = torch.tensor([timestep_val])

    tt_out = tt_model.inner_step(
        video_1BNI_torch=spatial_torch,
        video_prompt_1BLP=tt_prompt,
        video_rope_cos=tt_cos,
        video_rope_sin=tt_sin,
        trans_mat=None,
        video_N=seq_len,
        timestep_torch=timestep_torch,
    )

    tt_out_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0]).squeeze(0)

    logger.info(
        f"TT model output shape: {tt_out_torch.shape}, range=[{tt_out_torch.min():.4f}, {tt_out_torch.max():.4f}]"
    )
    logger.info(f"Ref output shape: {torch_out.shape}, range=[{torch_out.min():.4f}, {torch_out.max():.4f}]")
    assert_quality(torch_out, tt_out_torch, pcc=0.992, relative_rmse=0.15)
    logger.info("PASSED: LTX transformer model matches PyTorch reference")


@pytest.mark.parametrize(
    ("mesh_device", "sp_axis", "tp_axis", "num_links", "device_params", "topology", "is_fsdp"),
    _LTX_TRANSFORMER_MESH_PARAMS,
    indirect=["mesh_device", "device_params"],
)
def test_ltx_transformer_inner_step(
    mesh_device: ttnn.MeshDevice,
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    topology: ttnn.Topology,
    is_fsdp: bool,
    reset_seeds,
) -> None:
    """
    Test LTXTransformerModel.inner_step: validates the pipeline denoising loop path.
    Caches prompt/RoPE on device, then calls inner_step with torch spatial input.
    """
    from ltx_core.guidance.perturbations import BatchedPerturbationConfig, PerturbationConfig
    from ltx_core.model.transformer.model import LTXModel, LTXModelType, Modality
    from ltx_core.model.transformer.rope import LTXRopeType as RefRopeType

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

    # PyTorch reference. SPLIT rope_type matches production / block test
    # (see comment in test_ltx_transformer_model for rationale).
    torch_model = LTXModel(
        model_type=LTXModelType.VideoOnly,
        num_layers=num_layers,
        num_attention_heads=num_heads,
        attention_head_dim=head_dim,
        in_channels=in_channels,
        out_channels=out_channels,
        cross_attention_dim=cross_attention_dim,
        use_middle_indices_grid=True,
        cross_attention_adaln=True,
        rope_type=RefRopeType.SPLIT,
    )
    torch_model.eval()
    WEIGHT_SEED = 1234
    torch.manual_seed(WEIGHT_SEED)
    with torch.no_grad():
        for p in torch_model.parameters():
            p.copy_(torch.randn(p.shape, dtype=p.dtype) * 0.1)

    # Inputs
    torch.manual_seed(42)
    latent = torch.randn(B, seq_len, in_channels, dtype=torch.float32)
    context = torch.randn(B, prompt_len, cross_attention_dim, dtype=torch.float32)
    timestep_val = 0.01

    # Positions
    t_ids = torch.arange(F)
    h_ids = torch.arange(H)
    w_ids = torch.arange(W)
    grid_t, grid_h, grid_w = torch.meshgrid(t_ids, h_ids, w_ids, indexing="ij")
    indices = torch.stack([grid_t.flatten(), grid_h.flatten(), grid_w.flatten()], dim=0).float()
    positions = torch.stack([indices, indices], dim=-1).unsqueeze(0)

    # === PyTorch reference forward (compute BEFORE any TT operations) ===
    # sigma=timestep_val: TT's inner_step uses a single value for both
    # adaln_single and prompt_adaln_single (matches production where timestep=sigma).
    video = Modality(
        latent=latent,
        sigma=torch.tensor([timestep_val]),
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
    logger.info(f"PyTorch reference output range=[{torch_out.min():.4f}, {torch_out.max():.4f}]")

    # Snapshot state dict with cloned tensors so TT-side _prepare_torch_state
    # views/permutations cannot share storage with torch_model parameters.
    torch_state = {k: v.detach().clone() for k, v in torch_model.state_dict().items()}

    # TT model
    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
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
        is_fsdp=is_fsdp,
    )
    tt_model.load_torch_state_dict(torch_state)

    # Cache prompt and RoPE on device (step-independent)
    prompt = context.unsqueeze(0)
    tt_prompt = bf16_tensor(prompt, device=mesh_device)

    from models.tt_dit.models.transformers.ltx.rope_ltx import precompute_freqs_cis

    # SPLIT-format RoPE — matches torch_model's rope_type and TT's SPLIT path
    # (selected by trans_mat=None below).
    indices_grid = torch.stack([grid_t.flatten(), grid_h.flatten(), grid_w.flatten()], dim=-1).float().unsqueeze(0)
    cos_freq, sin_freq = precompute_freqs_cis(
        indices_grid,
        dim=dim,
        out_dtype=torch.float32,
        theta=10000.0,
        max_pos=[20, 2048, 2048],
        num_attention_heads=num_heads,
        rope_type=LTXRopeType.SPLIT,
    )
    tt_cos = bf16_tensor_2dshard(cos_freq, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 1})
    tt_sin = bf16_tensor_2dshard(sin_freq, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 1})

    # Call inner_step (torch spatial input, cached device tensors)
    spatial_torch = latent.unsqueeze(0)  # (1, B, N, in_channels)
    timestep_torch = torch.tensor([timestep_val])

    tt_out = tt_model.inner_step(
        video_1BNI_torch=spatial_torch,
        video_prompt_1BLP=tt_prompt,
        video_rope_cos=tt_cos,
        video_rope_sin=tt_sin,
        trans_mat=None,
        video_N=seq_len,
        timestep_torch=timestep_torch,
    )

    tt_out_torch = LTXTransformerModel.device_to_host(tt_out).squeeze(0)
    logger.info(f"inner_step output shape: {tt_out_torch.shape}")

    assert_quality(torch_out, tt_out_torch, pcc=0.992, relative_rmse=0.15)
    logger.info("PASSED: LTX inner_step matches PyTorch reference")
