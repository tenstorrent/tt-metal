# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Smoke tests for LTXTransformerBlock and LTXTransformerModel.

Mesh / fabric / CCL setup follows Wan2_2-style tests: ``line_params`` /
``ring_params`` from ``utils.test``, explicit ``num_links`` and ``topology``,
and ``is_fsdp``. Single-device uses empty ``device_params`` (no fabric).
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.models.transformers.ltx.rope_ltx import precompute_freqs_cis
from models.tt_dit.models.transformers.ltx.transformer_ltx import LTXTransformerBlock, LTXTransformerModel
from models.tt_dit.parallel.config import DiTParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils.mochi import get_rot_transformation_mat
from models.tt_dit.utils.tensor import bf16_tensor, bf16_tensor_2dshard
from models.tt_dit.utils.test import line_params, ring_params


def _make_parallel_config(mesh_device, sp_axis, tp_axis):
    return DiTParallelConfig(
        cfg_parallel=ParallelFactor(factor=1, mesh_axis=0),
        sequence_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[sp_axis], mesh_axis=sp_axis),
        tensor_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[tp_axis], mesh_axis=tp_axis),
    )


def _make_ltx_block_state(dim, num_heads, context_dim, ffn_dim, *, cross_attention_adaln=True):
    """Build a random torch state_dict in the format expected by LTXTransformerBlock.load_torch_state_dict.

    Key names match BasicAVTransformerBlock.state_dict() as expected before _prepare_torch_state renaming.
    """
    torch.manual_seed(42)
    head_dim = dim // num_heads
    scale = 0.02

    state = {}

    # Self-attention (attn1): separate to_q, to_k, to_v + to_out.0 + q_norm + k_norm
    for proj, in_dim, out_dim in [("to_q", dim, dim), ("to_k", dim, dim), ("to_v", dim, dim)]:
        state[f"attn1.{proj}.weight"] = torch.randn(out_dim, in_dim) * scale
        state[f"attn1.{proj}.bias"] = torch.zeros(out_dim)
    state["attn1.to_out.0.weight"] = torch.randn(dim, dim) * scale
    state["attn1.to_out.0.bias"] = torch.zeros(dim)
    state["attn1.q_norm.weight"] = torch.ones(dim)
    state["attn1.k_norm.weight"] = torch.ones(dim)

    # Cross-attention (attn2): to_q + to_k/to_v (KV from context) + to_out.0 + norms
    state["attn2.to_q.weight"] = torch.randn(dim, dim) * scale
    state["attn2.to_q.bias"] = torch.zeros(dim)
    state["attn2.to_k.weight"] = torch.randn(dim, context_dim) * scale
    state["attn2.to_k.bias"] = torch.zeros(dim)
    state["attn2.to_v.weight"] = torch.randn(dim, context_dim) * scale
    state["attn2.to_v.bias"] = torch.zeros(dim)
    state["attn2.to_out.0.weight"] = torch.randn(dim, dim) * scale
    state["attn2.to_out.0.bias"] = torch.zeros(dim)
    state["attn2.q_norm.weight"] = torch.ones(dim)
    state["attn2.k_norm.weight"] = torch.ones(dim)

    # FFN: ff.net.0.proj is (ffn_dim, dim) PyTorch-convention (transposed on load), ff.net.2 is (dim, ffn_dim)
    state["ff.net.0.proj.weight"] = torch.randn(ffn_dim, dim) * scale
    state["ff.net.0.proj.bias"] = torch.zeros(ffn_dim)
    state["ff.net.2.weight"] = torch.randn(dim, ffn_dim) * scale
    state["ff.net.2.bias"] = torch.zeros(dim)

    # AdaLN tables
    adaln_coeff = 9 if cross_attention_adaln else 6
    state["scale_shift_table"] = torch.zeros(adaln_coeff, dim)
    if cross_attention_adaln:
        state["prompt_scale_shift_table"] = torch.zeros(2, dim)

    return state


# Primary smoke shapes (Wan-style: line on 8-chip, ring on 32-chip).
_LTX_SMOKE_MESH_PARAMS = [
    pytest.param((1, 1), 0, 1, 1, {}, ttnn.Topology.Linear, False, id="1x1sp0tp1"),
    pytest.param((2, 4), 0, 1, 1, line_params, ttnn.Topology.Linear, True, id="2x4sp0tp1"),
    pytest.param((4, 8), 1, 0, 4, ring_params, ttnn.Topology.Ring, True, id="wh_4x8sp1tp0"),
]


@pytest.mark.parametrize(
    ("mesh_device", "sp_axis", "tp_axis", "num_links", "device_params", "topology", "is_fsdp"),
    _LTX_SMOKE_MESH_PARAMS,
    indirect=["mesh_device", "device_params"],
)
def test_ltx_transformer_block_smoke(
    mesh_device: ttnn.MeshDevice,
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    topology: ttnn.Topology,
    is_fsdp: bool,
) -> None:
    """Smoke test: run LTXTransformerBlock forward pass on device without PCC comparison.

    Validates:
    - Block loads onto mesh without errors
    - Forward pass runs to completion
    - Output shape matches input shape
    - Output values are finite (no NaN/Inf)
    """
    dim = 4096
    num_heads = 32
    head_dim = dim // num_heads
    context_dim = 4096
    ffn_dim = dim * 4
    B = 1
    F, H, W = 4, 8, 8
    seq_len = F * H * W
    prompt_len = 32

    logger.info(f"mesh shape: {tuple(mesh_device.shape)}, sp_axis={sp_axis}, tp_axis={tp_axis}")

    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
    parallel_config = _make_parallel_config(mesh_device, sp_axis, tp_axis)

    tt_block = LTXTransformerBlock(
        video_dim=dim,
        video_ffn_dim=ffn_dim,
        video_num_heads=num_heads,
        video_cross_attention_dim=context_dim,
        eps=1e-6,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_fsdp=is_fsdp,
        has_audio=False,
    )

    torch_state = _make_ltx_block_state(dim, num_heads, context_dim, ffn_dim)
    tt_block.load_torch_state_dict(torch_state)
    logger.info("Block loaded onto device")

    torch.manual_seed(123)
    x = torch.randn(B, seq_len, dim, dtype=torch.float32)
    context = torch.randn(B, prompt_len, context_dim, dtype=torch.float32)
    temb = torch.zeros(B, 9, dim, dtype=torch.float32)
    prompt_temb = torch.zeros(B, 2, dim, dtype=torch.float32)

    t_ids = torch.arange(F)
    h_ids = torch.arange(H)
    w_ids = torch.arange(W)
    grid_t, grid_h, grid_w = torch.meshgrid(t_ids, h_ids, w_ids, indexing="ij")
    indices_grid = torch.stack([grid_t.flatten(), grid_h.flatten(), grid_w.flatten()], dim=-1).float().unsqueeze(0)
    cos_freq, sin_freq = precompute_freqs_cis(
        indices_grid, dim=dim, out_dtype=torch.float32, max_pos=[20, 2048, 2048], num_attention_heads=num_heads
    )

    spatial = x.unsqueeze(0)
    tt_spatial = bf16_tensor_2dshard(spatial, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 3})

    prompt = context.unsqueeze(0)
    tt_prompt = bf16_tensor(prompt, device=mesh_device)

    temb_4d = temb.unsqueeze(0)
    tt_temb = bf16_tensor(temb_4d, device=mesh_device, mesh_axis=tp_axis, shard_dim=3)

    prompt_temb_4d = prompt_temb.unsqueeze(0)
    tt_prompt_temb = bf16_tensor(prompt_temb_4d, device=mesh_device)

    cos_heads = cos_freq.reshape(B, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
    sin_heads = sin_freq.reshape(B, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
    tt_cos = bf16_tensor_2dshard(cos_heads, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 1})
    tt_sin = bf16_tensor_2dshard(sin_heads, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 1})
    tt_trans_mat = bf16_tensor(get_rot_transformation_mat(), device=mesh_device)

    logger.info("Running forward pass...")
    tt_out = tt_block(
        video_1BND=tt_spatial,
        video_prompt=tt_prompt,
        video_temb=tt_temb,
        video_N=seq_len,
        video_rope_cos=tt_cos,
        video_rope_sin=tt_sin,
        trans_mat=tt_trans_mat,
        video_prompt_temb=tt_prompt_temb,
    )

    concat_dims = [None, None]
    concat_dims[sp_axis] = 2
    concat_dims[tp_axis] = 3
    tt_out_torch = ttnn.to_torch(
        tt_out,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=concat_dims, mesh_shape=tuple(mesh_device.shape)),
    ).squeeze(0)

    assert tt_out_torch.shape == (B, seq_len, dim), f"Output shape mismatch: {tt_out_torch.shape}"
    assert torch.isfinite(tt_out_torch).all(), "Output contains NaN or Inf"

    logger.info(
        f"PASSED: output shape={tuple(tt_out_torch.shape)}, "
        f"mean={tt_out_torch.mean():.4f}, std={tt_out_torch.std():.4f}"
    )


def _make_ltx_model_state(
    dim,
    num_heads,
    in_channels,
    out_channels,
    context_dim,
    ffn_dim,
    num_layers,
    *,
    cross_attention_adaln=True,
):
    """Build a random torch state_dict for LTXTransformerModel.load_torch_state_dict.

    Key names and shapes match ltx_core LTXModel.state_dict() format before
    _prepare_torch_state renaming.  All weights are (out, in) PyTorch convention.
    """
    scale = 0.02
    state = {}

    # patchify_proj (ColParallelLinear: in_channels → dim)
    state["patchify_proj.weight"] = torch.randn(dim, in_channels) * scale
    state["patchify_proj.bias"] = torch.zeros(dim)

    # adaln_single + prompt_adaln_single
    adaln_specs = [("adaln_single", 9 if cross_attention_adaln else 6)]
    if cross_attention_adaln:
        adaln_specs.append(("prompt_adaln_single", 2))
    for prefix, coeff in adaln_specs:
        state[f"{prefix}.emb.timestep_embedder.linear_1.weight"] = torch.randn(dim, 256) * scale
        state[f"{prefix}.emb.timestep_embedder.linear_1.bias"] = torch.zeros(dim)
        state[f"{prefix}.emb.timestep_embedder.linear_2.weight"] = torch.randn(dim, dim) * scale
        state[f"{prefix}.emb.timestep_embedder.linear_2.bias"] = torch.zeros(dim)
        state[f"{prefix}.linear.weight"] = torch.randn(coeff * dim, dim) * scale
        state[f"{prefix}.linear.bias"] = torch.zeros(coeff * dim)

    # transformer_blocks (reuse block state builder)
    for i in range(num_layers):
        for k, v in _make_ltx_block_state(
            dim, num_heads, context_dim, ffn_dim, cross_attention_adaln=cross_attention_adaln
        ).items():
            state[f"transformer_blocks.{i}.{k}"] = v

    # scale_shift_table: (2, dim) — _prepare_torch_state will unsqueeze to (1,1,2,dim)
    state["scale_shift_table"] = torch.zeros(2, dim)

    # proj_out (Linear: dim → out_channels)
    state["proj_out.weight"] = torch.randn(out_channels, dim) * scale
    state["proj_out.bias"] = torch.zeros(out_channels)

    return state


@pytest.mark.parametrize(
    ("mesh_device", "sp_axis", "tp_axis", "num_links", "device_params", "topology", "is_fsdp"),
    _LTX_SMOKE_MESH_PARAMS,
    indirect=["mesh_device", "device_params"],
)
def test_ltx_transformer_model_smoke(
    mesh_device: ttnn.MeshDevice,
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    topology: ttnn.Topology,
    is_fsdp: bool,
) -> None:
    """Smoke test: run LTXTransformerModel.forward on device without PCC comparison.

    Uses num_layers=2 (not 48) to stay within host RAM while exercising the full
    patchify → AdaLN → N×block → norm_out → proj_out pipeline.

    Validates:
    - Model loads onto mesh without errors
    - inner_step runs to completion
    - Output shape matches expected (B, seq_len, out_channels)
    - Output values are finite (no NaN/Inf)
    """
    dim = 4096
    num_heads = 32
    head_dim = dim // num_heads
    in_channels = 128
    out_channels = 128
    context_dim = 4096
    ffn_dim = dim * 4
    num_layers = 2
    B = 1
    F, H, W = 4, 8, 8
    seq_len = F * H * W
    prompt_len = 32

    logger.info(f"mesh shape: {tuple(mesh_device.shape)}, sp_axis={sp_axis}, tp_axis={tp_axis}")

    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
    parallel_config = _make_parallel_config(mesh_device, sp_axis, tp_axis)

    tt_model = LTXTransformerModel(
        num_attention_heads=num_heads,
        attention_head_dim=head_dim,
        in_channels=in_channels,
        out_channels=out_channels,
        num_layers=num_layers,
        cross_attention_dim=context_dim,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_fsdp=is_fsdp,
        has_audio=False,
    )

    torch.manual_seed(42)
    torch_state = _make_ltx_model_state(dim, num_heads, in_channels, out_channels, context_dim, ffn_dim, num_layers)
    tt_model.load_torch_state_dict(torch_state)
    logger.info("Model loaded onto device")

    # Inputs
    torch.manual_seed(123)
    latent = torch.randn(B, seq_len, in_channels, dtype=torch.float32)
    context = torch.randn(B, prompt_len, context_dim, dtype=torch.float32)
    timestep_val = 500.0

    # RoPE
    t_ids = torch.arange(F)
    h_ids = torch.arange(H)
    w_ids = torch.arange(W)
    grid_t, grid_h, grid_w = torch.meshgrid(t_ids, h_ids, w_ids, indexing="ij")
    indices_grid = torch.stack([grid_t.flatten(), grid_h.flatten(), grid_w.flatten()], dim=-1).float().unsqueeze(0)
    cos_freq, sin_freq = precompute_freqs_cis(
        indices_grid, dim=dim, out_dtype=torch.float32, max_pos=[20, 2048, 2048], num_attention_heads=num_heads
    )
    cos_heads = cos_freq.reshape(B, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
    sin_heads = sin_freq.reshape(B, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
    tt_cos = bf16_tensor_2dshard(cos_heads, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 1})
    tt_sin = bf16_tensor_2dshard(sin_heads, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 1})
    tt_trans_mat = bf16_tensor(get_rot_transformation_mat(), device=mesh_device)

    # Prompt
    tt_prompt = bf16_tensor(context.unsqueeze(0), device=mesh_device)

    spatial_torch = latent.unsqueeze(0)  # (1, B, N, in_channels)
    timestep_torch = torch.tensor([timestep_val])

    logger.info("Running forward...")
    tt_out = tt_model.forward(
        video_1BNI_torch=spatial_torch,
        video_prompt_1BLP=tt_prompt,
        video_rope_cos=tt_cos,
        video_rope_sin=tt_sin,
        trans_mat=tt_trans_mat,
        video_N=seq_len,
        timestep_torch=timestep_torch,
    )

    tt_out_torch = LTXTransformerModel.device_to_host(tt_out).squeeze(0)

    assert tt_out_torch.shape == (B, seq_len, out_channels), f"Output shape mismatch: {tt_out_torch.shape}"
    assert torch.isfinite(tt_out_torch).all(), "Output contains NaN or Inf"

    logger.info(
        f"PASSED: output shape={tuple(tt_out_torch.shape)}, "
        f"mean={tt_out_torch.mean():.4f}, std={tt_out_torch.std():.4f}"
    )
