# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
LTX-2 attention unit tests. Mesh / fabric / topology parametrization mirrors
``test_wan_attention`` in ``tests/models/wan2_2/test_attention_wan.py``:

- ``line_params`` / ``ring_params`` from ``utils.test`` (``FABRIC_1D`` vs ``FABRIC_1D_RING``)
- explicit ``num_links`` and ``ttnn.Topology.{Linear,Ring}``
- ``is_fsdp`` passed through to ``LTXAttention``

LTX-only: ``1x1`` with empty ``device_params`` (no fabric) for single-chip runs.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.models.transformers.ltx.attention_ltx import LTXAttention
from models.tt_dit.parallel.config import DiTParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils.check import assert_quality
from models.tt_dit.utils.mochi import get_rot_transformation_mat
from models.tt_dit.utils.tensor import bf16_tensor, bf16_tensor_2dshard
from models.tt_dit.utils.test import line_params, ring_params

# Reference model + RoPE come from the installed ``ltx_core`` package (imported lazily
# inside each test), exactly like ``test_wan_attention`` pulls its reference from
# diffusers. No ``sys.path.insert`` into the in-repo ``LTX-2`` clone.

# Wan ``test_wan_attention`` grid + LTX single-device row (empty device_params).
_LTX_ATTENTION_MESH_PARAMS = [
    pytest.param((1, 1), 0, 1, 1, {}, ttnn.Topology.Linear, False, id="1x1sp0tp1"),
    pytest.param((2, 4), 0, 1, 1, line_params, ttnn.Topology.Linear, True, id="2x4sp0tp1"),
    pytest.param((2, 4), 1, 0, 1, line_params, ttnn.Topology.Linear, True, id="2x4sp1tp0"),
    pytest.param((4, 8), 1, 0, 4, ring_params, ttnn.Topology.Ring, True, id="wh_4x8sp1tp0"),
    pytest.param((4, 8), 1, 0, 2, line_params, ttnn.Topology.Linear, False, id="bh_4x8sp1tp0"),
]


@pytest.mark.parametrize(
    ("mesh_device", "sp_axis", "tp_axis", "num_links", "device_params", "topology", "is_fsdp"),
    _LTX_ATTENTION_MESH_PARAMS,
    indirect=["mesh_device", "device_params"],
)
def test_ltx_self_attention(
    mesh_device: ttnn.MeshDevice,
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    topology: ttnn.Topology,
    is_fsdp: bool,
) -> None:
    """
    Test LTX-2 self-attention: compare TT LTXAttention vs PyTorch Attention.
    """
    from ltx_core.model.transformer.attention import Attention as TorchAttention
    from ltx_core.model.transformer.rope import LTXRopeType, precompute_freqs_cis

    dim = 4096
    num_heads = 32
    head_dim = dim // num_heads
    B = 1
    seq_len = 256  # Small for fast test

    # INTERLEAVED matches ltx_core precompute_freqs_cis which returns (B, N, D) format
    torch_model = TorchAttention(
        query_dim=dim, heads=num_heads, dim_head=head_dim, norm_eps=1e-6, rope_type=LTXRopeType.INTERLEAVED
    )
    torch_model.eval()
    torch_state = torch_model.state_dict()

    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(factor=1, mesh_axis=0),
        sequence_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[sp_axis], mesh_axis=sp_axis),
        tensor_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[tp_axis], mesh_axis=tp_axis),
    )

    tt_model = LTXAttention(
        dim=dim,
        num_heads=num_heads,
        eps=1e-6,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_fsdp=is_fsdp,
        is_self=True,
    )
    tt_model.load_torch_state_dict(torch_state)

    # seq_len=256 → N_local=32 for 4x8 (sp=8); exactly 1 tile, the minimum
    torch.manual_seed(42)
    x = torch.randn(B, seq_len, dim, dtype=torch.float32)

    # Build RoPE (small grid for test)
    F, H, W = 4, 8, 8  # 4*8*8=256 = seq_len
    t_ids = torch.arange(F)
    h_ids = torch.arange(H)
    w_ids = torch.arange(W)
    grid_t, grid_h, grid_w = torch.meshgrid(t_ids, h_ids, w_ids, indexing="ij")
    # ltx_core's precompute_freqs_cis expects (B, n_dims, N) and defaults to SPLIT, so
    # stack on dim=0 and request INTERLEAVED (the mode this attention block was built with).
    indices_grid = torch.stack([grid_t.flatten(), grid_h.flatten(), grid_w.flatten()], dim=0).float().unsqueeze(0)

    cos_freq, sin_freq = precompute_freqs_cis(
        indices_grid,
        dim=dim,
        out_dtype=torch.float32,
        max_pos=[20, 2048, 2048],
        num_attention_heads=num_heads,
        rope_type=LTXRopeType.INTERLEAVED,
    )
    # Reshape for apply: (B, seq_len, num_heads, head_dim)
    cos_apply = cos_freq.reshape(B, seq_len, num_heads, head_dim)
    sin_apply = sin_freq.reshape(B, seq_len, num_heads, head_dim)

    # PyTorch forward
    # LTX-2 Attention applies RoPE to q/k in (B, T, inner_dim) shape before multi-head split.
    # So pe should be (B, T, dim) for interleaved mode.
    pe_flat_cos = cos_freq.squeeze(0) if cos_freq.ndim == 3 else cos_freq  # (B, T, dim)
    pe_flat_sin = sin_freq.squeeze(0) if sin_freq.ndim == 3 else sin_freq
    with torch.no_grad():
        torch_out = torch_model(x, pe=(pe_flat_cos, pe_flat_sin))

    logger.info(f"PyTorch output shape: {torch_out.shape}")

    # Prepare TT tensors
    # spatial: (1, B, N, D) fractured on SP/TP
    spatial = x.unsqueeze(0)  # (1, B, N, D)
    tt_spatial = bf16_tensor_2dshard(spatial, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 3})

    # RoPE for rotary_embedding_llama: expects (B, H, N, head_dim)
    # LTX-2 has per-head RoPE (different frequencies per head), so pass full (B, H, N, D)
    rope_cos_bhnd = cos_apply.permute(0, 2, 1, 3)  # (B, H, N, D)
    rope_sin_bhnd = sin_apply.permute(0, 2, 1, 3)
    tt_cos = bf16_tensor_2dshard(rope_cos_bhnd, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 1})
    tt_sin = bf16_tensor_2dshard(rope_sin_bhnd, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 1})
    tt_trans_mat = bf16_tensor(get_rot_transformation_mat(), device=mesh_device)

    # TT forward
    tt_out = tt_model(
        spatial_1BND=tt_spatial,
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

    assert_quality(torch_out, tt_out_torch, pcc=0.988)
    logger.info("PASSED: LTX self-attention matches PyTorch reference")


@pytest.mark.parametrize(
    ("mesh_device", "sp_axis", "tp_axis", "num_links", "device_params", "topology", "is_fsdp"),
    _LTX_ATTENTION_MESH_PARAMS,
    indirect=["mesh_device", "device_params"],
)
def test_ltx_cross_attention(
    mesh_device: ttnn.MeshDevice,
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    topology: ttnn.Topology,
    is_fsdp: bool,
) -> None:
    """
    Test LTX-2 cross-attention: compare TT LTXAttention vs PyTorch Attention.
    """
    from ltx_core.model.transformer.attention import Attention as TorchAttention

    dim = 4096
    context_dim = 4096
    num_heads = 32
    head_dim = dim // num_heads
    B = 1
    seq_len = 256
    prompt_len = 32

    # Create PyTorch reference (cross-attention: context_dim != None)
    torch_model = TorchAttention(
        query_dim=dim, context_dim=context_dim, heads=num_heads, dim_head=head_dim, norm_eps=1e-6
    )
    torch_model.eval()
    torch_state = torch_model.state_dict()

    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(factor=1, mesh_axis=0),
        sequence_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[sp_axis], mesh_axis=sp_axis),
        tensor_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[tp_axis], mesh_axis=tp_axis),
    )

    tt_model = LTXAttention(
        dim=dim,
        num_heads=num_heads,
        eps=1e-6,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_fsdp=is_fsdp,
        is_self=False,
        context_dim=context_dim,
    )
    tt_model.load_torch_state_dict(torch_state)

    torch.manual_seed(42)
    x = torch.randn(B, seq_len, dim, dtype=torch.float32)
    context = torch.randn(B, prompt_len, context_dim, dtype=torch.float32)

    with torch.no_grad():
        torch_out = torch_model(x, context=context)

    logger.info(f"PyTorch cross-attn output shape: {torch_out.shape}")

    spatial = x.unsqueeze(0)
    prompt = context.unsqueeze(0)
    tt_spatial = bf16_tensor_2dshard(spatial, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 3})
    tt_prompt = bf16_tensor(prompt, device=mesh_device)

    tt_out = tt_model(spatial_1BND=tt_spatial, N=seq_len, prompt_1BLP=tt_prompt)

    concat_dims = [None, None]
    concat_dims[sp_axis] = 2
    concat_dims[tp_axis] = 3
    tt_out_torch = ttnn.to_torch(
        tt_out,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=concat_dims, mesh_shape=tuple(mesh_device.shape)),
    ).squeeze(0)

    assert_quality(torch_out, tt_out_torch, pcc=0.988)
    logger.info("PASSED: LTX cross-attention matches PyTorch reference")
