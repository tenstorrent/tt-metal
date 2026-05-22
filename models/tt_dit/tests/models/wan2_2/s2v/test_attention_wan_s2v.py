# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Sub-component parity test for ``WanAttention`` with the S2V audio frame-diagonal mask.

Mirrors ``tests/models/wan2_2/test_attention_wan.py`` (T2V attention parity)
at the same granularity, but tests the **audio cross-attention** path used
by ``AudioInjector_WAN`` rather than the text cross-attention used by T2V.

Two tests:
  * ``test_wan_attention_s2v`` — block-diagonal mask flat cross-attn ≡
    per-frame cross-attn (the math the production ``after_transformer_block``
    relies on).
  * ``test_wan_attention_s2v_staged`` — per-stage breakdown (to_q → to_kv →
    norm_q → norm_k → SDPA → to_out) with PCC compare at each stage. Useful
    for debugging which stage drops parity.

PCC bar: 0.99 across the board.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F  # noqa: N812
from loguru import logger

import ttnn

from .....models.transformers.wan2_2.attention_wan import WanAttention
from .....parallel.config import DiTParallelConfig, ParallelFactor
from .....parallel.manager import CCLManager
from .....utils.check import assert_quality
from .....utils.tensor import bf16_tensor, from_torch, local_device_to_torch
from .....utils.test import line_params, ring_params

# Production Wan-AI/Wan2.2-S2V-14B model config (resolution-independent).
DIM = 5120
NUM_HEADS = 40
HEAD_DIM = DIM // NUM_HEADS  # 128
N_AUDIO_PER_FRAME = 5  # num_audio_token + 1 (padding token), production config
B = 1

MASK_NEG = -1e9


def _build_block_diagonal_audio_mask(
    *, n_noisy: int, t_video: int, n_audio_per_frame: int, mask_neg: float = MASK_NEG
) -> torch.Tensor:
    """``[1, 1, n_noisy, t_video * n_audio_per_frame]`` block-diagonal mask.

    Each spatial frame's tokens attend only to that frame's audio K/V slot.
    """
    sk = t_video * n_audio_per_frame
    assert n_noisy % t_video == 0, f"n_noisy={n_noisy} must be divisible by t_video={t_video}"
    hw_per_frame = n_noisy // t_video
    mask = torch.full((1, 1, n_noisy, sk), mask_neg, dtype=torch.float32)
    for t in range(t_video):
        mask[
            ...,
            t * hw_per_frame : (t + 1) * hw_per_frame,
            t * n_audio_per_frame : (t + 1) * n_audio_per_frame,
        ] = 0.0
    return mask


def _make_synth_weights() -> dict[str, torch.Tensor]:
    """Random Q/K/V/O + qk-norm weights for a `WanAttention(is_self=False)` load."""
    return {
        "to_q.weight": torch.randn(DIM, DIM, dtype=torch.float32),
        "to_q.bias": torch.randn(DIM, dtype=torch.float32),
        "to_k.weight": torch.randn(DIM, DIM, dtype=torch.float32),
        "to_k.bias": torch.randn(DIM, dtype=torch.float32),
        "to_v.weight": torch.randn(DIM, DIM, dtype=torch.float32),
        "to_v.bias": torch.randn(DIM, dtype=torch.float32),
        "to_out.0.weight": torch.randn(DIM, DIM, dtype=torch.float32),
        "to_out.0.bias": torch.randn(DIM, dtype=torch.float32),
        "norm_q.weight": torch.randn(DIM, dtype=torch.float32),
        "norm_k.weight": torch.randn(DIM, dtype=torch.float32),
    }


def _host_per_frame_cross_attn(
    q_BTNI: torch.Tensor,
    kv_BTAI: torch.Tensor,
    *,
    weights: dict[str, torch.Tensor],
    eps: float = 1e-5,
) -> torch.Tensor:
    """Reference per-frame cross-attention via per-frame iteration.

    Mirrors the upstream ``rearrange("b (t n) c -> (b t) n c")`` flow.
    Returns ``[B, T, N, dim]``.
    """
    q_w, q_b = weights["to_q.weight"], weights["to_q.bias"]
    k_w, k_b = weights["to_k.weight"], weights["to_k.bias"]
    v_w, v_b = weights["to_v.weight"], weights["to_v.bias"]
    o_w, o_b = weights["to_out.0.weight"], weights["to_out.0.bias"]
    norm_q_w = weights["norm_q.weight"]
    norm_k_w = weights["norm_k.weight"]

    B, T, N, _ = q_BTNI.shape
    _, _, A, _ = kv_BTAI.shape
    out = torch.zeros_like(q_BTNI)
    for t in range(T):
        q_frame = q_BTNI[:, t]
        kv_frame = kv_BTAI[:, t]
        q = q_frame @ q_w.T + q_b
        k = kv_frame @ k_w.T + k_b
        v = kv_frame @ v_w.T + v_b
        q = q * torch.rsqrt(q.float().pow(2).mean(-1, keepdim=True) + eps).to(q.dtype) * norm_q_w
        k = k * torch.rsqrt(k.float().pow(2).mean(-1, keepdim=True) + eps).to(k.dtype) * norm_k_w
        qh = q.view(B, N, NUM_HEADS, HEAD_DIM).permute(0, 2, 1, 3)
        kh = k.view(B, A, NUM_HEADS, HEAD_DIM).permute(0, 2, 1, 3)
        vh = v.view(B, A, NUM_HEADS, HEAD_DIM).permute(0, 2, 1, 3)
        attn = F.scaled_dot_product_attention(qh, kh, vh)
        attn = attn.permute(0, 2, 1, 3).reshape(B, N, NUM_HEADS * HEAD_DIM)
        out[:, t] = attn @ o_w.T + o_b
    return out


def _host_block_mask_cross_attn(
    q_flat: torch.Tensor,
    kv_flat: torch.Tensor,
    mask: torch.Tensor,
    *,
    weights: dict[str, torch.Tensor],
    eps: float = 1e-5,
) -> torch.Tensor:
    """Reference block-mask cross-attention (single flat sequence + mask).

    The math the production TT path implements; should equal the per-frame
    reference up to numerical noise.
    """
    q_w, q_b = weights["to_q.weight"], weights["to_q.bias"]
    k_w, k_b = weights["to_k.weight"], weights["to_k.bias"]
    v_w, v_b = weights["to_v.weight"], weights["to_v.bias"]
    o_w, o_b = weights["to_out.0.weight"], weights["to_out.0.bias"]
    norm_q_w = weights["norm_q.weight"]
    norm_k_w = weights["norm_k.weight"]

    B_, N_noisy, _ = q_flat.shape
    L_ = kv_flat.shape[1]
    q = q_flat @ q_w.T + q_b
    k = kv_flat @ k_w.T + k_b
    v = kv_flat @ v_w.T + v_b
    q = q * torch.rsqrt(q.float().pow(2).mean(-1, keepdim=True) + eps).to(q.dtype) * norm_q_w
    k = k * torch.rsqrt(k.float().pow(2).mean(-1, keepdim=True) + eps).to(k.dtype) * norm_k_w
    qh = q.view(B_, N_noisy, NUM_HEADS, HEAD_DIM).permute(0, 2, 1, 3)
    kh = k.view(B_, L_, NUM_HEADS, HEAD_DIM).permute(0, 2, 1, 3)
    vh = v.view(B_, L_, NUM_HEADS, HEAD_DIM).permute(0, 2, 1, 3)
    attn = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask.squeeze(0))
    attn = attn.permute(0, 2, 1, 3).reshape(B_, N_noisy, NUM_HEADS * HEAD_DIM)
    return attn @ o_w.T + o_b


@pytest.mark.parametrize(
    ("mesh_device", "mesh_shape", "sp_axis", "tp_axis", "num_links", "device_params", "topology", "is_fsdp"),
    [
        pytest.param((2, 4), (2, 4), 1, 0, 2, line_params, ttnn.Topology.Linear, False, id="bh_2x4sp1tp0"),
        pytest.param((4, 8), (4, 8), 1, 0, 2, ring_params, ttnn.Topology.Ring, False, id="bh_4x8sp1tp0"),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    ("T_VIDEO", "N_PER_FRAME"),
    [
        pytest.param(20, 30 * 52, id="480p"),  # patched (H/2 * W/2) at 480p latent (60, 104)
        pytest.param(20, 45 * 80, id="720p"),  # patched at 720p latent (90, 160)
    ],
)
def test_wan_attention_s2v(
    mesh_device: ttnn.MeshDevice,
    mesh_shape: tuple[int, int],
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    is_fsdp: bool,
    topology: ttnn.Topology,
    T_VIDEO: int,
    N_PER_FRAME: int,
) -> None:
    """Block-diagonal mask flat cross-attn ≡ per-frame cross-attn (reference).

    Focuses on the noisy-portion cross-attention. Const tokens are zeroed
    after the residual by ``_cached_mask_noisy`` in production, so they're
    out of scope for this test.
    """
    torch.manual_seed(0)
    parent_mesh = mesh_device
    mesh_device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))

    parallel_config = DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tuple(mesh_device.shape)[tp_axis]),
        sequence_parallel=ParallelFactor(mesh_axis=sp_axis, factor=tuple(mesh_device.shape)[sp_axis]),
        cfg_parallel=None,
    )
    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)

    # ---- Build cross-attention module + load synthetic weights ----
    attn = WanAttention(
        dim=DIM,
        num_heads=NUM_HEADS,
        qk_norm=True,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_fsdp=is_fsdp,
        is_self=False,
    )
    weights = _make_synth_weights()
    incompat = attn.load_torch_state_dict(weights, strict=False)
    logger.info(f"WanAttention load: missing={len(incompat.missing_keys)} unexpected={len(incompat.unexpected_keys)}")

    # ---- Inputs ----
    q_BTNI = torch.randn(B, T_VIDEO, N_PER_FRAME, DIM, dtype=torch.float32)
    kv_BTAI = torch.randn(B, T_VIDEO, N_AUDIO_PER_FRAME, DIM, dtype=torch.float32)

    N_noisy = T_VIDEO * N_PER_FRAME
    L = T_VIDEO * N_AUDIO_PER_FRAME
    q_flat = q_BTNI.reshape(B, N_noisy, DIM).unsqueeze(0)
    kv_flat = kv_BTAI.reshape(B, L, DIM).unsqueeze(0)
    mask = _build_block_diagonal_audio_mask(
        n_noisy=N_noisy, t_video=T_VIDEO, n_audio_per_frame=N_AUDIO_PER_FRAME, mask_neg=MASK_NEG
    )

    # ---- Reference paths ----
    with torch.no_grad():
        ref_per_frame = _host_per_frame_cross_attn(q_BTNI, kv_BTAI, weights=weights)
        ref_flat = ref_per_frame.reshape(B, N_noisy, DIM)
        ref_block = _host_block_mask_cross_attn(q_flat.squeeze(0), kv_flat.squeeze(0), mask, weights=weights).float()
    logger.info(f"reference per-frame: {tuple(ref_flat.shape)}; block-mask: {tuple(ref_block.shape)}")

    # ---- TT block-diagonal path ----
    sp_factor = parallel_config.sequence_parallel.factor
    q_dev = from_torch(
        q_flat,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_axes=[None, None, sp_axis, tp_axis],
    )
    kv_dev = bf16_tensor(kv_flat, device=mesh_device, layout=ttnn.TILE_LAYOUT)
    # Critical: pad_value=MASK_NEG so the TILE-padded Sk columns are masked
    # out. Default pad fill is 0 → SDPA would attend to zero-filled K positions.
    mask_dev = from_torch(
        mask,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_axes=[None, None, sp_axis, None],
        pad_value=MASK_NEG,
    )

    out_dev = attn.forward(
        spatial_1BND=q_dev,
        N=N_noisy,
        prompt_1BLP=kv_dev,
        cross_attn_mask=mask_dev,
    )
    out_gather = ccl_manager.all_gather_persistent_buffer(out_dev, dim=2, mesh_axis=sp_axis)
    out_gather = ccl_manager.all_gather_persistent_buffer(out_gather, dim=3, mesh_axis=tp_axis)
    out_flat = local_device_to_torch(out_gather).squeeze(0).float()
    logger.info(f"TT output: {tuple(out_flat.shape)}")

    # Three-way parity: (1) ref math is consistent, (2) TT matches its torch
    # reference, (3) TT matches the per-frame reference.
    assert_quality(ref_block, ref_flat.float(), pcc=0.99)
    assert_quality(out_flat, ref_block, pcc=0.99)
    assert_quality(out_flat, ref_flat.float(), pcc=0.99)


@pytest.mark.parametrize(
    ("mesh_device", "mesh_shape", "sp_axis", "tp_axis", "num_links", "device_params", "topology", "is_fsdp"),
    [
        pytest.param((2, 4), (2, 4), 1, 0, 2, line_params, ttnn.Topology.Linear, False, id="bh_2x4sp1tp0"),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    ("T_VIDEO", "N_PER_FRAME"),
    [
        pytest.param(20, 30 * 52, id="480p"),
        pytest.param(20, 45 * 80, id="720p"),
    ],
)
def test_wan_attention_s2v_staged(
    mesh_device: ttnn.MeshDevice,
    mesh_shape: tuple[int, int],
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    is_fsdp: bool,
    topology: ttnn.Topology,
    T_VIDEO: int,
    N_PER_FRAME: int,
) -> None:
    """Step-through TT cross-attn with per-stage PCC compare against pytorch.

    Stages: A. post-to_q, B. post-to_kv, C. post-norm_q, D. post-norm_k,
    E. post-SDPA (after concat_heads + TP-gather), F. post-to_out (final).

    Useful for isolating which stage drops parity when the full test fails.
    Only run on (2, 4) since the staged math is mesh-shape-agnostic.
    """
    torch.manual_seed(0)
    parent_mesh = mesh_device
    mesh_device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))

    parallel_config = DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tuple(mesh_device.shape)[tp_axis]),
        sequence_parallel=ParallelFactor(mesh_axis=sp_axis, factor=tuple(mesh_device.shape)[sp_axis]),
        cfg_parallel=None,
    )
    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)

    attn = WanAttention(
        dim=DIM,
        num_heads=NUM_HEADS,
        qk_norm=True,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_fsdp=is_fsdp,
        is_self=False,
    )
    weights = _make_synth_weights()
    attn.load_torch_state_dict(weights, strict=False)

    N_noisy = T_VIDEO * N_PER_FRAME
    L = T_VIDEO * N_AUDIO_PER_FRAME
    q_BTNI = torch.randn(B, T_VIDEO, N_PER_FRAME, DIM, dtype=torch.float32)
    kv_BTAI = torch.randn(B, T_VIDEO, N_AUDIO_PER_FRAME, DIM, dtype=torch.float32)
    q_flat = q_BTNI.reshape(B, N_noisy, DIM).unsqueeze(0)
    kv_flat = kv_BTAI.reshape(B, L, DIM).unsqueeze(0)
    mask = _build_block_diagonal_audio_mask(
        n_noisy=N_noisy, t_video=T_VIDEO, n_audio_per_frame=N_AUDIO_PER_FRAME, mask_neg=MASK_NEG
    )

    # Pytorch reference per stage.
    pyt_q = q_flat.squeeze(0) @ weights["to_q.weight"].T + weights["to_q.bias"]
    pyt_k = kv_flat.squeeze(0) @ weights["to_k.weight"].T + weights["to_k.bias"]
    pyt_v = kv_flat.squeeze(0) @ weights["to_v.weight"].T + weights["to_v.bias"]
    eps = 1e-5
    pyt_q_normed = (
        pyt_q
        * torch.rsqrt(pyt_q.float().pow(2).mean(-1, keepdim=True) + eps).to(pyt_q.dtype)
        * weights["norm_q.weight"]
    )
    pyt_k_normed = (
        pyt_k
        * torch.rsqrt(pyt_k.float().pow(2).mean(-1, keepdim=True) + eps).to(pyt_k.dtype)
        * weights["norm_k.weight"]
    )
    pyt_qh = pyt_q_normed.view(B, N_noisy, NUM_HEADS, HEAD_DIM).permute(0, 2, 1, 3)
    pyt_kh = pyt_k_normed.view(B, L, NUM_HEADS, HEAD_DIM).permute(0, 2, 1, 3)
    pyt_vh = pyt_v.view(B, L, NUM_HEADS, HEAD_DIM).permute(0, 2, 1, 3)
    pyt_attn_BHNE = F.scaled_dot_product_attention(pyt_qh, pyt_kh, pyt_vh, attn_mask=mask.squeeze(0))
    pyt_attn_BND = pyt_attn_BHNE.permute(0, 2, 1, 3).reshape(B, N_noisy, DIM)
    pyt_out = pyt_attn_BND @ weights["to_out.0.weight"].T + weights["to_out.0.bias"]

    # Upload inputs.
    q_dev = from_torch(
        q_flat,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_axes=[None, None, sp_axis, tp_axis],
    )
    kv_dev = bf16_tensor(kv_flat, device=mesh_device, layout=ttnn.TILE_LAYOUT)
    mask_dev = from_torch(
        mask,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_axes=[None, None, sp_axis, None],
        pad_value=MASK_NEG,
    )

    # Step 0: TP all-gather spatial.
    spatial = ccl_manager.all_gather_persistent_buffer(q_dev, dim=3, mesh_axis=tp_axis)

    # Stage A: to_q.
    q_1BNF = attn.to_q(spatial, compute_kernel_config=attn.mm_compute_kernel_config, parallel_config=None)
    q_full_dev = ccl_manager.all_gather_persistent_buffer(q_1BNF, dim=2, mesh_axis=sp_axis)
    q_full_dev = ccl_manager.all_gather_persistent_buffer(q_full_dev, dim=3, mesh_axis=tp_axis)
    tt_q_BND = local_device_to_torch(q_full_dev).squeeze(0).float()
    assert_quality(tt_q_BND, pyt_q.float(), pcc=0.99)

    # Stage B: to_kv.
    k_1BNF, v_1BNF = attn.to_kv(kv_dev, compute_kernel_config=attn.mm_compute_kernel_config)
    k_full = ccl_manager.all_gather_persistent_buffer(k_1BNF, dim=3, mesh_axis=tp_axis)
    v_full = ccl_manager.all_gather_persistent_buffer(v_1BNF, dim=3, mesh_axis=tp_axis)
    assert_quality(local_device_to_torch(k_full).squeeze(0).float(), pyt_k.float(), pcc=0.99)
    assert_quality(local_device_to_torch(v_full).squeeze(0).float(), pyt_v.float(), pcc=0.99)

    # Stage C: norm_q (with head split).
    q_BHNE = attn.norm_q(q_1BNF, num_heads_per_device=attn.n_local_heads)
    q_BHNE_sp = ccl_manager.all_gather_persistent_buffer(q_BHNE, dim=2, mesh_axis=sp_axis)
    q_BHNE_full = ccl_manager.all_gather_persistent_buffer(q_BHNE_sp, dim=1, mesh_axis=tp_axis)
    assert_quality(local_device_to_torch(q_BHNE_full).float(), pyt_qh.float(), pcc=0.99)

    # Stage D: norm_k.
    k_BHNE = attn.norm_k(k_1BNF, num_heads_per_device=attn.n_local_heads)
    k_BHNE_full = ccl_manager.all_gather_persistent_buffer(k_BHNE, dim=1, mesh_axis=tp_axis)
    assert_quality(local_device_to_torch(k_BHNE_full).float(), pyt_kh.float(), pcc=0.99)

    # Stage E: SDPA.
    def _create_heads(inp):
        out, _, _ = ttnn.experimental.nlp_create_qkv_heads(
            inp, num_heads=attn.n_local_heads, num_kv_heads=0, transpose_k_heads=False
        )
        return out

    v_BHNE_dev = _create_heads(v_1BNF)
    spatial_BHNE = ttnn.transformer.scaled_dot_product_attention(
        q_BHNE,
        k_BHNE,
        v_BHNE_dev,
        is_causal=False,
        attn_mask=mask_dev,
        program_config=attn.sdpa_program_config,
        compute_kernel_config=attn.sdpa_compute_kernel_config,
    )
    sd_sp = ccl_manager.all_gather_persistent_buffer(spatial_BHNE, dim=2, mesh_axis=sp_axis)
    sd_full = ccl_manager.all_gather_persistent_buffer(sd_sp, dim=1, mesh_axis=tp_axis)
    assert_quality(local_device_to_torch(sd_full).float(), pyt_attn_BHNE.float(), pcc=0.99)

    # Stage F: concat heads + to_out.
    spatial_BND_dev = ttnn.transformer.concatenate_heads(spatial_BHNE)
    spatial_BND_dev = ttnn.unsqueeze(spatial_BND_dev, 0)
    spatial_BND_dev = ccl_manager.all_gather_persistent_buffer(spatial_BND_dev, dim=3, mesh_axis=tp_axis)
    out_dev = attn.to_out(spatial_BND_dev, compute_kernel_config=attn.mm_compute_kernel_config, parallel_config=None)
    out_sp = ccl_manager.all_gather_persistent_buffer(out_dev, dim=2, mesh_axis=sp_axis)
    out_full = ccl_manager.all_gather_persistent_buffer(out_sp, dim=3, mesh_axis=tp_axis)
    assert_quality(local_device_to_torch(out_full).squeeze(0).float(), pyt_out.float(), pcc=0.99)
    logger.info("staged attention parity ok across all 6 stages")
