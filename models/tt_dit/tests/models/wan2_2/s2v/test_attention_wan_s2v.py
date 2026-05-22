# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""WanAttention parity with the S2V audio frame-diagonal mask."""

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


def _project_qkv(
    q_in: torch.Tensor,
    kv_in: torch.Tensor,
    weights: dict[str, torch.Tensor],
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Q/K/V projection + qk-norm shared by both reference paths."""
    q = q_in @ weights["to_q.weight"].T + weights["to_q.bias"]
    k = kv_in @ weights["to_k.weight"].T + weights["to_k.bias"]
    v = kv_in @ weights["to_v.weight"].T + weights["to_v.bias"]
    q = q * torch.rsqrt(q.float().pow(2).mean(-1, keepdim=True) + eps).to(q.dtype) * weights["norm_q.weight"]
    k = k * torch.rsqrt(k.float().pow(2).mean(-1, keepdim=True) + eps).to(k.dtype) * weights["norm_k.weight"]
    return q, k, v


def _split_heads(x: torch.Tensor) -> torch.Tensor:
    B, S, _ = x.shape
    return x.view(B, S, NUM_HEADS, HEAD_DIM).permute(0, 2, 1, 3)


def _host_per_frame_cross_attn(
    q_BTNI: torch.Tensor,
    kv_BTAI: torch.Tensor,
    *,
    weights: dict[str, torch.Tensor],
    eps: float = 1e-5,
) -> torch.Tensor:
    """Reference per-frame cross-attention; returns ``[B, T, N, dim]``."""
    o_w, o_b = weights["to_out.0.weight"], weights["to_out.0.bias"]
    B, T, N, _ = q_BTNI.shape
    out = torch.zeros_like(q_BTNI)
    for t in range(T):
        q, k, v = _project_qkv(q_BTNI[:, t], kv_BTAI[:, t], weights, eps)
        attn = F.scaled_dot_product_attention(_split_heads(q), _split_heads(k), _split_heads(v))
        out[:, t] = attn.permute(0, 2, 1, 3).reshape(B, N, NUM_HEADS * HEAD_DIM) @ o_w.T + o_b
    return out


def _host_block_mask_cross_attn(
    q_flat: torch.Tensor,
    kv_flat: torch.Tensor,
    mask: torch.Tensor,
    *,
    weights: dict[str, torch.Tensor],
    eps: float = 1e-5,
) -> torch.Tensor:
    """Reference block-mask cross-attention (flat sequence + additive mask)."""
    o_w, o_b = weights["to_out.0.weight"], weights["to_out.0.bias"]
    B_, N_noisy, _ = q_flat.shape
    q, k, v = _project_qkv(q_flat, kv_flat, weights, eps)
    attn = F.scaled_dot_product_attention(_split_heads(q), _split_heads(k), _split_heads(v), attn_mask=mask.squeeze(0))
    return attn.permute(0, 2, 1, 3).reshape(B_, N_noisy, NUM_HEADS * HEAD_DIM) @ o_w.T + o_b


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
    """Block-diagonal mask flat cross-attn equals per-frame cross-attn."""
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
    incompat = attn.load_torch_state_dict(weights, strict=False)
    logger.info(f"WanAttention load: missing={len(incompat.missing_keys)} unexpected={len(incompat.unexpected_keys)}")

    q_BTNI = torch.randn(B, T_VIDEO, N_PER_FRAME, DIM, dtype=torch.float32)
    kv_BTAI = torch.randn(B, T_VIDEO, N_AUDIO_PER_FRAME, DIM, dtype=torch.float32)

    N_noisy = T_VIDEO * N_PER_FRAME
    L = T_VIDEO * N_AUDIO_PER_FRAME
    q_flat = q_BTNI.reshape(B, N_noisy, DIM).unsqueeze(0)
    kv_flat = kv_BTAI.reshape(B, L, DIM).unsqueeze(0)
    mask = _build_block_diagonal_audio_mask(
        n_noisy=N_noisy, t_video=T_VIDEO, n_audio_per_frame=N_AUDIO_PER_FRAME, mask_neg=MASK_NEG
    )

    with torch.no_grad():
        ref_per_frame = _host_per_frame_cross_attn(q_BTNI, kv_BTAI, weights=weights)
        ref_flat = ref_per_frame.reshape(B, N_noisy, DIM)
        ref_block = _host_block_mask_cross_attn(q_flat.squeeze(0), kv_flat.squeeze(0), mask, weights=weights).float()
    logger.info(f"reference per-frame: {tuple(ref_flat.shape)}; block-mask: {tuple(ref_block.shape)}")

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

    # Two-way parity is enough; out ≈ ref_flat follows transitively.
    assert_quality(ref_block, ref_flat.float(), pcc=0.99)
    assert_quality(out_flat, ref_block, pcc=0.99)
