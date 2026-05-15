# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Parity test: AudioInjector block-diagonal mask vs per-frame rearrange.

Our :meth:`WanS2VTransformer3DModel.after_transformer_block` flattens
per-frame audio K/V to a single sequence ``[1, B, T*(N_audio+1), dim]`` and
applies a block-diagonal cross-attention mask so each Q token attends only
to its own frame's audio K/V tokens. The reference does this differently —
``rearrange("b (t n) c -> (b t) n c")`` Q + audio + per-frame cross-attn.

Mathematically they should be equivalent (softmax with -inf mask zeros out
the off-frame contributions). This test verifies that on real hardware.

Test bar: PCC ≥ 0.99.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F  # noqa: N812
from loguru import logger

import ttnn

from ....models.transformers.wan2_2.attention_wan import WanAttention
from ....parallel.config import DiTParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....utils.check import assert_quality
from ....utils.tensor import bf16_tensor, from_torch, local_device_to_torch
from ....utils.test import line_params

# Reduced config that exercises the block-diagonal mask machinery.
DIM = 256
NUM_HEADS = 4
HEAD_DIM = DIM // NUM_HEADS  # 64
T_VIDEO = 8
N_PER_FRAME = 32  # spatial tokens per frame (after patch)
N_AUDIO_PER_FRAME = 5  # num_audio_token + 1 (padding token)
B = 1


def _host_per_frame_cross_attn(
    q_BTNI: torch.Tensor,
    kv_BTAI: torch.Tensor,
    *,
    q_w: torch.Tensor,
    q_b: torch.Tensor,
    k_w: torch.Tensor,
    k_b: torch.Tensor,
    v_w: torch.Tensor,
    v_b: torch.Tensor,
    o_w: torch.Tensor,
    o_b: torch.Tensor,
    norm_q_w: torch.Tensor,
    norm_k_w: torch.Tensor,
    num_heads: int,
    head_dim: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Reference per-frame cross-attention.

    Mirrors the reference's ``rearrange("b (t n) c -> (b t) n c")`` flow
    with a standard QKV → SDPA → output projection. Reads `q` from
    `q_BTNI` (per-frame spatial), `kv` from `kv_BTAI` (per-frame audio).
    """
    B, T, N, _ = q_BTNI.shape
    _, _, A, _ = kv_BTAI.shape
    out_BTNI = torch.zeros_like(q_BTNI)
    for t in range(T):
        q_frame = q_BTNI[:, t]  # [B, N, dim]
        kv_frame = kv_BTAI[:, t]  # [B, A, dim]
        q = q_frame @ q_w.T + q_b
        k = kv_frame @ k_w.T + k_b
        v = kv_frame @ v_w.T + v_b
        # RMSNorm over last dim, per-token.
        q = q * torch.rsqrt(q.float().pow(2).mean(-1, keepdim=True) + eps).to(q.dtype) * norm_q_w
        k = k * torch.rsqrt(k.float().pow(2).mean(-1, keepdim=True) + eps).to(k.dtype) * norm_k_w
        # Split heads: [B, N, num_heads, head_dim] → [B, num_heads, N, head_dim]
        qh = q.view(B, N, num_heads, head_dim).permute(0, 2, 1, 3)
        kh = k.view(B, A, num_heads, head_dim).permute(0, 2, 1, 3)
        vh = v.view(B, A, num_heads, head_dim).permute(0, 2, 1, 3)
        attn = F.scaled_dot_product_attention(qh, kh, vh)
        attn = attn.permute(0, 2, 1, 3).reshape(B, N, num_heads * head_dim)
        out_BTNI[:, t] = attn @ o_w.T + o_b
    return out_BTNI


@pytest.mark.parametrize(
    ("mesh_device", "mesh_shape", "sp_axis", "tp_axis", "num_links", "device_params", "topology", "is_fsdp"),
    [
        pytest.param(
            (4, 8),
            (4, 8),
            1,
            0,
            2,
            line_params,
            ttnn.Topology.Linear,
            False,
            id="bh_4x8sp1tp0",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_audio_injector_block_diagonal_vs_per_frame(
    mesh_device: ttnn.MeshDevice,
    mesh_shape: tuple[int, int],
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    is_fsdp: bool,
    topology: ttnn.Topology,
) -> None:
    """Block-diagonal mask flat cross-attn ≅ per-frame cross-attn (reference).

    Skips the ref/motion const tokens — this test focuses purely on the
    noisy-portion cross-attn equivalence (which is the math the production
    `after_transformer_block` relies on, with the const region zeroed out
    by `_cached_mask_noisy` after the fact).
    """
    torch.manual_seed(0)

    parallel_config = DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tuple(mesh_device.shape)[tp_axis]),
        sequence_parallel=ParallelFactor(mesh_axis=sp_axis, factor=tuple(mesh_device.shape)[sp_axis]),
        cfg_parallel=None,
    )
    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)

    # ---- Build cross-attention module. ----
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
    # Pull initialized weights to host for the reference reproduction.
    # WanAttention stores to_q/to_kv/to_out separately when is_self=False.
    q_w = local_device_to_torch(attn.to_q.weight.data).float()
    q_b = (
        local_device_to_torch(attn.to_q.bias.data).float().squeeze() if attn.to_q.bias is not None else torch.zeros(DIM)
    )
    kv_w = local_device_to_torch(attn.to_kv.weight.data).float()  # [2*dim, dim] fused
    kv_b = (
        local_device_to_torch(attn.to_kv.bias.data).float().squeeze()
        if attn.to_kv.bias is not None
        else torch.zeros(2 * DIM)
    )
    # Split fused KV.
    k_w, v_w = kv_w[:DIM], kv_w[DIM:]
    k_b, v_b = kv_b[:DIM], kv_b[DIM:]
    o_w = local_device_to_torch(attn.to_out.weight.data).float()
    o_b = (
        local_device_to_torch(attn.to_out.bias.data).float().squeeze()
        if attn.to_out.bias is not None
        else torch.zeros(DIM)
    )
    norm_q_w = local_device_to_torch(attn.norm_q.weight.data).float().reshape(-1)
    norm_k_w = local_device_to_torch(attn.norm_k.weight.data).float().reshape(-1)

    # ---- Build inputs. ----
    # Spatial Q: per-frame [B, T, N, dim]. We pass it as a flat sequence to the TT module.
    q_BTNI_torch = torch.randn(B, T_VIDEO, N_PER_FRAME, DIM, dtype=torch.float32)
    # Audio K/V: per-frame [B, T, A, dim].
    kv_BTAI_torch = torch.randn(B, T_VIDEO, N_AUDIO_PER_FRAME, DIM, dtype=torch.float32)

    # ---- Reference per-frame path. ----
    with torch.no_grad():
        ref_out_BTNI = _host_per_frame_cross_attn(
            q_BTNI_torch,
            kv_BTAI_torch,
            q_w=q_w,
            q_b=q_b,
            k_w=k_w,
            k_b=k_b,
            v_w=v_w,
            v_b=v_b,
            o_w=o_w,
            o_b=o_b,
            norm_q_w=norm_q_w,
            norm_k_w=norm_k_w,
            num_heads=NUM_HEADS,
            head_dim=HEAD_DIM,
        )
    ref_flat = ref_out_BTNI.reshape(B, T_VIDEO * N_PER_FRAME, DIM)  # [B, N_total, dim]
    logger.info(f"Reference per-frame output: shape={tuple(ref_flat.shape)}")

    # ---- TT block-diagonal path. ----
    N_noisy = T_VIDEO * N_PER_FRAME
    # Flatten Q to [1, B, N_noisy, dim].
    q_flat_torch = q_BTNI_torch.reshape(B, N_noisy, DIM).unsqueeze(0)
    # Flatten audio K/V to [1, B, T*A, dim].
    kv_flat_torch = kv_BTAI_torch.reshape(B, T_VIDEO * N_AUDIO_PER_FRAME, DIM).unsqueeze(0)
    # Build block-diagonal mask. Shape: [1, 1, N_noisy, T*A].
    mask_torch = torch.full((1, 1, N_noisy, T_VIDEO * N_AUDIO_PER_FRAME), float("-inf"), dtype=torch.float32)
    for t in range(T_VIDEO):
        mask_torch[
            ..., t * N_PER_FRAME : (t + 1) * N_PER_FRAME, t * N_AUDIO_PER_FRAME : (t + 1) * N_AUDIO_PER_FRAME
        ] = 0.0

    # Upload tensors with the right sharding.
    sp_factor = parallel_config.sequence_parallel.factor
    # Q is SP-sharded on N.
    q_dev = from_torch(
        q_flat_torch,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_axes=[None, None, sp_axis, None],
    )
    # K/V is replicated.
    kv_dev = bf16_tensor(kv_flat_torch, device=mesh_device, layout=ttnn.TILE_LAYOUT)
    # Mask is SP-sharded on Sq.
    mask_dev = from_torch(
        mask_torch,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_axes=[None, None, sp_axis, None],
    )

    # Run our cross-attention.
    out_dev = attn.forward(
        spatial_1BND=q_dev,
        N=N_noisy,
        prompt_1BLP=kv_dev,
        cross_attn_mask=mask_dev,
    )
    # Gather across SP (N) and TP (D).
    out_gather = ccl_manager.all_gather_persistent_buffer(out_dev, dim=2, mesh_axis=sp_axis)
    out_gather = ccl_manager.all_gather_persistent_buffer(out_gather, dim=3, mesh_axis=tp_axis)
    out_flat = local_device_to_torch(out_gather).squeeze(0).float()  # [B, N_noisy, dim]
    logger.info(f"TT block-diagonal output: shape={tuple(out_flat.shape)}")

    # ---- Compare. ----
    assert_quality(out_flat, ref_flat.float(), pcc=0.99)
