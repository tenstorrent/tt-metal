# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Bias-free linear layer matching Hugging Face ``nn.Linear(..., bias=False)`` (Mistral4 projections)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

import ttnn

_LINEAR_CK_ATTR = "_mistral_small_4_linear_compute_kernel_config"


def _linear_compute_kernel_config(mesh_device):
    cfg = getattr(mesh_device, _LINEAR_CK_ATTR, None)
    if cfg is None:
        cfg = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        setattr(mesh_device, _LINEAR_CK_ATTR, cfg)
    return cfg


def linear_bf16_no_bias(
    mesh_device,
    x_bsh: torch.Tensor,
    weight_out_in: torch.Tensor,
) -> torch.Tensor:
    """
    Compute ``y = x @ W.T`` on device (same as ``F.linear(x, W)``) and return host bf16 ``[B, S, out]``.

    Args:
        x_bsh: ``[B, S, in_features]`` (bfloat16 or float; cast to bf16 on upload).
        weight_out_in: HF ``Linear.weight`` layout ``[out_features, in_features]``.

    ``in_features`` and ``out_features`` should be multiples of 32 for TILE paths.
    """
    if x_bsh.ndim != 3:
        raise ValueError(f"expected x [B,S,in], got {tuple(x_bsh.shape)}")
    if weight_out_in.ndim != 2:
        raise ValueError(f"expected weight [out,in], got {tuple(weight_out_in.shape)}")
    out_f, in_f = int(weight_out_in.shape[0]), int(weight_out_in.shape[1])
    b, s, in_x = int(x_bsh.shape[0]), int(x_bsh.shape[1]), int(x_bsh.shape[2])
    if in_x != in_f:
        raise ValueError(f"x last dim {in_x} != weight in_features {in_f}")

    tt_x = to_tt_x_bsh_flat(mesh_device, x_bsh)
    tt_y = linear_bf16_no_bias_device(mesh_device, tt_x, weight_out_in)
    ttnn.deallocate(tt_x)
    y = tt_flat_to_torch_bsh(mesh_device, tt_y, b=b, s=s, out_f=out_f)
    ttnn.deallocate(tt_y)
    return y


def linear_bf16_no_bias_reference_torch(x_bsh: torch.Tensor, weight_out_in: torch.Tensor) -> torch.Tensor:
    """CPU reference: ``F.linear(x, weight)`` with HF weight layout ``[out, in]``."""
    return F.linear(x_bsh.to(torch.bfloat16), weight_out_in.to(torch.bfloat16), bias=None)


def to_tt_x_bsh_flat(mesh_device, x_bsh: torch.Tensor) -> ttnn.Tensor:
    """Host ``[B,S,H]`` bf16 → device ``[1, 1, B*S, H]`` TILE (replicated)."""
    if x_bsh.ndim != 3:
        raise ValueError(f"expected x [B,S,H], got {tuple(x_bsh.shape)}")
    b, s, h = (int(x_bsh.shape[0]), int(x_bsh.shape[1]), int(x_bsh.shape[2]))
    x_flat = x_bsh.to(torch.bfloat16).reshape(1, 1, b * s, h)
    return ttnn.from_torch(
        x_flat,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def linear_bf16_no_bias_device(
    mesh_device,
    tt_x_flat: ttnn.Tensor,
    weight_out_in: torch.Tensor,
) -> ttnn.Tensor:
    """
    On-device linear: ``tt_x_flat`` is ``[1, 1, N, in]`` TILE; HF weight ``[out, in]``.

    Returns ``[1, 1, N, out]`` on device. Does **not** deallocate ``tt_x_flat``. Deallocates uploaded weight.
    """
    w_bf16 = weight_out_in.to(torch.bfloat16)
    out_f, in_f = int(w_bf16.shape[0]), int(w_bf16.shape[1])
    w_tt_layout = w_bf16.transpose(0, 1).contiguous().reshape(1, 1, in_f, out_f)
    tt_w = ttnn.from_torch(
        w_tt_layout,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_y = ttnn.linear(
        tt_x_flat,
        tt_w,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=_linear_compute_kernel_config(mesh_device),
    )
    ttnn.deallocate(tt_w)
    return tt_y


def tt_flat_to_torch_bsh(mesh_device, tt_y: ttnn.Tensor, *, b: int, s: int, out_f: int) -> torch.Tensor:
    """Device ``[1,1,B*S,out]`` → host ``[B,S,out]`` bf16."""
    y = ttnn.to_torch(tt_y, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    while y.ndim > 3:
        y = y.squeeze(0)
    y = y.reshape(b, s, out_f)
    return y.to(torch.bfloat16)
