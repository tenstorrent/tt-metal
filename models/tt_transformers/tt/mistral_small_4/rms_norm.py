# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""RMSNorm matching Hugging Face ``Mistral4RMSNorm`` (T5-style, no mean centering)."""

from __future__ import annotations

import torch

import ttnn

_SHARD_HEIGHT = 32  # ttnn RMSNorm weight tiling (see ``models.common.rmsnorm``)
_CK_ATTR = "_mistral_small_4_rms_norm_compute_kernel_config"


def _norm_compute_kernel_config(mesh_device):
    # One config per mesh (avoids extra nanobind-wrapped objects every call).
    cfg = getattr(mesh_device, _CK_ATTR, None)
    if cfg is None:
        cfg = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        setattr(mesh_device, _CK_ATTR, cfg)
    return cfg


def rms_norm_bf16(
    mesh_device,
    hidden_states_bsh: torch.Tensor,
    weight_1d: torch.Tensor,
    *,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """
    Run RMSNorm on device; return host ``torch.Tensor`` ``[B, S, H]`` in bfloat16.

    Reference: ``transformers.models.mistral4.modeling_mistral4.Mistral4RMSNorm``.
    """
    if hidden_states_bsh.ndim != 3:
        raise ValueError(f"expected hidden_states [B,S,H], got shape {tuple(hidden_states_bsh.shape)}")
    if weight_1d.ndim != 1 or int(weight_1d.shape[0]) != int(hidden_states_bsh.shape[-1]):
        raise ValueError(f"expected weight [H] with H={hidden_states_bsh.shape[-1]}, got {tuple(weight_1d.shape)}")

    b, s, h = hidden_states_bsh.shape
    if h % _SHARD_HEIGHT != 0:
        raise ValueError(
            f"hidden_size {h} must be a multiple of {_SHARD_HEIGHT} for current ttnn RMSNorm weight layout"
        )

    x_bf16 = hidden_states_bsh.to(torch.bfloat16)
    w_bf16 = weight_1d.to(torch.bfloat16)
    w_tiled = w_bf16.unsqueeze(0).view(1, 1, h // _SHARD_HEIGHT, _SHARD_HEIGHT)

    tt_x = ttnn.from_torch(
        x_bf16,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_w = ttnn.from_torch(
        w_tiled,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_y = ttnn.rms_norm(
        tt_x,
        weight=tt_w,
        epsilon=epsilon,
        compute_kernel_config=_norm_compute_kernel_config(mesh_device),
    )
    ttnn.deallocate(tt_x)
    ttnn.deallocate(tt_w)

    y = ttnn.to_torch(tt_y, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    ttnn.deallocate(tt_y)

    while y.ndim > 3:
        y = y.squeeze(0)
    if y.ndim == 3 and y.shape[0] != b:
        y = y[:b]
    assert y.shape == (b, s, h), f"unexpected output shape {tuple(y.shape)}"
    return y.to(torch.bfloat16)
