# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host I/O helpers for VAE PCC tests (torch boundaries live here, not in tt modules)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

import ttnn
from models.experimental.hunyuan_image_3_0.ref.vae.encoder import IN_CHANNELS
from models.experimental.hunyuan_image_3_0.tt.vae.decoder import bcthw_to_bthwc, bthwc_to_bcthw
from models.tt_dit.utils.conv3d import aligned_channels


def pad_encoder_channels_bcthw(x_bcthw: torch.Tensor) -> torch.Tensor:
    """Pad RGB input channels to conv3d-aligned width (host-side, before upload)."""
    padded_c = aligned_channels(IN_CHANNELS)
    if x_bcthw.shape[1] >= padded_c:
        return x_bcthw
    return F.pad(x_bcthw, (0, 0, 0, 0, 0, 0, 0, padded_c - x_bcthw.shape[1]))


def upload_bcthw(
    mesh_device: ttnn.MeshDevice,
    z_bcthw: torch.Tensor,
    *,
    dtype: ttnn.DataType = ttnn.bfloat16,
) -> ttnn.Tensor:
    """Host BCTHW -> device BTHWC."""
    host = z_bcthw.bfloat16() if dtype == ttnn.bfloat16 else z_bcthw.float()
    x_bcthw = ttnn.from_torch(
        host,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    x_bthwc = bcthw_to_bthwc(x_bcthw)
    ttnn.deallocate(x_bcthw, force=False)
    return x_bthwc


def upload_bcthw_spatial(
    mesh_device: ttnn.MeshDevice,
    z_bcthw: torch.Tensor,
    *,
    h_mesh_axis: int = 0,
    w_mesh_axis: int = 1,
    dtype: ttnn.DataType = ttnn.bfloat16,
) -> ttnn.Tensor:
    """Host BCTHW -> H/W-sharded device BTHWC (channels already padded on host)."""
    from models.experimental.hunyuan_image_3_0.tt.vae.spatial import mesh_mapper_hw_spatial

    host = (z_bcthw.bfloat16() if dtype == ttnn.bfloat16 else z_bcthw.float()).permute(0, 2, 3, 4, 1).contiguous()
    return ttnn.from_torch(
        host,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper_hw_spatial(mesh_device, h_mesh_axis=h_mesh_axis, w_mesh_axis=w_mesh_axis),
    )


def download_bcthw(mesh_device: ttnn.MeshDevice, tensor_bthwc: ttnn.Tensor) -> torch.Tensor:
    """Device BTHWC -> host BCTHW."""
    x_bcthw = bthwc_to_bcthw(tensor_bthwc)
    out_host = ttnn.to_torch(
        ttnn.from_device(x_bcthw),
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
    )
    ttnn.deallocate(x_bcthw, force=False)
    num_devices = mesh_device.get_num_devices()
    return out_host[: out_host.shape[0] // num_devices].float()


def run_bcthw_module(
    mesh_device: ttnn.MeshDevice,
    module,
    x_bcthw: torch.Tensor,
    *,
    dtype: ttnn.DataType = ttnn.bfloat16,
) -> torch.Tensor:
    """Upload -> module.forward(ttnn) -> download."""
    x_bthwc = upload_bcthw(mesh_device, x_bcthw, dtype=dtype)
    out_bthwc = module(x_bthwc)
    ttnn.deallocate(x_bthwc, force=False)
    result = download_bcthw(mesh_device, out_bthwc)
    ttnn.deallocate(out_bthwc, force=False)
    return result
