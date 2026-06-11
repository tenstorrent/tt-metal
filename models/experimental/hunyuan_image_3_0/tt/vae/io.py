# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host/device tensor I/O for Hunyuan VAE (replicated mesh)."""

from __future__ import annotations

import torch

import ttnn


def bcthw_to_bthwc(z_bcthw: torch.Tensor) -> torch.Tensor:
    return z_bcthw.float().permute(0, 2, 3, 4, 1).contiguous()


def bthwc_to_bcthw(x_bthwc: torch.Tensor) -> torch.Tensor:
    return x_bthwc.permute(0, 4, 1, 2, 3).contiguous()


def upload_bthwc(
    mesh_device: ttnn.MeshDevice,
    tensor_bthwc: torch.Tensor,
    dtype: ttnn.DataType = ttnn.bfloat16,
) -> ttnn.Tensor:
    host = tensor_bthwc.bfloat16() if dtype == ttnn.bfloat16 else tensor_bthwc.float()
    return ttnn.from_torch(
        host,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def download_bthwc(mesh_device: ttnn.MeshDevice, tensor_bthwc: ttnn.Tensor) -> torch.Tensor:
    out_host = ttnn.to_torch(
        ttnn.from_device(tensor_bthwc),
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
    )
    num_devices = mesh_device.get_num_devices()
    return out_host[: out_host.shape[0] // num_devices]


def upload_bcthw(
    mesh_device: ttnn.MeshDevice,
    z_bcthw: torch.Tensor,
    dtype: ttnn.DataType = ttnn.bfloat16,
) -> ttnn.Tensor:
    return upload_bthwc(mesh_device, bcthw_to_bthwc(z_bcthw), dtype=dtype)


def download_bcthw(mesh_device: ttnn.MeshDevice, tensor_bthwc: ttnn.Tensor) -> torch.Tensor:
    return bthwc_to_bcthw(download_bthwc(mesh_device, tensor_bthwc).float())
