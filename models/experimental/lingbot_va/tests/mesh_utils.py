# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Lingbot-VA mesh shape, inference submesh, VAE/UMT5 parallel helpers.

Matches tt-metal root ``conftest`` ``mesh_device`` / ``MESH_DEVICE`` and Lingbot PCC tests.
"""

from __future__ import annotations

import os
import torch
import ttnn

from models.tt_dit.parallel.config import EncoderParallelConfig, ParallelFactor, VaeHWParallelConfig
from models.tt_dit.parallel.manager import CCLManager

_MESH_DEVICE_SHAPES = {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}


def _mesh_shape_from_lingbot_va_mesh_shape_env() -> ttnn.MeshShape | None:
    """Parse ``LINGBOT_VA_MESH_SHAPE`` (``R,C`` or ``N`` for ``1×N``); ``None`` if unset."""
    raw = os.environ.get("LINGBOT_VA_MESH_SHAPE", "").strip()
    if not raw:
        return None
    normalized = raw.replace("x", ",").replace("*", ",").replace(" ", "")
    parts = [int(p) for p in normalized.split(",") if p]
    if len(parts) == 1:
        return ttnn.MeshShape(1, parts[0])
    if len(parts) == 2:
        return ttnn.MeshShape(parts[0], parts[1])
    raise ValueError(f"LINGBOT_VA_MESH_SHAPE={raw!r}: expected one int (1×N) or two (rows,cols), e.g. '1,8' or '8x4'")


def mesh_shape_request_param() -> tuple[int, int] | int:
    """Pytest indirect ``mesh_device`` value: ``(rows, cols)`` or ``N`` for ``(1, N)`` from device count."""
    return _MESH_DEVICE_SHAPES.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))


def ttnn_mesh_shape_from_env() -> ttnn.MeshShape:
    """``MeshShape`` for ``ttnn.open_mesh_device`` (same rules as ``mesh_shape_request_param``)."""
    param = mesh_shape_request_param()
    if isinstance(param, tuple):
        return ttnn.MeshShape(*param)
    return ttnn.MeshShape(1, param)


def ttnn_mesh_shape_for_inference_demo() -> ttnn.MeshShape:
    """Demo open shape: ``LINGBOT_VA_MESH_SHAPE`` if set, else ``ttnn_mesh_shape_from_env()``."""
    override = _mesh_shape_from_lingbot_va_mesh_shape_env()
    if override is not None:
        return override
    return ttnn_mesh_shape_from_env()


def inference_work_mesh_from_opened(
    opened_mesh: ttnn.MeshDevice,
) -> tuple[ttnn.MeshDevice, ttnn.MeshDevice | None]:
    """``(work_mesh, parent)``; with ``LINGBOT_VA_INFERENCE_SINGLE_CHIP_MESH=1``, work is a ``(1,1)`` submesh."""
    single = os.environ.get("LINGBOT_VA_INFERENCE_SINGLE_CHIP_MESH", "").strip().lower() in ("1", "true", "yes")
    if single and opened_mesh.get_num_devices() > 1:
        sub = opened_mesh.create_submesh(ttnn.MeshShape(1, 1))
        return sub, opened_mesh
    return opened_mesh, None


def mesh_num_devices(mesh_device: ttnn.MeshDevice) -> int:
    return mesh_device.get_num_devices()


def vae_hw_parallel_config_for_mesh(mesh_device: ttnn.MeshDevice) -> VaeHWParallelConfig:
    if mesh_num_devices(mesh_device) <= 1:
        return VaeHWParallelConfig(
            height_parallel=ParallelFactor(factor=1, mesh_axis=0),
            width_parallel=ParallelFactor(factor=1, mesh_axis=1),
        )
    rows, cols = tuple(mesh_device.shape)
    return VaeHWParallelConfig(
        height_parallel=ParallelFactor(factor=rows, mesh_axis=0),
        width_parallel=ParallelFactor(factor=cols, mesh_axis=1),
    )


def vae_bthwc_to_torch(
    tt_BTHWC: ttnn.Tensor,
    mesh_device: ttnn.MeshDevice,
    parallel_config: VaeHWParallelConfig,
    ccl_manager: CCLManager,
) -> torch.Tensor:
    """Host readback for ``[B,T,H,W,C]`` VAE activations; all-gathers H/W when spatially sharded on mesh."""
    if mesh_num_devices(mesh_device) <= 1:
        return ttnn.to_torch(tt_BTHWC)
    x = ttnn.to_layout(tt_BTHWC, ttnn.TILE_LAYOUT)
    if parallel_config.height_parallel.factor > 1:
        x = ccl_manager.all_gather_persistent_buffer(
            x, dim=2, mesh_axis=parallel_config.height_parallel.mesh_axis, use_hyperparams=True
        )
    if parallel_config.width_parallel.factor > 1:
        x = ccl_manager.all_gather_persistent_buffer(
            x, dim=3, mesh_axis=parallel_config.width_parallel.mesh_axis, use_hyperparams=True
        )
    x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    return ttnn.to_torch(ttnn.get_device_tensors(x)[0])


def encoder_parallel_config_for_mesh(mesh_device: ttnn.MeshDevice) -> EncoderParallelConfig:
    cols = tuple(mesh_device.shape)[1]
    return EncoderParallelConfig(tensor_parallel=ParallelFactor(factor=cols, mesh_axis=1))


def _umt5_dp_axis_and_count(
    mesh_device: ttnn.MeshDevice, encoder_parallel_config: EncoderParallelConfig
) -> tuple[int, int]:
    tp_axis = encoder_parallel_config.tensor_parallel.mesh_axis
    dp_axis = 1 - tp_axis
    return dp_axis, tuple(mesh_device.shape)[dp_axis]


def umt5_mesh_mapper_for_text_inputs(
    mesh_device: ttnn.MeshDevice, encoder_parallel_config: EncoderParallelConfig
) -> ttnn.ReplicateTensorToMesh | ttnn.ShardTensor2dMesh:
    """Replicate on ``(1,N)``; else shard batch on the DP mesh axis."""
    dp_axis, num_dp = _umt5_dp_axis_and_count(mesh_device, encoder_parallel_config)
    if num_dp > 1:
        dims: list = [None, None]
        dims[dp_axis] = 0
        return ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=dims)
    return ttnn.ReplicateTensorToMesh(mesh_device)


def umt5_pad_input_ids_and_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    mesh_device: ttnn.MeshDevice,
    encoder_parallel_config: EncoderParallelConfig,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Pad batch to a multiple of the DP axis size; returns ``(ids, mask, pad_n)``."""
    _, num_dp = _umt5_dp_axis_and_count(mesh_device, encoder_parallel_config)
    b = input_ids.shape[0]
    pad_n = (num_dp - (b % num_dp)) % num_dp
    if pad_n == 0:
        return input_ids, attention_mask, 0
    _, seq_len = input_ids.shape
    pad_ids = torch.zeros((pad_n, seq_len), dtype=input_ids.dtype, device=input_ids.device)
    pad_mask = torch.zeros((pad_n, seq_len), dtype=attention_mask.dtype, device=attention_mask.device)
    return torch.cat([input_ids, pad_ids], dim=0), torch.cat([attention_mask, pad_mask], dim=0), pad_n


def umt5_post_encoder_hidden_states(
    ccl_manager: CCLManager,
    last_hidden: ttnn.Tensor,
    tt_mask: ttnn.Tensor,
    mesh_device: ttnn.MeshDevice,
    encoder_parallel_config: EncoderParallelConfig,
) -> ttnn.Tensor:
    """Mask on device; DP ``all_gather`` on batch when ``num_dp > 1`` (no TP gather—FFN already full width)."""
    out = last_hidden * ttnn.unsqueeze(tt_mask, -1)
    dp_axis, num_dp = _umt5_dp_axis_and_count(mesh_device, encoder_parallel_config)
    if num_dp > 1:
        out = ccl_manager.all_gather(out, dim=0, mesh_axis=dp_axis, use_hyperparams=True)
    return out


def umt5_encoder_hidden_states_to_torch(tt_tensor: ttnn.Tensor) -> torch.Tensor:
    """Single-device readback for mesh-backed encoder output (see ``test_umt5.py``)."""
    return ttnn.to_torch(ttnn.get_device_tensors(tt_tensor)[0])
