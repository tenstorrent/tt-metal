# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import ttnn.experimental

import ttnn
from models.tt_dit.utils import tensor

if TYPE_CHECKING:
    import torch

    from models.tt_dit.parallel.config import DiTParallelConfig


class CFGCombiner:
    """Classifier-free guidance combiner.

    Operates either on a single device or on two devices using mesh sockets for communication.
    Assumes that the unconditional prediction comes first - either in the first half of the batch
    dimension (single-device case) or on the first device (multi-device case).
    """

    def __init__(self, /, devices: tuple[ttnn.MeshDevice] | tuple[ttnn.MeshDevice, ttnn.MeshDevice]) -> None:
        match devices:
            case (_,):
                self._inner = _CombinerSingle()
            case (uncond_device, cond_device):
                self._inner = _CombinerParallel(uncond_device, cond_device)
            case _:
                msg = "devices must be a tuple of one or two MeshDevices"
                raise ValueError(msg)

    def combine(self, prediction: ttnn.Tensor, cfg_scale: float) -> ttnn.Tensor:
        """Compute the CFG-combined noise prediction on one submesh device."""
        return self._inner.combine(prediction, cfg_scale)


class _CombinerSingle:
    def combine(self, prediction: ttnn.Tensor, cfg_scale: float) -> ttnn.Tensor:
        n = prediction.shape[0]

        if n % 2 != 0:
            msg = "batch dimension must be even when using single-device CFGCombiner"
            raise ValueError(msg)

        split_pos = n // 2
        uncond = prediction[0:split_pos]
        cond = prediction[split_pos:]
        return uncond + cfg_scale * (cond - uncond)


class _CombinerParallel:
    def __init__(self, uncond_device: ttnn.MeshDevice, cond_device: ttnn.MeshDevice) -> None:
        self._devices = (uncond_device, cond_device)
        self._sockets = _create_sockets(uncond_device, cond_device)
        self._recv_buffers: list[ttnn.Tensor | None] = [None, None]

    def combine(self, local_pred: ttnn.Tensor, cfg_scale: float) -> ttnn.Tensor:
        idx = 0 if local_pred.device() == self._devices[0] else 1

        remote_pred = self._recv_buffers[idx]
        if remote_pred is None:
            remote_pred = ttnn.allocate_tensor_on_device(local_pred.spec, self._devices[idx])
            self._recv_buffers[idx] = remote_pred

        # call in different order on each device to avoid deadlock
        if idx == 0:
            ttnn.experimental.send_async(local_pred, self._sockets[0].tx)
            ttnn.experimental.recv_async(remote_pred, self._sockets[1].rx)
        else:
            ttnn.experimental.recv_async(remote_pred, self._sockets[0].rx)
            ttnn.experimental.send_async(local_pred, self._sockets[1].tx)

        uncond, cond = (local_pred, remote_pred) if idx == 0 else (remote_pred, local_pred)
        return uncond + cfg_scale * (cond - uncond)


@dataclass
class _SocketPair:
    tx: ttnn.MeshSocket
    rx: ttnn.MeshSocket


def _create_sockets(device0: ttnn.MeshDevice, device1: ttnn.MeshDevice) -> tuple[_SocketPair, _SocketPair]:
    """Create two unidirectional socket pairs."""
    if device0.shape != device1.shape:
        msg = "devices must have the same mesh shape to create sockets"
        raise ValueError(msg)

    connections = [
        ttnn.SocketConnection(
            ttnn.MeshCoreCoord(coord, ttnn.CoreCoord(0, 0)),
            ttnn.MeshCoreCoord(coord, ttnn.CoreCoord(0, 0)),
        )
        for coord in ttnn.MeshCoordinateRange(device0.shape)
    ]
    socket_config = ttnn.SocketConfig(connections, ttnn.SocketMemoryConfig(ttnn.BufferType.L1, 4096))
    tx_0to1, rx_0to1 = ttnn.create_socket_pair(device0, device1, socket_config)
    tx_1to0, rx_1to0 = ttnn.create_socket_pair(device1, device0, socket_config)

    return _SocketPair(tx_0to1, rx_0to1), _SocketPair(tx_1to0, rx_1to0)


def create_submeshes(
    device: ttnn.MeshDevice, parallel_config: DiTParallelConfig
) -> tuple[ttnn.MeshDevice] | tuple[ttnn.MeshDevice, ttnn.MeshDevice]:
    """Slice the mesh into cfg-parallel submeshes sized for tensor and sequence parallelism."""
    tp = parallel_config.tensor_parallel
    sp = parallel_config.sequence_parallel
    cp = parallel_config.cfg_parallel

    if cp.factor not in (1, 2):
        msg = "cfg parallel factor must be 1 or 2"
        raise ValueError(msg)

    submesh_shape = [1] * device.shape.dims()
    submesh_shape[sp.mesh_axis] *= sp.factor
    submesh_shape[tp.mesh_axis] *= tp.factor

    devices = device.create_submeshes(ttnn.MeshShape(*submesh_shape))
    if len(devices) < cp.factor:
        msg = f"not enough submeshes created: expected {cp.factor}, got {len(devices)}"
        raise ValueError(msg)

    return (devices[0],) if cp.factor == 1 else (devices[0], devices[1])


def distribute_cfg(
    x: torch.Tensor,
    /,
    *,
    devices: tuple[ttnn.MeshDevice] | tuple[ttnn.MeshDevice, ttnn.MeshDevice],
    on_host: bool = False,
) -> tuple[ttnn.Tensor] | tuple[ttnn.Tensor, ttnn.Tensor]:
    """Return one tensor per submesh from a conditioning batch."""
    match devices:
        case [device]:
            return (tensor.from_torch(x, device=device, on_host=on_host),)
        case [device1, device2]:
            half = x.shape[0] // 2
            return (
                tensor.from_torch(x[:half], device=device1, on_host=on_host),
                tensor.from_torch(x[half:], device=device2, on_host=on_host),
            )
        case _:
            msg = f"unsupported number of submeshes: expected 1 or 2, got {len(devices)}"
            raise ValueError(msg)
