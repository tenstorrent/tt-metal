# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import ttnn.experimental

import ttnn
from models.tt_dit.parallel.config import DiTParallelConfig
from models.tt_dit.utils import tensor


class CFGCombiner:
    """Classifier-free guidance combiner.

    Operates either on a single device or on two devices. Assumes that the unconditional prediction
    comes first - either in the first half of the batch dimension (single-device case) or on the
    first device (multi-device case).

    """

    def __init__(
        self,
        devices: tuple[ttnn.MeshDevice] | tuple[ttnn.MeshDevice, ttnn.MeshDevice],
        *,
        via_host: bool = True,
    ) -> None:
        match devices:
            case (_,):
                self._inner = _SingleCombiner()
            case (uncond_device, cond_device):
                if via_host:
                    self._inner = _HostCombiner(uncond_device, cond_device)
                else:
                    self._inner = _SocketCombiner(uncond_device, cond_device)
            case _:
                msg = "devices must be a tuple of one or two MeshDevices"
                raise ValueError(msg)

    def combine(self, predictions: Sequence[ttnn.Tensor], cfg_scale: float) -> tuple[ttnn.Tensor, ...]:
        """Compute the CFG-combined noise prediction on each submesh device."""
        return self._inner.combine(predictions, cfg_scale)


class _SingleCombiner:
    def combine(self, predictions: Sequence[ttnn.Tensor], cfg_scale: float) -> tuple[ttnn.Tensor]:
        if len(predictions) != 1:
            msg = f"expected 1 prediction tensor for single-device combiner, got {len(predictions)}"
            raise ValueError(msg)

        prediction = predictions[0]
        n = prediction.shape[0]

        if n % 2 != 0:
            msg = "batch dimension must be even when using single-device CFGCombiner"
            raise ValueError(msg)

        split_pos = n // 2
        uncond = prediction[0:split_pos]
        cond = prediction[split_pos:]
        combined = ttnn.lerp(uncond, cond, cfg_scale)

        return (combined,)


class _HostCombiner:
    def __init__(self, uncond_device: ttnn.MeshDevice, cond_device: ttnn.MeshDevice) -> None:
        self._uncond_device = uncond_device
        self._cond_device = cond_device

    def combine(self, predictions: Sequence[ttnn.Tensor], cfg_scale: float) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        if len(predictions) != 2:
            msg = f"expected 2 prediction tensors for host-based combiner, got {len(predictions)}"
            raise ValueError(msg)

        uncond, cond = predictions

        received_uncond = uncond.cpu(blocking=False)
        received_cond = cond.cpu(blocking=False)

        ttnn.synchronize_device(self._uncond_device)
        ttnn.synchronize_device(self._cond_device)

        received_uncond = received_uncond.to(self._cond_device)
        received_cond = received_cond.to(self._uncond_device)

        combined0 = ttnn.lerp(uncond, received_cond, cfg_scale)
        combined1 = ttnn.lerp(received_uncond, cond, cfg_scale)

        return combined0, combined1


class _SocketCombiner:
    def __init__(self, uncond_device: ttnn.MeshDevice, cond_device: ttnn.MeshDevice) -> None:
        self._devices = (uncond_device, cond_device)
        self._sockets = _create_sockets(uncond_device, cond_device)
        self._recv_buffers: tuple[ttnn.Tensor, ttnn.Tensor] | None = None

    def combine(self, predictions: Sequence[ttnn.Tensor], cfg_scale: float) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        if len(predictions) != 2:
            msg = f"expected 2 prediction tensors for socket-based combiner, got {len(predictions)}"
            raise ValueError(msg)

        if self._recv_buffers is None:
            self._recv_buffers = (
                ttnn.allocate_tensor_on_device(predictions[0].spec, self._devices[0]),
                ttnn.allocate_tensor_on_device(predictions[1].spec, self._devices[1]),
            )

        uncond, cond = predictions
        received_cond, received_uncond = self._recv_buffers

        # on device 0: send uncond, receive cond
        ttnn.experimental.send_async(uncond, self._sockets[0].tx)
        ttnn.experimental.recv_async(received_cond, self._sockets[1].rx)

        # on device 1: receive uncond, send cond
        ttnn.experimental.recv_async(received_uncond, self._sockets[0].rx)
        ttnn.experimental.send_async(cond, self._sockets[1].tx)

        combined0 = ttnn.lerp(uncond, received_cond, cfg_scale)
        combined1 = ttnn.lerp(received_uncond, cond, cfg_scale)

        return combined0, combined1


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
