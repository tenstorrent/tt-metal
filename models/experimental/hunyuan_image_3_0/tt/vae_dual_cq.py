# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Two-command-queue (2CQ) coordination for HunyuanImage-3.0 VAE decode.
#
# CQ0: VAE decoder forward
# CQ1: async RGB output D2H (1024² image is the large transfer)

from __future__ import annotations


import ttnn

from models.experimental.hunyuan_image_3_0.tt.ar_dual_cq import (
    COMPUTE_CQ,
    IO_CQ,
    device_num_command_queues,
)


def vae_2cq_enabled(device) -> bool:
    from models.experimental.hunyuan_image_3_0.tt.trace_config import vae_2cq_enabled as _vae_2cq

    return _vae_2cq(device)


def open_vae_mesh(mesh_shape, *, l1_small_size: int = 32768, enable_2cq: bool | None = None):
    """Open a mesh for VAE decode with optional two command queues."""
    from models.experimental.hunyuan_image_3_0.tt.trace_config import hy_trace_enabled, open_traced_mesh

    if enable_2cq is None:
        enable_2cq = hy_trace_enabled()
    num_cq = 2 if enable_2cq else 1
    mesh = open_traced_mesh(mesh_shape, l1_small_size=l1_small_size, num_cq=num_cq)
    if enable_2cq and device_num_command_queues(mesh) < 2:
        print("[vae] warning: requested 2CQ but mesh opened with 1 CQ", flush=True)
    elif enable_2cq:
        print(
            f"[vae] 2CQ enabled (HY_TRACE=1, num_command_queues={device_num_command_queues(mesh)})",
            flush=True,
        )
    return mesh


class VaeDualCQCoordinator:
    """CQ0 VAE decode + CQ1 async output D2H."""

    def __init__(self, device):
        self.device = device
        if device_num_command_queues(device) < 2:
            raise ValueError(
                f"VaeDualCQCoordinator requires num_command_queues>=2, got {device_num_command_queues(device)}"
            )
        self._read_event = None
        self._write_event = None
        self._pending_host = None
        self._pending_dev_tt = None
        self.d2h_transfers = 0

    def fence_compute_before_forward(self) -> None:
        """CQ0: wait for any prior CQ1 work before decoder forward."""
        if self._write_event is not None:
            ttnn.wait_for_event(COMPUTE_CQ, self._write_event)
            self._write_event = None

    def fence_io_before_launch(self) -> None:
        """CQ1: wait for the previous async D2H before reusing the host buffer."""
        if self._read_event is not None:
            ttnn.wait_for_event(IO_CQ, self._read_event)
            self._read_event = None

    def launch_output_d2h(self, output_tt: ttnn.Tensor) -> None:
        """After decoder forward on CQ0, enqueue async output D2H on CQ1."""
        if self._pending_host is not None:
            raise RuntimeError("launch_output_d2h called before consume_output_host")
        self.fence_io_before_launch()
        compute_done = ttnn.record_event(self.device, COMPUTE_CQ)
        ttnn.wait_for_event(IO_CQ, compute_done)
        self._pending_host = ttnn.from_device(output_tt, blocking=False, cq_id=IO_CQ)
        self._pending_dev_tt = output_tt
        self._read_event = ttnn.record_event(self.device, IO_CQ)
        self.d2h_transfers += 1

    def consume_output_host(self) -> ttnn.Tensor:
        """Synchronize CQ1 D2H and return the host output tensor."""
        if self._pending_host is None:
            raise RuntimeError("consume_output_host called with no pending D2H")
        ttnn.event_synchronize(self._read_event)
        self._read_event = None
        host = self._pending_host
        self._pending_host = None
        if self._pending_dev_tt is not None:
            ttnn.deallocate(self._pending_dev_tt, force=False)
            self._pending_dev_tt = None
        return host
