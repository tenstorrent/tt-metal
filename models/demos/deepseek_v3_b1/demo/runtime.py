# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
from typing import Generator

import torch

import ttnn
from models.demos.deepseek_v3_b1.micro_ops.host_io.op import HostInterface
from models.demos.deepseek_v3_b1.model import TOKEN_ID_BYTES, DeepSeekV3, page_size_bytes, to_padded_input


@contextlib.contextmanager
def create_model(
    mesh_device: ttnn.MeshDevice,
    *,
    batch_size: int = 1,
) -> Generator[DeepSeekV3, None, None]:
    """
    Context manager that creates sockets and HostInterface (loopback mode), runs it,
    yields a DeepSeekV3 wired to write_fn/read_fn, then terminates HostInterface on exit.
    """
    fifo_size = page_size_bytes(batch_size)
    socket_core = ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(0, 0))
    h2d_socket = ttnn.H2DSocket(
        mesh_device,
        socket_core,
        ttnn.BufferType.L1,
        fifo_size,
        ttnn.H2DMode.HOST_PUSH,
    )
    d2h_socket = ttnn.D2HSocket(mesh_device, socket_core, fifo_size)
    host_io = HostInterface(
        h2d_socket,
        d2h_socket,
        fifo_size,
        fifo_size,
        core_to_core_socket_buffer_size=fifo_size,
        loopback_mode=True,
    )
    host_io.run()
    try:
        model = DeepSeekV3(
            write_fn=h2d_socket.write_tensor,
            read_fn=d2h_socket.read_tensor,
            batch_size=batch_size,
        )
        yield model
    finally:
        host_io.terminate(sync_devices=True)


class TokenCodec:
    def __init__(self, *, batch_size: int) -> None:
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        self.page_size_datums = page_size_bytes(self.batch_size) // TOKEN_ID_BYTES

    def make_input(self, token_id: int) -> ttnn.Tensor:
        torch_token = torch.full((self.batch_size, 1), int(token_id), dtype=torch.int32)
        return to_padded_input(torch_token, batch_size=self.batch_size, page_size_datums=self.page_size_datums)

    def make_prefill_inputs(self, token_ids: list[int]) -> list[ttnn.Tensor]:
        return [self.make_input(token_id) for token_id in token_ids]

    def extract_token_id(self, output_tensor: ttnn.Tensor) -> int:
        torch_output = ttnn.to_torch(output_tensor).reshape(-1)
        return int(torch_output[0].item())
