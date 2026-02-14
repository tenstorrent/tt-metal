# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

"""
DeepSeek V3 B1 model (host interface).

Orchestrates prefill (token-by-token prompt processing) and autoregressive decode
over the socket interface. Uses HostInterface in loopback mode as a mock decoder
until the real decoder pipeline is composed from Pre-SDPA, Flash MLA, Post-SDPA, MoE.

Algorithm (prefill-by-decode then generation):
  - Prefill: for i = 0..S-1, call with input_ids = x[i] (B, 1); device uses/updates
    cache; ignore logits for i < S-1. prefill() returns the last step output (logits
    in real decoder) so caller can sample y0.
  - Start generation: last_logits = prefill(prompt_tokens); y0 = sample(last_logits).
  - Generation loop: for t = 0,1,..., feed y[t] (B, 1) via decode_step(), get logits,
    sample y[t+1], repeat.

Input tensor shape (H2D):
  - Only (B, 1) is supported: one token per batch element per step. The embedding layer
    runs on device; the host sends token IDs (int32). Payload size is B * TOKEN_ID_BYTES.

Interface vs real decoder:
  - One H2D write and one D2H read per step. The real decoder will also need per-step
    position (cur_pos_tensor / kv_cache_write_index); the engine tracks position for
    when the protocol is extended.
"""

import torch

import ttnn
from models.demos.deepseek_v3_b1.micro_ops.host_io.op import HostInterface

# Token IDs are int32 over the socket; payload size per step is B * TOKEN_ID_BYTES.
TOKEN_ID_BYTES: int = torch.tensor(0, dtype=torch.int32).element_size()

# Socket page_size must be PCIe-aligned (see h2d_socket.cpp). Use 64 to work across devices.
PCIE_PAGE_ALIGNMENT_BYTES: int = 64


def align_up(value: int, alignment: int) -> int:
    """Round value up to the next multiple of alignment."""
    return (value + alignment - 1) // alignment * alignment


def page_size_bytes(batch_size: int) -> int:
    """PCIe-aligned page (and FIFO) size in bytes for (B, 1) token IDs. Use for socket creation."""
    return align_up(batch_size * TOKEN_ID_BYTES, PCIE_PAGE_ALIGNMENT_BYTES)


def create_output_buffer(page_size_datums: int) -> ttnn.Tensor:
    """Allocate a host output tensor (1, page_size_datums) int32 for socket read_tensor."""
    torch_output = torch.zeros(1, page_size_datums, dtype=torch.int32)
    return ttnn.from_torch(torch_output, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)


def to_padded_input(
    token: torch.Tensor | ttnn.Tensor,
    batch_size: int,
    page_size_datums: int,
) -> ttnn.Tensor:
    """Copy (B, 1) token into a PCIe-aligned padded buffer for write_tensor."""
    if isinstance(token, ttnn.Tensor):
        token = ttnn.to_torch(token)
    torch_padded = torch.zeros(1, page_size_datums, dtype=torch.int32)
    torch_padded[0, :batch_size] = token.flatten()[:batch_size]
    return ttnn.from_torch(torch_padded, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)


class DeepSeekV3:
    """
    Host-side model interface for prefill and decode over H2D/D2H sockets.
    Tracks position for compatibility with the real decoder (position_ids, kv_cache_write_index).
    """

    def __init__(
        self,
        h2d_socket: ttnn.H2DSocket,
        d2h_socket: ttnn.D2HSocket,
        batch_size: int,
        h2d_mode: ttnn.H2DMode,
    ) -> None:
        """
        Args:
            h2d_socket: ttnn.H2DSocket for host-to-device communication.
            d2h_socket: ttnn.D2HSocket for device-to-host communication.
            batch_size: Batch size B. Only (B, 1) token IDs per step are supported;
                payload size is B * TOKEN_ID_BYTES (int32).
            h2d_mode: ttnn.H2DMode (HOST_PUSH or DEVICE_PULL).

        Sockets must be created with FIFO size equal to page_size_bytes(batch_size).
        """
        self.h2d_socket = h2d_socket
        self.d2h_socket = d2h_socket
        self.batch_size = batch_size
        payload_bytes: int = batch_size * TOKEN_ID_BYTES
        self._tensor_size_bytes: int = align_up(payload_bytes, PCIE_PAGE_ALIGNMENT_BYTES)
        self._page_size_datums: int = self._tensor_size_bytes // TOKEN_ID_BYTES
        self.h2d_mode = h2d_mode

        self._host_io: HostInterface = HostInterface(
            h2d_socket,
            d2h_socket,
            self._tensor_size_bytes,
            core_to_core_socket_buffer_size=self._tensor_size_bytes,
            loopback_mode=True,
        )
        self._position: int = 0
        self._output_buffer: ttnn.Tensor = create_output_buffer(self._page_size_datums)

    def start(self) -> None:
        """Launch the mock decoder program on device (HostInterface in loopback mode)."""
        self._host_io.run()

    def prefill(self, prompt_tokens: list[torch.Tensor | ttnn.Tensor]) -> ttnn.Tensor | None:
        """
        Prefill-by-decode: for i = 0..S-1, send input_ids = x[i] (B, 1), get logits
        (and device updates cache). Outputs for i < S-1 are discarded. Returns the
        last step output so the caller can sample y0 (first generated token).

        Args:
            prompt_tokens: List of token ID tensors, each of shape (B, 1) int32.
                Each is sent over H2D; device runs embedding + decoder.

        Returns:
            Last step output tensor; valid data is first batch_size elements (logits
            (B, V) in real decoder). None if prompt_tokens is empty. Caller uses this
            to sample(logits) -> y0 for the generation loop.
        """
        last_output: ttnn.Tensor | None = None
        for token in prompt_tokens:
            input_ttnn = to_padded_input(token, self.batch_size, self._page_size_datums)
            self.h2d_socket.write_tensor(input_ttnn)
            self.d2h_socket.read_tensor(self._output_buffer)
            last_output = self._output_buffer
            self._position += 1
        return last_output

    def decode_step(self, input_tensor: torch.Tensor | ttnn.Tensor) -> ttnn.Tensor:
        """
        Single decode step: send input via H2D, receive output via D2H.
        Returns the output tensor. Increments position.

        Args:
            input_tensor: Token IDs (B, 1), torch or ttnn.

        Returns:
            Output tensor; valid data is first batch_size elements. In loopback mode
            that matches the input.
        """
        padded_input = to_padded_input(input_tensor, self.batch_size, self._page_size_datums)
        self.h2d_socket.write_tensor(padded_input)
        self.d2h_socket.read_tensor(self._output_buffer)
        self._position += 1
        return self._output_buffer

    @property
    def position(self) -> int:
        """Current sequence position (number of tokens processed so far)."""
        return self._position

    def stop(self) -> None:
        """Clean shutdown of the mock decoder."""
        self._host_io.terminate()
