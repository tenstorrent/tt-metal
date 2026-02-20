# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

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
from loguru import logger

import ttnn
from models.demos.deepseek_v3_b1.micro_ops.host_io.op import HostInterface

# Token IDs are int32 over the socket; payload size per step is B * TOKEN_ID_BYTES.
TOKEN_ID_BYTES: int = 4

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

    Prefill uses DEVICE_PULL H2D; decode uses HOST_PUSH H2D. Two HostInterface programs
    run one at a time on the same core; the first decode_step() transitions from prefill
    to decode host I/O.
    """

    def __init__(
        self,
        h2d_socket_prefill: ttnn.H2DSocket,
        h2d_socket_decode: ttnn.H2DSocket,
        d2h_socket: ttnn.D2HSocket,
        batch_size: int = 1,
        loopback_mode: bool = False,
    ) -> None:
        """
        Args:
            h2d_socket_prefill: H2D socket for prefill; must use DEVICE_PULL mode.
            h2d_socket_decode: H2D socket for decode; must use HOST_PUSH mode.
            d2h_socket: D2H socket for device-to-host (shared by prefill and decode).
            batch_size: Batch size B. Current implementation supports only B=1;
                payload size is B * TOKEN_ID_BYTES (int32).
            loopback_mode: If True, host I/O uses circular buffers for H2D/D2H loopback
                If False, host I/O forwards to downstream/upstream cores via D2D sockets.

        Sockets must be created with FIFO size equal to page_size_bytes(batch_size).
        """
        if batch_size != 1:
            raise ValueError(f"DeepSeekV3 currently supports only batch_size=1, got {batch_size}")
        if h2d_socket_prefill.get_h2d_mode() != ttnn.H2DMode.DEVICE_PULL:
            raise ValueError(
                "h2d_socket_prefill must use H2DMode.DEVICE_PULL, got " f"{h2d_socket_prefill.get_h2d_mode()}"
            )
        if h2d_socket_decode.get_h2d_mode() != ttnn.H2DMode.HOST_PUSH:
            raise ValueError("h2d_socket_decode must use H2DMode.HOST_PUSH, got " f"{h2d_socket_decode.get_h2d_mode()}")

        self.h2d_socket_prefill = h2d_socket_prefill
        self.h2d_socket_decode = h2d_socket_decode
        self.d2h_socket = d2h_socket
        self.batch_size = batch_size
        payload_bytes: int = batch_size * TOKEN_ID_BYTES
        logger.debug(f"Payload bytes: {payload_bytes} bytes")
        self._tensor_size_bytes: int = align_up(payload_bytes, PCIE_PAGE_ALIGNMENT_BYTES)
        self._page_size_datums: int = self._tensor_size_bytes // TOKEN_ID_BYTES

        logger.debug(f"Creating DeepSeekV3 model with batch size {batch_size}")
        logger.debug(
            f"Creating host I/O for prefill and decode with aligned page size {self._tensor_size_bytes} bytes (payload size is {payload_bytes} bytes)"
        )

        self._host_io_prefill: HostInterface = HostInterface(
            h2d_socket_prefill,
            d2h_socket,
            self._tensor_size_bytes,
            self._tensor_size_bytes,
            core_to_core_socket_buffer_size=self._tensor_size_bytes,
            loopback_mode=loopback_mode,
        )
        self._host_io_decode: HostInterface = HostInterface(
            h2d_socket_decode,
            d2h_socket,
            self._tensor_size_bytes,
            self._tensor_size_bytes,
            core_to_core_socket_buffer_size=self._tensor_size_bytes,
            loopback_mode=loopback_mode,
        )
        self._prefill_active: bool = False
        self._decode_active: bool = False
        self._position: int = 0
        self._output_buffer: ttnn.Tensor = create_output_buffer(self._page_size_datums)

    def start(self) -> None:
        """Launch the prefill mock decoder program (DEVICE_PULL H2D) on device."""
        self._host_io_prefill.run()
        self._prefill_active = True
        self._decode_active = False

    def _switch_to_decode(self) -> None:
        """One-time transition: terminate prefill host I/O, launch decode host I/O (HOST_PUSH)."""
        if self._decode_active:
            return
        if self._prefill_active:
            logger.debug(f"Terminating prefill host I/O")
            self._host_io_prefill.terminate()
            self._prefill_active = False
        logger.debug(f"Switching to decode host I/O")
        self._host_io_decode.run()
        self._decode_active = True

    def prefill(self, prompt_tokens: list[ttnn.Tensor]) -> ttnn.Tensor:
        """
        Prefill-by-decode: for i = 0..S-1, send input_ids = x[i], get logits
        (and device updates cache). Outputs for i < S-1 are discarded. Returns the
        last step output so the caller can sample y0 (first generated token).

        Args:
            prompt_tokens: List of ttnn.Tensor, each already padded for the socket
                (PCIe-aligned, size in bytes equal to page_size_bytes(batch_size)).
                Caller is responsible for padding; use to_padded_input() if needed.

        Returns:
            Last step output tensor; valid data is first batch_size elements (logits
            (B, V) in real decoder). None if prompt_tokens is empty. Caller uses this
            to sample(logits) -> y0 for the generation loop.
        """
        if len(prompt_tokens) == 0:
            raise ValueError("Expected at least one prompt token")

        last_output: ttnn.Tensor | None = None
        for token in prompt_tokens:
            self.h2d_socket_prefill.write_tensor(token)
            self.d2h_socket.read_tensor(self._output_buffer)
            last_output = self._output_buffer
            self._position += 1
        assert last_output is not None, "Last output tensor is None"
        return last_output

    def decode_step(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        """
        Single decode step: send input via H2D, receive output via D2H.
        On first call, transitions from prefill host I/O to decode host I/O (HOST_PUSH).
        Returns the output tensor. Increments position.

        Args:
            input_tensor: Token IDs (B, 1), torch or ttnn.

        Returns:
            Output tensor; valid data is first batch_size elements. In loopback mode
            that matches the input.
        """
        assert len(input_tensor.shape) == 2, f"Input tensor shape must be (B, 1), got {input_tensor.shape}"
        assert (
            input_tensor.shape[0] == self.batch_size
        ), f"Input tensor batch size must be {self.batch_size}, got {input_tensor.shape[0]}"

        self._switch_to_decode()
        padded_input = to_padded_input(input_tensor, self.batch_size, self._page_size_datums)
        self.h2d_socket_decode.write_tensor(padded_input)
        self.d2h_socket.read_tensor(self._output_buffer)
        self._position += 1
        return self._output_buffer

    @property
    def position(self) -> int:
        """Current sequence position (number of tokens processed so far)."""
        return self._position

    def stop(self) -> None:
        """Clean shutdown of whichever mock decoder (prefill or decode) is active."""
        if self._prefill_active:
            self._host_io_prefill.terminate()
            self._prefill_active = False
        if self._decode_active:
            self._host_io_decode.terminate()
            self._decode_active = False
