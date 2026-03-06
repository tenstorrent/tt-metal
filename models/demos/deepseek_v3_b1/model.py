# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
DeepSeek V3 B1 model (host interface).

Orchestrates prefill (token-by-token prompt processing) and autoregressive decode
via injectable write/read callables. Caller provides write_fn(token_tensor) and
read_fn(output_tensor); e.g. Pipeline.write_token / Pipeline.read_output.

Algorithm (prefill-by-decode then generation):
  - Prefill: for i = 0..S-1, call with input_ids = x[i] (B, 1); device uses/updates
    cache; ignore outputs for i < S-1. prefill() returns the last step's output token id.
  - Start generation: y0 = prefill(prompt_tokens).
  - Generation loop: for t = 0,1,..., feed y[t] (B, 1) via decode_step(), get y[t+1], repeat.

Input tensor shape (H2D):
  - Only (B, 1) is supported: one token per batch element per step. The embedding layer
    runs on device; the host sends token IDs (int32). Payload size is B * TOKEN_ID_BYTES.

Interface vs real decoder:
  - One write and one read per step. The real decoder will also need per-step
    position (cur_pos_tensor / kv_cache_write_index); the engine tracks position for
    when the protocol is extended.
"""

from __future__ import annotations

from typing import Callable

import torch
from loguru import logger

import ttnn

# Token IDs are int32 over the socket; payload size per step is B * TOKEN_ID_BYTES.
TOKEN_ID_BYTES: int = 4

# Socket page_size must be PCIe-aligned (see h2d_socket.cpp). Must match demo stage TOKEN_PAGE_SIZE_BYTES (64).
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
    token: torch.Tensor,
    batch_size: int,
    page_size_datums: int,
) -> ttnn.Tensor:
    """Copy (B, 1) token into a PCIe-aligned padded buffer for write_tensor."""
    torch_padded = torch.zeros(1, page_size_datums, dtype=torch.int32)
    torch_padded[0, :batch_size] = token.flatten()[:batch_size]
    return ttnn.from_torch(torch_padded, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)


class DeepSeekV3:
    """
    Host-side model interface for prefill and decode via injectable write/read.
    Tracks position for compatibility with the real decoder (position_ids, kv_cache_write_index).
    Caller manages I/O lifecycle (e.g. Pipeline.setup_and_run() or HostInterface.run/terminate).
    """

    def __init__(
        self,
        write_fn: Callable[[ttnn.Tensor], None],
        read_fn: Callable[[ttnn.Tensor], int | None],
        batch_size: int = 1,
    ) -> None:
        """
        Args:
            write_fn: Called with a token tensor (PCIe-aligned, page_size_bytes(batch_size)).
            read_fn: Called with an output tensor; implementation returns an int token id
            on the input-stage rank (pipeline stage 0) and None on other ranks (e.g. Pipeline.read_output).
            batch_size: Batch size B. Current implementation supports only B=1;
                payload size is B * TOKEN_ID_BYTES (int32).
        """
        if batch_size != 1:
            raise ValueError(f"DeepSeekV3 currently supports only batch_size=1, got {batch_size}")
        self._write_fn = write_fn
        self._read_fn = read_fn
        self.batch_size = batch_size
        payload_bytes: int = batch_size * TOKEN_ID_BYTES
        logger.debug(f"Payload bytes: {payload_bytes} bytes")
        self._tensor_size_bytes: int = align_up(payload_bytes, PCIE_PAGE_ALIGNMENT_BYTES)
        self._page_size_datums: int = self._tensor_size_bytes // TOKEN_ID_BYTES
        self._position: int = 0
        self._output_buffer: ttnn.Tensor = create_output_buffer(self._page_size_datums)
        logger.debug(f"Creating DeepSeekV3 model with batch size {batch_size}")

    def prefill(self, prompt_tokens: list[torch.Tensor]) -> int | None:
        """
        Prefill-by-decode: for i = 0..S-1, send input_ids = x[i], get logits
        (and device updates cache). Outputs for i < S-1 are discarded. Returns the
        last step output so the caller can sample y0 (first generated token).

        Args:
            prompt_tokens: List of torch tensors, each shaped (B, 1) with token IDs (int32).

        Returns:
            Token ID of last prefill step on input-stage rank (rank 0), otherwise None.
        """
        if len(prompt_tokens) == 0:
            raise ValueError("Expected at least one prompt token")

        last_maybe_token: int | None = None
        for token in prompt_tokens:
            if not isinstance(token, torch.Tensor):
                raise TypeError(f"Expected prompt token to be a torch.Tensor, got {type(token)}")
            padded_token = to_padded_input(token, self.batch_size, self._page_size_datums)
            self._write_fn(padded_token)
            last_maybe_token = self._read_fn(self._output_buffer)
            self._position += 1

        return last_maybe_token

    def decode_step(self, input_tensor: torch.Tensor) -> int | None:
        """
        Single decode step: send input via write_fn, receive output via read_fn.
        Returns the output token ID. Increments position.

        Args:
            input_tensor: Token IDs (B, 1), torch or ttnn.

        Returns:
            Token ID on input-stage rank (rank 0), otherwise None.
        """
        if not isinstance(input_tensor, torch.Tensor):
            raise TypeError(f"Expected input_tensor to be a torch.Tensor, got {type(input_tensor)}")
        assert len(input_tensor.shape) == 2, f"Input tensor shape must be (B, 1), got {input_tensor.shape}"
        assert (
            input_tensor.shape[0] == self.batch_size
        ), f"Input tensor batch size must be {self.batch_size}, got {input_tensor.shape[0]}"

        padded_input = to_padded_input(input_tensor, self.batch_size, self._page_size_datums)
        self._write_fn(padded_input)
        maybe_token = self._read_fn(self._output_buffer)
        self._position += 1
        return maybe_token

    @property
    def position(self) -> int:
        """Current sequence position (number of tokens processed so far)."""
        return self._position
