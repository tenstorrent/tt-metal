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
    cache; ignore logits for i < S-1. prefill() returns the last step output (logits
    in real decoder) so caller can sample y0.
  - Start generation: last_logits = prefill(prompt_tokens); y0 = sample(last_logits).
  - Generation loop: for t = 0,1,..., feed y[t] (B, 1) via decode_step(), get logits,
    sample y[t+1], repeat.

Input tensor shape (H2D):
  - Only (B, 1) is supported: one token per batch element per step. The embedding layer
    runs on device; the host sends token IDs (int32). Payload size is B * TOKEN_ID_BYTES.

Interface vs real decoder:
  - One write and one read per step. The real decoder will also need per-step
    position (cur_pos_tensor / kv_cache_write_index); the engine tracks position for
    when the protocol is extended.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from loguru import logger

import ttnn

# Token IDs are int32 over the socket; payload size per step is B * TOKEN_ID_BYTES.
TOKEN_ID_BYTES: int = 4

PCIE_PAGE_ALIGNMENT_BYTES: int = 64


# ---------------------------------------------------------------------------
# Speculative-decode page layout (64 bytes = 16 uint32 words)
# ---------------------------------------------------------------------------


class OutputField:
    """uint32 indices within the 16-word output page."""

    TOKEN_0 = 0
    TOKEN_0_TYPE = 1
    TOKEN_0_POS = 2
    TOKEN_1 = 3
    TOKEN_1_TYPE = 4
    TOKEN_1_POS = 5


class InputField:
    """uint32 indices within the 16-word input page."""

    TOKEN_TYPE = 1
    USER_ID = 6
    TOKEN_ID = 7
    POSITION_ID = 8
    PREFILL_TOKEN_ID = 9
    TOKEN0_POSITION_ID = 2


class TokenType:
    BASE = 0
    SPEC = 1


@dataclass
class DecodeResult:
    """Parsed output page from the pipeline."""

    token_0: int
    token_0_type: int
    token_0_pos: int
    token_1: int | None = None
    token_1_type: int | None = None
    token_1_pos: int | None = None
    slot_id: int | None = None


def parse_output_page(output_buffer: ttnn.Tensor) -> DecodeResult:
    """Parse a 16-word output page into a structured DecodeResult."""
    raw = ttnn.to_torch(output_buffer).to(torch.int32).flatten()
    return DecodeResult(
        token_0=int(raw[OutputField.TOKEN_0].item()),
        token_0_type=int(raw[OutputField.TOKEN_0_TYPE].item()),
        token_0_pos=int(raw[OutputField.TOKEN_0_POS].item()),
        token_1=int(raw[OutputField.TOKEN_1].item()),
        token_1_type=int(raw[OutputField.TOKEN_1_TYPE].item()),
        token_1_pos=int(raw[OutputField.TOKEN_1_POS].item()),
        slot_id=int(raw[InputField.USER_ID].item()),
    )


def to_spec_input(
    token_id: int, prefill_token_id: int, user_id: int, position_id: int, page_size_datums: int, token_type: TokenType
) -> ttnn.Tensor:
    """Build a PCIe-aligned input page carrying (token_id, user_id, position_id)."""
    torch_padded = torch.zeros(1, page_size_datums, dtype=torch.int32)
    torch_padded[0, InputField.TOKEN_ID] = token_id
    torch_padded[0, InputField.PREFILL_TOKEN_ID] = prefill_token_id
    torch_padded[0, InputField.TOKEN_TYPE] = token_type
    torch_padded[0, InputField.USER_ID] = user_id
    torch_padded[0, InputField.POSITION_ID] = position_id
    torch_padded[0, InputField.TOKEN0_POSITION_ID] = position_id
    return ttnn.from_torch(torch_padded, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)


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
    Host-side model interface for prefill and decode via injectable write/read.
    Tracks position for compatibility with the real decoder (position_ids, kv_cache_write_index).
    Caller manages I/O lifecycle (e.g. Pipeline.setup_and_run() or HostInterface.run/terminate).
    """

    def __init__(
        self,
        write_fn: Callable[[ttnn.Tensor], None],
        read_fn: Callable[[ttnn.Tensor], None],
        batch_size: int = 1,
        pipeline_depth: int = 1,
    ) -> None:
        """
        Args:
            write_fn: Called with a token tensor (PCIe-aligned, page_size_bytes(batch_size)).
            read_fn: Called with an output tensor; implementation fills it (e.g. Pipeline.read_output).
            batch_size: Batch size B. Current implementation supports only B=1;
                payload size is B * TOKEN_ID_BYTES (int32).
            pipeline_depth: Number of pipeline stages between host write and readable
                output. Prefill queues this many tokens before overlapping writes with
                readback. Use 1 for direct loopback/no pipeline.
        """
        if batch_size != 1:
            raise ValueError(f"DeepSeekV3 currently supports only batch_size=1, got {batch_size}")
        if pipeline_depth <= 0:
            raise ValueError(f"pipeline_depth must be > 0, got {pipeline_depth}")
        self._write_fn = write_fn
        self._read_fn = read_fn
        self.batch_size = batch_size
        self._pipeline_depth = pipeline_depth
        payload_bytes: int = batch_size * TOKEN_ID_BYTES
        logger.debug(f"Payload bytes: {payload_bytes} bytes")
        self._tensor_size_bytes: int = align_up(payload_bytes, PCIE_PAGE_ALIGNMENT_BYTES)
        self._page_size_datums: int = self._tensor_size_bytes // TOKEN_ID_BYTES
        self._position: list[int] = [0 for _ in range(2)]
        self._output_buffer: ttnn.Tensor = create_output_buffer(self._page_size_datums)
        logger.debug(f"Creating DeepSeekV3 model with batch size {batch_size}")

    def prefill(self, prompt_tokens: list[ttnn.Tensor], slot_id: int) -> list[DecodeResult]:
        """
        Prefill-by-decode with overlapped I/O: enqueue tokens until the pipeline is
        saturated, then overlap readback per additional write, and finally drain
        the remaining in-flight outputs.

        Each write produces READS_PER_WRITE output pages (base + spec).
        All outputs are discarded except the last READS_PER_WRITE (from the final
        prompt token), which are parsed and returned so the decode state machine
        can process both the base and speculative results.

        Args:
            prompt_tokens: List of ttnn.Tensor, each already padded for the socket
                (PCIe-aligned, size in bytes equal to page_size_bytes(batch_size)).
                Caller is responsible for padding; use to_spec_input() if needed.

        Returns:
            List of READS_PER_WRITE DecodeResults from the last prompt token's outputs.
        """
        if len(prompt_tokens) == 0:
            raise ValueError("Expected at least one prompt token")

        num_writes_before_readback = min(self._pipeline_depth, len(prompt_tokens))
        total_reads = len(prompt_tokens)

        write_idx = 0
        read_count = 0

        # Phase 1: saturate the pipeline (no reads yet)
        while write_idx < num_writes_before_readback:
            print(f"Writing to index {write_idx}")
            self._write_fn(prompt_tokens[write_idx])
            write_idx += 1

        # Phase 2: overlap — drain outputs and issue remaining writes in steady state
        while write_idx < len(prompt_tokens):
            self._read_fn(self._output_buffer)
            read_count += 1
            self._write_fn(prompt_tokens[write_idx])
            write_idx += 1

        # Phase 3: drain remaining outputs; save the last output
        last_results: list[DecodeResult] = []
        while read_count < total_reads:
            print(f"Reading from index {read_count}")
            self._read_fn(self._output_buffer)
            print(f"Done reading from index {read_count}")
            read_count += 1
            if read_count > total_reads - 1:
                last_results.append(parse_output_page(self._output_buffer))

        self._position[slot_id] += len(prompt_tokens)
        return last_results

    def decode_step(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        """
        Single decode step: send input via write_fn, receive output via read_fn.
        Returns the output tensor. Increments position.

        Args:
            input_tensor: Token IDs (B, 1), torch or ttnn.

        Returns:
            Output tensor; valid data is first batch_size elements.
        """
        assert len(input_tensor.shape) == 2, f"Input tensor shape must be (B, 1), got {input_tensor.shape}"
        assert (
            input_tensor.shape[0] == self.batch_size
        ), f"Input tensor batch size must be {self.batch_size}, got {input_tensor.shape[0]}"

        self._write_fn(padded_input)
        self._read_fn(self._output_buffer)
        self._position += 1
        return self._output_buffer

    def write_input(
        self, token_id: int, prefill_token_id: int, user_id: int, position_id: int, token_type: TokenType
    ) -> None:
        """Write a single spec-decode input page (token_id, user_id, position_id) to the pipeline."""
        input_tensor = to_spec_input(
            token_id, prefill_token_id, user_id, position_id, self._page_size_datums, token_type
        )
        self._write_fn(input_tensor)

    def read_result(self) -> DecodeResult:
        """Read one output page from the pipeline and return the parsed DecodeResult."""
        self._read_fn(self._output_buffer)
        return parse_output_page(self._output_buffer)

    @property
    def position(self) -> int:
        """Current sequence position (number of tokens processed so far)."""
        return self._position
