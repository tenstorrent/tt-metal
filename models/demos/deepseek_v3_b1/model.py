# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
DeepSeek V3 B1 model (host interface).

Orchestrates prefill (token-by-token prompt processing) and autoregressive decode
via injectable write/read callables. Caller provides write_fn(token_tensor) and
read_fn(output_tensor); e.g. Pipeline.write_token / Pipeline.read_output.

Algorithm (prefill-by-decode then generation):
  - Prefill: for i = 0..S-1, call with input_ids = x[i] (B, 1); device uses/updates
    cache; ignore outputs for i < S-1. prefill() returns the last output token.
  - Start generation: y0 = prefill(prompt_tokens).
  - Generation loop: for t = 0,1,..., feed y[t] (B, 1) via decode_step(), get y[t+1],
    repeat.

Input tensor shape (H2D):
  - Only (B, 1) is supported: one token per batch element per step. The embedding layer
    runs on device; the host sends token IDs (int32). Payload size is B * TOKEN_ID_BYTES.

Interface vs real decoder:
  - One write and one read per step. The real decoder will also need per-step
    position (cur_pos_tensor / kv_cache_write_index); the engine tracks position for
    when the protocol is extended.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_b1.metadata.metadata import DeepseekMetadata
from models.demos.deepseek_v3_b1.utils import float_to_uint32

# Token IDs are int32 over the socket; payload size per step is B * TOKEN_ID_BYTES.
TOKEN_ID_BYTES: int = 4
MAX_SPECULATIVE_TOKENS: int = DeepseekMetadata.MAX_SPECULATIVE_TOKENS
MAX_WINDOW_TOKENS: int = DeepseekMetadata.MAX_WINDOW_TOKENS
RELAXED_ACCEPT_TOPN: int = DeepseekMetadata.RELAXED_ACCEPT_TOPN
TOPK_METADATA_COUNT: int = DeepseekMetadata.TOPK_METADATA_COUNT
SPEC_DECODE_PAGE_SIZE_BYTES: int = DeepseekMetadata.aligned_size_bytes()

PCIE_PAGE_ALIGNMENT_BYTES: int = 64


# ---------------------------------------------------------------------------
# Speculative-decode page layout.
# ---------------------------------------------------------------------------


class OutputField:
    """uint32 indices within the fixed metadata output page."""

    TOKEN_TYPE = 0
    USER_ID = 1
    TOKEN_ID = 2
    POSITION_ID = 3
    LANE_IDX = 4
    TEMPERATURE = 5
    TOP_K = 6
    TOP_P = 7
    CANDIDATE_TOKEN_IDS = 8
    PREFILL_TOKEN_IDS = 13
    P_TOP32_INDICES = 17
    P_TOP32_SCORES = 49
    Q_TOP32_INDICES = 65
    Q_TOP32_SCORES = 97


class InputField:
    """uint32 indices within the fixed metadata input page."""

    TOKEN_TYPE = 0
    USER_ID = 1
    TOKEN_ID = 2
    POSITION_ID = 3
    LANE_IDX = 4
    TEMPERATURE = 5
    TOP_K = 6
    TOP_P = 7
    CANDIDATE_TOKEN_IDS = 8
    PREFILL_TOKEN_IDS = 13


class TokenType:
    PREFILL = 2
    BASE = 0
    SPEC = 1


@dataclass
class CandidateToken:
    token_id: int
    pos: int

    def __post_init__(self) -> None:
        self.token_id = int(self.token_id)
        self.pos = int(self.pos)


@dataclass
class DecodeResult:
    """Parsed dynamic-depth output page from the pipeline."""

    token_type: int
    tokens: list[CandidateToken]
    user_id: int = 0
    lane_idx: int = 0
    position_id: int = 0
    p_top32_indices: list[int] = field(default_factory=list)
    p_top32_scores: list[float] = field(default_factory=list)
    q_top32_indices: list[int] = field(default_factory=list)
    q_top32_scores: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.token_type = int(self.token_type)
        self.user_id = int(self.user_id)
        self.lane_idx = int(self.lane_idx)
        self.position_id = int(self.position_id)
        self.tokens = [
            token if isinstance(token, CandidateToken) else CandidateToken(**token)  # type: ignore[arg-type]
            for token in self.tokens
        ]
        if not self.tokens:
            raise ValueError("DecodeResult requires at least one candidate token")
        self.p_top32_indices = [int(token) for token in self.p_top32_indices]
        self.p_top32_scores = [float(prob) for prob in self.p_top32_scores]
        self.q_top32_indices = [int(token) for token in self.q_top32_indices]
        self.q_top32_scores = [float(prob) for prob in self.q_top32_scores]

    @property
    def token_ids(self) -> list[int]:
        return [token.token_id for token in self.tokens]

    @property
    def positions(self) -> list[int]:
        return [token.pos for token in self.tokens]

    @property
    def window_start_pos(self) -> int:
        return self.position_id - self.lane_idx + 1

    @property
    def num_window_tokens(self) -> int:
        return len(self.tokens)

    @property
    def target_topn_tokens(self) -> list[int]:
        return self.p_top32_indices

    @property
    def target_topn_probs(self) -> list[float]:
        return self.p_top32_scores


def parse_output_page(output_buffer: ttnn.Tensor) -> DecodeResult:
    """Parse a fixed metadata output page into a structured DecodeResult."""
    raw = ttnn.to_torch(output_buffer).to(torch.int32).flatten()

    def raw_int(idx: int, default: int = 0) -> int:
        return int(raw[idx].item()) if idx < raw.numel() else default

    def raw_bf16_pair_word(idx: int, default: int = 0) -> int:
        return raw_int(idx, default) & 0xFFFFFFFF

    def unpack_bf16_scores(start_word: int) -> list[float]:
        scores = []
        for word_idx in range(TOPK_METADATA_COUNT // 2):
            word = raw_bf16_pair_word(start_word + word_idx)
            for shift in (0, 16):
                bf16_bits = (word >> shift) & 0xFFFF
                scores.append(torch.tensor(bf16_bits << 16, dtype=torch.uint32).view(torch.float32).item())
        return scores

    position_id = raw_int(OutputField.POSITION_ID)
    lane_idx = raw_int(OutputField.LANE_IDX)
    first_candidate_pos = position_id + 1
    tokens = [
        CandidateToken(
            raw_int(OutputField.CANDIDATE_TOKEN_IDS + slot_idx),
            first_candidate_pos + slot_idx,
        )
        for slot_idx in range(MAX_WINDOW_TOKENS)
    ]

    return DecodeResult(
        token_type=int(raw[OutputField.TOKEN_TYPE].item()),
        user_id=raw_int(OutputField.USER_ID),
        lane_idx=lane_idx,
        position_id=position_id,
        tokens=tokens,
        p_top32_indices=[raw_int(OutputField.P_TOP32_INDICES + idx) for idx in range(TOPK_METADATA_COUNT)],
        p_top32_scores=unpack_bf16_scores(OutputField.P_TOP32_SCORES),
        q_top32_indices=[raw_int(OutputField.Q_TOP32_INDICES + idx) for idx in range(TOPK_METADATA_COUNT)],
        q_top32_scores=unpack_bf16_scores(OutputField.Q_TOP32_SCORES),
    )


def to_spec_input(
    token_id: int,
    prefill_token_id: int | list[int],
    user_id: int,
    position_id: int,
    page_size_datums: int,
    token_type: TokenType,
    lane_idx: int = 0,
    temperature: float = 1.0,
    top_k: int = 32,
    top_p: float = 1.0,
) -> ttnn.Tensor:
    """Build a PCIe-aligned input page carrying (token_id, user_id, position_id)."""
    torch_padded = torch.zeros(1, page_size_datums, dtype=torch.int32)
    torch_padded[0, InputField.TOKEN_ID] = token_id
    torch_padded[0, InputField.TOKEN_TYPE] = token_type
    torch_padded[0, InputField.USER_ID] = user_id
    torch_padded[0, InputField.POSITION_ID] = position_id
    torch_padded[0, InputField.LANE_IDX] = lane_idx
    torch_padded[0, InputField.TEMPERATURE] = float_to_uint32(temperature)
    torch_padded[0, InputField.TOP_K] = top_k
    torch_padded[0, InputField.TOP_P] = float_to_uint32(top_p)
    prefill_token_ids = (
        [int(prefill_token_id)] if isinstance(prefill_token_id, int) else [int(value) for value in prefill_token_id]
    )
    for idx, value in enumerate(prefill_token_ids[:MAX_SPECULATIVE_TOKENS]):
        torch_padded[0, InputField.PREFILL_TOKEN_IDS + idx] = value
    return ttnn.from_torch(torch_padded, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)


def align_up(value: int, alignment: int) -> int:
    """Round value up to the next multiple of alignment."""
    return (value + alignment - 1) // alignment * alignment


def page_size_bytes(batch_size: int) -> int:
    """PCIe-aligned page (and FIFO) size in bytes for (B, 1) token IDs. Use for socket creation."""
    return max(align_up(batch_size * TOKEN_ID_BYTES, PCIE_PAGE_ALIGNMENT_BYTES), SPEC_DECODE_PAGE_SIZE_BYTES)


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
        self._tensor_size_bytes: int = page_size_bytes(batch_size)
        self._page_size_datums: int = self._tensor_size_bytes // TOKEN_ID_BYTES
        self._position: int = 0
        self._output_buffer: ttnn.Tensor = create_output_buffer(self._page_size_datums)
        logger.debug(f"Creating DeepSeekV3 model with batch size {batch_size}")

    def prefill(self, prompt_tokens: list[ttnn.Tensor]) -> list[DecodeResult]:
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
            self._read_fn(self._output_buffer)
            read_count += 1
            if read_count > total_reads - 1:
                last_results.append(parse_output_page(self._output_buffer))

        self._position += len(prompt_tokens)
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
        if isinstance(input_tensor, ttnn.Tensor) and int(input_tensor.shape[-1]) == self._page_size_datums:
            padded_input = input_tensor
        else:
            padded_input = to_padded_input(input_tensor, self.batch_size, self._page_size_datums)
        self._write_fn(padded_input)
        self._read_fn(self._output_buffer)
        self._position += 1
        return self._output_buffer

    def write_input(
        self,
        token_id: int,
        prefill_token_id: int,
        user_id: int,
        position_id: int,
        token_type: TokenType,
        lane_idx: int = 0,
        temperature: float = 1.0,
        top_k: int = 32,
        top_p: float = 1.0,
    ) -> None:
        """Write a single spec-decode input page (token_id, user_id, position_id) to the pipeline."""
        input_tensor = to_spec_input(
            token_id,
            prefill_token_id,
            user_id,
            position_id,
            self._page_size_datums,
            token_type,
            lane_idx=lane_idx,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
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
