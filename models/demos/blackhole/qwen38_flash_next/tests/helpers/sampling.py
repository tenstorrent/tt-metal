# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Synthetic logit rows, parameters, candidate rows, the traced chain's 1-row primitives and the sampling extension's
host reads (no device)."""

from __future__ import annotations

import torch

from models.demos.blackhole.qwen38_flash_next.chat import EOS_TOKEN_IDS
from models.demos.blackhole.qwen38_flash_next.ttnn import mtp_v2
from models.demos.blackhole.qwen38_flash_next.ttnn.embedding import VOCAB_SIZE
from models.demos.blackhole.qwen38_flash_next.ttnn.sampling import Qwen38CandidateRow, Qwen38SamplingParameters

SAMPLING_PROBE_IDS = (95_859, 248_044, 248_319, 7, 62_086, 131_071)
SAMPLING_PEAK_LIMIT = 200_000  # peaks below the tokenizer size: a synthetic stream never ends with ``error``


def sampling_logits(
    seed: int,
    *,
    peaks: int = 24,
    below_tokenizer: bool = False,
    scale: float = 2.5,
    shift: float = 2.0,
    lift: float = 16.0,
    spread: float = 8.0,
) -> torch.Tensor:
    """A bf16 logit row at model scale: a wide body plus ``peaks`` peaks above its maximum (about 13) and peaks at
    the probe ids; ``below_tokenizer`` draws the peaks below the tokenizer size and adds no probe ids."""

    generator = torch.Generator().manual_seed(seed)
    row = torch.randn(VOCAB_SIZE, generator=generator) * scale + shift
    ids = torch.randperm(SAMPLING_PEAK_LIMIT if below_tokenizer else VOCAB_SIZE, generator=generator)[:peaks].tolist()
    ids += [] if below_tokenizer else list(SAMPLING_PROBE_IDS)
    row[ids] = lift + torch.rand(len(ids), generator=generator) * spread
    return row.to(torch.bfloat16)


def sampling_history_logits(
    history: list[int], *, eos_at: int | None = None, peak_limit: int = VOCAB_SIZE
) -> torch.Tensor:
    """A fake target's bf16 row, determined by the history's length and last token; EOS dominates from ``eos_at``."""

    generator = torch.Generator().manual_seed(len(history) * 7919 + (history[-1] if history else 0))
    row = torch.randn(VOCAB_SIZE, generator=generator) * 2.0
    peaks = torch.randperm(peak_limit, generator=generator)[:8]
    row[peaks] = 12.0 + torch.rand(8, generator=generator) * 6.0
    if eos_at is not None and len(history) >= eos_at:
        row[EOS_TOKEN_IDS[0]] = 60.0
    return row.to(torch.bfloat16)


def sampling_parameters(**fields) -> Qwen38SamplingParameters:
    """Custom parameters over temperature 1, top_p 1, top_k 20, no penalties, seed 0."""

    base = {"temperature": 1.0, "top_p": 1.0, "top_k": 20, "presence_penalty": 0.0, "seed": 0}
    return Qwen38SamplingParameters(**{**base, **fields})


def sampling_host_rows(rows_logits: list[torch.Tensor]) -> torch.Tensor:
    """The fp32 ``[rows, 256]`` candidate rows the device reads back for bf16 logit rows."""

    return torch.stack([Qwen38CandidateRow.emulate(logits).to_host_row().reshape(-1) for logits in rows_logits])


def sampling_verify_head(rows_logits: list[torch.Tensor], drafts: list[int]) -> mtp_v2.Qwen38TTNNVerifyHeadReadback:
    """The split verify's head readback for the pass rows' logits: the greedy accept count, its token, the argmax
    lanes and the candidate rows."""

    argmaxes = [int(torch.argmax(logits.to(torch.float32))) for logits in rows_logits]
    accepted = 0
    while accepted < len(drafts) and argmaxes[accepted] == drafts[accepted]:
        accepted += 1
    return mtp_v2.Qwen38TTNNVerifyHeadReadback(
        accepted, argmaxes[accepted], tuple(argmaxes), sampling_host_rows(rows_logits)
    )


class SamplingFakeChain:
    """The traced chain's 1-row primitives; TAIL(t) 'computes' the row of the logits after the tokens written so far."""

    peak_limit = VOCAB_SIZE

    def __init__(self, prefix: list[int], *, eos_at: int | None = None) -> None:
        self.inputs = list(prefix)  # every token HEAD consumed
        self.eos_at = eos_at
        self.log: list[str] = []
        self.tails = 0
        self.row: int | None = None

    def logits(self) -> torch.Tensor:
        return sampling_history_logits(self.inputs, eos_at=self.eos_at, peak_limit=self.peak_limit)

    def write_token_row(self, token_id: int) -> None:
        self.log.append(f"write:{token_id}")
        self.row = token_id

    def execute_head(self, residue: int) -> None:
        self.log.append(f"head:{residue}")
        self.inputs.append(self.row)

    def execute_tail(self, residue: int) -> None:
        self.log.append(f"tail:{residue}")
        self.tails += 1
        self.row = int(torch.argmax(self.logits().to(torch.float32)))

    def refresh_ple_row(self, token_id: int, context):
        self.log.append(f"ple:{token_id}")
        return (token_id, 0 if context is None else context[0])

    def record_event(self):
        self.log.append("event")
        return object()

    def event_synchronize(self, event) -> None:
        self.log.append("sync")

    def read_token_row(self) -> int:
        self.log.append("read_token_row")
        return self.row

    def read_token_row_nonblocking(self):
        self.log.append("read_nonblocking")
        return self.row

    @staticmethod
    def pending_value(pending) -> int:
        return pending


class SamplingFakeExtension:
    """The sampling chain extension's host reads over a fake chain whose ``logits()`` is the row TAIL computed."""

    def __init__(self, chain) -> None:
        self.chain = chain
        self.full_reads = 0

    def read_candidate_row(self) -> Qwen38CandidateRow:
        self.chain.log.append("read_row")
        return Qwen38CandidateRow.emulate(self.chain.logits())

    def read_full_logits(self, residue: int) -> torch.Tensor:
        self.full_reads += 1
        self.chain.log.append(f"full_gather:{residue}")
        return self.chain.logits().to(torch.float32)
