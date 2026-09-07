# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Host-side assembly of sampled tokens and their behavior log probabilities."""

from __future__ import annotations

from collections.abc import Sequence

from .rollout_engine import RolloutOutput


class RolloutOutputCollector:
    """Collect aligned decode steps while applying the worker's EOS policy."""

    def __init__(
        self,
        *,
        batch_size: int,
        active_batch_size: int,
        stop_token_ids: Sequence[int],
        stop_at_eos: bool,
    ) -> None:
        if not 0 <= active_batch_size <= batch_size:
            raise ValueError("active_batch_size must be between zero and batch_size")
        self._active_batch_size = active_batch_size
        self._stop_token_ids = frozenset(int(token) for token in stop_token_ids) if stop_at_eos else frozenset()
        self._tokens: list[list[int]] = [[] for _ in range(batch_size)]
        self._logprobs: list[list[float]] = [[] for _ in range(batch_size)]
        self._done = [row >= active_batch_size for row in range(batch_size)]

    @property
    def all_done(self) -> bool:
        return all(self._done)

    def add_step(self, tokens: Sequence[int], logprobs: Sequence[float]) -> None:
        if len(tokens) != len(self._tokens) or len(logprobs) != len(self._tokens):
            raise ValueError(
                f"decode step has {len(tokens)} tokens and {len(logprobs)} logprobs; "
                f"expected {len(self._tokens)} of each"
            )
        for row, (token, logprob) in enumerate(zip(tokens, logprobs)):
            if self._done[row]:
                continue
            token = int(token)
            if token in self._stop_token_ids:
                self._done[row] = True
                continue
            self._tokens[row].append(token)
            self._logprobs[row].append(float(logprob))

    def finish(self) -> RolloutOutput:
        return RolloutOutput.from_sequences(
            self._tokens[: self._active_batch_size],
            self._logprobs[: self._active_batch_size],
        )


def sampled_token_logprobs(tokens: Sequence[int], logprob_payload) -> list[float]:
    """Extract one sampled-token score per row from scalar or top-k output.

    ``Generator`` returns either a flat sampled-token logprob tensor or a
    ``(topk_logprobs, topk_indices)`` pair.  Inputs here are already converted
    to ordinary host lists so this helper remains device-independent.
    """
    token_rows = [int(token) for token in tokens]
    if logprob_payload is None:
        raise RuntimeError("on-device sampling did not return behavior-policy logprobs")
    if not isinstance(logprob_payload, tuple):
        values = [float(value) for value in logprob_payload]
        if len(values) != len(token_rows):
            raise ValueError(f"received {len(values)} sampled logprobs for {len(token_rows)} tokens")
        return values

    topk_logprobs, topk_indices = logprob_payload
    if len(topk_logprobs) != len(token_rows) or len(topk_indices) != len(token_rows):
        raise ValueError("top-k logprob rows do not match the sampled-token batch")
    sampled = []
    for row, token in enumerate(token_rows):
        try:
            position = [int(index) for index in topk_indices[row]].index(token)
        except ValueError as error:
            raise ValueError(f"sampled token {token} is absent from top-k logprobs for row {row}") from error
        sampled.append(float(topk_logprobs[row][position]))
    return sampled


__all__ = ["RolloutOutputCollector", "sampled_token_logprobs"]
