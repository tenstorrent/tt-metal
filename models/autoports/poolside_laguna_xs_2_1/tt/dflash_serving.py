# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Default-off, batch-one/cache-off serving controller for Laguna DFlash.

The controller is intentionally device-agnostic.  The TT core supplies draft
proposals and capture slicing; the target adapter supplies one greedy contiguous
verify.  This module owns request continuity, exact greedy acceptance, rolling
auxiliary-state commit, rollback metadata, and the one-token-at-a-time vLLM
buffer.  It never mutates the normal target forward path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from .dflash_reference import DFlashTargetAuxCapture


@dataclass(frozen=True)
class DFlashServingEnvelope:
    """Fail-closed constraints for the experimental controller tranche.

    This envelope describes where correctness was exercised; it is not a
    performance-qualification claim.  DFlash stays default-off because the
    first full-target warm gate regressed per-committed-token latency.
    """

    enabled: bool = False
    batch_size: int = 1
    greedy: bool = True
    prefix_caching: bool = False
    hybrid_kv: bool = False
    cache_off: bool = True

    def validate(self) -> None:
        if not self.enabled:
            raise RuntimeError("DFlash served decoding is experimental and default-off")
        if int(self.batch_size) != 1:
            raise RuntimeError(f"DFlash served decoding requires exactly one request, got B={self.batch_size}")
        if not bool(self.greedy):
            raise RuntimeError("DFlash served decoding is greedy-only")
        if bool(self.prefix_caching):
            raise RuntimeError("DFlash served decoding is not qualified with prefix caching")
        if bool(self.hybrid_kv):
            raise RuntimeError("DFlash served decoding is not qualified with hybrid KV")
        if not bool(self.cache_off):
            raise RuntimeError("DFlash served decoding requires cache-off request ownership")


@dataclass(frozen=True)
class DFlashServingRound:
    position: int
    drafts: tuple[int, ...]
    target_greedy: tuple[int, ...]
    accepted_drafts: int
    committed: tuple[int, ...]
    target_only: bool = False


class DFlashServedController:
    """Single-request exact-greedy controller with a vLLM token buffer."""

    def __init__(
        self,
        *,
        core,
        proposal_cache,
        target_model,
        verify_greedy: Callable[..., tuple[Sequence[int], DFlashTargetAuxCapture]],
        draft_argmax: Callable[[object], Sequence[int]],
        envelope: DFlashServingEnvelope,
    ):
        envelope.validate()
        if proposal_cache.core is not core:
            raise ValueError("DFlash controller cache was not allocated by its core")
        self.core = core
        self.cache = proposal_cache
        self.target_model = target_model
        self.verify_greedy = verify_greedy
        self.draft_argmax = draft_argmax
        self.envelope = envelope
        self._request_id = None
        self._pending: list[int] = []
        self._expected_input_position = None
        self._expected_input_token = None
        self._closed = False
        self.rounds: list[DFlashServingRound] = []

    @property
    def active(self) -> bool:
        return self._request_id is not None and not self._closed

    @property
    def pending_tokens(self) -> tuple[int, ...]:
        return tuple(self._pending)

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("DFlash served controller is closed")

    def begin_request(self, request_id, capture: DFlashTargetAuxCapture) -> None:
        self._require_open()
        if self._request_id is not None:
            self.end_request(self._request_id)
        capture.validate(self.core.config)
        self.cache.begin_request(request_id)
        self.cache.update_target_capture(capture, replace=True)
        self._request_id = request_id
        self._pending.clear()
        self._expected_input_position = capture.end_position + 1
        self._expected_input_token = None
        self.rounds.clear()

    def ingest_prefill_capture(
        self,
        request_id,
        capture: DFlashTargetAuxCapture,
        *,
        new_request: bool = False,
    ) -> None:
        """Ingest one target prefill chunk while retaining only a 511-row tail."""

        self._require_open()
        capture.validate(self.core.config)
        if new_request or self._request_id is None:
            self.begin_request(request_id, capture)
            return
        if request_id != self._request_id:
            raise RuntimeError(
                f"DFlash prefill request {request_id!r} does not match active request {self._request_id!r}"
            )
        current = self.cache.target_capture()
        # A full 511-row chunk supersedes all older state.  A short terminal
        # chunk is adjacent and must be appended to preserve the cross-chunk tail.
        replace = int(capture.row_count) == self.cache.max_context_rows
        if replace and int(capture.end_position) <= int(current.end_position):
            raise ValueError(
                f"full DFlash prefill tail ends at {capture.end_position}, "
                f"not after retained end {current.end_position}"
            )
        if not replace and int(capture.start_position) != current.end_position + 1:
            raise ValueError(
                f"short DFlash prefill capture starts at {capture.start_position}, "
                f"expected {current.end_position + 1}"
            )
        self.cache.update_target_capture(capture, replace=replace)
        retained = self.cache.target_capture()
        self._expected_input_position = retained.end_position + 1
        self._expected_input_token = None

    @staticmethod
    def _accept_greedy(drafts: Sequence[int], target_greedy: Sequence[int]) -> tuple[int, list[int]]:
        drafts = [int(token) for token in drafts]
        target = [int(token) for token in target_greedy]
        if len(target) != len(drafts) + 1:
            raise ValueError(
                f"DFlash target verify returned {len(target)} rows for {len(drafts)} drafts; "
                "expected one target bonus row"
            )
        accepted = len(drafts)
        for index, draft in enumerate(drafts):
            if draft != target[index]:
                accepted = index
                break
        return accepted, drafts[:accepted] + [target[accepted]]

    def _validate_input(self, known_bonus: int, position: int) -> None:
        if not self.active:
            raise RuntimeError("DFlash served controller has no active prefilled request")
        if int(position) != int(self._expected_input_position):
            raise RuntimeError(
                f"DFlash served position discontinuity: expected {self._expected_input_position}, got {position}"
            )
        if self._expected_input_token is not None and int(known_bonus) != int(self._expected_input_token):
            raise RuntimeError(
                f"DFlash served token discontinuity: expected {self._expected_input_token}, got {known_bonus}"
            )

    def _pop_committed(self, input_position: int) -> int:
        if not self._pending:
            raise RuntimeError("DFlash internal commit buffer is empty")
        token = int(self._pending.pop(0))
        self._expected_input_position = int(input_position) + 1
        self._expected_input_token = token
        return token

    def _verify_contiguous(
        self,
        input_tokens: Sequence[int],
        position: int,
        verify_kwargs: Mapping[str, object] | None,
    ) -> tuple[list[int], DFlashTargetAuxCapture]:
        input_tokens = [int(token) for token in input_tokens]
        if not input_tokens:
            raise ValueError("DFlash target verify requires at least one input row")
        if any(token < 0 or token >= int(self.core.config.vocab_size) for token in input_tokens):
            raise ValueError("DFlash target verify input is outside the shared target vocabulary")
        verify_kwargs = dict(verify_kwargs or {})
        if verify_kwargs.get("page_tables_per_layer") is not None:
            raise RuntimeError("DFlash served verification does not support hybrid per-layer page tables")
        verify_positions = list(range(int(position), int(position) + len(input_tokens)))
        target_greedy, verify_capture = self.verify_greedy(
            input_tokens,
            verify_positions,
            **verify_kwargs,
        )
        target_greedy = [int(token) for token in target_greedy]
        if len(target_greedy) != len(input_tokens):
            raise ValueError(
                f"DFlash target verify returned {len(target_greedy)} outputs for " f"{len(input_tokens)} input rows"
            )
        if not isinstance(verify_capture, DFlashTargetAuxCapture):
            raise TypeError("DFlash target verify did not return an auxiliary capture")
        verify_capture.validate(self.core.config)
        if int(verify_capture.start_position) != int(position) or int(verify_capture.row_count) != len(input_tokens):
            raise ValueError(
                "DFlash target verify capture does not match its contiguous input rows: "
                f"start={verify_capture.start_position} rows={verify_capture.row_count}, "
                f"expected start={position} rows={len(input_tokens)}"
            )
        return target_greedy, verify_capture

    def serve_target_token(
        self,
        *,
        known_bonus: int,
        position: int,
        verify_kwargs: Mapping[str, object] | None = None,
    ) -> int:
        """Execute one target+aux row when a fixed proposal cannot fit safely."""

        self._require_open()
        known_bonus = int(known_bonus)
        position = int(position)
        self._validate_input(known_bonus, position)
        if self._pending:
            raise RuntimeError("DFlash target-only fallback cannot bypass buffered committed tokens")
        capture = self.cache.target_capture()
        if capture.end_position + 1 != position:
            raise RuntimeError(
                f"DFlash auxiliary context ends at {capture.end_position}, but known bonus is at {position}"
            )
        target_greedy, verify_capture = self._verify_contiguous(
            [known_bonus],
            position,
            verify_kwargs,
        )
        self.cache.update_target_capture(verify_capture)
        self._pending = [target_greedy[0]]
        self.rounds.append(
            DFlashServingRound(
                position=position,
                drafts=(),
                target_greedy=tuple(target_greedy),
                accepted_drafts=0,
                committed=(target_greedy[0],),
                target_only=True,
            )
        )
        return self._pop_committed(position)

    def serve_token(
        self,
        *,
        known_bonus: int,
        position: int,
        verify_kwargs: Mapping[str, object] | None = None,
    ) -> int:
        """Return one committed token, running a full round only when needed."""

        self._require_open()
        known_bonus = int(known_bonus)
        position = int(position)
        self._validate_input(known_bonus, position)
        if self._pending:
            return self._pop_committed(position)

        capture = self.cache.target_capture()
        if capture.end_position + 1 != position:
            raise RuntimeError(
                f"DFlash auxiliary context ends at {capture.end_position}, but known bonus is at {position}"
            )
        proposal = self.core.proposal_round(
            self.cache,
            target_model=self.target_model,
            bonus_token_id=known_bonus,
            enable_experimental=True,
        )
        drafts = [int(token) for token in self.draft_argmax(proposal)]
        expected_drafts = int(self.core.config.max_speculative_tokens)
        if len(drafts) != expected_drafts:
            raise ValueError(f"DFlash drafter returned {len(drafts)} tokens, expected {expected_drafts}")
        verify_tokens = [known_bonus, *drafts]
        target_greedy, verify_capture = self._verify_contiguous(
            verify_tokens,
            position,
            verify_kwargs,
        )

        accepted, committed = self._accept_greedy(drafts, target_greedy)
        # Commit auxiliary states only for rows that are now authoritative:
        # known bonus + accepted drafts.  The trailing target bonus has not been
        # executed yet and becomes the known bonus of the next round.
        committed_capture = self.core.capture_prefix(verify_capture, 1 + accepted)
        self.cache.update_target_capture(committed_capture)
        self._pending = list(committed)
        self.rounds.append(
            DFlashServingRound(
                position=position,
                drafts=tuple(drafts),
                target_greedy=tuple(target_greedy),
                accepted_drafts=accepted,
                committed=tuple(committed),
            )
        )
        return self._pop_committed(position)

    def end_request(self, request_id=None) -> None:
        self._require_open()
        if self._request_id is None:
            raise RuntimeError("DFlash served controller has no active request")
        if request_id is not None and request_id != self._request_id:
            raise RuntimeError(
                f"cannot end DFlash served request {request_id!r}; active request is {self._request_id!r}"
            )
        self.cache.end_request(self._request_id)
        self._request_id = None
        self._pending.clear()
        self._expected_input_position = None
        self._expected_input_token = None

    def close(self) -> None:
        if self._closed:
            return
        if self._request_id is not None:
            self.end_request(self._request_id)
        self.cache.close()
        self._closed = True


__all__ = [
    "DFlashServedController",
    "DFlashServingEnvelope",
    "DFlashServingRound",
]
