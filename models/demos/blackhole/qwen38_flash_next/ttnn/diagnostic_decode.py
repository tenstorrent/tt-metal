# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""One-token, explicitly nonqualifying Qwen3.8 decode mechanics.

This adapter is intentionally smaller than a serving session.  It consumes an
already-built, caller-pinned target, executes the existing serialized 48-layer
ordinary-decode path once, and closes the state owner.  The underlying path is
the same one used by :class:`Qwen38OrdinaryDecodeSession`: embedding, all 48
backbone layers, terminal hyper-connection RMS normalization, the untied
vocabulary-sharded LM head, and exact TP4 greedy argmax.

The returned token proves only that those mechanics executed.  It is always
marked ``NONQUALIFYING`` and this module performs no golden comparison,
production admission, publication, device discovery, mesh open, or weight
staging.  Those concerns remain with the bounded diagnostic runner that owns
the already-open mesh and its full-lifetime locks.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

from models.demos.blackhole.qwen38_flash_next.ttnn.builder import Qwen38BuildProvenance, Qwen38TTNNBuiltTarget
from models.demos.blackhole.qwen38_flash_next.ttnn.decode import (
    Qwen38OrdinaryDecodeSession,
    Qwen38OrdinarySessionStatus,
    Qwen38OrdinaryStopReason,
    Qwen38OrdinaryToken,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.layer import BACKBONE_LAYERS
from models.demos.blackhole.qwen38_flash_next.ttnn.model import VOCAB_SIZE

NONQUALIFYING = "NONQUALIFYING"
GREEDY_MECHANIC = "tp4_vocab_sharded_sparse_argmax"
TERMINAL_MECHANIC = "terminal_hyper_connection_rms_norm_and_untied_lm_head"

Synchronize = Callable[[], None]
Clock = Callable[[], int]


@dataclass(frozen=True)
class Qwen38NonqualifyingDecodedToken:
    """One CPU token from the mechanics-only diagnostic path."""

    input_token_id: int
    token_id: int
    cache_position: int
    qualification: Literal["NONQUALIFYING"] = field(default=NONQUALIFYING, init=False)
    layers_executed: int = field(default=BACKBONE_LAYERS, init=False)
    terminal_mechanic: str = field(default=TERMINAL_MECHANIC, init=False)
    sampling_mechanic: str = field(default=GREEDY_MECHANIC, init=False)

    def __post_init__(self) -> None:
        for name in ("input_token_id", "token_id"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < VOCAB_SIZE:
                raise ValueError(f"{name} must be an integer in [0,{VOCAB_SIZE}), got {value!r}")
        if isinstance(self.cache_position, bool) or self.cache_position != 1:
            raise ValueError(f"first-token cache position must be 1, got {self.cache_position!r}")


def decode_first_nonqualifying_token(
    built_target: Qwen38TTNNBuiltTarget,
    input_token_id: int,
    *,
    expected_provenance: Qwen38BuildProvenance,
    expected_physical_ids: Sequence[int],
    expected_identity_key: str,
    eos_token_ids: Sequence[int],
    synchronize: Synchronize | None = None,
    clock_ns: Clock = time.perf_counter_ns,
) -> Qwen38NonqualifyingDecodedToken:
    """Consume one token and return one explicitly nonqualifying greedy token.

    ``built_target`` must already own the exact 48-layer target.  Constructor
    validation in :class:`Qwen38OrdinaryDecodeSession` remains unchanged and
    verifies its independently supplied provenance, route, and identity.  A
    healthy session is always closed; a poisoned session is deliberately left
    for process-level diagnostic cleanup without attempting state reuse.
    """

    if isinstance(input_token_id, bool) or not isinstance(input_token_id, int) or not 0 <= input_token_id < VOCAB_SIZE:
        raise ValueError(f"input_token_id must be an integer in [0,{VOCAB_SIZE}), got {input_token_id!r}")

    session = Qwen38OrdinaryDecodeSession(
        built_target,
        expected_provenance=expected_provenance,
        expected_physical_ids=expected_physical_ids,
        expected_identity_key=expected_identity_key,
        eos_token_ids=eos_token_ids,
        synchronize=synchronize,
        clock_ns=clock_ns,
    )
    try:
        event = session.begin((input_token_id,), max_new_tokens=1)
        if type(event) is not Qwen38OrdinaryToken:
            raise TypeError("ordinary decode returned a non-Qwen38 token record")
        if event.token_index != 0 or event.cache_position != 1:
            raise RuntimeError(
                f"first ordinary emission has index/position {(event.token_index, event.cache_position)}, expected (0,1)"
            )
        if event.stop_reason not in (Qwen38OrdinaryStopReason.EOS, Qwen38OrdinaryStopReason.MAX_NEW_TOKENS):
            raise RuntimeError(f"one-token diagnostic did not stop after its first emission: {event.stop_reason!r}")
        if session.status is not Qwen38OrdinarySessionStatus.FINISHED:
            raise RuntimeError(f"one-token diagnostic ended with session status {session.status.value!r}")
        return Qwen38NonqualifyingDecodedToken(
            input_token_id=input_token_id,
            token_id=event.token_id,
            cache_position=event.cache_position,
        )
    finally:
        if session.status is not Qwen38OrdinarySessionStatus.POISONED:
            session.close()


__all__ = [
    "GREEDY_MECHANIC",
    "NONQUALIFYING",
    "TERMINAL_MECHANIC",
    "Qwen38NonqualifyingDecodedToken",
    "decode_first_nonqualifying_token",
]
