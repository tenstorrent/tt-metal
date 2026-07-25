# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Measure and gate degenerate committed blocks (#48291).

A block-diffusion step commits a whole 256-token canvas at once, so degeneration has a shape the
autoregressive detectors do not have: the committed canvas itself collapses onto one token id, or
onto a short cycle. The observed GPQA failure emitted a canvas of ``\\ \\ \\ ...`` and then a
canvas that was a solid wall of ``1``.

That is directly measurable on the committed tensor, before it reaches the KV cache -- no entropy
proxy needed. The entropy scalars the halt gate already reads cannot separate this from a healthy
finish: a block that has legitimately run out of things to say also has near-zero entropy.

This module is measurement plus a policy hook; it does not decide the default.
"""

from __future__ import annotations

import os

import torch

POLICIES = ("off", "warn", "stop")
DEFAULT_POLICY = "off"
# A healthy 256-token canvas of prose has well over 100 distinct ids and no long single-id run.
# These bounds are deliberately far from healthy so the gate cannot fire on ordinary text; they
# are calibrated in doc/decision_fidelity/degeneracy_calibration.md.
DEFAULT_TOP_FRAC = 0.5
DEFAULT_MAX_RUN = 32


class DegenerateBlockError(RuntimeError):
    """Raised when a committed canvas is degenerate and the policy says stop.

    Carries the offending tokens so a caller can log or return the partial generation; the block
    has NOT been committed to the KV cache when this is raised.
    """

    def __init__(self, message: str, *, tokens: torch.Tensor, stats: dict):
        super().__init__(message)
        self.tokens = tokens
        self.stats = stats


def longest_run(ids: torch.Tensor) -> int:
    """Length of the longest run of one repeated id."""
    if ids.numel() == 0:
        return 0
    changes = torch.nonzero(ids[1:] != ids[:-1]).flatten() + 1
    bounds = torch.cat([torch.zeros(1, dtype=torch.long), changes, torch.tensor([ids.numel()], dtype=torch.long)])
    return int((bounds[1:] - bounds[:-1]).max())


def block_degeneracy(tokens: torch.Tensor) -> dict:
    """Degeneracy statistics for one committed canvas ``[B, L]`` (or ``[L]``).

    ``top_frac`` is the share of the canvas taken by its single most frequent id, and ``max_run``
    the longest consecutive repeat. A wall of one token scores ``top_frac == 1.0``; a short cycle
    (``\\ \\ \\``) scores a high ``top_frac`` with a small ``max_run``, so both are needed.
    """
    ids = tokens.flatten().to(torch.long)
    total = int(ids.numel())
    if total == 0:
        return {"tokens": 0, "distinct": 0, "distinct_frac": 1.0, "top_frac": 0.0, "top_id": -1, "max_run": 0}
    counts = torch.bincount(ids)
    top_count = int(counts.max())
    return {
        "tokens": total,
        "distinct": int((counts > 0).sum()),
        "distinct_frac": float(int((counts > 0).sum()) / total),
        "top_frac": float(top_count / total),
        "top_id": int(counts.argmax()),
        "max_run": longest_run(ids),
    }


def is_degenerate(
    stats: dict,
    *,
    top_frac: float = DEFAULT_TOP_FRAC,
    max_run: int = DEFAULT_MAX_RUN,
    stop_token_ids=None,
) -> bool:
    """True when the canvas collapsed onto CONTENT, not when it terminated.

    A finished answer fills the canvas with <eos>, which scores top_frac 1.0 / max_run 256 —
    the same numbers as degeneration but the opposite meaning, and the stop-token path already
    ends generation on it. So a canvas dominated by a stop token is never degenerate here. The
    observed degenerate canvases collapse onto content ids instead (id 621 ' \\' and id 236770
    '1' for the GPQA physics prompt), so this exclusion does not weaken the gate.
    """
    if stop_token_ids and stats.get("top_id") in set(stop_token_ids):
        return False
    return stats["top_frac"] >= top_frac or stats["max_run"] >= max_run


def resolve_policy() -> str:
    policy = os.environ.get("DG_DEGENERACY_POLICY", DEFAULT_POLICY).strip().lower()
    if policy not in POLICIES:
        raise ValueError(f"DG_DEGENERACY_POLICY must be one of {POLICIES}, got {policy!r}")
    return policy


def _resolve_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    value = float(raw)
    if not 0.0 < value <= 1.0:
        raise ValueError(f"{name} must be in (0, 1], got {value}")
    return value


def _resolve_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def check_committed_block(
    tokens: torch.Tensor, *, block_idx: int | None = None, logger=None, stop_token_ids=None
) -> dict:
    """Apply ``DG_DEGENERACY_POLICY`` to a committed canvas. Returns the stats either way.

    ``off`` (default) measures nothing and costs nothing. ``warn`` logs. ``stop`` raises
    :class:`DegenerateBlockError` so the caller can end the generation without committing.
    """
    policy = resolve_policy()
    if policy == "off":
        return {}
    stats = block_degeneracy(tokens)
    if not is_degenerate(
        stats,
        top_frac=_resolve_float("DG_DEGENERACY_TOP_FRAC", DEFAULT_TOP_FRAC),
        max_run=_resolve_int("DG_DEGENERACY_MAX_RUN", DEFAULT_MAX_RUN),
        stop_token_ids=stop_token_ids,
    ):
        return stats
    where = "" if block_idx is None else f" at block {block_idx}"
    message = (
        f"degenerate committed canvas{where}: {stats['distinct']}/{stats['tokens']} distinct ids, "
        f"top id {stats['top_id']} covers {stats['top_frac']:.1%}, longest run {stats['max_run']}"
    )
    if policy == "warn":
        if logger is not None:
            logger.warning(message)
        return stats
    raise DegenerateBlockError(message, tokens=tokens, stats=stats)
