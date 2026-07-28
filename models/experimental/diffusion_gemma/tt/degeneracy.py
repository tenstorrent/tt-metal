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

POLICIES = ("off", "warn", "stop", "retry")
# `stop` is the default: it refuses to commit a collapsed canvas, which is the only setting that
# actually prevents degenerate output. The `host` Gumbel default removes the FREQUENT corruption
# but not degeneration itself -- across 10 GPQA docs re-run under it, one still committed a canvas
# that was 85.2% a single content token. `warn` would have logged that and emitted it anyway.
# `off` disables the measurement entirely.
DEFAULT_POLICY = "stop"
# Calibrated on 192 committed canvases (the 10-doc GPQA re-check plus both 4-seed sweeps), taking
# only the ones NOT dominated by a stop token, since those are terminations rather than content:
#
#   healthy (n=136):  max top_frac 0.1836,  max max_run 18
#   degenerate (n=1):     top_frac 0.8516,      max_run 86
#
# 0.5 sits 2.7x above the healthy maximum and 1.7x below the degenerate one. max_run 64 is 3.5x
# above the healthy maximum -- the margin that ordinary long runs (markdown rules, table
# separators, padding in code blocks) need -- and still under the observed 86.
DEFAULT_TOP_FRAC = 0.5
DEFAULT_MAX_RUN = 64


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
    if stop_token_ids is not None:
        # Accept a bare id as well as a collection: sessions initialised from `eos_token_id`
        # carry a scalar, and a TypeError here would surface as a failed generation.
        benign = {int(stop_token_ids)} if isinstance(stop_token_ids, int) else {int(i) for i in stop_token_ids}
        if benign and stats.get("top_id") in benign:
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


DEFAULT_RETRIES = 2


def resolve_retries() -> int:
    """How many extra denoise attempts ``retry`` may spend on one block."""
    raw = os.environ.get("DG_DEGENERACY_RETRIES")
    if raw is None or raw == "":
        return DEFAULT_RETRIES
    value = int(raw)
    if value < 1:
        raise ValueError(f"DG_DEGENERACY_RETRIES must be >= 1, got {value}")
    return value


def evaluate(tokens: torch.Tensor, *, stop_token_ids=None) -> tuple:
    """Return ``(stats, degenerate)`` for a committed canvas, applying no policy.

    Split out from :func:`check_committed_block` so the commit path can decide between raising,
    warning and retrying without re-resolving the thresholds itself.
    """
    stats = block_degeneracy(tokens)
    degenerate = is_degenerate(
        stats,
        top_frac=_resolve_float("DG_DEGENERACY_TOP_FRAC", DEFAULT_TOP_FRAC),
        max_run=DEFAULT_MAX_RUN,
        stop_token_ids=stop_token_ids,
    )
    return stats, degenerate


def describe(stats: dict, *, block_idx: int | None = None) -> str:
    where = "" if block_idx is None else f" at block {block_idx}"
    return (
        f"degenerate committed canvas{where}: {stats['distinct']}/{stats['tokens']} distinct ids, "
        f"top id {stats['top_id']} covers {stats['top_frac']:.1%}, longest run {stats['max_run']}"
    )


def check_committed_block(
    tokens: torch.Tensor, *, block_idx: int | None = None, logger=None, stop_token_ids=None
) -> dict:
    """Apply ``DG_DEGENERACY_POLICY`` to a committed canvas. Returns the stats either way.

    ``off`` measures nothing and costs nothing. ``warn`` logs. ``stop`` (the DEFAULT) raises
    :class:`DegenerateBlockError` so the caller can end the generation without committing.

    NOTE: production does not call this — ``tt/generate.py`` inlines the same policy over
    :func:`evaluate` + :func:`describe`. That duplication is real and the fix is to unify on one
    of them, not to delete this one: the ten policy assertions in ``tests/test_degeneracy.py``
    exercise this function, and moving them onto the generate.py branch would trade tested pure
    logic for an untested integration path.
    """
    policy = resolve_policy()
    if policy == "off":
        return {}
    stats = block_degeneracy(tokens)
    if not is_degenerate(
        stats,
        top_frac=_resolve_float("DG_DEGENERACY_TOP_FRAC", DEFAULT_TOP_FRAC),
        max_run=DEFAULT_MAX_RUN,
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
