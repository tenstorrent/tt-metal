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
#
# These numbers hold for the CONTENT region only, which is why :func:`is_degenerate` measures
# there. Applied to the whole canvas they are wrong by construction: the terminal block of every
# answer shorter than the canvas ends in a stop-token run, and a >=64-long run or a >=50% share of
# <eos> padding is what a NORMAL completion looks like. Measured on the tt-shield eval of
# 2026-07-27 (run 30285823000), that mistake ended 110 of 198 requests mid-answer; the discarded
# blocks held a median of 107 real tokens each, 11135 in total. See
# doc/decision_fidelity/degenerate_output_fix.md.
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


def terminal_stop_run(ids: torch.Tensor, stop_ids) -> int:
    """Length of the trailing run of stop tokens — the padding a finished answer leaves behind.

    A canvas is committed whole, so an answer that ends at position 149 pads the remaining 107
    positions with <eos>. That tail is not content and must not be measured as if it were: it is
    what makes a NORMAL completion score top_frac 0.58 / max_run 107 and trip a gate calibrated on
    content-only canvases (see :func:`is_degenerate`).
    """
    if not stop_ids or ids.numel() == 0:
        return 0
    is_stop = torch.isin(ids, torch.tensor(sorted(stop_ids), dtype=ids.dtype))
    if not bool(is_stop[-1]):
        return 0
    content = torch.nonzero(~is_stop).flatten()
    return int(ids.numel()) if content.numel() == 0 else int(ids.numel() - 1 - int(content[-1]))


def _measure(ids: torch.Tensor) -> dict:
    counts = torch.bincount(ids)
    return {
        "distinct": int((counts > 0).sum()),
        "top_frac": float(int(counts.max()) / int(ids.numel())),
        "top_id": int(counts.argmax()),
        "max_run": longest_run(ids),
    }


def block_degeneracy(tokens: torch.Tensor, *, stop_token_ids=None) -> dict:
    """Degeneracy statistics for one committed canvas ``[B, L]`` (or ``[L]``).

    ``top_frac`` is the share of the canvas taken by its single most frequent id, and ``max_run``
    the longest consecutive repeat. A wall of one token scores ``top_frac == 1.0``; a short cycle
    (``\\ \\ \\``) scores a high ``top_frac`` with a small ``max_run``, so both are needed.

    The ``top_frac``/``max_run``/``distinct``/``top_id`` keys always describe the WHOLE canvas, so
    telemetry stays comparable across runs. When ``stop_token_ids`` is known the same four
    statistics are also reported for the canvas with its terminal stop-token run removed
    (``content_*``, plus ``stop_tail`` and ``content_tokens``); :func:`is_degenerate` prefers those
    because the content region is what the thresholds were calibrated on.
    """
    ids = tokens.flatten().to(torch.long)
    total = int(ids.numel())
    if total == 0:
        return {"tokens": 0, "distinct": 0, "distinct_frac": 1.0, "top_frac": 0.0, "top_id": -1, "max_run": 0}
    stats = {"tokens": total, **_measure(ids)}
    stats["distinct_frac"] = float(stats["distinct"] / total)
    if stop_token_ids:
        tail = terminal_stop_run(ids, {int(i) for i in _as_id_set(stop_token_ids)})
        stats["stop_tail"] = tail
        stats["content_tokens"] = total - tail
        if total - tail > 0:
            stats.update({f"content_{k}": v for k, v in _measure(ids[: total - tail]).items()})
    return stats


def _as_id_set(stop_token_ids) -> set:
    """Accept a bare id as well as a collection.

    Sessions initialised from `eos_token_id` carry a scalar, and a TypeError here would surface as
    a failed generation.
    """
    if stop_token_ids is None:
        return set()
    if isinstance(stop_token_ids, int) and not isinstance(stop_token_ids, bool):
        return {int(stop_token_ids)}
    return {int(i) for i in stop_token_ids}


def is_degenerate(stats: dict, *, stop_token_ids=None) -> bool:
    """True when the canvas collapsed onto CONTENT, not when it terminated.

    Measured on the CONTENT region — the canvas minus its terminal stop-token run — whenever
    :func:`block_degeneracy` was given the stop ids. That is the only region the thresholds were
    calibrated on (see DEFAULT_TOP_FRAC), and it is what separates the two shapes that score
    identically over the whole canvas:

      answer, then <eos> padding    content 149 ids, top_frac 0.07  ->  healthy
      wall of one content id        content 256 ids, top_frac 1.00  ->  degenerate

    Two special cases fall out of that: a canvas that is ALL stop tokens has no content region at
    all (a pure termination), and a content region still dominated by a stop token is a termination
    with stray content, not a collapse. Both are benign.

    Without stop ids the whole-canvas statistics are used, which cannot make the distinction. That
    is the pre-2026-07-28 behaviour and it is deliberately kept for callers that declare no stop
    set, because narrowing it there would silently weaken the gate rather than fix it — the fix is
    for the caller to declare its stop ids (``tt/serving.py`` does).

    The thresholds are read from the module globals on every call, so a test may tighten them with
    ``monkeypatch.setattr``. They are NOT overridable from the environment: ``DG_DEGENERACY_TOP_FRAC``
    was deleted 2026-08-03 (nothing in the tree ever set it, so every shipped block already took the
    ``0.5`` branch), and ``DG_DEGENERACY_MAX_RUN`` went in the 2026-07-28 triage for the same reason.
    """
    benign = _as_id_set(stop_token_ids)
    if benign and stats.get("top_id") in benign:
        return False
    if "content_tokens" in stats:
        if stats["content_tokens"] == 0:
            return False
        if stats.get("content_top_id") in benign:
            return False
        return stats["content_top_frac"] >= DEFAULT_TOP_FRAC or stats["content_max_run"] >= DEFAULT_MAX_RUN
    return stats["top_frac"] >= DEFAULT_TOP_FRAC or stats["max_run"] >= DEFAULT_MAX_RUN


def resolve_policy() -> str:
    policy = os.environ.get("DG_DEGENERACY_POLICY", DEFAULT_POLICY).strip().lower()
    if policy not in POLICIES:
        raise ValueError(f"DG_DEGENERACY_POLICY must be one of {POLICIES}, got {policy!r}")
    return policy


# How many extra denoise attempts the ``retry`` policy may spend on one block. Not overridable from
# the environment: ``DG_DEGENERACY_RETRIES`` was deleted 2026-08-03 — nothing in the tree set it, and
# the attempt count is pinned behaviourally by tests/test_degeneracy.py instead.
DEFAULT_RETRIES = 2


def evaluate(tokens: torch.Tensor, *, stop_token_ids=None) -> tuple:
    """Return ``(stats, degenerate)`` for a committed canvas, applying no policy.

    Split out from :func:`check_committed_block` so the commit path can decide between raising,
    warning and retrying.
    """
    stats = block_degeneracy(tokens, stop_token_ids=stop_token_ids)
    degenerate = is_degenerate(stats, stop_token_ids=stop_token_ids)
    return stats, degenerate


def describe(stats: dict, *, block_idx: int | None = None) -> str:
    where = "" if block_idx is None else f" at block {block_idx}"
    # The content region is what the verdict was taken on, so report it when it exists; a reader
    # who only sees the whole-canvas numbers cannot tell why the block was rejected.
    content = ""
    if "content_tokens" in stats:
        content = f" (content {stats['content_tokens']}/{stats['tokens']}, stop tail {stats['stop_tail']}"
        if stats["content_tokens"]:
            content += (
                f", content top id {stats['content_top_id']} covers {stats['content_top_frac']:.1%}, "
                f"content longest run {stats['content_max_run']}"
            )
        content += ")"
    return (
        f"degenerate committed canvas{where}: {stats['distinct']}/{stats['tokens']} distinct ids, "
        f"top id {stats['top_id']} covers {stats['top_frac']:.1%}, longest run {stats['max_run']}{content}"
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
    stats = block_degeneracy(tokens, stop_token_ids=stop_token_ids)
    if not is_degenerate(stats, stop_token_ids=stop_token_ids):
        return stats
    message = describe(stats, block_idx=block_idx)
    if policy == "warn":
        if logger is not None:
            logger.warning(message)
        return stats
    raise DegenerateBlockError(message, tokens=tokens, stats=stats)
