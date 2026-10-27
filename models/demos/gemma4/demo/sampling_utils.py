# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for wiring Gemma4 on-device sampling through Generator demos."""

from __future__ import annotations

from loguru import logger

from models.common.sampling.generator import SamplingParams


def model_can_sample_on_device(model) -> bool:
    """True when ``Gemma4Model`` constructed a ``SamplingGenerator`` (tp>1, vocab shard OK)."""
    return bool(getattr(model, "_supports_on_device_sampling", False) and getattr(model, "sampling", None) is not None)


def build_device_sampling_params(sampling_params: dict | None, *, can_sample: bool) -> SamplingParams | None:
    """Map demo CLI sampling dict → ``SamplingParams`` for Generator, or None for host path.

    Greedy (temperature≤0) uses top_k=1 / top_p=1.0 so device sampling matches host argmax.
    """
    if not can_sample:
        return None
    sp = sampling_params or {}
    t = sp.get("temperature", 0) or 0
    if t <= 0:
        return SamplingParams(temperature=0.0, top_k=1, top_p=1.0)
    return SamplingParams(
        temperature=float(t),
        top_k=int(sp.get("top_k", 32)),
        top_p=float(sp.get("top_p", 1.0)),
    )


def log_sampling_mode(can_sample: bool, sampling_params: dict | None = None) -> None:
    sp = sampling_params or {}
    greedy = (sp.get("temperature", 0) or 0) <= 0
    logger.info(
        f"Gemma4 decode sampling: device={can_sample}, "
        f"mode={'greedy' if greedy else 'sample'}, "
        f"temp={sp.get('temperature', 0)}"
    )
