# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""FLUX 3 Image Klein admission policy (the OOM seam of the bringup plan, section 0 decision 10 (a)-(b)).

M7 ships the structural limits only: sides multiples of 16 inside ``[MIN_SIDE, MAX_SIDE]`` (reference
``pipeline.py:26-28``), no references (they land in M8), text at most ``FLUX3_KLEIN_MAX_TEXT_LENGTH`` tokens. The
capacity table (max total tokens per mesh shape from measured peak DRAM / L1 per length bucket) is filled at M8/M10;
until then ``max_total_tokens`` is ``None`` and every structurally valid request is admitted. The device-allocation
recovery (c) and the mock-cluster test (d) are M8/M10 items and are not started here.

``admit`` runs on the host BEFORE any device work and returns a structured :class:`Admission`; the pipeline raises
:class:`Flux3KleinAdmissionError` (the 422 analogue) carrying it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .packing import FLUX3_KLEIN_DOWNSAMPLING_RATIO, FLUX3_KLEIN_MAX_TEXT_LENGTH

if TYPE_CHECKING:
    from .pipeline_flux3_klein import Flux3KleinResolved

# reference pipeline.py:26
MIN_SIDE, MAX_SIDE = 64, 4096
REFERENCES_REASON = "references land in M8"


@dataclass(frozen=True)
class Flux3KleinCapacity:
    """One row per mesh shape. M7: the 4x8 row with ONLY the structural limits."""

    mesh_shape: tuple[int, int]
    max_side: int = MAX_SIDE
    min_side: int = MIN_SIDE
    max_total_tokens: int | None = None
    max_references: int = 0
    max_text_length: int = FLUX3_KLEIN_MAX_TEXT_LENGTH


CAPACITY: dict[tuple[int, int], Flux3KleinCapacity] = {
    (4, 8): Flux3KleinCapacity(mesh_shape=(4, 8)),
}


def capacity_for(mesh_shape: tuple[int, int]) -> Flux3KleinCapacity:
    """The row of ``mesh_shape``; an unlisted mesh gets the structural limits (no token budget yet)."""
    return CAPACITY.get(tuple(mesh_shape), Flux3KleinCapacity(mesh_shape=tuple(mesh_shape)))


@dataclass(frozen=True)
class Admission:
    ok: bool
    reason: str | None = None
    budget: str | None = None
    measured: dict[str, int] = field(default_factory=dict)


class Flux3KleinAdmissionError(ValueError):
    """A rejected request; ``admission`` carries the budget that was exceeded."""

    def __init__(self, admission: Admission) -> None:
        super().__init__(f"{admission.reason} (budget {admission.budget}, measured {admission.measured})")
        self.admission = admission


def admit(resolved: Flux3KleinResolved, mesh_shape: tuple[int, int]) -> Admission:
    cap = capacity_for(mesh_shape)
    h, w = resolved.height // FLUX3_KLEIN_DOWNSAMPLING_RATIO, resolved.width // FLUX3_KLEIN_DOWNSAMPLING_RATIO
    text_length = resolved.text_length
    measured = {
        "image_tokens": h * w,
        "references": len(resolved.images),
        "text_length": text_length if text_length is not None else -1,
        "width": resolved.width,
        "height": resolved.height,
    }
    for name, side in (("width", resolved.width), ("height", resolved.height)):
        if side % FLUX3_KLEIN_DOWNSAMPLING_RATIO:
            return Admission(
                False, f"{name} {side} is not a multiple of {FLUX3_KLEIN_DOWNSAMPLING_RATIO}", "side", measured
            )
        if not cap.min_side <= side <= cap.max_side:
            return Admission(False, f"{name} {side} outside [{cap.min_side}, {cap.max_side}]", "side", measured)
    if len(resolved.images) > cap.max_references:
        return Admission(False, REFERENCES_REASON, "references", measured)
    if text_length is not None and text_length > cap.max_text_length:
        return Admission(False, f"text length {text_length} > {cap.max_text_length}", "text_length", measured)
    if cap.max_total_tokens is not None and h * w + (text_length or 0) > cap.max_total_tokens:
        return Admission(False, f"total tokens exceed {cap.max_total_tokens}", "total_tokens", measured)
    return Admission(True, None, None, measured)
