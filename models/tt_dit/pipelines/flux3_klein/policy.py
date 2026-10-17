# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""FLUX 3 Image Klein admission policy (the OOM seam of the bringup plan, section 0 decision 10 (a)-(b); design record
``docs/M8_DESIGN.md`` section 6).

``admit`` runs on the host BEFORE any device work, on a request whose references are already PREPARED (sized on the
host by ``references.prepare_references``), and returns a structured :class:`Admission`; the pipeline raises
:class:`Flux3KleinAdmissionError` (the 422 analogue) carrying it. A rejected request never allocates on the device.

Checks, in order: canvas sides (multiples of 16 inside ``[MIN_SIDE, MAX_SIDE]``, reference ``pipeline.py:26``);
references at most ``FLUX3_KLEIN_MAX_REFERENCES`` (``pipeline.py:27``); every reference's four VAE encoder levels
tile-aligned (the plain ``ttnn.group_norm`` reduces over whole 32-row tiles, so a level whose ``H * W`` is not a
multiple of 32 would fold padding rows into the statistics; ``budget="reference_size"``), the canvas likewise
(``budget="canvas_size"``); text at most ``FLUX3_KLEIN_MAX_TEXT_LENGTH`` tokens; and the per-sample token total
``image + image_cond + text`` against ``max_total_tokens``, the LARGEST MEASURED bucket of the mesh (the number is a
measured row with its log cite in ``measured_buckets`` or it is not in the table). The worst case the API admits
structurally (4096 x 4096 + 10 references of <= 1 MP + 4096 text = 65536 + 40960 + 4096 = 110,592 tokens per sample)
is therefore refused with ``budget="total_tokens"`` until M10 measures it bucket by bucket. The device-allocation
recovery (c) and the mock-cluster test (d) are M10 items.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .packing import (
    FLUX3_KLEIN_DOWNSAMPLING_RATIO,
    FLUX3_KLEIN_MAX_REFERENCES,
    FLUX3_KLEIN_MAX_TEXT_LENGTH,
    FLUX3_KLEIN_TILE,
)
from .references import FLUX3_KLEIN_MAX_SIDE, FLUX3_KLEIN_MIN_SIDE, TOO_MANY_REFERENCES

if TYPE_CHECKING:
    from .pipeline_flux3_klein import Flux3KleinResolved

# reference pipeline.py:26 (kept under the M7 names; the values live in references.py)
MIN_SIDE, MAX_SIDE = FLUX3_KLEIN_MIN_SIDE, FLUX3_KLEIN_MAX_SIDE
REFERENCES_REASON = TOO_MANY_REFERENCES
# The VAE's channel count per level (encoder: full, /2, /4, /8 resolution; the decoder mirrors it).
VAE_LEVEL_CHANNELS = (128, 256, 512, 512)
# The empty prompt's padded text length (text_encoder.py:35): the base model's unconditional sample.
NULL_TEXT_LENGTH = 80


@dataclass(frozen=True)
class Flux3KleinCapacity:
    """One row per mesh shape."""

    mesh_shape: tuple[int, int]
    max_side: int = MAX_SIDE
    min_side: int = MIN_SIDE
    max_references: int = FLUX3_KLEIN_MAX_REFERENCES
    max_text_length: int = FLUX3_KLEIN_MAX_TEXT_LENGTH
    max_total_tokens: int | None = None
    """Per-sample single-stage rows (image + image_cond + text) of the LARGEST MEASURED bucket; ``None`` = no token
    budget (an unlisted mesh)."""
    measured_buckets: tuple[str, ...] = ()
    """Documentation: the measured rows that set ``max_total_tokens``, with their log cites."""


CAPACITY: dict[tuple[int, int], Flux3KleinCapacity] = {
    # (4, 8) ring: the largest bucket that has run on the ring with its DRAM / TRACE line in a log. M7 shipped the
    # T2I 1024 row (4096 image + 320 text = 4416 per sample, + 80 null text for the base CFG pair):
    # logs/m4_fix_full_4x8.log:619 (DRAM 190.6 MB/bank after capture, TRACE 25.6 MB base / 13.0 distilled),
    # logs/m67_builder_pipeline_4x8.log:94 (3.41 GB/chip residency).
    # M8 (fixer, 2026-10-17 23:5x): raised to the 2-reference bucket 4096 + 8192 + 320 = 12608 (distilled/2_3_mugprint,
    # two 1024x1024 references), measured on the ring through the M4 denoise method with the real distilled DiT:
    # logs/m8_verifier_numerics_denoise_refs_4x8.log:1912 (final pcc 0.999552 / rel_rmse 0.0301 vs the golden, teacher-
    # forced floor 0.998987 at :1904) with TRACE 1,859,584 of 8,003,584 B per bank (23.2 %) and DRAM 204.7 MB per bank
    # after capture (the RECORD line after :1907); the 1-reference bucket 8512 ran through the full Flux3KleinPipeline
    # with everything resident: DRAM 513.1 MB/bank = 4.11 GB/chip, TRACE 2,121,728 B/bank
    # (logs/m8_verifier_device_probe_4x8_run3.log:351-354). The 12608 bucket's own line through the pipeline process
    # (test_edit_golden_inputs[distilled-2_3_mugprint]) is the next device session's; the denoise-process numbers are
    # the interim evidence (same slots, same traces, 9 MB/bank more than the 1-reference bucket).
    (4, 8): Flux3KleinCapacity(
        mesh_shape=(4, 8),
        max_total_tokens=12608,
        measured_buckets=(
            "t2i 1024 (4096 + 320, base +80 null): logs/m4_fix_full_4x8.log:619, logs/m67_builder_pipeline_4x8.log:94",
            "1 ref 1024 (4096 + 4096 + 320 = 8512, distilled/3_bandana) through Flux3KleinPipeline: "
            "logs/m8_verifier_device_probe_4x8_run3.log:351-354 (DRAM 513.1 MB/bank, TRACE 2,121,728 B/bank); denoise "
            "method logs/m8_verifier_numerics_denoise_refs_4x8.log:2258",
            "2 refs 1024 (4096 + 8192 + 320 = 12608, distilled/2_3_mugprint) through the denoiser (M4 method): "
            "logs/m8_verifier_numerics_denoise_refs_4x8.log:1904,1912 (TRACE 1,859,584 B/bank, DRAM 204.7 MB/bank)",
        ),
    ),
}


def capacity_for(mesh_shape: tuple[int, int]) -> Flux3KleinCapacity:
    """The row of ``mesh_shape``; an unlisted mesh gets the structural limits (no token budget)."""
    return CAPACITY.get(tuple(mesh_shape), Flux3KleinCapacity(mesh_shape=tuple(mesh_shape)))


@dataclass(frozen=True)
class Admission:
    ok: bool
    reason: str | None = None
    budget: str | None = None
    measured: dict = field(default_factory=dict)


class Flux3KleinAdmissionError(ValueError):
    """A rejected request; ``admission`` carries the budget that was exceeded."""

    def __init__(self, admission: Admission) -> None:
        super().__init__(f"{admission.reason} (budget {admission.budget}, measured {admission.measured})")
        self.admission = admission


def vae_level_rows(width: int, height: int) -> tuple[int, ...]:
    """``H * W`` of the four VAE levels (full, /2, /4, /8; the strided downsample ``conv(F.pad(x, (0, 1, 0, 1)))`` with
    stride 2 gives ``floor((H + 1 - 3) / 2) + 1 = floor(H / 2)``, ``autoencoder.py:34-40``; a 16-multiple side never
    reaches an odd extent before a downsample, so floor and ceil agree on every admitted size)."""
    rows, h, w = [], height, width
    for _ in VAE_LEVEL_CHANNELS:
        rows.append(h * w)
        h, w = (h + 1 - 3) // 2 + 1, (w + 1 - 3) // 2 + 1
    return tuple(rows)


def vae_levels_tile_aligned(width: int, height: int) -> bool:
    """True when every VAE level of a ``width x height`` image has a row count that is a multiple of the tile height."""
    return all(rows % FLUX3_KLEIN_TILE == 0 for rows in vae_level_rows(width, height))


def admit(resolved: Flux3KleinResolved, mesh_shape: tuple[int, int]) -> Admission:
    cap = capacity_for(mesh_shape)
    r = FLUX3_KLEIN_DOWNSAMPLING_RATIO
    width, height = resolved.width, resolved.height
    prepared = resolved.prepared
    reference_sizes = list(prepared.sizes) if prepared is not None else []
    cond_tokens = prepared.cond_tokens() if prepared is not None else 0
    num_references = len(reference_sizes) if prepared is not None else len(resolved.images)
    text_length = resolved.text_length
    image_tokens = (height // r) * (width // r) if height % r == 0 and width % r == 0 else -1
    text_tokens = text_length if text_length is not None else 0
    measured = {
        "image_tokens": image_tokens,
        "cond_tokens": cond_tokens,
        "references": num_references,
        "reference_sizes": [list(s) for s in reference_sizes],
        "text_length": text_length if text_length is not None else -1,
        "text_null": 0 if resolved.flow_map else NULL_TEXT_LENGTH,
        "total_tokens": image_tokens + cond_tokens + text_tokens,
        "samples": 1 if resolved.flow_map else 2,
        "width": width,
        "height": height,
        "max_total_tokens": cap.max_total_tokens,
    }
    for name, side in (("width", width), ("height", height)):
        if side % r:
            return Admission(False, f"{name} {side} is not a multiple of {r}", "side", measured)
        if not cap.min_side <= side <= cap.max_side:
            return Admission(False, f"{name} {side} outside [{cap.min_side}, {cap.max_side}]", "side", measured)
    if num_references > cap.max_references:
        return Admission(False, REFERENCES_REASON, "references", measured)
    for index, (ref_w, ref_h) in enumerate(reference_sizes):
        if not vae_levels_tile_aligned(ref_w, ref_h):
            return Admission(
                False,
                f"reference {index} prepared to {ref_w}x{ref_h} has a VAE level that is not tile-aligned "
                f"(level rows {vae_level_rows(ref_w, ref_h)})",
                "reference_size",
                measured,
            )
    if not vae_levels_tile_aligned(width, height):
        return Admission(
            False,
            f"canvas {width}x{height} has a VAE level that is not tile-aligned (level rows {vae_level_rows(width, height)})",
            "canvas_size",
            measured,
        )
    if text_length is not None and text_length > cap.max_text_length:
        return Admission(False, f"text length {text_length} > {cap.max_text_length}", "text_length", measured)
    if cap.max_total_tokens is not None and measured["total_tokens"] > cap.max_total_tokens:
        return Admission(
            False,
            f"total tokens {measured['total_tokens']} per sample exceed the measured capacity {cap.max_total_tokens} "
            f"of mesh {cap.mesh_shape}",
            "total_tokens",
            measured,
        )
    return Admission(True, None, None, measured)


__all__ = [
    "CAPACITY",
    "MAX_SIDE",
    "MIN_SIDE",
    "NULL_TEXT_LENGTH",
    "REFERENCES_REASON",
    "VAE_LEVEL_CHANNELS",
    "Admission",
    "Flux3KleinAdmissionError",
    "Flux3KleinCapacity",
    "admit",
    "capacity_for",
    "vae_level_rows",
    "vae_levels_tile_aligned",
]
