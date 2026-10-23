# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""FLUX 3 Image Klein admission policy (the OOM seam of the bringup plan, section 0 decision 10 (a)-(b); design record
``docs/M8_DESIGN.md`` section 6).

``admit`` runs on the host BEFORE any device work, on a request whose references are already PREPARED (sized on the
host by ``references.prepare_references``), and returns a structured :class:`Admission`; the pipeline raises
:class:`Flux3KleinAdmissionError` (the 422 analogue) carrying it. A rejected request never allocates on the device.

Checks, in order: canvas sides (multiples of 16 inside ``[MIN_SIDE, MAX_SIDE]``, reference ``pipeline.py:26``);
references at most ``FLUX3_KLEIN_MAX_REFERENCES`` (``pipeline.py:27``); the canvas's four VAE decoder levels
tile-aligned (``budget="canvas_size"``; the H-partitioned decoder's fused group norm reduces whole 32-row tiles).
References of any prepared size are admitted since 2026-10-21 (the encoder's plain GroupNorm has an exact path for
unaligned row counts); text at most ``FLUX3_KLEIN_MAX_TEXT_LENGTH`` tokens. These are the model's own
structural limits. There is NO token budget: the measured-bucket gate (``max_total_tokens``, refusing any request
larger than the biggest bucket that had run on the ring) was removed on 2026-10-19 by the user's decision; a request
of any admitted shape goes to the device, and a device allocation failure surfaces as the ttnn exception. The
``measured`` dict still reports ``total_tokens`` (image + image_cond + text per sample) for logging. The structural
worst case is 4096 x 4096 + 10 references of <= 1 MP + 4096 text = 65536 + 40960 + 4096 = 110,592 tokens per sample.
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


CAPACITY: dict[tuple[int, int], Flux3KleinCapacity] = {
    # (4, 8) ring: structural limits only (no token budget, see the module docstring). Footprints measured so far,
    # for reference: T2I 1024 (4096 + 320 tokens) DRAM 190.6 MB/bank after capture, TRACE 25.6 MB base / 13.0 MB
    # distilled (logs/m4_fix_full_4x8.log:619), 3.41 GB/chip residency (logs/m67_builder_pipeline_4x8.log:94);
    # 1 ref 1024 (8512 tokens) through the full pipeline DRAM 513.1 MB/bank = 4.11 GB/chip, TRACE 2,121,728 B/bank
    # (logs/m8_verifier_device_probe_4x8_run3.log:351-354); 2 refs 1024 (12608 tokens) through the denoiser TRACE
    # 1,859,584 B/bank, DRAM 204.7 MB/bank (logs/m8_verifier_numerics_denoise_refs_4x8.log:1904,1912).
    (4, 8): Flux3KleinCapacity(mesh_shape=(4, 8)),
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
    }
    for name, side in (("width", width), ("height", height)):
        if side % r:
            return Admission(False, f"{name} {side} is not a multiple of {r}", "side", measured)
        if not cap.min_side <= side <= cap.max_side:
            return Admission(False, f"{name} {side} outside [{cap.min_side}, {cap.max_side}]", "side", measured)
    if num_references > cap.max_references:
        return Admission(False, REFERENCES_REASON, "references", measured)
    # References of any prepared size are admitted (2026-10-21): the encoder's GroupNorm computes exact fp32 statistics
    # for levels whose row count is not a multiple of the tile height (layers/normalization.py _forward_unaligned;
    # test_encode_unaligned_reference: 880x1168 / 400x736 / 1232x816 at pcc 0.9998-0.9999 vs the CPU encode). The
    # former refusal (``reference_size``) rejected most real photos at BFL's own prepared sizes.
    if not vae_levels_tile_aligned(width, height):
        return Admission(
            False,
            f"canvas {width}x{height} has a VAE level that is not tile-aligned (level rows {vae_level_rows(width, height)})",
            "canvas_size",
            measured,
        )
    if text_length is not None and text_length > cap.max_text_length:
        return Admission(False, f"text length {text_length} > {cap.max_text_length}", "text_length", measured)
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
