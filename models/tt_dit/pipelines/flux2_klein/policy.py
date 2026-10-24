# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""FLUX.2 Klein admission policy: the host-side gate of ``Flux2KleinPipeline.generate`` (the FLUX 3 Klein
``pipelines/flux3_klein/policy.py`` pattern).

``admit`` runs on the host BEFORE any device work, on a request whose references are already PREPARED (sized on the
host by ``references.prepare_references``, ``pipeline_flux2_klein.py:764-787``), and returns a structured
:class:`Admission`; the pipeline raises :class:`Flux2KleinAdmissionError` carrying it. A rejected request never
allocates on the device.

Checks, in order: canvas sides (multiples of 16, ``pipeline_flux2_klein.py:778-780`` / ``:493-494``, inside
``[MIN_SIDE, MAX_SIDE]``); references at most ``FLUX2_KLEIN_MAX_REFERENCES`` (a port limit, the reference has none);
the canvas's four VAE decoder levels tile-aligned (``budget="canvas_size"``; the H-partitioned decoder's fused group
norm reduces whole 32-row tiles); and, given the DiT layout's ``sp_factor``, that the spatial stream's per-device
rows differ from the prompt stream's (``budget="stream_rows"``): the ring joint attention gathers the spatial K / V
and the prompt K / V over SP into the CCL manager's ping-pong buffers, which are keyed by shape and alternate between
TWO buffers per key (``parallel/manager.py`` ``get_ag_ping_pong_buffer``; ``blocks/attention_opt.py`` requests four
per forward under ``shard_prompt``), and the reduce-scatter ping-pong of the dual blocks' two feed-forwards alternates
the same way, so two streams of ONE per-device shape alias each other's buffers. The coincidence is
``padded_length(S_img + S_cond, SP) / SP == 512 / SP`` under ``shard_prompt`` (``L_pad == 512``: canvases of 257 to
512 tokens on SP8) and ``L_pad == 512 * SP`` without it. References of any prepared size are admitted (the encoder is
not H-partitioned). There is NO token budget: a request of any admitted shape goes to the device, and a device
allocation failure surfaces as the ttnn exception. The ``measured`` dict reports ``total_tokens`` (image + reference
+ text rows per sample) for logging; the prompt stream is always ``FLUX2_KLEIN_TEXT_LENGTH`` rows
(``pipeline_flux2_klein.py:204``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .packing import (
    FLUX2_KLEIN_DOWNSAMPLING_RATIO,
    FLUX2_KLEIN_MAX_REFERENCES,
    FLUX2_KLEIN_TEXT_LENGTH,
    FLUX2_KLEIN_TILE,
    padded_length,
)
from .references import FLUX2_KLEIN_MAX_SIDE, FLUX2_KLEIN_MIN_SIDE, TOO_MANY_REFERENCES

if TYPE_CHECKING:
    from .pipeline_flux2_klein import Flux2KleinResolved

MIN_SIDE, MAX_SIDE = FLUX2_KLEIN_MIN_SIDE, FLUX2_KLEIN_MAX_SIDE
REFERENCES_REASON = TOO_MANY_REFERENCES
# The VAE's channel count per level (encoder: full, /2, /4, /8 resolution; the decoder mirrors it; vae/config.json).
VAE_LEVEL_CHANNELS = (128, 256, 512, 512)
# The empty prompt's encoding is a full 512-row stream like any other (the base model's unconditional sample).
NULL_TEXT_LENGTH = FLUX2_KLEIN_TEXT_LENGTH


@dataclass(frozen=True)
class Flux2KleinCapacity:
    """One row per mesh shape."""

    mesh_shape: tuple[int, int]
    max_side: int = MAX_SIDE
    min_side: int = MIN_SIDE
    max_references: int = FLUX2_KLEIN_MAX_REFERENCES


CAPACITY: dict[tuple[int, int], Flux2KleinCapacity] = {
    # (4, 8) ring: structural limits only (no token budget, see the module docstring).
    (4, 8): Flux2KleinCapacity(mesh_shape=(4, 8)),
}


def capacity_for(mesh_shape: tuple[int, int]) -> Flux2KleinCapacity:
    """The row of ``mesh_shape``; an unlisted mesh gets the structural limits (no token budget)."""
    return CAPACITY.get(tuple(mesh_shape), Flux2KleinCapacity(mesh_shape=tuple(mesh_shape)))


@dataclass(frozen=True)
class Admission:
    ok: bool
    reason: str | None = None
    budget: str | None = None
    measured: dict = field(default_factory=dict)


class Flux2KleinAdmissionError(ValueError):
    """A rejected request; ``admission`` carries the check that failed."""

    def __init__(self, admission: Admission) -> None:
        super().__init__(f"{admission.reason} (budget {admission.budget}, measured {admission.measured})")
        self.admission = admission


def vae_level_rows(width: int, height: int) -> tuple[int, ...]:
    """``H * W`` of the four VAE levels (full, /2, /4, /8; the strided downsample ``conv(F.pad(x, (0, 1, 0, 1)))`` with
    stride 2 gives ``floor((H + 1 - 3) / 2) + 1 = floor(H / 2)``; a 16-multiple side never reaches an odd extent
    before a downsample, so floor and ceil agree on every admitted size)."""
    rows, h, w = [], height, width
    for _ in VAE_LEVEL_CHANNELS:
        rows.append(h * w)
        h, w = (h + 1 - 3) // 2 + 1, (w + 1 - 3) // 2 + 1
    return tuple(rows)


def vae_levels_tile_aligned(width: int, height: int) -> bool:
    """True when every VAE level of a ``width x height`` image has a row count that is a multiple of the tile height."""
    return all(rows % FLUX2_KLEIN_TILE == 0 for rows in vae_level_rows(width, height))


def stream_rows_coincide(spatial_rows: int, *, sp_factor: int, shard_prompt: bool) -> bool:
    """True when the ``spatial_rows`` real spatial rows, padded for ``sp_factor``, give the same per-device row count
    as the ``FLUX2_KLEIN_TEXT_LENGTH`` prompt rows (sharded over SP under ``shard_prompt``, replicated otherwise): the
    CCL ping-pong aliasing of the module docstring."""
    per_device_spatial = padded_length(spatial_rows, sp_factor) // sp_factor
    per_device_prompt = FLUX2_KLEIN_TEXT_LENGTH // sp_factor if shard_prompt else FLUX2_KLEIN_TEXT_LENGTH
    return per_device_spatial == per_device_prompt


def admit(
    resolved: Flux2KleinResolved,
    mesh_shape: tuple[int, int],
    *,
    sp_factor: int | None = None,
    shard_prompt: bool = True,
) -> Admission:
    """The structural checks of the module docstring. ``sp_factor`` is the sequence-parallel factor of the DiT layout
    the request renders on (``None`` skips the stream-rows check, for a caller without a layout)."""
    cap = capacity_for(mesh_shape)
    r = FLUX2_KLEIN_DOWNSAMPLING_RATIO
    width, height = resolved.width, resolved.height
    prepared = resolved.prepared
    reference_sizes = list(prepared.sizes) if prepared is not None else []
    cond_tokens = prepared.cond_tokens() if prepared is not None else 0
    num_references = len(reference_sizes) if prepared is not None else len(resolved.images)
    text_length = int(resolved.text_length)
    image_tokens = (height // r) * (width // r) if height % r == 0 and width % r == 0 else -1
    measured = {
        "image_tokens": image_tokens,
        "cond_tokens": cond_tokens,
        "references": num_references,
        "reference_sizes": [list(s) for s in reference_sizes],
        "text_length": text_length,
        "text_null": NULL_TEXT_LENGTH if resolved.cfg else 0,
        "total_tokens": image_tokens + cond_tokens + text_length,
        "samples": 2 if resolved.cfg else 1,
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
    if not vae_levels_tile_aligned(width, height):
        return Admission(
            False,
            f"canvas {width}x{height} has a VAE level that is not tile-aligned (level rows {vae_level_rows(width, height)})",
            "canvas_size",
            measured,
        )
    if sp_factor is not None:
        measured["padded_rows"] = padded_length(image_tokens + cond_tokens, sp_factor)
        if stream_rows_coincide(image_tokens + cond_tokens, sp_factor=sp_factor, shard_prompt=shard_prompt):
            return Admission(
                False,
                f"{image_tokens + cond_tokens} spatial rows pad to {measured['padded_rows']} on SP {sp_factor}, the "
                f"per-device row count of the {text_length}-row prompt stream (the two streams would alias their CCL "
                "ping-pong buffers); change the canvas or reference size",
                "stream_rows",
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
    "Flux2KleinAdmissionError",
    "Flux2KleinCapacity",
    "admit",
    "capacity_for",
    "stream_rows_coincide",
    "vae_level_rows",
    "vae_levels_tile_aligned",
]
