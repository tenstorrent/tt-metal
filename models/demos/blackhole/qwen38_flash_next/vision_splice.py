# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The host side of the vision splice into a prefill chunk (pure torch, no TTNN).

The reference replaces the token embedding of every ``<|image_pad|>`` with the vision tower's row for it
(``masked_scatter``).  The port's chunk body embeds its token rows on device, so the splice is two host-prepared
inputs: the token rows carry the zero-embedding sentinel (-1) at the image lanes, which every device localizes to
its zero row, and the feature rows ``[1, 1, rows, 2560]`` hold the tower's rows at the image lanes and -0.0 at the
text lanes; the body adds the feature rows to the embedding.  ``x + (-0.0) == x`` for every x (+0.0 and -0.0
included), so a text row is bitwise the embedding and an image row is bitwise the feature (``0.0 + f == f``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from models.demos.blackhole.qwen38_flash_next.mrope import IMAGE_TOKEN_ID, Qwen38MRoPEPositions

IMAGE_LANE_SENTINEL_TOKEN = -1  # embedding.ZERO_EMBEDDING_TOKEN: the lane embeds to an exact zero row
HIDDEN_SIZE = 2560
NEGATIVE_ZERO_BF16_BITS = -0x8000  # 0x8000 as the int16 view: the add's identity element


@dataclass(frozen=True)
class Qwen38VisionPrompt:
    """An image prompt's device inputs beside its token ids: the (t, h, w) rotary positions of every prompt token
    (``mrope.mrope_positions`` over the expanded ids) and the tower's feature rows ``[N, 2560]`` BF16, one per
    ``<|image_pad|>`` in prompt order; ``digest`` identifies the images' pixels (the prefix-reuse key: two prompts
    with equal ids but different pixels are different prompts)."""

    positions: Qwen38MRoPEPositions
    features: torch.Tensor
    digest: str = ""
    # The per-image keys in prompt order (``qwen38_vision_inputs.image_digests``): with them a committed prefix
    # that covers some of a prompt's images is reusable when those images match; without them every span carries
    # the request digest (correct, and a new image appended behind a covered one then reads as a different prompt).
    image_digests: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            self.features.ndim != 2
            or int(self.features.shape[1]) != HIDDEN_SIZE
            or self.features.dtype != torch.bfloat16
        ):
            raise ValueError(
                f"vision features must be BF16 [N, {HIDDEN_SIZE}], got {self.features.dtype} {tuple(self.features.shape)}"
            )

    def validate_prompt(self, token_ids: Sequence[int]) -> None:
        """The prompt these inputs belong to: as many positions as ids, one feature row per image pad, and a tail
        the decode's block-start rule admits (``Qwen38MRoPEPositions.tail_is_plain``)."""

        ids = [int(token) for token in token_ids]
        if self.positions.length != len(ids):
            raise ValueError(f"{self.positions.length} rotary positions for a prompt of {len(ids)} tokens")
        pads = len(image_lanes(ids))
        if int(self.features.shape[0]) != pads:
            raise ValueError(f"{int(self.features.shape[0])} feature rows for {pads} image pads")
        if not self.positions.tail_is_plain(len(ids)):
            raise ValueError(
                "an image ends within the last index block of the prompt: the prompt must end with text after its "
                "last image (the chat template's assistant header does)"
            )
        if self.image_digests and len(self.image_digests) != len(image_spans(ids)):
            raise ValueError(f"{len(self.image_digests)} image digests for {len(image_spans(ids))} image spans")

    def spans(self, token_ids: Sequence[int]) -> tuple[tuple[int, int, str], ...]:
        """The images' pad spans of ``token_ids`` with their digests, ``(start, stop, digest)`` per image in prompt
        order: the prefix-reuse key beside the ids (``Qwen38ChatSession.reusable_prefix``).  The per-image digests
        when the request gave them, else the request digest on every span."""

        runs = image_spans(token_ids)
        digests = self.image_digests if len(self.image_digests) == len(runs) else (self.digest,) * len(runs)
        return tuple((start, stop, digest) for (start, stop), digest in zip(runs, digests))


def image_spans(token_ids: Sequence[int], *, image_token_id: int = IMAGE_TOKEN_ID) -> list[tuple[int, int]]:
    """The runs of consecutive image pads in ``token_ids`` as ``(start, stop)`` lane ranges, one per image (the
    chat template wraps every image's pads in its own vision markers, so two images never share a run)."""

    spans: list[tuple[int, int]] = []
    start = None
    for lane, token in enumerate(token_ids):
        if int(token) == image_token_id:
            if start is None:
                start = lane
        elif start is not None:
            spans.append((start, lane))
            start = None
    if start is not None:
        spans.append((start, len(token_ids)))
    return spans


def image_lanes(token_ids: Sequence[int], *, image_token_id: int = IMAGE_TOKEN_ID) -> list[int]:
    """The lanes (row indices) of a chunk's token ids that are image pads, in order."""

    return [lane for lane, token in enumerate(token_ids) if int(token) == image_token_id]


def sentinel_token_rows(token_rows: torch.Tensor, lanes: Sequence[int]) -> torch.Tensor:
    """A copy of the host token-rows image (``[1, 1, tiles, 32]`` fp32, lane j = token j) with the zero-embedding
    sentinel at ``lanes``."""

    rows = token_rows.clone()
    flat = rows.reshape(-1)
    for lane in lanes:
        if not 0 <= int(lane) < flat.numel():
            raise ValueError(f"image lane {lane} is outside the {flat.numel()} token lanes")
        flat[int(lane)] = float(IMAGE_LANE_SENTINEL_TOKEN)
    return rows


def clean_feature_rows(rows: int, *, hidden: int = HIDDEN_SIZE) -> torch.Tensor:
    """The feature rows of a chunk without images: ``[1, 1, rows, hidden]`` BF16, every element -0.0."""

    if isinstance(rows, bool) or type(rows) is not int or rows <= 0:
        raise ValueError(f"feature rows need a positive row count, got {rows!r}")
    return torch.full((1, 1, rows, hidden), -0.0, dtype=torch.bfloat16)


def feature_rows_image(
    token_ids: Sequence[int],
    features: torch.Tensor | None,
    *,
    hidden: int = HIDDEN_SIZE,
    image_token_id: int = IMAGE_TOKEN_ID,
) -> torch.Tensor | None:
    """The chunk's feature rows: -0.0 everywhere, the tower's rows ``features`` (``[n, hidden]`` BF16, one per image
    pad of ``token_ids`` in order) at the image lanes.  None when the chunk has no image pads (``features`` must then
    be None or empty); image pads without features, or a count mismatch, are refused."""

    lanes = image_lanes(token_ids, image_token_id=image_token_id)
    count = 0 if features is None else int(features.shape[0])
    if count != len(lanes):
        raise ValueError(f"{len(lanes)} image pads in the chunk vs {count} feature rows")
    if not lanes:
        return None
    if features.ndim != 2 or int(features.shape[1]) != hidden or features.dtype != torch.bfloat16:
        raise ValueError(f"vision features must be BF16 [n, {hidden}], got {features.dtype} {tuple(features.shape)}")
    rows = clean_feature_rows(len(token_ids), hidden=hidden)
    rows[0, 0, torch.tensor(lanes, dtype=torch.int64)] = features
    return rows


def split_features(
    token_ids: Sequence[int], features: torch.Tensor | None, cursor: int, *, image_token_id: int = IMAGE_TOKEN_ID
) -> tuple[torch.Tensor | None, int]:
    """The rows of ``features`` that belong to the image pads of ``token_ids`` starting at ``cursor``, and the cursor
    after them (the prefill driver walks a prompt's features chunk by chunk)."""

    count = len(image_lanes(token_ids, image_token_id=image_token_id))
    if count == 0:
        return None, cursor
    if features is None or cursor + count > int(features.shape[0]):
        have = 0 if features is None else int(features.shape[0]) - cursor
        raise ValueError(f"the chunk holds {count} image pads but {have} feature rows remain")
    return features[cursor : cursor + count], cursor + count
