# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Multi-axis rotary positions (MRoPE) for prompts with images: the host side of the Qwen4Exp position rule.

Pure torch; no TTNN, no Transformers.  The formulas follow ``Qwen4ExpModel.get_rope_index`` /
``get_vision_position_ids`` and ``Qwen4ExpTextRotaryEmbedding.apply_interleaved_mrope`` of the pinned Transformers
implementation and are checked against them in ``tests/test_mrope_transformers.py``.

Positions.  Every token has three positions (t, h, w).  A run of text tokens advances all three together.  An image
whose grid is (t, h, w) patches contributes ``t * (h // 2) * (w // 2)`` merged tokens (one per ``<|image_pad|>``) in
row-major order over (t, h // 2, w // 2); merged token (ti, hi, wi) sits at (p0 + ti, p0 + hi, p0 + wi), p0 being
the position the run started at, and the next text token continues at ``p0 + max(h, w) // 2``.  A text-only prompt
is the plain arange on every axis.  The rope delta is ``max(position) + 1 - length`` (0 for text, negative with
images): a generated token at index i sits at ``i + delta`` on every axis, so the device's rotary index is the token
index minus the shift ``-delta``.

Rows.  With head_dim 256 and partial_rotary_factor 0.25 the rotary width is 64 = 32 frequency pairs; pair i takes the
t axis when ``i % 3 == 0`` (11 pairs), h when ``i % 3 == 1`` (11 pairs) and w when ``i % 3 == 2`` (10 pairs)
(``mrope_section`` [11, 11, 10], interleaved), and the row is ``cat(freqs, freqs)`` in the HF split-halves layout.
A token whose three positions agree reads the plain row of that position, so the rows of text tokens are exactly
the rows of the resident device table (row p = :func:`rope_row` at p) and an image token's row is a column select
over three table rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

IMAGE_TOKEN_ID = 248_056
VIDEO_TOKEN_ID = 248_057
VISION_START_TOKEN_ID = 248_053
VISION_END_TOKEN_ID = 248_054
SPATIAL_MERGE_SIZE = 2
ROPE_DIM = 64
ROPE_PAIRS = ROPE_DIM // 2
ROPE_THETA = 10_000_000.0
MROPE_SECTION = (11, 11, 10)
# Frequency pair i (columns i and i + 32 of a row) reads axis i % 3: t = 0, h = 1, w = 2.
MROPE_AXIS_OF_PAIR = tuple(pair % 3 for pair in range(ROPE_PAIRS))
MROPE_AXIS_OF_COLUMN = torch.tensor([MROPE_AXIS_OF_PAIR[column % ROPE_PAIRS] for column in range(ROPE_DIM)])
assert tuple(MROPE_AXIS_OF_PAIR.count(axis) for axis in range(3)) == MROPE_SECTION


def rope_inverse_frequency(theta: float = ROPE_THETA, dim: int = ROPE_DIM) -> torch.Tensor:
    """The 32 inverse frequencies ``1 / theta ** (2k / dim)``, fp32."""

    return 1.0 / (float(theta) ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))


def rope_row(position: int, inverse_frequency: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The plain RoPE row of one position: BF16 ``[1,1,1,64]`` cos and sin (the per-position scalar path the
    resident device table is built from; the batched ``torch.outer(arange, ...)`` path may differ in the last bit)."""

    if isinstance(position, bool) or type(position) is not int or position < 0:
        raise ValueError(f"RoPE position must be a non-negative int, got {position!r}")
    frequencies = torch.outer(torch.tensor([position], dtype=torch.float32), inverse_frequency)
    embedding = torch.cat((frequencies, frequencies), dim=-1).reshape(1, 1, 1, ROPE_DIM)
    return embedding.cos().to(torch.bfloat16), embedding.sin().to(torch.bfloat16)


def rope_table(length: int, inverse_frequency: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Rows 0 .. length - 1 of :func:`rope_row` as two BF16 ``[length, 64]`` tables (the host image of the device
    table; row p is bitwise :func:`rope_row` at p)."""

    if isinstance(length, bool) or type(length) is not int or length <= 0:
        raise ValueError(f"RoPE table length must be a positive int, got {length!r}")
    rows = [rope_row(position, inverse_frequency) for position in range(length)]
    cos = torch.cat([cos.reshape(1, ROPE_DIM) for cos, _ in rows], dim=0)
    sin = torch.cat([sin.reshape(1, ROPE_DIM) for _, sin in rows], dim=0)
    return cos, sin


@dataclass(frozen=True)
class Qwen38ImageGrid:
    """One image's patch grid (t, h, w) as the image processor reports it (``image_grid_thw``)."""

    t: int
    h: int
    w: int

    def __post_init__(self) -> None:
        for name, value in (("t", self.t), ("h", self.h), ("w", self.w)):
            if isinstance(value, bool) or type(value) is not int or value <= 0:
                raise ValueError(f"image grid {name} must be a positive int, got {value!r}")
        for name, value in (("h", self.h), ("w", self.w)):
            if value % SPATIAL_MERGE_SIZE:
                raise ValueError(f"image grid {name} must be a multiple of {SPATIAL_MERGE_SIZE}, got {value}")

    @property
    def merged_tokens(self) -> int:
        """The number of ``<|image_pad|>`` tokens the image occupies (``prod(grid) // merge ** 2``)."""

        return self.t * (self.h // SPATIAL_MERGE_SIZE) * (self.w // SPATIAL_MERGE_SIZE)

    @property
    def position_span(self) -> int:
        """How far the positions advance over the image (``max(h, w) // merge``)."""

        return max(self.h, self.w) // SPATIAL_MERGE_SIZE

    def token_positions(self, start: int) -> torch.Tensor:
        """The ``[3, merged_tokens]`` (t, h, w) positions of the image's merged tokens from ``start``, row-major over
        (t, h // 2, w // 2) (``get_vision_position_ids`` with temporal merge 1 and time interval 1)."""

        t_axis = torch.arange(self.t, dtype=torch.int64) + start
        h_axis = torch.arange(self.h // SPATIAL_MERGE_SIZE, dtype=torch.int64) + start
        w_axis = torch.arange(self.w // SPATIAL_MERGE_SIZE, dtype=torch.int64) + start
        grids = torch.meshgrid(t_axis, h_axis, w_axis, indexing="ij")
        return torch.stack(grids, dim=0).reshape(3, -1)

    def as_tuple(self) -> tuple[int, int, int]:
        return (self.t, self.h, self.w)


@dataclass(frozen=True)
class Qwen38MRoPEPositions:
    """The (t, h, w) positions of a token sequence and its rope delta.

    ``axes`` is int64 ``[3, length]`` (rows t, h, w); ``delta`` is ``max(axes) + 1 - length`` (0 for text, negative
    with images); ``shift`` is ``-delta``: the device rotary index of a generated token at index i is ``i - shift``.
    """

    axes: torch.Tensor
    delta: int

    def __post_init__(self) -> None:
        if self.axes.dtype != torch.int64 or self.axes.ndim != 2 or self.axes.shape[0] != 3:
            raise ValueError(f"MRoPE axes must be int64 [3, length], got {self.axes.dtype} {tuple(self.axes.shape)}")
        if isinstance(self.delta, bool) or type(self.delta) is not int or self.delta > 0:
            raise ValueError(f"MRoPE delta must be an int <= 0, got {self.delta!r}")

    @property
    def length(self) -> int:
        return int(self.axes.shape[1])

    @property
    def shift(self) -> int:
        return -self.delta

    @property
    def text_only(self) -> bool:
        """Every token has t == h == w == its index (the plain RoPE)."""

        return self.delta == 0 and bool(torch.equal(self.axes, self.axes[0:1].expand(3, -1)))

    def rows(self, start: int, stop: int) -> torch.Tensor:
        """The ``[3, stop - start]`` positions of tokens ``start .. stop - 1``."""

        if not 0 <= start <= stop <= self.length:
            raise ValueError(f"row range [{start}, {stop}) is outside the {self.length} positions")
        return self.axes[:, start:stop]

    def generated(self, index: int) -> int:
        """The (shared) position of a token generated at index ``index >= length``."""

        if index < self.length:
            raise ValueError(f"generated index {index} lies inside the {self.length} prompt positions")
        return index + self.delta

    def shift_at(self, length: int) -> int:
        """The rotary shift after ``length`` consumed tokens: ``length - (max position among them + 1)``, what a
        token at index ``length`` subtracts from its index when it is a text token continuing the sequence (0 for
        text; ``shift_at(self.length) == self.shift``).  Never negative: inside an image the positions stay below
        the index."""

        if isinstance(length, bool) or type(length) is not int or not 0 <= length <= self.length:
            raise ValueError(f"length must be an int in [0, {self.length}], got {length!r}")
        if length == 0:
            return 0
        return length - (int(self.axes[:, :length].max().item()) + 1)

    def tail_is_plain(self, length: int) -> bool:
        """Whether every token from the index block start of token ``length - 1`` (``(length - 1) & ~3``) through
        ``length - 1`` is a plain text token of the final shift, i.e. sits at ``index - shift_at(length)`` on every
        axis.  The device's 1-row bodies rotate a block's pooled index key at the block's FIRST token's row, which
        they derive as ``(P & ~3) - S``: exact only when that token is such a text token (the reference reads the
        token's own three-axis row).  The chat template ends a prompt with at least four text tokens after the last
        image, so this holds for every rendered prompt; a caller checks it before the decode."""

        if isinstance(length, bool) or type(length) is not int or not 1 <= length <= self.length:
            raise ValueError(f"length must be an int in [1, {self.length}], got {length!r}")
        shift = self.shift_at(length)
        start = (length - 1) & ~3
        expected = torch.arange(start, length, dtype=torch.int64) - shift
        return bool(torch.equal(self.axes[:, start:length], expected.reshape(1, -1).expand(3, -1)))


def image_groups(token_ids: Sequence[int], *, image_token_id: int = IMAGE_TOKEN_ID) -> list[tuple[int, int]]:
    """The ``(start, stop)`` spans of consecutive ``<|image_pad|>`` tokens, in order."""

    groups: list[tuple[int, int]] = []
    start = None
    for index, token in enumerate(token_ids):
        if token == image_token_id:
            if start is None:
                start = index
        elif start is not None:
            groups.append((start, index))
            start = None
    if start is not None:
        groups.append((start, len(token_ids)))
    return groups


def mrope_positions(
    token_ids: Sequence[int],
    image_grids: Sequence[Qwen38ImageGrid] = (),
    *,
    image_token_id: int = IMAGE_TOKEN_ID,
    video_token_id: int = VIDEO_TOKEN_ID,
) -> Qwen38MRoPEPositions:
    """``get_rope_index`` for one unpadded sequence: the (t, h, w) positions of every token and the rope delta.

    ``token_ids`` carries the expanded pads (``merged_tokens`` copies of ``image_token_id`` per image, see
    :func:`expand_image_pads`); ``image_grids`` are the images in prompt order.  Video pads are refused.
    """

    ids = [int(token) for token in token_ids]
    if any(token == video_token_id for token in ids):
        raise ValueError("video tokens are not supported")
    groups = image_groups(ids, image_token_id=image_token_id)
    grids = list(image_grids)
    if len(groups) != len(grids):
        raise ValueError(f"{len(groups)} image pad runs in the prompt vs {len(grids)} image grids")
    axes = torch.empty((3, len(ids)), dtype=torch.int64)
    position = 0
    cursor = 0
    for (start, stop), grid in zip(groups, grids):
        if stop - start != grid.merged_tokens:
            raise ValueError(
                f"image pad run of {stop - start} tokens at {start} vs {grid.merged_tokens} merged tokens of grid "
                f"{grid.as_tuple()}"
            )
        if start > cursor:
            axes[:, cursor:start] = torch.arange(start - cursor, dtype=torch.int64) + position
            position += start - cursor
        axes[:, start:stop] = grid.token_positions(position)
        position += grid.position_span
        cursor = stop
    if cursor < len(ids):
        axes[:, cursor:] = torch.arange(len(ids) - cursor, dtype=torch.int64) + position
    delta = int(axes.max().item()) + 1 - len(ids) if ids else 0
    return Qwen38MRoPEPositions(axes, delta)


def expand_image_pads(
    token_ids: Sequence[int], image_grids: Sequence[Qwen38ImageGrid], *, image_token_id: int = IMAGE_TOKEN_ID
) -> list[int]:
    """The chat template's one ``<|image_pad|>`` per image expanded to ``merged_tokens`` pads (the processor's
    ``replace_image_token``); the pads must be exactly one per grid, in order."""

    ids = [int(token) for token in token_ids]
    pads = [index for index, token in enumerate(ids) if token == image_token_id]
    grids = list(image_grids)
    if len(pads) != len(grids):
        raise ValueError(f"{len(pads)} image pads in the rendered prompt vs {len(grids)} images")
    expanded: list[int] = []
    cursor = 0
    for index, grid in zip(pads, grids):
        expanded.extend(ids[cursor:index])
        expanded.extend([image_token_id] * grid.merged_tokens)
        cursor = index + 1
    expanded.extend(ids[cursor:])
    return expanded


def mrope_rows(
    axes: torch.Tensor, table_cos: torch.Tensor, table_sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The cos/sin rows ``[R, 64]`` of tokens with (t, h, w) positions ``axes`` ``[3, R]`` from the plain tables
    (:func:`rope_table`): column c reads axis ``(c % 32) % 3``.  Rows whose three positions agree are bitwise the
    table rows."""

    if axes.dtype != torch.int64 or axes.ndim != 2 or axes.shape[0] != 3:
        raise ValueError(f"MRoPE axes must be int64 [3, R], got {axes.dtype} {tuple(axes.shape)}")
    for name, table in (("cos", table_cos), ("sin", table_sin)):
        if table.ndim != 2 or table.shape[1] != ROPE_DIM:
            raise ValueError(f"RoPE {name} table must be [length, {ROPE_DIM}], got {tuple(table.shape)}")
    length = int(table_cos.shape[0])
    if axes.numel() and (int(axes.min().item()) < 0 or int(axes.max().item()) >= length):
        raise ValueError(f"MRoPE positions must lie in [0, {length}), got [{axes.min().item()}, {axes.max().item()}]")
    column_axis = MROPE_AXIS_OF_COLUMN.to(axes.device)
    outputs = []
    for table in (table_cos, table_sin):
        per_axis = torch.stack([table[axes[axis]] for axis in range(3)], dim=0)  # [3, R, 64]
        index = column_axis.reshape(1, 1, ROPE_DIM).expand(1, axes.shape[1], ROPE_DIM)
        outputs.append(torch.gather(per_axis, 0, index).squeeze(0))
    return outputs[0], outputs[1]
