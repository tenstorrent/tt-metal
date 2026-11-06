# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-side bookkeeping of multimodal (image / video) prompt inputs: no ttnn, no device.

Shared by the served adapter (``tt/generator_vllm.py``, which must import without ttnn) and the device path
(``tt/multimodal.py``). Three facts of the GLM-5.3-Flash prompt format drive everything here (HF
``Glm5NextProcessor`` / ``Glm5NextModel.get_placeholder_mask``):

* every vision token in the ids is ``image_token_id`` (``<|image|>``, 154854); the ``<|video|>`` token (154855)
  only ever appears in the chat template and is expanded by the processor;
* an image is one run of ``grid.prod() // 4`` image tokens between ``<|begin_of_image|>`` / ``<|end_of_image|>``;
* a video is ``<|begin_of_video|>`` + one such run PER TEMPORAL GROUP (``video_grid_thw[0]`` groups of
  ``temporal_patch_size`` = 2 frames each, ``h * w // 4`` tokens per group) each followed by the text
  ``"<t>.<d> seconds"`` + ``<|end_of_video|>``; the vision tower runs every group as its own ``(1, h, w)`` image
  (HF ``get_video_features`` flattens ``video_grid_thw`` to ``[1, h, w] * t``) and the features are scattered
  into the image-token rows in prompt order. Whether an image token belongs to a video is decided by the
  enclosing ``<|begin_of_video|>`` .. ``<|end_of_video|>`` span (cumsum of the two markers), as HF does.

The vLLM TT plugin (``model_runner._gather_multi_modal_inputs``) hands ``prefill_forward`` one list per field,
aligned with the batch rows; a row's entry is ``None`` (text) or a per-item list aligned with the request's
``mm_features``: ``pixel_values`` / ``image_grid_thw`` hold a tensor on image items, ``pixel_values_videos`` /
``video_grid_thw`` on video items (``None`` elsewhere), and ``mm_placeholder_offsets`` / ``mm_placeholder_lengths``
the item's placeholder range (``vLLM mm_position``). ``mm_items`` parses that, ``feature_rows`` maps every item to
the absolute prompt rows its features scatter into, ``chunk_scatter`` restricts those rows to one device
prefill chunk ``[start, end)`` (an item straddling a chunk boundary is split: the rows before ``end`` take the
first part of its features, the next chunk takes the rest; nothing is ever written past the chunk end).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

IMAGE_TOKEN_ID = 154854  # <|image|>: every vision token in the ids (images AND video frames)
VIDEO_TOKEN_ID = 154855  # <|video|>: template-only, expanded by the processor
IMAGE_START_ID = 154830  # <|begin_of_image|>
IMAGE_END_ID = 154831  # <|end_of_image|>
VIDEO_START_ID = 154832  # <|begin_of_video|>
VIDEO_END_ID = 154833  # <|end_of_video|>
MERGE_TOKENS = 4  # spatial_merge_size ** 2: patches per vision token
PATCH_DIM = 1176  # in_channels * temporal_patch_size * patch_size ** 2

MM_IMAGE_FIELDS = ("pixel_values", "image_grid_thw")
MM_VIDEO_FIELDS = ("pixel_values_videos", "video_grid_thw")
MM_PLACEHOLDER_FIELDS = ("mm_placeholder_offsets", "mm_placeholder_lengths")
MM_FIELDS = MM_IMAGE_FIELDS + MM_VIDEO_FIELDS


@dataclass
class MMItem:
    """One multimodal item of a prompt: an image (``grid_thw`` one row ``[1, h, w]``) or a video (one row
    ``[t, h, w]``: ``t`` temporal groups of 2 frames). ``pixel_values`` ``[n_patches, PATCH_DIM]`` fp32 (the
    processor's patches, both modalities share the layout). ``offset`` / ``length``: the item's placeholder range
    in the prompt (vLLM ``mm_position``) when the plugin forwarded it, else None (rows are then found by order)."""

    kind: str  # "image" | "video"
    pixel_values: torch.Tensor
    grid_thw: torch.Tensor
    offset: int | None = None
    length: int | None = None

    def __post_init__(self):
        if self.kind not in ("image", "video"):
            raise ValueError(f"multimodal item kind {self.kind!r}: expected image or video")
        self.grid_thw = torch.as_tensor(self.grid_thw).reshape(-1, 3).long()
        if self.pixel_values.dim() != 2:
            raise ValueError(
                f"{self.kind} pixel_values must be [n_patches, patch_dim], got {tuple(self.pixel_values.shape)}"
            )
        if int(self.pixel_values.shape[0]) != self.n_patches:
            raise ValueError(
                f"{self.kind} pixel_values hold {int(self.pixel_values.shape[0])} patches, grid {self.grid_thw.tolist()} "
                f"needs {self.n_patches}"
            )

    @property
    def n_patches(self) -> int:
        return int(self.grid_thw.prod(-1).sum())

    @property
    def n_tokens(self) -> int:
        """Vision tokens (= feature rows) of the item: ``prod(grid) // 4`` summed over its grid rows."""
        return int((self.grid_thw.prod(-1) // MERGE_TOKENS).sum())

    @property
    def n_groups(self) -> int:
        """Attention windows the tower runs: 1 per image, ``t`` per video."""
        return int(self.grid_thw[:, 0].sum())


# ------------------------------------------------------------------------------------------ plugin kwargs
def mm_present(kwargs: dict) -> bool:
    """True when some batch row carries a real multimodal item (a text batch on a multimodal class still
    arrives with ``pixel_values=[None] * B``)."""
    for key in MM_FIELDS:
        v = kwargs.get(key)
        if v is None:
            continue
        if isinstance(v, (list, tuple)):
            for row in v:
                if row is None:
                    continue
                if isinstance(row, (list, tuple)):
                    if any(x is not None for x in row):
                        return True
                else:
                    return True
        else:
            return True
    return False


def _row_entries(kwargs: dict, key: str, row: int):
    v = kwargs.get(key)
    if v is None or not isinstance(v, (list, tuple)) or row >= len(v):
        return None
    entry = v[row]
    if entry is None:
        return None
    return list(entry) if isinstance(entry, (list, tuple)) else [entry]


def mm_items(kwargs: dict, row: int) -> list[MMItem]:
    """The multimodal items of batch row ``row`` in the request's feature order (the plugin's per-item lists;
    an item is an image when its ``pixel_values`` entry is a tensor, a video when ``pixel_values_videos`` is).
    An item with neither (a payload stripped by prefix caching, which this adapter declares off) is refused."""
    pv = _row_entries(kwargs, "pixel_values", row) or []
    ig = _row_entries(kwargs, "image_grid_thw", row) or []
    pvv = _row_entries(kwargs, "pixel_values_videos", row) or []
    vg = _row_entries(kwargs, "video_grid_thw", row) or []
    offs = _row_entries(kwargs, "mm_placeholder_offsets", row) or []
    lens = _row_entries(kwargs, "mm_placeholder_lengths", row) or []
    n = max(len(pv), len(pvv))
    if n == 0:
        return []
    items: list[MMItem] = []
    for j in range(n):
        image = pv[j] if j < len(pv) else None
        video = pvv[j] if j < len(pvv) else None
        off = offs[j] if j < len(offs) else None
        length = lens[j] if j < len(lens) else None
        off = None if off is None else int(off)
        length = None if length is None else int(length)
        if image is not None:
            grid = ig[j] if j < len(ig) else None
            if grid is None:
                raise ValueError(f"row {row} item {j}: pixel_values without image_grid_thw")
            items.append(MMItem("image", torch.as_tensor(image).float(), grid, off, length))
        elif video is not None:
            grid = vg[j] if j < len(vg) else None
            if grid is None:
                raise ValueError(f"row {row} item {j}: pixel_values_videos without video_grid_thw")
            items.append(MMItem("video", torch.as_tensor(video).float(), grid, off, length))
        else:
            raise ValueError(
                f"row {row} item {j}: no pixel payload (a prefix-cached item? the GLM53 adapter declares "
                "supports_prefix_caching False) and nothing to encode"
            )
    return items


# ------------------------------------------------------------------------------------------ prompt rows
def video_spans(ids: torch.Tensor, *, video_start_id: int = VIDEO_START_ID, video_end_id: int = VIDEO_END_ID):
    """Per position: the 1-based index of the enclosing ``<|begin_of_video|>`` .. ``<|end_of_video|>`` span, 0
    outside (HF: ``cumsum(start) > cumsum(end)``)."""
    ids = ids.reshape(-1)
    starts = (ids == video_start_id).long().cumsum(0)
    ends = (ids == video_end_id).long().cumsum(0)
    inside = starts > ends
    return torch.where(inside, starts, torch.zeros_like(starts))


def placeholder_runs(
    ids: torch.Tensor,
    *,
    image_token_id: int = IMAGE_TOKEN_ID,
    video_start_id: int = VIDEO_START_ID,
    video_end_id: int = VIDEO_END_ID,
) -> list[tuple[str, torch.Tensor]]:
    """The prompt's vision items in order, from the ids alone: ``("image", rows)`` for every maximal run of
    consecutive image tokens outside a video span, ``("video", rows)`` for all the image tokens of one video span
    (its frame runs are separated by ``<|end_of_image|>`` + timestamp + ``<|begin_of_image|>`` tokens)."""
    ids = ids.reshape(-1).long()
    rows = torch.nonzero(ids == image_token_id).reshape(-1)
    if rows.numel() == 0:
        return []
    span = video_spans(ids, video_start_id=video_start_id, video_end_id=video_end_id)[rows]
    out: list[tuple[str, torch.Tensor]] = []
    i = 0
    n = int(rows.numel())
    while i < n:
        if int(span[i]) > 0:
            s = int(span[i])
            j = i
            while j < n and int(span[j]) == s:
                j += 1
            out.append(("video", rows[i:j].clone()))
        else:
            j = i
            while j + 1 < n and int(span[j + 1]) == 0 and int(rows[j + 1]) == int(rows[j]) + 1:
                j += 1
            j += 1
            out.append(("image", rows[i:j].clone()))
        i = j
    return out


def feature_rows(
    ids: torch.Tensor,
    items: list[MMItem],
    *,
    image_token_id: int = IMAGE_TOKEN_ID,
    video_start_id: int = VIDEO_START_ID,
    video_end_id: int = VIDEO_END_ID,
) -> list[torch.Tensor]:
    """Per item (in ``items`` order) the absolute prompt rows its features scatter into. ``ids`` is the prefix up
    to the end of the chunk being prefilled (video spans are detected from it); the plugin forwards EVERY item of
    a request on every engine chunk, so an item entirely past the prefix gets no rows, an item the prefix cuts
    (the engine chunk ends inside its placeholder; vLLM's ``disable_chunked_mm_input`` avoids it, the adapter
    tolerates it) gets its rows so far (the next chunk, with a longer prefix, re-derives the full list), and a
    complete item exactly ``item.n_tokens`` rows. With placeholder ranges on every item: the image-token rows
    inside ``[offset, offset + length)``; otherwise the items are matched by order and kind to
    ``placeholder_runs`` (only the last run of the prefix may be cut)."""
    ids = ids.reshape(-1).long()
    n = int(ids.numel())
    mask = ids == image_token_id
    empty = torch.zeros(0, dtype=torch.long)
    out: list[torch.Tensor] = []
    if items and all(it.offset is not None and it.length is not None for it in items):
        for k, it in enumerate(items):
            lo, hi = it.offset, it.offset + it.length
            if lo < 0 or hi <= lo:
                raise ValueError(f"item {k} ({it.kind}): placeholder range [{lo}, {hi}) is empty")
            if lo >= n:
                out.append(empty)
                continue
            cut = hi > n
            rows = torch.nonzero(mask[lo : min(hi, n)]).reshape(-1) + lo
            if (not cut and int(rows.numel()) != it.n_tokens) or int(rows.numel()) > it.n_tokens:
                raise ValueError(
                    f"item {k} ({it.kind}, grid {it.grid_thw.tolist()}): {int(rows.numel())} image tokens in its "
                    f"placeholder range [{lo}, {min(hi, n)}) vs {it.n_tokens} feature rows"
                )
            out.append(rows)
        return out
    runs = placeholder_runs(
        ids, image_token_id=image_token_id, video_start_id=video_start_id, video_end_id=video_end_id
    )
    if len(runs) > len(items):
        raise ValueError(
            f"{len(runs)} vision runs in the prompt ({[k for k, _ in runs]}) vs {len(items)} multimodal items"
        )
    # the last run is cut when the prefix ends inside it (its last token is a vision token) or inside its video span
    open_span = n > 0 and int(video_spans(ids, video_start_id=video_start_id, video_end_id=video_end_id)[-1]) > 0
    for k, it in enumerate(items):
        if k >= len(runs):
            out.append(empty)
            continue
        kind, rows = runs[k]
        if kind != it.kind:
            raise ValueError(f"item {k} is a {it.kind} but prompt run {k} is a {kind}")
        last = k == len(runs) - 1
        cut = last and (bool(mask[-1]) or (kind == "video" and open_span))
        if (not cut and int(rows.numel()) != it.n_tokens) or int(rows.numel()) > it.n_tokens:
            raise ValueError(
                f"item {k} ({it.kind}, grid {it.grid_thw.tolist()}): prompt run holds {int(rows.numel())} image "
                f"tokens vs {it.n_tokens} feature rows"
            )
        out.append(rows)
    return out


@dataclass(frozen=True)
class ScatterEntry:
    """Rows of one item inside a device chunk: ``local_rows`` (chunk-relative, ascending) take the item's
    feature rows ``[lo, hi)``."""

    item: int
    local_rows: torch.Tensor
    lo: int
    hi: int

    @property
    def n(self) -> int:
        return int(self.local_rows.numel())


def chunk_scatter(rows_per_item: list[torch.Tensor], start: int, end: int) -> list[ScatterEntry]:
    """The scatter plan of the device chunk ``[start, end)``: every item whose rows intersect it, with the
    chunk-local rows and the slice of its features those rows take (rows ascend, so the slice is contiguous).
    An item entirely before ``start`` (prefilled by an earlier chunk) or at / past ``end`` (a later chunk)
    contributes nothing; a straddling item is split at the boundary."""
    out: list[ScatterEntry] = []
    for k, rows in enumerate(rows_per_item):
        rows = rows.reshape(-1)
        if rows.numel() == 0:
            continue
        lo = int(torch.searchsorted(rows, torch.tensor(start)))
        hi = int(torch.searchsorted(rows, torch.tensor(end)))
        if hi <= lo:
            continue
        out.append(ScatterEntry(k, rows[lo:hi] - start, lo, hi))
    return out


def items_in_range(rows_per_item: list[torch.Tensor], start: int, end: int) -> list[int]:
    """The item indices whose rows intersect ``[start, end)`` (the items a prefill call must encode)."""
    return [e.item for e in chunk_scatter(rows_per_item, start, end)]


# ------------------------------------------------------------------------------------------ tower geometry
def flatten_grid(grid_thw: torch.Tensor) -> torch.Tensor:
    """``[n, 3]`` rows ``[t, h, w]`` -> ``[sum t, 3]`` rows ``[1, h, w]``: one tower window per temporal group
    (HF ``get_video_features``: ``repeat_interleave(hw, t)`` with a leading 1)."""
    grid = torch.as_tensor(grid_thw).reshape(-1, 3).long()
    if grid.numel() == 0:
        return grid
    t = grid[:, 0]
    hw = torch.repeat_interleave(grid[:, 1:], t, dim=0)
    return torch.cat([torch.ones(hw.shape[0], 1, dtype=torch.long), hw], dim=1)


def vision_batches(grid_rows: torch.Tensor, max_patches: int) -> list[tuple[int, int]]:
    """Consecutive ``[lo, hi)`` ranges of flattened grid rows (whole windows) whose patch count stays within
    ``max_patches`` (a single window above it runs alone): the tower's per-call working set is bounded by the
    largest batch, not by the whole prompt's media."""
    grid = torch.as_tensor(grid_rows).reshape(-1, 3).long()
    out: list[tuple[int, int]] = []
    lo, acc = 0, 0
    for i in range(grid.shape[0]):
        p = int(grid[i].prod())
        if i > lo and acc + p > max_patches:
            out.append((lo, i))
            lo, acc = i, 0
        acc += p
    if grid.shape[0] > lo:
        out.append((lo, grid.shape[0]))
    return out


def window_patch_counts(items: list[MMItem]) -> list[int]:
    """Patches per tower window over all items (``flatten_grid`` row products), in item order."""
    out: list[int] = []
    for it in items:
        out.extend(int(r.prod()) for r in flatten_grid(it.grid_thw))
    return out


__all__ = [
    "IMAGE_TOKEN_ID",
    "VIDEO_TOKEN_ID",
    "IMAGE_START_ID",
    "IMAGE_END_ID",
    "VIDEO_START_ID",
    "VIDEO_END_ID",
    "MERGE_TOKENS",
    "PATCH_DIM",
    "MM_FIELDS",
    "MM_IMAGE_FIELDS",
    "MM_VIDEO_FIELDS",
    "MM_PLACEHOLDER_FIELDS",
    "MMItem",
    "ScatterEntry",
    "mm_present",
    "mm_items",
    "video_spans",
    "placeholder_runs",
    "feature_rows",
    "chunk_scatter",
    "items_in_range",
    "flatten_grid",
    "vision_batches",
    "window_patch_counts",
]
