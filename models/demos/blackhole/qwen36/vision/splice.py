# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The two ends of a spliced (encoder-service) Qwen3.5 prefill, host only:

- adapter side: ``expand`` replaces the k-th image marker of the chat template's token ids by the k-th item's expansion
  runs (what the tt-d-gen adapters do with the encoder's reply) and returns each item's [start, end) span;
- engine -> runner: ``MediaRef`` is tt-d-gen's per-chunk ``PrefillMediaRef`` (#2467 / tt-metal #63999 layout: u32
  chunk_index, u32 start_pos, u32 end_pos, 32-byte media key = 44 bytes);
- runner side: ``fetch_media`` reads every ref's stored value from the embedding store (one batched lookup), checks the
  header against the ref's span and returns (header, rows) in prompt order; ``image_grid_thw`` gives the grids for
  M-RoPE straight from the headers; ``mrope_positions`` / ``rope_delta`` are HF get_rope_index for single-turn image
  prompts (positions [3, L], and the decode offset of the generated tokens).
"""

import struct
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from models.demos.blackhole.qwen36.vision import format as media_format

_REF = struct.Struct("<III32s")
REF_BYTES = _REF.size  # 44


@dataclass(frozen=True)
class MediaRef:
    chunk_index: int
    start_pos: int
    end_pos: int
    media_key: bytes

    def pack(self) -> bytes:
        return _REF.pack(self.chunk_index, self.start_pos, self.end_pos, self.media_key)

    @classmethod
    def unpack(cls, raw, offset: int = 0) -> "MediaRef":
        return cls(*_REF.unpack_from(raw, offset))


def expand(template_ids: Sequence[int], marker: int, runs_per_item: Sequence[Sequence[tuple[int, int]]]):
    """Adapter splice: the i-th ``marker`` in template_ids becomes item i's runs. Returns (ids, spans) with spans[i] =
    [start, end) of item i's embedding positions in ids (the marker run; the template keeps vision_start / vision_end
    around it). Raises ValueError when markers and items don't pair up."""
    n_markers = sum(1 for t in template_ids if t == marker)
    if n_markers != len(runs_per_item):
        raise ValueError(f"{n_markers} image markers in the prompt for {len(runs_per_item)} images")
    ids, spans, k = [], [], 0
    for t in template_ids:
        if t != marker:
            ids.append(t)
            continue
        start = len(ids)
        for token, count in runs_per_item[k]:
            ids.extend([token] * count)
        spans.append((start, len(ids)))
        k += 1
    return ids, spans


def refs_for_chunk(chunk_index: int, chunk_start: int, chunk_end: int, spans, media_keys) -> list[MediaRef]:
    """The refs a chunk [chunk_start, chunk_end) carries: one per item overlapping it (tt-d-gen #2467 semantics)."""
    return [
        MediaRef(chunk_index, s, e, k) for (s, e), k in zip(spans, media_keys) if s < chunk_end and e > chunk_start
    ]


def fetch_media(reader, refs: Sequence[MediaRef]):
    """Runner side: [(header, rows)] for refs, from ONE batched store read (EmbeddingReader.fetch_batch). Each value's
    header must match its ref's span (n_tokens == end - start); rows are uint16 bf16 views, in place in the store
    segment when the value lives on this host."""
    from models.common.multimedia_store.reader import FetchRequest

    values = reader.fetch_batch([FetchRequest(r.media_key) for r in refs])
    out = []
    for r, v in zip(refs, values):
        header, rows = media_format.parse(v)
        if header.n_tokens != r.end_pos - r.start_pos:
            raise ValueError(
                f"media {r.media_key.hex()[:16]}: stored {header.n_tokens} tokens for a span of "
                f"{r.end_pos - r.start_pos} ({r.start_pos}, {r.end_pos})"
            )
        out.append((header, rows))
    return out


def image_grid_thw(headers):
    """[n_images, 3] int64 (t, h, w) per image, prompt order: HF's image_grid_thw, from the stored values."""
    import torch

    return torch.tensor([list(h.grid_thw) for h in headers], dtype=torch.long)


def mrope_positions(ids: Sequence[int], grids, image_token: int, merge: int = 2) -> np.ndarray:
    """HF Qwen3.5 get_rope_index for one prompt whose images are contiguous runs of image_token (no video): [3, L]
    (t, h, w) positions. Text advances all three by 1; an image run of grid (t, h, w) occupies (t, h/merge, w/merge)
    positions offset by the position after the preceding text, and the next text resumes max(h, w) / merge past it."""
    ids = np.asarray(ids)
    L = ids.shape[0]
    pos = np.zeros((3, L), dtype=np.int64)
    i, k, nxt = 0, 0, 0
    grids = [tuple(int(v) for v in g) for g in grids]
    while i < L:
        if ids[i] != image_token:
            pos[:, i] = nxt
            nxt += 1
            i += 1
            continue
        t, h, w = grids[k]
        gh, gw = h // merge, w // merge
        n = t * gh * gw
        if i + n > L or not (ids[i : i + n] == image_token).all():
            raise ValueError(f"image {k}: run at {i} shorter than its grid's {n} tokens")
        tt, hh, ww = np.meshgrid(np.arange(t), np.arange(gh), np.arange(gw), indexing="ij")
        pos[0, i : i + n] = tt.reshape(-1) + nxt
        pos[1, i : i + n] = hh.reshape(-1) + nxt
        pos[2, i : i + n] = ww.reshape(-1) + nxt
        nxt += max(gh, gw)  # HF: the text after an image resumes max(h, w) / merge past the image's start
        i += n
        k += 1
    if k != len(grids):
        raise ValueError(f"{k} image runs in the prompt for {len(grids)} grids")
    return pos


def rope_delta(ids: Sequence[int], grids, image_token: int, merge: int = 2) -> int:
    """Decode's offset: generated token j (0-based, after the L prompt tokens) sits at rope position L + j + delta."""
    pos = mrope_positions(ids, grids, image_token, merge)
    return int(pos.max()) + 1 - len(ids)
