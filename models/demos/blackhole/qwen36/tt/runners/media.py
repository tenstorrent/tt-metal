# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PUP prefill media: the image rows of a forward come from the encoder service's embedding store.

Same flow as MiniMax-M3's splice (tt-metal #64000 on #63999 / #64095): the engine lists, per chunk, the media items
the chunk overlaps as ``MediaRef(chunk_index, start_pos, end_pos, media_key)`` (tt-d-gen #2467's 44-byte
``PrefillMediaRef``); the runtime reads every item of the forward with ONE ``fetch_batch`` on the store reader (a value
on another host is a view the next fetch overwrites), checks each value against its ref, and hands the rows to the
model. Qwen3.5 additionally needs each image's grid for M-RoPE; it is in the stored value's 64-byte header
(``vision/format.py``), so nothing but the media key passes through the engine.

The model splices the rows itself (``Qwen36Model.media_values_to_device`` -> the prefill's ``vision_tokens``): the image
token positions of the request are where its ``<|image_pad|>`` runs are, which the runtime checks against the refs.
"""

from __future__ import annotations

from typing import NamedTuple, Sequence

import numpy as np

from models.demos.blackhole.qwen36.vision import format as media_format


class MediaSpanError(ValueError):
    """A ref the runtime cannot serve (straddles the chunk, wrong size, not on image tokens): fail the request."""


class FetchRequest(NamedTuple):
    """``models.common.multimedia_store.reader.FetchRequest`` (#64095) when that module is not installed."""

    media_key: bytes
    offset: int = 0
    length: int = -1


def _fetch_request_cls():
    try:
        from models.common.multimedia_store.reader import FetchRequest as fr

        return fr
    except Exception:  # noqa: BLE001
        return FetchRequest


class DictEmbeddingReader:
    """An in-process store (tests, the harness): ``{media_key: stored value bytes}``. Same reads as the Mooncake one."""

    def __init__(self, values=None):
        self.values = dict(values or {})
        self.fetches = 0

    def put(self, key: bytes, value: bytes) -> None:
        self.values[bytes(key)] = bytes(value)

    def fetch(self, media_key: bytes) -> memoryview:
        return self.fetch_batch([FetchRequest(media_key)])[0]

    def fetch_batch(self, requests) -> list:
        self.fetches += 1
        out = []
        for r in requests:
            try:
                v = self.values[bytes(r.media_key)]
            except KeyError:
                raise KeyError(f"media {bytes(r.media_key).hex()[:16]} not in the store") from None
            end = len(v) if r.length < 0 else r.offset + r.length
            out.append(memoryview(v)[r.offset : end])
        return out


def refs_in_chunk(media_refs, chunk_index: int = 0):
    """The refs of one chunk of a forward (refs carry ``chunk_index``; a single-chunk forward passes index 0)."""
    return [r for r in (media_refs or ()) if int(r.chunk_index) == chunk_index]


def check_refs(refs, actual_start: int, actual_end: int, token_ids=None, image_token: int | None = None):
    """Refs sorted by position, each wholly inside the chunk (the PUP runtime does not resume an image across chunks)
    and, when the host ids are known, on a run of image tokens."""
    refs = sorted(refs, key=lambda r: int(r.start_pos))
    prev = actual_start
    for r in refs:
        s, e = int(r.start_pos), int(r.end_pos)
        if not (actual_start <= s < e <= actual_end):
            raise MediaSpanError(
                f"media {bytes(r.media_key).hex()[:16]} [{s}, {e}) is not inside the chunk [{actual_start}, "
                f"{actual_end}): the PUP prefill runs a request in one chunk"
            )
        if s < prev:
            raise MediaSpanError(f"media refs overlap at {s}")
        prev = e
        if token_ids is not None and image_token is not None:
            run = np.asarray(token_ids[s - actual_start : e - actual_start])
            if run.size != e - s or not (run == image_token).all():
                raise MediaSpanError(f"media [{s}, {e}) is not a run of image tokens ({image_token})")
    return refs


def fetch_values(reader, refs: Sequence):
    """[(header, rows)] in ref order from ONE batched read; each header checked against its ref's span."""
    req = _fetch_request_cls()
    values = reader.fetch_batch([req(bytes(r.media_key)) for r in refs])
    out = []
    for r, v in zip(refs, values):
        try:
            header, rows = media_format.parse(v)
        except ValueError as e:
            raise MediaSpanError(f"media {bytes(r.media_key).hex()[:16]}: {e}") from None
        if header.n_tokens != int(r.end_pos) - int(r.start_pos):
            raise MediaSpanError(
                f"media {bytes(r.media_key).hex()[:16]}: stored {header.n_tokens} rows for a span of "
                f"{int(r.end_pos) - int(r.start_pos)}"
            )
        out.append((header, rows))
    return out


def stored_value(rows_bf16_u16: np.ndarray, grid_thw, n_refs: int = 1, merge: int = 2) -> bytes:
    """Build a stored value (header + rows) the way the encoder service writes it (harness / tests)."""
    rows = np.ascontiguousarray(rows_bf16_u16, dtype=np.uint16)
    n, hidden = rows.shape
    buf = bytearray(media_format.stored_nbytes(n, hidden))
    media_format.write_header(
        memoryview(buf), media_format.Header(n, hidden, tuple(int(v) for v in grid_thw), n_refs=n_refs, merge=merge)
    )
    buf[media_format.HEADER_BYTES :] = rows.tobytes()
    return bytes(buf)
