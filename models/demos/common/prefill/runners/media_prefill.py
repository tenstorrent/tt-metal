# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Stage-0 media: a chunk's items come off the media ring, their rows from a row source (the pixels are unused)."""

import os
from typing import Optional

import torch
from loguru import logger

from models.demos.common.prefill.media import RowSource, build_injection, media_rows_for_chunk, stub_row_source
from models.demos.common.prefill.runners.media_ring import MediaRing

DEFAULT_MEDIA_TIMEOUT_S = 30.0


def row_source_from_env(hidden: int) -> RowSource:
    if os.environ.get("M3_MEDIA_STUB_EMBEDDINGS", "0") == "1":
        return stub_row_source(hidden)
    raise NotImplementedError(
        "PREFILL_MEDIA=1 needs an image embedding source; until one exists, set M3_MEDIA_STUB_EMBEDDINGS=1"
    )


class MediaPrefill:
    def __init__(
        self,
        ring: MediaRing,
        source: RowSource,
        *,
        sp: int,
        chunk_local: int,
        timeout_s: float = DEFAULT_MEDIA_TIMEOUT_S,
    ):
        self.ring = ring
        self.source = source
        self.sp = sp
        self.chunk_local = chunk_local
        self.timeout_s = timeout_s
        self.media_chunks = 0

    @classmethod
    def from_env(cls, ring: MediaRing, *, hidden: int, sp: int, chunk_local: int) -> "MediaPrefill":
        return cls(
            ring,
            row_source_from_env(hidden),
            sp=sp,
            chunk_local=chunk_local,
            timeout_s=float(os.environ.get("PREFILL_MEDIA_TIMEOUT_S", DEFAULT_MEDIA_TIMEOUT_S)),
        )

    def rows_for(self, meta: dict) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """Blocks until the chunk's media is on the ring. None for a chunk without media."""
        count = meta["media_count"]
        if count == 0:
            return None
        slot, start, end = meta["slot_id"], meta["actual_start"], meta["actual_end"]
        payloads = self.ring.read_chunk(slot, start, count, self.timeout_s)
        try:
            items = [p.desc.item() for p in payloads]
        finally:
            del payloads
            self.ring.release()
        slices = media_rows_for_chunk(items, start, end, self.source)
        self.media_chunks += 1
        logger.info(
            f"MEDIA_CHUNK slot={slot} [{start},{end}) items={count} rows={sum(s.rows.shape[0] for s in slices)}"
        )
        return build_injection(slices, start, end, self.sp, self.chunk_local)
