# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Stage-0 media for one chunk: read its images off the media ring, embed them through the encoder cache, and lay
out the rows the chunk injects. Host-only: the device injection is the runtime's (inject_media_embeddings)."""

import os
from typing import Callable, Optional

import torch
from loguru import logger

from models.demos.common.prefill.media import EncoderCache, build_injection, stub_embeddings
from models.demos.common.prefill.runners.media_ring import MediaPayload, MediaRing

# Encodes one image: its ring payload -> [num_tokens, hidden] bf16 embeddings.
Encoder = Callable[[MediaPayload], torch.Tensor]

DEFAULT_ENCODER_CACHE_MB = 2048
DEFAULT_MEDIA_TIMEOUT_S = 30.0


def stub_encoder(hidden: int) -> Encoder:
    """Placeholder embeddings seeded from the media hash, until the vision encoder (tt-metal#62156) is wired in."""
    return lambda p: stub_embeddings(p.desc.hash, p.desc.num_tokens, hidden)


def encoder_from_env(hidden: int) -> Encoder:
    if os.environ.get("M3_MEDIA_STUB_EMBEDDINGS", "0") == "1":
        return stub_encoder(hidden)
    raise NotImplementedError(
        "PREFILL_MEDIA=1 needs an image encoder; until tt-metal#62156 lands, set M3_MEDIA_STUB_EMBEDDINGS=1"
    )


class MediaPrefill:
    def __init__(
        self,
        ring: MediaRing,
        encoder: Encoder,
        *,
        sp: int,
        chunk_local: int,
        cache_bytes: int,
        timeout_s: float = DEFAULT_MEDIA_TIMEOUT_S,
    ):
        self.ring = ring
        self.encoder = encoder
        self.sp = sp
        self.chunk_local = chunk_local
        self.cache = EncoderCache(cache_bytes)
        self.timeout_s = timeout_s
        self.media_chunks = 0

    @classmethod
    def from_env(cls, ring: MediaRing, *, hidden: int, sp: int, chunk_local: int) -> "MediaPrefill":
        return cls(
            ring,
            encoder_from_env(hidden),
            sp=sp,
            chunk_local=chunk_local,
            cache_bytes=int(os.environ.get("PREFILL_ENCODER_CACHE_MB", DEFAULT_ENCODER_CACHE_MB)) << 20,
            timeout_s=float(os.environ.get("PREFILL_MEDIA_TIMEOUT_S", DEFAULT_MEDIA_TIMEOUT_S)),
        )

    def rows_for(self, meta: dict) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """(rows, mask) for the chunk `meta` names, or None for a chunk with no media. Blocks until its media is
        published on the ring. A cache hit never looks at the pixels."""
        count = meta["media_count"]
        if count == 0:
            return None
        slot, start, end = meta["slot_id"], meta["actual_start"], meta["actual_end"]
        payloads = self.ring.read_chunk(slot, start, count, self.timeout_s)
        try:
            embeddings = {
                p.desc.hash: self.cache.get_or_compute(p.desc.hash, lambda p=p: self.encoder(p)) for p in payloads
            }
            items = [p.desc.item() for p in payloads]
        finally:
            del payloads
            self.ring.release()
        self.media_chunks += 1
        logger.info(f"MEDIA_CHUNK slot={slot} [{start},{end}) items={count} {self.cache.stats_line()}")
        return build_injection(embeddings, items, start, end, self.sp, self.chunk_local)
