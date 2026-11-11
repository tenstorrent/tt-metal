# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Multimodal prefix caching (QWEN36_MM_PREFIX_CACHE=1, default off). Host-only helpers, no ttnn.

A multimodal prompt can resume its prefill from a cached prefix like a text prompt, if three things hold:

1. The GDN prefix-state key covers the images. GdnPrefixStateCache keys a snapshot by (pos, digest of tokens[:pos]). The
   placeholder ids of two different images with the same grid are equal, so a token digest alone would hand prompt B
   the GDN state computed from prompt A's image. ``MMKeyedTokens`` is the key material: the token ids with each item's
   256-bit content hash (vLLM's mm identifier, the same value vLLM puts in its KV block hashes) inserted in front of
   the item's placeholders. The digest of tokens[:pos] then covers every item that starts before pos.
2. Only the items the recomputed suffix [s, L) reaches are encoded. Items that end at or before s are skipped; an item
   that straddles s is encoded whole and spliced from its row (s - offset) on. ``MMVision.rows_for(s)`` returns the
   packed rows of items[first:] and ``row_base`` = the placeholder rows of items[:first]; the model's
   ``_vis_row_offset_for`` subtracts row_base, so every segment at a >= s splices the right rows.
3. M-RoPE positions come from the grids of ALL items (the cached ones included): ``MMVision.grids`` is restaged as the
   model's grid stash before the request RoPE table is built. Grids are CPU metadata vLLM keeps even when it strips a
   covered item's pixels.
"""

import bisect
import hashlib
import os
from typing import Callable, NamedTuple, Sequence

import numpy as np
import torch

KEY_WORDS = 8  # 256-bit item hash, as int32 words


def mm_prefix_cache_enabled():
    """QWEN36_MM_PREFIX_CACHE (default "0"): "1" lets an image prompt resume from the GDN prefix-state cache."""
    return os.environ.get("QWEN36_MM_PREFIX_CACHE", "0") == "1"


class MMItem(NamedTuple):
    offset: int  # first placeholder position
    length: int  # placeholder count == packed vision rows of the item
    identifier: str  # vLLM mm identifier (content hash)


def parse_items(spans) -> list:
    """The plugin's ``mm_item_spans`` row ([(offset, length, identifier), ...]) as MMItems in position order. Raises on
    overlapping or empty spans."""
    items = sorted(MMItem(int(o), int(n), str(i)) for o, n, i in spans)
    end = 0
    for it in items:
        if it.length <= 0 or it.offset < end or not it.identifier:
            raise ValueError(f"bad multimodal item spans {list(spans)}")
        end = it.offset + it.length
    return items


def item_hash_words(identifier: str) -> np.ndarray:
    return np.frombuffer(hashlib.blake2b(identifier.encode(), digest_size=4 * KEY_WORDS).digest(), dtype=np.int32)


class MMKeyedTokens:
    """GDN prefix-state key material for a multimodal prompt (see the module docstring, point 1).

    ``buf`` = the int32 token ids with KEY_WORDS hash words of item k inserted right before position offset_k.
    ``kpos(pos)`` = the length of buf that stands for tokens[:pos]: pos + KEY_WORDS * (items starting before pos). A
    snapshot taken exactly at an item's offset does not depend on that item and does not include its hash. For a prompt
    without items buf == tokens and kpos is the identity, so the digests are the plain token digests."""

    def __init__(self, tokens, items: Sequence[MMItem]):
        t = np.ascontiguousarray(np.asarray(_as_list(tokens), dtype=np.int32).reshape(-1))
        self.n = int(t.shape[0])
        self._starts = [int(it.offset) for it in items]
        parts, prev = [], 0
        for it in items:
            if it.offset + it.length > self.n:
                raise ValueError(f"item {it} past the prompt ({self.n} tokens)")
            parts.append(t[prev : it.offset])
            parts.append(item_hash_words(it.identifier))
            prev = it.offset
        parts.append(t[prev:])
        self.buf = np.ascontiguousarray(np.concatenate(parts)) if parts else t

    def __len__(self):
        return self.n

    def kpos(self, pos: int) -> int:
        pos = int(pos)
        return pos + KEY_WORDS * bisect.bisect_left(self._starts, pos)


def _as_list(tokens):
    if isinstance(tokens, torch.Tensor):
        return tokens.reshape(-1).tolist()
    return tokens


def placeholders_match(tokens, items: Sequence[MMItem], placeholder_id: int) -> bool:
    """True when the prompt's placeholder ids are exactly the items' spans (no stray placeholder id outside them, none
    missing inside). The splice counts placeholder ids to find its rows, so a mismatch would shift every later row."""
    t = torch.as_tensor(_as_list(tokens)).reshape(-1)
    is_ph = t == int(placeholder_id)
    want = torch.zeros_like(is_ph)
    for it in items:
        if it.offset + it.length > t.numel():
            return False
        want[it.offset : it.offset + it.length] = True
    return bool(torch.equal(is_ph, want))


class MMVision:
    """Vision side of one image prompt whose prefill may resume at a cached position s (module docstring, points 2-3).

    ``pixel_values`` / ``grids``: per item, in position order (pixels may be None for an item the caller has no payload
    for; ``rows_for`` raises MMPayloadMissing if the resumed prefill needs it). ``encode(pixels, grids)`` runs the
    vision tower over the concatenated pixels of several items and returns their packed rows (device tensor); it is
    called at most once per distinct first item (a re-prefill of the row reuses the result). ``key`` is the prompt's
    MMKeyedTokens."""

    def __init__(self, tokens, items, pixel_values, grids, encode: Callable):
        if not (len(items) == len(pixel_values) == len(grids)):
            raise ValueError("one pixel tensor and one grid per item")
        self.items = list(items)
        self.pixel_values = list(pixel_values)
        self.grids = torch.stack([torch.as_tensor(g, dtype=torch.int32).reshape(3) for g in grids], dim=0)
        self._encode = encode
        self._memo = {}
        self.key = MMKeyedTokens(tokens, self.items)
        self.encoded_items = 0  # stats: items run through the tower (memo hits not counted)
        self.skipped_items = 0  # stats: items a resume skipped

    def plan(self, s: int):
        """(first, row_base) for a prefill that recomputes [s, L): items[first:] end after s and need rows."""
        s = int(s)
        first = 0
        while first < len(self.items) and self.items[first].offset + self.items[first].length <= s:
            first += 1
        row_base = sum(it.length for it in self.items[:first])
        return first, row_base

    def rows_for(self, s: int):
        """(vision rows or None, row_base) for a prefill of [s, L). None: no item reaches [s, L) (all are cached)."""
        first, row_base = self.plan(s)
        if first == len(self.items):
            self.skipped_items += len(self.items)
            return None, row_base
        if first not in self._memo:
            missing = [k for k in range(first, len(self.items)) if self.pixel_values[k] is None]
            if missing:
                raise MMPayloadMissing(f"no pixels for item(s) {missing} of a prefill resuming at {s}")
            pv = torch.cat([self.pixel_values[k] for k in range(first, len(self.items))], dim=0)
            self._memo[first] = self._encode(pv, self.grids[first:])
            self.encoded_items += len(self.items) - first
        self.skipped_items += first
        return self._memo[first], row_base


class MMPayloadMissing(RuntimeError):
    pass


def row_offset(placeholders_before: int, row_base: int) -> int:
    """Packed-row index of the first placeholder at or after a segment start, for rows that begin at item ``first``."""
    off = int(placeholders_before) - int(row_base)
    if off < 0:
        raise ValueError(f"segment starts {-off} placeholder rows before the first encoded item")
    return off
