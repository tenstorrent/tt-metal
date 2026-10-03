# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Paged KV / state layout, page allocator and DRAM capacity model of DeepSeek-V4.1-Flash decode (host side only).

Design: docs/superpowers/specs/2026-10-02-dsv41-kv-paged-capacity-design.md. Nothing here touches the device; the device side
(tt/attention.py, tt/indexer.py) consumes the layout constants, the page table built by ``PageAllocator.page_table`` and the
index translation ``PageLayout.phys_rows``.

What is cached per user (config.json of the checkpoint):
  * window ring: every attention layer (40 backbone + MTP layers) keeps the last ``window`` K==V latents (128 x 512). Fixed size.
  * compressed latents: only the kv-SOURCE layers (2, 8, 14: one entry per 2 tokens, 20: one entry per token) compress; every
    other compressed layer READS its source's entries (layers 3-7 read 2, 9-13 read 8, 15-19 read 14, 21-39 read 20).
  * index keys (128 wide): the index-source layers that own keys (2, 8, 14, 20); layers 24, 28, 32, 36 read layer 20's keys.
  * compressor partial state (ratio 2 only) and the Engram n-gram history: tiny, fixed per user.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

KiB, MiB, GiB = 1 << 10, 1 << 20, 1 << 30

# bytes per element of the storage formats a ttnn tensor can use (block float formats include the shared exponents)
DTYPE_BYTES = {"fp32": 4.0, "bf16": 2.0, "fp8": 1.0, "bfp8": 1088 / 1024, "bfp4": 576 / 1024}

CKPT_DIR = os.environ.get("DSV41_CKPT", "/mnt/tt-data/ssinghal/deepseek-v41-flash")


@dataclass(frozen=True)
class V41Geometry:
    n_layers: int = 40
    n_mtp_layers: int = 3
    window: int = 128
    head_dim: int = 512
    index_head_dim: int = 128
    index_topk: int = 512
    index_n_heads: int = 32
    compress_ratios: tuple = ()
    kv_source_layers: tuple = (2, 8, 14, 20)
    index_source_layers: tuple = (2, 8, 14, 20, 24, 28, 32, 36)
    candidate_source_layer: int = 20
    candidate_topk_blocks: int = 2048
    candidate_block_size: int = 8
    compressor_state_ratio_gt1_layers: tuple = (2, 8, 14)
    max_context: int = 1 << 20

    @classmethod
    def from_config(cls, path: str | None = None) -> "V41Geometry":
        path = path or os.path.join(CKPT_DIR, "inference", "config.json")
        c = json.load(open(path))
        return cls(
            n_layers=c["n_layers"],
            n_mtp_layers=c["n_mtp_layers"],
            window=c["window_size"],
            head_dim=c["head_dim"],
            index_head_dim=c["index_head_dim"],
            index_topk=c["index_topk"],
            index_n_heads=c["index_n_heads"],
            compress_ratios=tuple(c["compress_ratios"]),
            kv_source_layers=tuple(c["kv_source_layers"]),
            index_source_layers=tuple(c["index_source_layers"]),
            candidate_source_layer=c["candidate_source_layer"],
            candidate_topk_blocks=c["candidate_topk_blocks"],
            candidate_block_size=c["candidate_block_size"],
        )

    # ---- structure ---------------------------------------------------------------------------------------------
    def ratio(self, layer: int) -> int:
        return self.compress_ratios[layer]

    def kv_source(self, layer: int) -> int | None:
        """The kv-source layer whose compressed entries ``layer`` reads (None: window-only layer)."""
        if self.ratio(layer) == 0:
            return None
        return max(s for s in self.kv_source_layers if s <= layer)

    def index_source(self, layer: int) -> int | None:
        """The index-source layer whose top-k selection ``layer`` uses (None: window-only layer)."""
        if self.ratio(layer) == 0:
            return None
        return max(s for s in self.index_source_layers if s <= layer)

    def selection_groups(self) -> dict:
        """(kv source, index source) -> layers sharing one selected-entry set (the 512 compressed rows are identical for all of them)."""
        groups: dict = {}
        for layer in range(self.n_layers):
            if self.ratio(layer):
                groups.setdefault((self.kv_source(layer), self.index_source(layer)), []).append(layer)
        return groups

    def readers(self, source: int) -> list:
        return [layer for layer in range(self.n_layers) if self.kv_source(layer) == source]


@dataclass(frozen=True)
class Formats:
    """Storage formats (keys of DTYPE_BYTES)."""

    ring: str = "bf16"  # window ring (TILE, written by paged_update_cache)
    comp: str = "bf16"  # compressed latents (K == V)
    idx: str = "bf16"  # index keys
    name: str = "bf16/bf16/bf16"

    @staticmethod
    def presets() -> dict:
        return {
            "bf16": Formats("bf16", "bf16", "bf16", "all bf16"),
            "lean": Formats("bf16", "fp8", "bfp8", "ring bf16, latents fp8_e4m3, index keys bfp8"),
            "leaner": Formats("bfp8", "fp8", "bfp4", "ring bfp8, latents fp8_e4m3, index keys bfp4"),
        }


# ---- per-user memory ---------------------------------------------------------------------------------------------
def comp_bytes_per_token(g: V41Geometry, fmt: Formats, shared: bool = True) -> float:
    """Bytes of compressed latents per context token and user. ``shared=False`` is today's code: every reading layer keeps a copy."""
    b = g.head_dim * DTYPE_BYTES[fmt.comp]
    if shared:
        return sum(b / g.ratio(s) for s in g.kv_source_layers)
    return sum(b / g.ratio(layer) for layer in range(g.n_layers) if g.ratio(layer))


def idx_bytes_per_token(g: V41Geometry, fmt: Formats) -> float:
    """Index keys exist only in the index-source layers that own keys (= kv sources); 24/28/32/36 read layer 20's."""
    b = g.index_head_dim * DTYPE_BYTES[fmt.idx]
    return sum(b / g.ratio(s) for s in g.kv_source_layers if s in g.index_source_layers)


def fixed_bytes_per_user(g: V41Geometry, fmt: Formats, spec_k: int = 0, mtp: bool = False) -> dict:
    """Context-independent per-user state. ``spec_k``: extra tokens a verify step writes speculatively (the ring needs that many
    spare slots so a rollback does not lose window entries; rounded up to a tile row block of 32)."""
    ring_rows = g.window + (-(-spec_k // 32) * 32 if spec_k else 0)
    layers = g.n_layers + (g.n_mtp_layers if mtp else 0)
    ring = layers * ring_rows * g.head_dim * DTYPE_BYTES[fmt.ring]
    # compressor state of the ratio-2 sources: [kv | score] fp32 for the previous token (+ spec_k snapshots)
    comp_state = len(g.compressor_state_ratio_gt1_layers) * (1 + spec_k) * 2 * g.head_dim * 4
    return {"ring": ring, "compressor_state": comp_state, "total": ring + comp_state}


def page_waste_bytes(g: V41Geometry, fmt: Formats, page_tokens: int) -> float:
    """Average internal fragmentation per user: half a page of every per-token pool."""
    return 0.5 * page_tokens * (comp_bytes_per_token(g, fmt) + idx_bytes_per_token(g, fmt))


# ---- prefill reservation (ESTIMATE: to be replaced by the prefill agent's measurement) -------------------------------
PREFILL_BYTES_PER_TOKEN = int(
    0.6 * MiB
)  # per chip per chunk token: mHC streams (3 x 80 KiB fp32), q/o heads, MoE dispatch/combine/inter


def prefill_reserve_bytes(tokens_per_chip: int, ctx: int, shard_cols: int = 1, g: V41Geometry | None = None) -> float:
    """Chunk activations + the indexer score block of the densest layer (ratio 1: one score per context token, bf16)."""
    return tokens_per_chip * PREFILL_BYTES_PER_TOKEN + tokens_per_chip * ctx * 2.0 / shard_cols


# ---- capacity ----------------------------------------------------------------------------------------------------
def max_context(
    users_per_row: int,
    free_bytes: float,
    g: V41Geometry,
    fmt: Formats,
    shard_cols: int = 1,
    spec_k: int = 0,
    mtp: bool = False,
    reserve: float = 0.0,
    page_tokens: int = 128,
    prefill_chunk: int = 0,
    shared: bool = True,
) -> int:
    """Largest uniform per-user context (tokens) so that ``users_per_row`` users fit in ``free_bytes - reserve`` of one chip.

    shard_cols = 1: the compressed/index caches are replicated over the 8 mesh columns (every chip of a row holds every user of that
    row); shard_cols = 8: they are page-sharded / sequence-sharded over the columns (window ring and state stay replicated).
    """
    per_tok = (comp_bytes_per_token(g, fmt, shared) + idx_bytes_per_token(g, fmt)) / shard_cols
    fixed = fixed_bytes_per_user(g, fmt, spec_k, mtp)["total"] + page_waste_bytes(g, fmt, page_tokens) / shard_cols
    # linear in ctx: free - reserve - chunk*(P + 2*ctx/shard) = U * (fixed + per_tok * ctx)
    c = prefill_chunk
    num = free_bytes - reserve - c * PREFILL_BYTES_PER_TOKEN - users_per_row * fixed
    den = users_per_row * per_tok + 2.0 * c / shard_cols
    ctx = max(0, int(num / den))
    return min(ctx, g.max_context)


def capacity_table(
    free_gib: float,
    g: V41Geometry,
    fmt: Formats,
    shard_cols: int,
    reserve_gib: float = 0.0,
    prefill_chunk: int = 0,
    spec_k: int = 0,
    mtp: bool = False,
    users=(1, 2, 4, 8, 16, 32),
) -> list:
    return [
        max_context(u, free_gib * GiB, g, fmt, shard_cols, spec_k, mtp, reserve_gib * GiB, prefill_chunk=prefill_chunk)
        for u in users
    ]


def fmt_tokens(n: int, cap: int = 1 << 20) -> str:
    if n >= cap:
        return "1M"
    if n >= 1024:
        return f"{n // 1024}k"
    return str(n)


# ---- paging ----------------------------------------------------------------------------------------------------------
@dataclass
class PageLayout:
    """One page table for every token-indexed pool: a page = ``page_tokens`` context tokens. A page of the compressed pool holds
    the rows ``[src2: P/2 | src8: P/2 | src14: P/2 | src20: P]`` (P = page_tokens), i.e. 2.5 * P rows of 512, so ONE page id addresses
    the latents of all four kv sources. A logical entry ``j`` of source ``s`` (ratio r_s) lives in page ``(j * r_s) // P``.
    """

    page_tokens: int = 128
    sources: tuple = (2, 8, 14, 20)
    ratios: tuple = (2, 2, 2, 1)

    @property
    def rows_per_page(self) -> int:
        return sum(self.page_tokens // r for r in self.ratios)

    def row_offset(self, src_pos: int) -> int:
        return sum(self.page_tokens // r for r in self.ratios[:src_pos])

    def phys_rows(self, page_table_row, src_pos: int, entries):
        """Logical entry ids of source ``src_pos`` -> physical row ids of the flat [num_pages * rows_per_page, 512] pool (torch/numpy)."""
        r = self.ratios[src_pos]
        per_page = self.page_tokens // r
        page = page_table_row[entries // per_page]
        return page * self.rows_per_page + self.row_offset(src_pos) + entries % per_page


class PageAllocator:
    """Host-owned allocator of token pages (one pool shared by all users of a mesh row; every column holds the same page ids)."""

    FREE = 0xFFFF  # page-table padding (int32 on device: -1)

    def __init__(self, num_pages: int, page_tokens: int = 128, max_pages_per_user: int | None = None):
        self.num_pages, self.page_tokens = num_pages, page_tokens
        self.max_pages = max_pages_per_user or num_pages
        self._free = list(range(num_pages - 1, -1, -1))  # stack: low page ids first
        self.pages: dict = {}  # user -> [page ids]
        self.length: dict = {}  # user -> tokens

    def free_pages(self) -> int:
        return len(self._free)

    def _need(self, tokens: int) -> int:
        return -(-tokens // self.page_tokens)

    def admit(self, user, tokens: int, reserve_tokens: int = 0) -> list:
        """Allocate pages for ``tokens`` (+ ``reserve_tokens`` of look-ahead, e.g. the 1 + k tokens a spec-decode step may write)."""
        assert user not in self.pages, f"user {user} already admitted"
        self.pages[user], self.length[user] = [], 0
        try:
            self.grow(user, tokens, reserve_tokens)
        except MemoryError:
            self.release(user)
            raise
        return self.pages[user]

    def grow(self, user, tokens: int, reserve_tokens: int = 0) -> list:
        need = self._need(tokens + reserve_tokens)
        if need > self.max_pages:
            raise MemoryError(f"user {user}: {need} pages exceed the per-user limit {self.max_pages}")
        have = self.pages[user]
        if need - len(have) > len(self._free):
            raise MemoryError(f"pool exhausted: need {need - len(have)} pages, {len(self._free)} free")
        while len(have) < need:
            have.append(self._free.pop())
        self.length[user] = max(self.length[user], tokens)
        return have

    def rollback(self, user, tokens: int, keep_reserve_tokens: int = 0) -> None:
        """Spec-decode rollback: the accepted length shrinks to ``tokens``; pages beyond the look-ahead go back to the pool. The data of
        rejected positions is never cleared: it is overwritten by the next steps and masked by the position until then.
        """
        assert tokens <= self.length[user]
        self.length[user] = tokens
        need = self._need(tokens + keep_reserve_tokens)
        have = self.pages[user]
        while len(have) > max(need, 1):
            self._free.append(have.pop())

    def release(self, user) -> None:
        self._free.extend(reversed(self.pages.pop(user, [])))
        self.length.pop(user, None)

    def page_table(self, users, max_pages: int | None = None):
        """int32 [len(users), max_pages] torch tensor, -1 padded: the tensor uploaded (replicated over columns, sharded over rows)."""
        import torch

        mp = max_pages or self.max_pages
        t = torch.full((len(users), mp), -1, dtype=torch.int32)
        for i, u in enumerate(users):
            p = self.pages.get(u, [])
            t[i, : len(p)] = torch.tensor(p, dtype=torch.int32)
        return t
