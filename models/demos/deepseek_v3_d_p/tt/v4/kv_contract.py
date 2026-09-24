# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The DeepSeek-V4-Flash prefill <-> decode KV contract, as plain data (no ttnn, no torch).

Both sides of the KV migration import THIS module: the tt-metal prefill worker builds its KvChunkAddressTable
configs from it, and the tt-blaze decode ``kv_migration_spec`` (blaze/models/deepseek_v4_flash/kv_migration.py)
asserts against it. Migration is a verbatim byte copy per (config, layer, 32-row chunk, slot), so the two sides
must agree on: config ORDER and names, per-config dtype/layout and row width (hence ``chunk_size_bytes``), the
meaning of the position axis, and the migratable extent per layer.

Decode's caches, which fix the dtypes (tt-blaze ``tests/blaze/fused_ops/dsv4_hca_layer/harness.py:545, :4103-4124``;
``blaze/ops/csa_attention/op.py:6-31``):
  * CSA layers: ONE unified ROW_MAJOR bf16 cache per layer -- rows [0, 128) are the sliding window at row
    ``p % 128``, row ``128 + w`` is compressed entry ``w = p // 4``.
  * HCA layers: the same window + entries layout at ``128 + p // 128`` in ``_CACHE_DTYPE`` (bfp8_b TILE by
    default, bfp4_b when the ring runs with DSV4_FLASH_CACHE_BF4=1).
  * SWA layers (0, 1): window only, ``_CACHE_DTYPE``.
  * CSA indexer key cache: bfp8_b TILE, 128 wide, one row per compressed entry (``p >> 2``).
  * Pending compressor state (configs 4-5): the partial-window ``[kv | gate]`` projections the compressor needs
    to finish its next entry. A contract GAP found 2026-09-24: decode cannot rebuild it from attention KV; where
    decode keeps it must be confirmed with the decode owner (postmortem DS4F-0242). Kept LAST so the first four
    configs are stable whatever the resolution.

Position axis: the UNIFIED ROW ``r``. Window rows hold token ``p`` at ``r = p % 128`` ("copy the ring
verbatim"); entry rows hold entry ``w`` at ``r = 128 + w``. The index-key cache is indexed by ``w`` directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

TILE = 32
CHUNK_N_TOKENS = 32  # rows per migration chunk; every config uses the same granularity

WINDOW = 128
HEAD_DIM = 512
INDEX_HEAD_DIM = 128
CSA_RATE = 4
HCA_RATE = 128

# dtype tags (ttnn-free); bytes per 32-row chunk of ``width`` columns
_BFP8_TILE_BYTES = 1088  # 32x32 bfp8_b tile: 1024 B mantissas + 64 B shared exponents
_BFP4_TILE_BYTES = 576  # 32x32 bfp4_b tile: 512 B mantissas + 64 B shared exponents
DTYPE_TAGS = ("bfp8_tile", "bfp4_tile", "bf16_rm")


def chunk_size_bytes(dtype_tag: str, width: int, chunk_n_tokens: int = CHUNK_N_TOKENS) -> int:
    """Bytes of one ``chunk_n_tokens``-row chunk of a ``width``-wide cache in ``dtype_tag``."""
    if width % TILE:
        raise ValueError(f"width {width} must be a multiple of {TILE}")
    if chunk_n_tokens % TILE:
        raise ValueError(f"chunk_n_tokens {chunk_n_tokens} must be a multiple of {TILE}")
    tiles = (chunk_n_tokens // TILE) * (width // TILE)
    if dtype_tag == "bfp8_tile":
        return tiles * _BFP8_TILE_BYTES
    if dtype_tag == "bfp4_tile":
        return tiles * _BFP4_TILE_BYTES
    if dtype_tag == "bf16_rm":
        return chunk_n_tokens * width * 2
    raise ValueError(f"unknown dtype tag {dtype_tag!r}; expected one of {DTYPE_TAGS}")


def window_cache_dtype_tag() -> str:
    """The ring's ``_CACHE_DTYPE`` for the HCA/SWA caches: bfp8_b unless DSV4_FLASH_CACHE_BF4=1 (same env var
    the decode harness reads, so one switch moves both sides)."""
    return "bfp4_tile" if os.environ.get("DSV4_FLASH_CACHE_BF4") == "1" else "bfp8_tile"


def tiles_up(rows: int) -> int:
    return -(-int(rows) // TILE) * TILE


def compressed_entries(tokens: int, compress_rate: int) -> int:
    """Compressed entries a compressor emits for ``tokens`` source tokens: one per full window."""
    return int(tokens) // int(compress_rate)


@dataclass(frozen=True)
class KvGroupSpec:
    """One migratable config. ``kind`` names the attention kind whose layers this config covers (the attention
    kind constants live in ``layer_kinds``; here they are plain strings). ``extent(max_seq_len)`` is the migratable
    row count per layer (``per_layer_seq_len`` on the decode side), always a multiple of ``CHUNK_N_TOKENS``."""

    name: str
    kind: str
    dtype_tag: str
    width: int
    pending: bool = False  # configs 4-5: compressor partial-window state (contract gap, see module docstring)
    # Rows ALLOCATED past the migratable extent: the entry writers write whole tiles from the tile boundary below
    # the entry count (TtHCA's tail-tile write covers up to 96 rows for a 5120-token chunk), so the tensor needs
    # that headroom; those rows are zero and are never migrated (extent() excludes them).
    write_headroom: int = 0

    @property
    def chunk_size_bytes(self) -> int:
        return chunk_size_bytes(self.dtype_tag, self.width)

    def alloc_rows(self, max_seq_len: int) -> int:
        return self.extent(max_seq_len) + self.write_headroom

    def extent(self, max_seq_len: int) -> int:
        s = int(max_seq_len)
        if self.name == "swa_window":
            return WINDOW
        if self.name == "hca_unified":
            return WINDOW + tiles_up(compressed_entries(s, HCA_RATE))
        if self.name == "csa_unified":
            return WINDOW + tiles_up(compressed_entries(s, CSA_RATE))
        if self.name == "csa_index_k":
            return tiles_up(compressed_entries(s, CSA_RATE))
        if self.name == "csa_pending":
            # rows 0..3: the last window's Ca-series [kv | gate] rows (HEAD_DIM each -> 1024 wide);
            # rows 4..7: the indexer compressor's Ca rows ([kv | gate] of INDEX_HEAD_DIM each, zero-padded)
            return TILE
        if self.name == "hca_pending":
            # up to 127 token rows of [kv | gate] past the last full 128-window, plus a count row
            return WINDOW
        raise ValueError(f"unknown group {self.name}")


def build_contract(window_dtype_tag: str | None = None) -> tuple[KvGroupSpec, ...]:
    """The ordered contract. ``window_dtype_tag`` defaults to the ring's ``_CACHE_DTYPE`` (bfp8_b)."""
    wd = window_dtype_tag or window_cache_dtype_tag()
    return (
        KvGroupSpec("swa_window", "sliding_attention", wd, HEAD_DIM),
        KvGroupSpec("hca_unified", "heavily_compressed_attention", wd, HEAD_DIM, write_headroom=96),
        KvGroupSpec("csa_unified", "compressed_sparse_attention", "bf16_rm", HEAD_DIM, write_headroom=TILE),
        KvGroupSpec("csa_index_k", "compressed_sparse_attention", "bfp8_tile", INDEX_HEAD_DIM, write_headroom=TILE),
        KvGroupSpec("csa_pending", "compressed_sparse_attention", "bf16_rm", 2 * HEAD_DIM, pending=True),
        KvGroupSpec("hca_pending", "heavily_compressed_attention", "bf16_rm", 2 * HEAD_DIM, pending=True),
    )


CONTRACT: tuple[KvGroupSpec, ...] = build_contract()
KV_GROUPS: tuple[str, ...] = tuple(g.name for g in CONTRACT)
MIGRATED_GROUPS: tuple[str, ...] = tuple(g.name for g in CONTRACT if not g.pending)
PENDING_GROUPS: tuple[str, ...] = tuple(g.name for g in CONTRACT if g.pending)


def spec(name: str, contract: tuple[KvGroupSpec, ...] = CONTRACT) -> KvGroupSpec:
    for g in contract:
        if g.name == name:
            return g
    raise KeyError(name)


def validate_contract(contract: tuple[KvGroupSpec, ...] = CONTRACT, max_seq_lens=(5120, 16384, 65536, 133120)) -> None:
    """Structural checks both repos run device-free: unique names, tile-aligned widths/extents, positive chunk
    bytes, and the pending configs last."""
    names = [g.name for g in contract]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate config names: {names}")
    seen_pending = False
    for g in contract:
        if g.pending:
            seen_pending = True
        elif seen_pending:
            raise ValueError("pending configs must come last in the contract")
        if g.dtype_tag not in DTYPE_TAGS:
            raise ValueError(f"{g.name}: bad dtype tag {g.dtype_tag}")
        if g.width % TILE or g.chunk_size_bytes <= 0:
            raise ValueError(f"{g.name}: width {g.width} / chunk bytes {g.chunk_size_bytes}")
        for s in max_seq_lens:
            e = g.extent(s)
            if e <= 0 or e % CHUNK_N_TOKENS:
                raise ValueError(f"{g.name}: extent {e} at max_seq_len {s} is not a multiple of {CHUNK_N_TOKENS}")
