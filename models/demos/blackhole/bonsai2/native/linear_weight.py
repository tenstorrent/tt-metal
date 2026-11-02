# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""A ternary weight prepared once, streamed many times.

The A0 profiling result drives this design. The DeepSeek streaming op was 20.6 ms
of wall clock for 1.99 ms of device kernel, because its Python entry point
rebuilds the per-core metadata (one L1 tensor of subblock byte sizes plus one of
per-tile format infos) on *every* call. Those tensors depend only on the weight,
never on the activation, so an engine that builds them once and reuses them pays
that cost once per model instead of once per token.

``TernaryWeight`` is that prepared object. It owns:

  codes      packed BFP2 tiles, in the column-major-within-bank order the
             streaming reader walks, as one DRAM tensor
  scales     fp16, one per (output column, group), in the same walk order
  meta       per-core subblock byte counts, built once
  per-core   compile-time args (bank id, VC, column range, DRAM offset)

and hands the kernel the addresses it needs. Nothing here is per-call state, so
a decode step is one dispatch with no host work beyond the activation.

Why a separate scale tensor rather than interleaving scales into the tile stream:
a CB page is a fixed 320 B and the unpacker is configured for that page size, so
scales cannot be spliced between tiles without either a second CB or a variable
page size. The plan lists the interleaved form first and a separate scale tensor
as the fallback; the separate tensor is what this implements, and the byte cost
is the same (the scales are read either way) -- it only costs one extra read
descriptor per subblock.
"""

from __future__ import annotations

import numpy as np

from .bfp2 import BFP2_TILE_BYTES, TILE
from .pack import padded_n, tile_column_major

#: Tiles per subblock, derived from the group size: a subblock is exactly one
#: scale group, so the epilogue applies one scale per subblock.
def subblock_k_for_group(group: int) -> int:
    if group % TILE:
        raise ValueError(f"group size {group} is not a whole number of {TILE}-wide tiles")
    return group // TILE


class TernaryWeight:
    """A ternary matrix laid out and pre-measured for the streaming engine.

    ``codes`` is ``[N, K]`` in {-1, 0, +1} int8 (matmul order, ``y = x @ W.T``)
    and ``scales`` is ``[N, K // group]`` fp16, both exactly as ``gguf_ternary``
    produces them.
    """

    def __init__(
        self,
        codes: np.ndarray,
        scales: np.ndarray,
        group: int,
        num_banks: int,
        cores_per_bank: int = 1,
        name: str = "",
    ):
        from .bfp2 import pack_tiles

        codes = np.asarray(codes)
        if codes.ndim != 2:
            raise ValueError(f"codes must be [N, K], got {codes.shape}")
        n, k = codes.shape
        if k % group:
            raise ValueError(f"K={k} is not a multiple of the group size {group}")
        if scales.shape != (n, k // group):
            raise ValueError(f"scales shape {scales.shape} != {(n, k // group)}")

        self.name = name
        self.N = n
        self.K = k
        self.group = group
        self.subblock_k = subblock_k_for_group(group)
        self.num_groups = k // group
        self.num_banks = num_banks
        self.cores_per_bank = cores_per_bank
        self.num_cores = num_banks * cores_per_bank

        self.n_padded = padded_n(n, num_banks, cores_per_bank)
        self.tiles_per_core = self.n_padded // self.num_cores // TILE

        # --- tile image, in the reader's walk order ---
        tiles = tile_column_major(codes, self.n_padded, num_banks)
        self.tile_bytes = pack_tiles(tiles.reshape(-1, TILE, TILE)).reshape(-1)
        expected_tiles = (self.n_padded // TILE) * (k // TILE)
        if tiles.shape[0] * tiles.shape[1] != expected_tiles:
            raise AssertionError(
                f"tile image has {tiles.shape[0] * tiles.shape[1]} tiles, expected {expected_tiles} "
                f"({self.n_padded} padded columns x {k // TILE} K tiles)"
            )
        if self.tile_bytes.size != expected_tiles * BFP2_TILE_BYTES:
            raise AssertionError(
                f"packed {self.tile_bytes.size} bytes, expected {expected_tiles * BFP2_TILE_BYTES}"
            )

        # --- scales, in the same (bank, group, column) walk order ---
        sc = np.asarray(scales, dtype=np.float16)
        if self.n_padded != n:
            sc = np.concatenate([sc, np.zeros((self.n_padded - n, self.num_groups), dtype=np.float16)], axis=0)
        self.scale_bytes = np.ascontiguousarray(
            sc.reshape(num_banks, self.n_padded // num_banks, self.num_groups).transpose(0, 2, 1)
        ).reshape(-1)

        # --- bytes the reader pulls per subblock, one entry per (column, subblock) ---
        # With a uniform encoding every subblock is the same size, so the metadata
        # is constant: this is why the engine needs no per-tile format table.
        self.subblock_bytes = self.subblock_k * BFP2_TILE_BYTES
        self.meta = np.full(self.tiles_per_core * self.num_groups, self.subblock_bytes, dtype=np.uint32)

    # -- geometry the kernel needs, all derivable without per-call work ----

    @property
    def bytes_per_token(self) -> int:
        """Weight bytes one decode step reads (codes + scales, padding excluded)."""
        return self.N * self.K // 4 + self.N * self.num_groups * 2

    @property
    def padded_bytes(self) -> int:
        """Bytes actually resident in DRAM, including the column padding."""
        return self.tile_bytes.size + self.scale_bytes.size * 2

    def __repr__(self) -> str:
        return (
            f"TernaryWeight({self.name or 'anon'} [{self.N}, {self.K}] G={self.group} "
            f"N_padded={self.n_padded} cores={self.num_cores} "
            f"{self.padded_bytes / 1e6:.1f} MB)"
        )


def weight_from_gguf(gguf, name: str, num_banks: int, cores_per_bank: int = 1) -> TernaryWeight:
    """Read one tensor out of a GGUF file and prepare it for the engine."""
    from .gguf_ternary import read_ternary

    t = read_ternary(gguf, name)
    return TernaryWeight(t.codes, t.scales, t.group, num_banks, cores_per_bank, name=name)
