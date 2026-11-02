# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The Bonsai blockwise Hadamard rotation, host-side reference and device layout.

The GGUF stores ``W' = ternary(W @ (H D)^T)``: the weights already absorbed the
rotation, so the *runtime* has to rotate activations instead::

    y = W' (H (D x))      # H block-diagonal, block 1024; D = diag(+-1)

``H`` is the normalized Sylvester Walsh-Hadamard matrix of order 1024,
``H H^T = I``.  ``D`` is a fixed explicit sign vector per *input width*
(5120, 6144, 17408 -- the three K dims the model uses).

``H`` is symmetric, so ``x @ H`` and ``H @ x`` coincide.  That lets the
reference use the fast Walsh-Hadamard butterfly (``O(n log n)``) instead of a
dense ``O(n^2)`` product: a float64 unrotation of a 17408-wide weight is
otherwise a 17408x17408 matmul per row.

For the kernel, ``H / 32`` is a scaled integer matrix: every entry is ``+-1/32``,
a power of two and therefore exact in BFP2 (1 sign bit + 1 magnitude bit, no
mantissa).  Dividing by 32 keeps the intermediate sum in range while staying
exactly representable, and the caller folds the ``* 32`` back into the
following norm weight or the projection's output scale.
"""

from __future__ import annotations

import numpy as np

TILE = 32
BFP2_TILE_BYTES = 320


def sylvester(n: int) -> np.ndarray:
    """Normalized Sylvester Walsh-Hadamard matrix, ``float64``, ``H @ H.T == I``."""
    if n & (n - 1):
        raise ValueError(f"Hadamard order must be a power of two, got {n}")
    h = np.ones((1, 1), dtype=np.float64)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h / np.sqrt(n)


def fwht(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Unnormalized fast Walsh-Hadamard transform along ``axis``, in float64.

    ``fwht(x)[..., i] == sum_j (+-1)**popcount(i & j) x[..., j]``, i.e.
    ``sylvester(n) * sqrt(n) @ x``.  ``O(n log n)``.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[axis]
    if n & (n - 1):
        raise ValueError(f"transform length must be a power of two, got {n}")
    y = np.moveaxis(x, axis, -1).copy()
    lead = y.shape[:-1]
    y = y.reshape(-1, n)  # each row is transformed independently
    h = 1
    while h < n:
        # Split the innermost axis into (chunk of 2h) -> (2, h) and butterfly.
        z = y.reshape(-1, n // (2 * h), 2, h)
        a = z[:, :, 0, :].copy()
        b = z[:, :, 1, :].copy()
        z[:, :, 0, :] = a + b
        z[:, :, 1, :] = a - b
        y = z.reshape(-1, n)
        h *= 2
    return np.moveaxis(y.reshape(*lead, n), -1, axis)


class BlockHadamard:
    """Block-diagonal ``H`` (order ``block``) plus the per-width sign vector ``D``.

    ``rotate(x)`` returns ``H (D x)`` for a row-major ``[M, K]`` batch, matching
    what the GGUF runtime does to activations before the matmul.
    """

    def __init__(self, block: int, signs: dict[int, np.ndarray]):
        self.block = block
        self.signs = {int(k): np.asarray(v, dtype=np.float32) for k, v in signs.items()}

    @classmethod
    def from_metadata(cls, kv: dict) -> "BlockHadamard":
        """Build from GGUF ``prism.hadamard.*`` metadata."""
        if kv.get("prism.hadamard.transform") != "normalized-sylvester-walsh-hadamard":
            raise ValueError(f"unsupported transform {kv.get('prism.hadamard.transform')!r}")
        if kv.get("prism.hadamard.sign_mode") != "explicit":
            raise ValueError(f"unsupported sign mode {kv.get('prism.hadamard.sign_mode')!r}")
        if kv.get("prism.hadamard.axis") != "input-last-dimension":
            raise ValueError(f"unsupported rotation axis {kv.get('prism.hadamard.axis')!r}")
        block = int(kv["prism.hadamard.block_size"])
        widths = [int(w) for w in kv["prism.hadamard.sign_widths"]]
        values = np.asarray(kv["prism.hadamard.sign_values"], dtype=np.float32)
        if values.size != sum(widths):
            raise ValueError(f"{values.size} sign values for widths summing to {sum(widths)}")
        signs, off = {}, 0
        for w in widths:
            signs[w] = values[off : off + w]
            off += w
        return cls(block, signs)

    # -- reference math ----------------------------------------------------

    def _rot(self, v: np.ndarray) -> np.ndarray:
        """Blockwise ``v @ blockdiag(H)`` via the butterfly."""
        k = v.shape[-1]
        if k % self.block:
            raise ValueError(f"K={k} is not a multiple of the block size {self.block}")
        blocks = v.reshape(*v.shape[:-1], k // self.block, self.block)
        return (fwht(blocks) / np.sqrt(self.block)).reshape(v.shape)

    def rotate(self, x: np.ndarray, width: int | None = None) -> np.ndarray:
        """``H (D x)`` for ``x`` shaped ``[..., K]`` with ``K`` a multiple of ``block``."""
        x = np.asarray(x, dtype=np.float64)
        width = x.shape[-1] if width is None else width
        if width not in self.signs:
            raise ValueError(f"no sign vector for width {width}; have {sorted(self.signs)}")
        return self._rot(x * self.signs[width].astype(np.float64))

    def unrotate(self, w: np.ndarray) -> np.ndarray:
        """``W' @ diag(H) @ diag(D)`` -> the equivalent dense ``W``, shaped ``[N, K]``."""
        w = np.asarray(w, dtype=np.float64)
        k = w.shape[-1]
        if k not in self.signs:
            raise ValueError(f"no sign vector for width {k}; have {sorted(self.signs)}")
        return self._rot(w) * self.signs[k].astype(np.float64)[None, :]

    # -- device layout -----------------------------------------------------

    def h_tiles_bfp2(self, scale_log2: int = 0) -> np.ndarray:
        """``H / 2**scale_log2`` as BFP2 bytes, ready to sit resident in L1.

        The normalized Sylvester matrix is already ``+-1/32`` at block 1024, so
        the default (``scale_log2=0``) is the exact device matrix the plan
        describes: every entry is a power of two, hence exact in BFP2 with no
        mantissa.  A larger ``scale_log2`` divides further, which callers can
        use to keep a 1024-term accumulate in range.

        Returns ``[block // 32, block // 32, 320]`` uint8: row-tile, col-tile,
        one packed BFP2 tile (64 exponent bytes then 256 code bytes).  The
        element order inside a tile is ttnn's face order, so the bytes can be
        copied straight into a CB.
        """
        from .bfp2 import pack_tiles

        t = self.block // TILE
        h = self._h() / float(1 << scale_log2)
        tgt = 1.0 / (1 << scale_log2) / np.sqrt(self.block)
        if not np.all(np.isin(h, (-tgt, tgt))):
            raise ValueError("a Hadamard entry scaled by a power of two must be exactly +-2**-k")
        tiles = np.sign(h).astype(np.int8).reshape(t, TILE, t, TILE).transpose(0, 2, 1, 3)
        return pack_tiles(np.ascontiguousarray(tiles).reshape(t * t, TILE, TILE)).reshape(t, t, BFP2_TILE_BYTES)

    def _h(self) -> np.ndarray:
        """The rotation matrix itself, materialized on demand (it is 8 MB at block 1024)."""
        return sylvester(self.block)

    def __repr__(self) -> str:
        return f"BlockHadamard(block={self.block}, widths={sorted(self.signs)})"
