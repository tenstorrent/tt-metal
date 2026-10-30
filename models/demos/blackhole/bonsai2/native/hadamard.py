# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The Bonsai blockwise Hadamard rotation, host-side reference and device layout.

The GGUF stores ``W' = ternary(W @ (H D)^T)``: the weights already absorbed the
rotation, so the *runtime* has to rotate activations instead::

    y = W' (H (D x))      # H block-diagonal, block 1024; D = diag(+-1)

``H`` is the normalized Sylvester Walsh-Hadamard matrix of order 1024,
``H H^T = I``.  ``D`` is a fixed explicit sign vector per *input width*
(5120, 6144, 17408 — the three K dims the model uses).

For the kernel, ``H / 32`` is a scaled integer matrix: every entry is ``+-1/32``,
which is a power of two and therefore exact in BFP2 (1 sign bit + 1 magnitude
bit, no mantissa).  Dividing by 32 keeps the intermediate sum in range for a
BF16 accumulate of 1024 terms while staying exactly representable, and the
caller folds the ``* 32`` back into the following RMSNorm/LayerNorm weight or
into the projection's output scale.

Device shape: one 32x32 BFP2 tile holds a 32x32 block of ``H/32``.  With block
1024 the matrix is 32x32 tiles, so the rotation is a 1024x1024 matmul per
1024-wide slice of the input; a 5120-wide input carries 5 such slices and is
computed as one 5120x5120 block-diagonal matmul (5 tiles of K per output tile,
of which only 1 row-block is nonzero — so it is cheaper to run it as 5 separate
1024-deep matvecs, which is what ``hadamard/op.py`` does).
"""

from __future__ import annotations

import numpy as np


def sylvester(n: int) -> np.ndarray:
    """Normalized Sylvester Walsh-Hadamard matrix, ``float64``, ``H @ H.T == I``."""
    if n & (n - 1):
        raise ValueError(f"Hadamard order must be a power of two, got {n}")
    h = np.ones((1, 1), dtype=np.float64)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h / np.sqrt(n)


class BlockHadamard:
    """Block-diagonal ``H`` (order ``block``) plus the per-width sign vector ``D``.

    ``rotate(x)`` returns ``H (D x)`` for a row-major ``[M, K]`` batch, matching
    what the GGUF runtime does to activations before the matmul.
    """

    def __init__(self, block: int, signs: dict[int, np.ndarray]):
        self.block = block
        self.signs = {int(k): np.asarray(v, dtype=np.float32) for k, v in signs.items()}
        self._h = sylvester(block).astype(np.float32)

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

    def rotate(self, x: np.ndarray, width: int | None = None) -> np.ndarray:
        """``H (D x)`` for ``x`` shaped ``[..., K]`` with ``K`` a multiple of ``block``."""
        x = np.asarray(x, dtype=np.float32)
        k = x.shape[-1]
        width = k if width is None else width
        if width not in self.signs:
            raise ValueError(f"no sign vector for width {width}; have {sorted(self.signs)}")
        if k % self.block:
            raise ValueError(f"K={k} is not a multiple of the block size {self.block}")
        xd = x * self.signs[width]
        return (xd.reshape(*xd.shape[:-1], k // self.block, self.block) @ self._h).reshape(x.shape)

    def rotate_float64(self, x: np.ndarray) -> np.ndarray:
        """``H (D x)`` in float64; the reference path."""
        x = np.asarray(x, dtype=np.float64)
        k = x.shape[-1]
        if k not in self.signs:
            raise ValueError(f"no sign vector for width {k}; have {sorted(self.signs)}")
        if k % self.block:
            raise ValueError(f"K={k} is not a multiple of the block size {self.block}")
        h = sylvester(self.block)  # float64
        xd = x * self.signs[k].astype(np.float64)
        return (xd.reshape(*xd.shape[:-1], k // self.block, self.block) @ h).reshape(x.shape)

    def unrotate(self, w: np.ndarray) -> np.ndarray:
        """``W' @ diag(H) @ diag(D)`` -> the equivalent dense ``W``, shaped ``[N, K]``."""
        w = np.asarray(w, dtype=np.float32)
        n, k = w.shape
        if k % self.block:
            raise ValueError(f"K={k} is not a multiple of the block size {self.block}")
        if k not in self.signs:
            raise ValueError(f"no sign vector for width {k}; have {sorted(self.signs)}")
        dense = (w.reshape(n, k // self.block, self.block) @ self._h).reshape(n, k)
        return dense * self.signs[k][None, :]

    def unrotate_float64(self, w: np.ndarray) -> np.ndarray:
        """``W' @ diag(H) @ diag(D)`` in float64; the reference path."""
        w = np.asarray(w, dtype=np.float64)
        n, k = w.shape
        if k not in self.signs:
            raise ValueError(f"no sign vector for width {k}; have {sorted(self.signs)}")
        if k % self.block:
            raise ValueError(f"K={k} is not a multiple of the block size {self.block}")
        h = sylvester(self.block)
        dense = (w.reshape(n, k // self.block, self.block) @ h).reshape(n, k)
        return dense * self.signs[k].astype(np.float64)[None, :]


    # -- device layout -----------------------------------------------------

    def h_tiles_bfp2(self, scale_log2: int = 5) -> np.ndarray:
        """``H / 2**scale_log2`` as BFP2 bytes, ready to sit resident in L1.

        Returns ``[block // 32, block // 32, 320]`` uint8: row-tile, col-tile,
        one packed BFP2 tile (64 exponent bytes then 256 code bytes).  The
        element order inside a tile is ttnn's face order, so the bytes can be
        copied straight into a CB.
        """
        from .bfp2 import pack_bfp2_tile

        h = self._h / float(1 << scale_log2)
        t = self.block // 32
        out = np.empty((t, t, 320), dtype=np.uint8)
        for tr in range(t):
            for tc in range(t):
                out[tr, tc] = pack_bfp2_tile(h[tr * 32 : (tr + 1) * 32, tc * 32 : (tc + 1) * 32])
        return out

    def __repr__(self) -> str:
        return f"BlockHadamard(block={self.block}, widths={sorted(self.signs)})"
