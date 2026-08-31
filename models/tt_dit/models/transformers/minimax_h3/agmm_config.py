# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Matmul block sizes for the MiniMax-H3 all-gather-matmul shapes.

Keyed on `(K, N)` only. `get_matmul_config` keys its own tables on `(M, K, N)`, but in
this model M is the per-device packed sequence length, which changes with the requested video
duration -- 4768 / 9216 / 13632 at 768P for 5s / 10s / 15s, and anything else a caller asks for. K and
N are fixed by the architecture and the TP factor. M only sets how many blocks each core walks
through, and it is large enough at every duration that the block shape does not want to change with
it, so keying on `(K, N)` gives one entry per matmul instead of one per matmul per duration.

Passed to the linear layers as `default_block_size`, which `get_matmul_config` uses when its own
`(M, K, N)` lookup misses -- so these apply without registering anything globally.

Two hard constraints the op asserts on, both hit while bringing this up:

`N_block` must be divisible by `subblock_w`, which `get_matmul_config` leaves at 2 -- so N_block must
be even. (For a fused-SwiGLU ff1 it must be even anyway: gate/up tile pairs interleave along N and a
block must never split a pair.)

`K_block` must divide `K_tiles_per_device`
(`K / 32 / tp_factor`). The ring delivers the gathered K in `K_block`-sized chunks and a partial
final chunk is unsupported; the op asserts on it. With TP=4 that means K_block divides 42 for
K=5376 and 56 for K=7168 -- 8 (the generic default) divides neither, which is why these shapes need
an entry at all rather than falling back.
Measured with `models/tt_dit/utils/sweep_mm_block_sizes.py` on 4x8 Blackhole Galaxy at the 12x9 grid
the model uses (one core column reserved for CCL), 811 combos over the three shapes, at M=4768:

    (5376, 5376)  to_qkv   (8, 7, 12)  1416 us   vs (8, 7, 8): 1634 us, -13.4%
    (7168, 1344)  to_out   (8,  8, 6)   890 us   best among usable combos
    (5376, 7168)  ff1      (8, 3, 14)  2089 us   vs (8, 7, 8): 2365 us, -11.7%

The sweep's own best for to_out was (8, 8, 5) at 875 us -- 1.7% faster -- but it needs subblock
(4, 1), and `get_matmul_config` hardcodes subblock (2, 2) for anything supplied via
`default_block_size`. Expressing it would mean registering a full `(M, K, N)` entry with an explicit
subblock, which reintroduces the M-keying this table exists to avoid, for 15 us on a 0.9 ms matmul.
So these are the best combos reachable at subblock (2, 2), which for the other two shapes is also the
global best.
"""

from __future__ import annotations

# (K, N) -> (M_block, K_block, N_block). N is the *per-device* output width.
#   (5376, 5376)  attention to_qkv   K_tiles_per_device = 42
#   (7168, 1344)  attention to_out   K_tiles_per_device = 56
#   (5376, 7168)  feed-forward ff1   K_tiles_per_device = 42, fused SwiGLU so N_block must be even
AGMM_BLOCK_SIZES: dict[tuple[int, int], tuple[int, int, int]] = {
    (5376, 5376): (8, 7, 12),
    (7168, 1344): (8, 8, 6),
    (5376, 7168): (8, 3, 14),
    # attention to_gate_compress (VSA): K_tiles_per_device = 42 -> K_block 7; N = 56 tiles -> N_block 8.
    # Valid-by-construction, not sweep-measured (the gate matmul is off the dense path).
    (5376, 1792): (8, 7, 8),
}


def agmm_block_size(k: int, n: int) -> tuple[int, int, int] | None:
    """Block sizes for an all-gather matmul of this `(K, N)`, or None to let the generic path decide."""
    return AGMM_BLOCK_SIZES.get((k, n))
