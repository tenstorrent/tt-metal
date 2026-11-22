# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""CPU (no device), P.78: tt/engram_bfp8.ngram_hashes (HOST_HIDE=on's numba hash) equals the reference's
EngramHasher.hashes bit for bit: table rows and the new tail, for random ids, a fresh tail (DEAD ids: the pad path),
a carried tail, multipliers whose products overflow int64, and every engram layer.

    scripts/run_safe_pytest.sh --run-all models/demos/deepseek_v41_flash_d_p/tests/test_engram_hash.py
"""

import torch

from models.demos.deepseek_v41_flash_d_p.reference.ngram import EngramHasher
from models.demos.deepseek_v41_flash_d_p.tt.engram_bfp8 import ngram_hashes


def _hasher(seed: int) -> EngramHasher:
    g = torch.Generator().manual_seed(seed)
    m, h = 4, 8
    return EngramHasher(
        layer_ids=(1, 14),
        max_ngram_size=m,
        n_heads=h,
        token_map=torch.randint(0, 100_000, (129_280,), generator=g),
        pad_id=2,
        # (odd 62-bit multipliers: every product of an id overflows int64, as the checkpoint's do)
        multipliers=torch.randint(1, 2**61, (2, m), generator=g) * 2 + 1,
        primes=torch.randint(1_000_000, 4_000_000, (2, m - 1, h), generator=g),
        offsets=torch.randint(0, 10_000_000, (2, (m - 1) * h), generator=g),
    )


def test_ngram_hashes_bit_equal():
    for seed in range(3):
        hs = _hasher(seed)
        g = torch.Generator().manual_seed(100 + seed)
        for n in (1, 3, 37, 5120):
            ids = torch.randint(0, 129_280, (n,), generator=g)
            tails = [hs.new_tail(), hs.token_map[torch.randint(0, 129_280, (3,), generator=g)]]
            tails.append(torch.tensor([-1, -1, int(hs.token_map[5])]))  # (a partly dead tail)
            for tail in tails:
                for layer in hs.layer_ids:
                    want, want_tail = hs.hashes(ids, layer, tail)
                    got, got_tail = ngram_hashes(hs, ids, layer, tail)
                    assert torch.equal(got, want), (seed, n, layer, (got != want).sum())
                    assert torch.equal(got_tail, want_tail), (seed, n, layer)


def test_tile_rows_chips_equals_tile_rows():
    import numpy as np

    from models.demos.deepseek_v41_flash_d_p.tt.engram_bfp8 import tile_rows, tile_rows_chips

    g = torch.Generator().manual_seed(7)
    packed = torch.randint(0, 256, (4096, 272), dtype=torch.uint8, generator=g).numpy()
    for chips, q in ((4, 64), (32, 160), (8, 32)):
        s = chips * q
        for valid in (s, s - 37, 5, 0):
            ids = torch.randint(0, 4096, (valid, 24), generator=g)
            nbytes = q * 24 * 256 * 17 // 16
            outs = tuple(np.full(nbytes, 0xAB, dtype=np.uint8) for _ in range(chips))
            tile_rows_chips(packed, ids, valid, outs, q, 8)
            for k in range(chips):
                lo, hi = k * q, (k + 1) * q
                share = ids[lo : min(hi, valid)] if lo < valid else ids[:0]
                want = tile_rows(packed, share, q, 4)
                assert torch.equal(torch.from_numpy(outs[k]), want), (chips, q, valid, k)


def test_tile_rows_chips_range_check():
    import numpy as np

    from models.demos.deepseek_v41_flash_d_p.tt.engram_bfp8 import tile_rows_chips

    packed = np.zeros((64, 272), dtype=np.uint8)
    ids = torch.zeros(4 * 32, 24, dtype=torch.int64)
    ids[70, 3] = 64  # (one past the table)
    outs = tuple(np.zeros(32 * 24 * 256 * 17 // 16, dtype=np.uint8) for _ in range(4))
    try:
        tile_rows_chips(packed, ids, 128, outs, 32, 4)
    except AssertionError as e:
        assert "out of range (1 ids)" in str(e), e
    else:
        raise AssertionError("an id past the table was not refused")
    tile_rows_chips(packed, ids, 70, outs, 32, 4)  # (rows past valid are never read)
