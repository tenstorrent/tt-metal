# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host-only check of the prefill matmul row chunking (tt/dram_sharded.py prefill_row_chunks): full slabs first, then
the leftover in chunks of at most the remainder size (the pre-slab configs), so no chunk sizes its own odd config
(4096 = 2560 + 1536 overflowed L1 on the 31B and clashed with L1 buffers on the 12B, CI 2026-10-24). No device.

    pytest models/demos/gemma4/tests/unit/test_prefill_row_chunks_host.py --noconftest -p no:cacheprovider
"""

from models.demos.gemma4.tt.dram_sharded import prefill_row_chunks


def test_full_slabs_then_small_remainders():
    assert prefill_row_chunks(4096, slab=2560, remainder=1024) == [(0, 2560), (2560, 3584), (3584, 4096)]
    assert prefill_row_chunks(8192, slab=2560, remainder=1024) == [(0, 2560), (2560, 5120), (5120, 7680), (7680, 8192)]
    assert prefill_row_chunks(3072, slab=2560, remainder=1024) == [(0, 2560), (2560, 3072)]
    assert prefill_row_chunks(2561, slab=2560, remainder=1024) == [(0, 2560), (2560, 2561)]


def test_chunks_cover_m_exactly_and_never_exceed_the_bounds():
    for m in range(32, 40000, 992):
        chunks = prefill_row_chunks(m, slab=2560, remainder=1024)
        assert chunks[0][0] == 0 and chunks[-1][1] == m
        assert all(a < b for a, b in chunks)
        assert all(chunks[i][1] == chunks[i + 1][0] for i in range(len(chunks) - 1))
        assert all(b - a == 2560 or b - a <= 1024 for a, b in chunks)
        assert all(b - a == 2560 for a, b in chunks[: m // 2560])


def test_defaults_are_the_blackhole_slab_and_the_pre_slab_chunk():
    from models.demos.gemma4.tt import dram_sharded as ds

    assert ds._PREFILL_REMAINDER_CHUNK <= 1024 and ds._PREFILL_REMAINDER_CHUNK <= ds._PREFILL_CUTOFF
    assert prefill_row_chunks(4096)[0] == (0, ds._PREFILL_M_CHUNK) if ds._PREFILL_M_CHUNK < 4096 else True


def test_slab_rows_follow_the_l1_headroom():
    import ttnn
    from models.demos.gemma4.tt.dram_sharded import prefill_cb_bytes, prefill_slab_rows

    # the two CI numbers the footprint model is calibrated on (Blackhole, 2D matmul, k-block 4)
    assert (
        prefill_cb_bytes(8, 48, 4, 2048) == 1815936
    )  # 31B gate_up with bf16 weights: "beyond max L1" in the unit test
    assert prefill_cb_bytes(8, 40, 4, 2048) == 1553792  # 12B gate_up bf16: the CB region end that clashed at 1539200
    bh = dict(grid=(8, 10), l1_bytes=1572864, headroom=128 * 1024, max_cols=8)
    assert prefill_slab_rows(5376, 2 * 21504 // 4, ttnn.bfloat8_b, **bh) == 2240  # the 31B model's MLP weights
    assert prefill_slab_rows(5376, 2 * 21504 // 4, ttnn.bfloat16, **bh) == 1280  # the same weight in the unit test
    assert prefill_slab_rows(3840, 2 * 15360 // 4, ttnn.bfloat16, **bh) == 1920  # the 12B's MLP weights
    assert prefill_slab_rows(3840, (4096 + 4096) // 4, ttnn.bfloat16, **bh) == 2560  # a narrow weight: the full slab
    assert prefill_slab_rows(5376, 2 * 21504 // 4, ttnn.bfloat16, **dict(bh, headroom=0)) == 2560  # 0 = no cap
    h64 = dict(bh, headroom=64 * 1024)  # the default: the 31B keeps its slab, the 12B's 2K prompt stays one matmul
    assert prefill_slab_rows(5376, 2 * 21504 // 4, ttnn.bfloat8_b, **h64) == 2560
    assert prefill_slab_rows(3840, 2 * 15360 // 4, ttnn.bfloat16, **h64) == 2240
    assert prefill_slab_rows(5376, 2 * 21504 // 4, ttnn.bfloat16, **h64) == 1600  # the unit test's bf16 31B weight
    from models.demos.gemma4.tt import dram_sharded as ds

    assert ds._PREFILL_L1_HEADROOM == 64 * 1024 or "GEMMA4_PREFILL_L1_HEADROOM_KB" in __import__("os").environ
    # every slab's CBs stay under the budget and the next size up would not
    for k, n, dt in ((5376, 10752, ttnn.bfloat8_b), (5376, 10752, ttnn.bfloat16), (3840, 7680, ttnn.bfloat16)):
        rows = prefill_slab_rows(k, n, dt, **bh)
        assert rows % 320 == 0 and 320 <= rows <= 2560
