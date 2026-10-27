# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only: msa_cache_read_extent sizes the cross-chunk MSA gather for mid-slab (multi-turn resume) starts.

The KV writer places global position g on SP rank (g // chunk_local) % sp at local row
(g // chunk_global) * chunk_local + g % chunk_local. After the chunk at cached_len is written, every written
position [0, end) must fall inside the local prefix high_bw_all_gather collects from its rank (n_rows), with no
more rows than the fullest rank needs; kv_len must cover end in whole blocks; and the chunk-aligned case must keep
its previous exact extent. With a TP-deduped cache (tp > 1) each rank's rows of a slab are split into tp stripes of
chunk_local/tp, one per chip, and n_rows must be the fullest CHIP's local row count.
"""

import pytest

from models.demos.minimax_m3.tt.attention.msa import msa_cache_read_extent

BLOCK = 128


@pytest.mark.parametrize("tp", [1, 4], ids=["tp_replicated", "tp_dedup"])
@pytest.mark.parametrize("sp,chunk_local", [(2, 256), (4, 384), (8, 640)])
def test_msa_cache_read_extent_covers_prefix(sp, chunk_local, tp):
    chunk_global = sp * chunk_local
    stripe = chunk_local // tp
    for cached_len in range(0, 3 * chunk_global, 32):
        kv_len, n_rows = msa_cache_read_extent(cached_len, chunk_local, sp, BLOCK, tp=tp)
        end = cached_len + chunk_global
        assert kv_len % BLOCK == 0 and end <= kv_len < end + BLOCK
        fill = [0] * (sp * tp)  # per-chip local rows needed to cover the written prefix [0, end)
        for g in range(end):
            chip = (g // stripe) % (sp * tp)  # == rank*tp + col: stripe col of rank's chunk_local rows
            fill[chip] = max(fill[chip], (g // chunk_global) * stripe + g % stripe + 1)
        assert n_rows == max(fill), f"cached_len={cached_len}: n_rows={n_rows}, fullest chip needs {max(fill)}"
        if cached_len % chunk_global == 0:
            assert (kv_len, n_rows) == (end, (cached_len // sp + chunk_local) // tp)


def test_msa_cache_read_extent_rejects_off_grid_start(expect_error):
    with expect_error(AssertionError, "must be a multiple of 32"):
        msa_cache_read_extent(5120 + 16, 640, 8, BLOCK)
