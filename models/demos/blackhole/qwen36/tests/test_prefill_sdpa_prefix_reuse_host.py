# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only (no device): the Q chunk and SDPAProgramConfig.prefix_reuse that tp_common.prefill_sdpa_prefix_reuse
picks for a chunked prefill SDPA (3 Q heads per device at TP=8; K chunk 128 on the tensor-start path)."""

import pytest

from models.demos.blackhole.qwen36.tt import tp_common as tpc


@pytest.mark.parametrize(
    "rows, n_cores, want",
    [
        # more Q chunks than cores: prefix chains, no split
        (8192, 120, (128, 2)),
        (8192, 130, (128, 2)),
        (4096, 64, (128, 2)),
        # Q chunks that fit the grid, a split would pass 2/5 of it: one Q chunk per core
        (4096, 120, (128, 4)),
        (2048, 120, (128, 4)),
        # split while the Q chunks stay within 2/5 of the grid (48 of 120, 52 of 130)
        (1024, 120, (64, 4)),
        (512, 120, (32, 4)),
        (512, 130, (32, 4)),
        # split calls of at most 1/5 of the grid's Q chunks also size the K/V read batches
        (256, 120, (32, 5)),
        (128, 120, (32, 5)),
        (256, 130, (32, 5)),
        # a small grid splits less
        (1024, 64, (128, 4)),
        (512, 64, (64, 4)),
    ],
)
def test_policy(rows, n_cores, want):
    assert tpc.prefill_sdpa_prefix_reuse(rows, 128, 128, 3, n_cores) == want


def test_fewer_rows_than_a_k_chunk_keep_the_call():
    # the 32-row spec verify bucket and short calls keep today's program
    assert tpc.prefill_sdpa_prefix_reuse(32, 32, 32, 3, 120) == (32, 0)
    assert tpc.prefill_sdpa_prefix_reuse(64, 64, 64, 3, 120) == (64, 0)


def test_other_k_chunks_are_not_split():
    # the scalar-start path's 64-row chunks below 2048 rows, and tuned configs (prefill_sdpa_config_fn)
    assert tpc.prefill_sdpa_prefix_reuse(1024, 64, 64, 3, 120) == (64, 4)
    assert tpc.prefill_sdpa_prefix_reuse(1024, 256, 256, 3, 120) == (256, 4)
    # a tuned Q chunk below the K chunk on few cores also takes the read batches
    assert tpc.prefill_sdpa_prefix_reuse(1024, 128, 256, 3, 120) == (128, 5)


def test_the_split_divides_the_chunk_start():
    assert tpc.prefill_sdpa_prefix_reuse(512, 128, 128, 3, 120, chunk_start=128 * 7) == (32, 4)
    # a scalar start that the halved Q chunk does not divide keeps the Q chunk
    assert tpc.prefill_sdpa_prefix_reuse(512, 128, 128, 3, 120, chunk_start=96) == (128, 4)
