# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host-only checks of the dispatch-prefill chunk sizing (tt/experts/dispatch_prefill.py): with
GEMMA4_MOE_DISPATCH_FIT_CHUNK=1 a call shorter than the full chunk runs on a chunk of its own tile-aligned length
(a 128-token prompt on a 128-token chunk instead of 4096 padded rows). No device.

    pytest models/demos/gemma4/tests/unit/test_dispatch_fit_chunk_host.py --noconftest -p no:cacheprovider
"""

import pytest

from models.demos.gemma4.tt.experts import dispatch_prefill as dp


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(dp.FIT_CHUNK_ENV, raising=False)


def test_fit_chunk_default_follows_the_dispatch_default():
    from models.demos.gemma4.tt.mesh_orientation import fit_chunk_default

    assert fit_chunk_default((4, 1), blackhole=True)  # the QB2 column mesh: the validated configuration
    assert not fit_chunk_default((1, 4), blackhole=True)
    assert not fit_chunk_default((4, 1), blackhole=False)
    assert not fit_chunk_default((8, 4), blackhole=True)
    assert dp.fit_chunk_tokens(128, 4, 4096, fit=False) == 4096
    assert dp.fit_chunk_tokens(4096, 4, 4096, fit=True) == 4096


def test_fit_chunk_env_overrides_the_default(monkeypatch):
    from models.demos.gemma4.tt.mesh_orientation import fit_chunk_default

    monkeypatch.setenv(dp.FIT_CHUNK_ENV, "0")
    assert not fit_chunk_default((4, 1), blackhole=True)
    monkeypatch.setenv(dp.FIT_CHUNK_ENV, "1")
    assert fit_chunk_default((1, 4), blackhole=True)
    assert fit_chunk_default((8, 4), blackhole=False)


def test_fit_chunk_rounds_to_the_dispatch_group_tile(monkeypatch):
    monkeypatch.setenv(dp.FIT_CHUNK_ENV, "1")
    assert dp.fit_chunk_enabled((1, 4))
    assert dp.fit_chunk_tokens(128, 4, 4096) == 256  # 64 rows per chip of the group (masked_bincount's grid)
    assert dp.fit_chunk_tokens(32, 4, 4096) == 256  # never below 64 rows per chip
    assert dp.fit_chunk_tokens(256, 4, 4096) == 256
    assert dp.fit_chunk_tokens(1024, 4, 4096) == 1024
    assert dp.fit_chunk_tokens(2065, 4, 4096) == 2304  # padded to the next 256
    assert dp.fit_chunk_tokens(4096, 4, 4096) == 4096
    assert dp.fit_chunk_tokens(4224, 4, 4096) == 4096  # the loop takes a full chunk, then the 128 remainder
    assert dp.fit_chunk_tokens(128, 8, 4096) == 512  # an 8-chip group: 64 rows per chip
    assert dp.fit_chunk_tokens(128, 4, 4096, fit=False) == 4096
    assert dp.MIN_ROWS_PER_CHIP % 32 == 0


def test_small_fit_rounds_to_the_dispatch_group_tile_without_the_dispatch_floor():
    from models.demos.gemma4.tt.experts.dispatch_prefill import small_fit_chunk_tokens

    # the small-row path has no dispatch histogram: a 128-token prompt on the 4-chip group runs 128 rows (32/chip)
    assert small_fit_chunk_tokens(1, 4) == 128
    assert small_fit_chunk_tokens(128, 4) == 128
    assert small_fit_chunk_tokens(129, 4) == 256
    assert small_fit_chunk_tokens(256, 4) == 256
    assert small_fit_chunk_tokens(64, 8) == 256  # eight chips: one tile per chip


def test_small_row_max_tokens_default_env_and_cap(monkeypatch):
    from models.demos.gemma4.custom_ops.expert_gather_small import ExpertGatherSmall
    from models.demos.gemma4.tt.experts.dispatch_prefill import small_row_max_tokens

    monkeypatch.delenv("GEMMA4_SMALL_ROW_MAX_TOKENS", raising=False)
    assert small_row_max_tokens() == 128
    monkeypatch.setenv("GEMMA4_SMALL_ROW_MAX_TOKENS", "256")
    assert small_row_max_tokens() == 256
    monkeypatch.setenv("GEMMA4_SMALL_ROW_MAX_TOKENS", "4096")
    assert small_row_max_tokens() == ExpertGatherSmall.MAX_TOKENS


def test_small_row_bounds_keep_the_cached_prefix_continuations_on_the_dispatch_path(monkeypatch):
    from models.demos.gemma4.tt.experts.dispatch_prefill import small_row_min_tokens, small_rows_admit

    monkeypatch.delenv("GEMMA4_SMALL_ROW_MIN_TOKENS", raising=False)
    lo = small_row_min_tokens()
    assert lo == 32
    for rows, ok in (
        (1, False),
        (21, False),
        (29, False),
        (31, False),
        (32, True),
        (64, True),
        (128, True),
        (129, False),
    ):
        assert small_rows_admit(rows, lo, 128) == ok, rows
    monkeypatch.setenv("GEMMA4_SMALL_ROW_MIN_TOKENS", "1")
    assert small_row_min_tokens() == 1 and small_rows_admit(8, small_row_min_tokens(), 128)
