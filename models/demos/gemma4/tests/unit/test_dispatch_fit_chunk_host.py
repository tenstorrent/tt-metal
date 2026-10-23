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


def test_fit_chunk_off_by_default():
    assert not dp.fit_chunk_enabled()
    assert dp.fit_chunk_tokens(128, 4, 4096) == 4096
    assert dp.fit_chunk_tokens(4096, 4, 4096) == 4096


def test_fit_chunk_rounds_to_the_dispatch_group_tile(monkeypatch):
    monkeypatch.setenv(dp.FIT_CHUNK_ENV, "1")
    assert dp.fit_chunk_enabled()
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
