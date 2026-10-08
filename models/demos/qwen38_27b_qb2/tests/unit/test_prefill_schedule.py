# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest

from models.demos.qwen38_27b_qb2.tt.prefill_schedule import prefill_chunk_size


def test_default_preserves_all_supported_prefill_batches():
    assert all(prefill_chunk_size(batch) == 4096 for batch in range(1, 33))


@pytest.mark.parametrize("budget", [4096, 8192, 16384, 32768, 65536])
def test_budget_is_respected_for_regular_and_odd_batches(budget):
    for batch in range(1, 33):
        chunk = prefill_chunk_size(batch, budget)
        assert 32 <= chunk <= 4096 and chunk % 32 == 0
        assert batch * chunk <= budget
        assert chunk == 4096 or batch * (chunk + 32) > budget


@pytest.mark.parametrize("budget", [0, -4096, 4095, 4097, 4096.0, True])
def test_invalid_budget_rejected(budget, expect_error):
    with expect_error(ValueError, "Prefill batch-token budget"):
        prefill_chunk_size(32, budget)


@pytest.mark.parametrize("batch", [0, 33, 64, 1.0, True])
def test_unsupported_batch_stays_rejected(batch, expect_error):
    with expect_error(ValueError, "1..32 users"):
        prefill_chunk_size(batch, 32768)
