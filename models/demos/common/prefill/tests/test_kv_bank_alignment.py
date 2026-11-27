# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest

from models.demos.common.prefill.runners.kv_bank_alignment import bank_aligned_seq_len_quantum, check_kv_bank_aligned


@pytest.mark.parametrize(
    "max_seq_len, stripes",
    [(1064960, 32), (286720, 32), (40960, 32), (61440, 8), (81920, 32), (262144, 8), (131072, 8), (32768, 4)],
)
def test_configured_lengths_pass(max_seq_len, stripes):
    check_kv_bank_aligned(max_seq_len, max_seq_len // stripes, 8)


@pytest.mark.parametrize("max_seq_len, stripes", [(1049600, 8), (56320, 8), (256000, 32), (61440, 32)])
def test_unaligned_lengths_fail(max_seq_len, stripes, expect_error):
    with expect_error(ValueError, f"multiple of {bank_aligned_seq_len_quantum(stripes, 8)}"):
        check_kv_bank_aligned(max_seq_len, max_seq_len // stripes, 8)
