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


# A Bin6 Galaxy has 7 DRAM banks per chip. GLM-5.3's Bin6 length (run_multirank_pcc.sh --bin6) is aligned
# on 7 banks, and neither it nor the 8-bank manifest length is aligned on the other bank count.
def test_bin6_length_passes_on_7_banks():
    check_kv_bank_aligned(1075200, 1075200 // 32, 7)


@pytest.mark.parametrize("max_seq_len, num_dram_banks", [(1064960, 7), (1075200, 8)])
def test_lengths_fail_on_the_other_bank_count(max_seq_len, num_dram_banks, expect_error):
    with expect_error(ValueError, f"multiple of {bank_aligned_seq_len_quantum(32, num_dram_banks)}"):
        check_kv_bank_aligned(max_seq_len, max_seq_len // 32, num_dram_banks)
