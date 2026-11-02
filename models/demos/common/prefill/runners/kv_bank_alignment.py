# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK = 32


def bank_aligned_seq_len_quantum(stripes: int, num_dram_banks: int) -> int:
    return stripes * NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK * num_dram_banks


def check_kv_bank_aligned(seq_len: int, rows_per_device: int, num_dram_banks: int) -> None:
    # Paging requires every KV slot to start on DRAM bank 0.
    if rows_per_device % (NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK * num_dram_banks) == 0:
        return
    quantum = bank_aligned_seq_len_quantum(seq_len // rows_per_device, num_dram_banks)
    lower = seq_len // quantum * quantum
    raise ValueError(
        f"seq_len {seq_len} gives {rows_per_device} KV rows per device, which does not start every KV slot "
        f"on DRAM bank 0 ({num_dram_banks} banks x {NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK} tokens), as paging "
        f"requires: use a multiple of {quantum}, e.g. {lower or quantum} or {lower + quantum}"
    )
