# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Llama KV chunk table address computation.

CPU-only: validates that the shard-index formula for n_kv_local > 1 produces
the correct bank-id / bank-offset pattern, matching what a running bank walk
through the [B, n_kv_local, seq_local, head_dim] tensor layout would produce.

Run: pytest models/demos/llama_3_1_8b/tests/test_kv_cache_table.py -v
"""

import pytest


def _running_bank_walk(num_batches, n_kv_local, seq_banks_per_chip, num_dram_banks):
    """Simulate a running ROUND_ROBIN_1D bank walk through the full tensor.

    Returns a dict mapping (batch, local_head, seq_bank) -> (bank_id, bank_offset_mult).
    """
    result = {}
    curr_bank_id = 0
    curr_offset_mult = 0
    for batch in range(num_batches):
        for lh in range(n_kv_local):
            for sb in range(seq_banks_per_chip):
                result[(batch, lh, sb)] = (curr_bank_id, curr_offset_mult)
                curr_bank_id = (curr_bank_id + 1) % num_dram_banks
                if curr_bank_id == 0:
                    curr_offset_mult += 1
    return result


def _direct_shard_index(batch, local_head, seq_bank, n_kv_local, seq_banks_per_chip):
    """Direct shard index formula used in the table builder."""
    return batch * (n_kv_local * seq_banks_per_chip) + local_head * seq_banks_per_chip + seq_bank


@pytest.mark.parametrize(
    "n_kv_local, num_users, num_layers, seq_banks_per_chip, num_dram_banks",
    [
        (1, 1, 4, 20, 12),  # GPT-OSS-like: 1 head per column
        (2, 1, 4, 20, 12),  # Llama: 2 heads per column
        (2, 1, 32, 80, 12),  # Full Llama: 32 layers, 80 banks/chip
        (2, 2, 4, 20, 12),  # Multi-user
        (4, 1, 4, 16, 12),  # 4 heads per column (hypothetical)
    ],
)
def test_shard_index_matches_running_walk(n_kv_local, num_users, num_layers, seq_banks_per_chip, num_dram_banks):
    """The direct shard_idx formula must produce the same bank_id and bank_offset as a running walk."""
    num_batches = num_users * num_layers
    walk = _running_bank_walk(num_batches, n_kv_local, seq_banks_per_chip, num_dram_banks)

    for batch in range(num_batches):
        for lh in range(n_kv_local):
            for sb in range(seq_banks_per_chip):
                shard_idx = _direct_shard_index(batch, lh, sb, n_kv_local, seq_banks_per_chip)
                expected_bank_id, expected_offset_mult = walk[(batch, lh, sb)]
                actual_bank_id = shard_idx % num_dram_banks
                actual_offset_mult = shard_idx // num_dram_banks
                assert actual_bank_id == expected_bank_id, (
                    f"bank_id mismatch at (batch={batch}, lh={lh}, sb={sb}): "
                    f"direct={actual_bank_id}, walk={expected_bank_id}"
                )
                assert actual_offset_mult == expected_offset_mult, (
                    f"offset_mult mismatch at (batch={batch}, lh={lh}, sb={sb}): "
                    f"direct={actual_offset_mult}, walk={expected_offset_mult}"
                )


def test_n_kv_local_1_matches_gpt_oss_pattern():
    """With n_kv_local=1, the direct formula should be identical to the GPT-OSS running walk."""
    num_layers = 36
    seq_banks_per_chip = 64
    num_dram_banks = 12

    walk = _running_bank_walk(num_layers, 1, seq_banks_per_chip, num_dram_banks)

    for layer in range(num_layers):
        for sb in range(seq_banks_per_chip):
            shard_idx = _direct_shard_index(layer, 0, sb, 1, seq_banks_per_chip)
            w_bank, w_off = walk[(layer, 0, sb)]
            assert shard_idx % num_dram_banks == w_bank
            assert shard_idx // num_dram_banks == w_off


@pytest.mark.parametrize("n_kv_local", [1, 2, 4])
def test_shard_indices_are_contiguous(n_kv_local):
    """Shard indices must be contiguous (0, 1, 2, ...) with no gaps or repeats."""
    num_batches = 4
    seq_banks = 20
    all_indices = []
    for b in range(num_batches):
        for lh in range(n_kv_local):
            for sb in range(seq_banks):
                all_indices.append(_direct_shard_index(b, lh, sb, n_kv_local, seq_banks))

    total = num_batches * n_kv_local * seq_banks
    assert len(all_indices) == total
    assert sorted(all_indices) == list(range(total))
