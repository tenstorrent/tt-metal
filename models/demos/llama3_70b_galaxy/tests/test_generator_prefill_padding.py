# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host-only checks (no device) for Galaxy prefill padding, the prefix-caching heuristic and the
prefill warmup lengths. Llama (pad_multiple None) must match the power-of-2 functions exactly;
Qwen (prefill_pad_multiple 8192) pads > 4096 to multiples of 8192 and warms every such bucket."""

import pytest

from models.demos.llama3_70b_galaxy.tt.generator import (
    MAX_TRACED_PREFILL_LEN,
    _should_skip_prefix_caching,
    get_padded_prefill_len,
    get_prefill_warmup_sequence_lengths,
)
from models.demos.llama3_70b_galaxy.tt.llama_ccl import PREFILL_SUPPORT_SEQLENS

QWEN_PAD_MULTIPLE = 8192  # TtQwenModelArgs.prefill_pad_multiple
MAX_LEN = 262144


# Reference: the functions as they were before pad_multiple (tt-metal main@8bdf164e, generator.py).
def ref_padded_len(seq_len):
    if seq_len <= 128:
        return 128
    if seq_len <= 1024:
        return 1024
    return 2 ** (seq_len - 1).bit_length()


def ref_warmup_lengths(max_seq_len):
    return [128] + [2**i for i in range(10, max_seq_len.bit_length()) if 2**i <= max_seq_len]


def ref_skip_prefix_caching(seq_len, num_cached):
    if num_cached == 0:
        return False
    padded_new = ref_padded_len(seq_len - num_cached)
    padded_total = ref_padded_len(seq_len)
    if padded_new >= padded_total:
        return True
    ratio = padded_new / padded_total
    if ratio <= 0.125:
        return False
    if ratio <= 0.25:
        return padded_total > 32768
    if ratio <= 0.5:
        return padded_total > 16384
    return True


def test_max_traced_prefill_len_is_largest_ccl_seqlen():
    assert MAX_TRACED_PREFILL_LEN == max(PREFILL_SUPPORT_SEQLENS) == 4096


def test_llama_padding_unchanged_exhaustive():
    for s in range(1, MAX_LEN + 1):
        expected = ref_padded_len(s)
        assert get_padded_prefill_len(s) == expected, s
        assert get_padded_prefill_len(s, None) == expected, s


def test_llama_warmup_lengths_unchanged():
    for max_seq_len in list(range(128, 70000, 61)) + [2**i for i in range(7, 19)] + [40960, 131072, 131071]:
        assert get_prefill_warmup_sequence_lengths(max_seq_len) == ref_warmup_lengths(max_seq_len), max_seq_len
        assert get_prefill_warmup_sequence_lengths(max_seq_len, None) == ref_warmup_lengths(max_seq_len)


def test_llama_skip_prefix_caching_unchanged():
    for seq_len in range(1, 140000, 97):
        for num_cached in range(0, seq_len, 128):
            assert _should_skip_prefix_caching(seq_len, num_cached) == ref_skip_prefix_caching(seq_len, num_cached)
            assert _should_skip_prefix_caching(seq_len, num_cached, None) == ref_skip_prefix_caching(
                seq_len, num_cached
            )


def test_qwen_padding_exhaustive():
    m = QWEN_PAD_MULTIPLE
    for s in range(1, MAX_LEN + 1):
        padded = get_padded_prefill_len(s, m)
        if s <= MAX_TRACED_PREFILL_LEN:
            assert padded == ref_padded_len(s), s  # traced buckets keep power-of-2 padding
        else:
            assert padded == -(-s // m) * m, s
            # never above the power of 2 (the dropped min() branch), and every bucket meets the
            # 2048 QKV reshape and the ring SDPA S % (2 * ring_size) constraints
            assert s <= padded <= ref_padded_len(s), s
            assert padded % 2048 == 0 and padded % 8 == 0, s
    assert get_padded_prefill_len(40928, m) == 40960
    assert get_padded_prefill_len(20000, m) == 24576
    assert get_padded_prefill_len(4097, m) == 8192


@pytest.mark.parametrize("bad_multiple", [3072, 6144, 16384])
def test_pad_multiple_must_divide_8192(bad_multiple, expect_error):
    with expect_error(AssertionError, "must divide 8192"):
        get_padded_prefill_len(5000, bad_multiple)


@pytest.mark.parametrize(
    "max_seq_len, expected",
    [
        (40960, [128, 1024, 2048, 4096, 8192, 16384, 24576, 32768, 40960]),
        (4096, [128, 1024, 2048, 4096]),
        (20000, [128, 1024, 2048, 4096, 8192, 16384]),
    ],
)
def test_qwen_warmup_lengths(max_seq_len, expected):
    assert get_prefill_warmup_sequence_lengths(max_seq_len, QWEN_PAD_MULTIPLE) == expected


@pytest.mark.parametrize("max_seq_len", [4096, 8192, 20000, 40960, 65536, 131072])
def test_qwen_warmup_covers_every_bucket(max_seq_len):
    # Every padded length a prompt of <= max_seq_len tokens can get is warmed (no first-request compile),
    # and each warmup length pads to itself.
    m = QWEN_PAD_MULTIPLE
    warm = get_prefill_warmup_sequence_lengths(max_seq_len, m)
    for length in warm:
        assert get_padded_prefill_len(length, m) == length
    buckets = {get_padded_prefill_len(s, m) for s in range(1, max_seq_len + 1)}
    assert {b for b in buckets if b <= max_seq_len} <= set(warm)
    assert set(ref_warmup_lengths(max_seq_len)) <= set(warm)  # superset of the power-of-2 list


@pytest.mark.parametrize("max_seq_len", [40960, 131072])
def test_qwen_prefix_cached_buckets_are_powers_of_two(max_seq_len):
    # The sp1 (prefix-cached) warmup keeps the power-of-2 list: the heuristic never keeps a cached
    # prefix whose new tokens pad to a non-power-of-2 bucket while max_seq_len < 8 * 24576.
    m = QWEN_PAD_MULTIPLE
    pow2 = set(ref_warmup_lengths(max_seq_len))
    for seq_len in range(129, max_seq_len + 1, 53):
        for num_cached in range(128, seq_len, 128):
            if not _should_skip_prefix_caching(seq_len, num_cached, m):
                assert get_padded_prefill_len(seq_len - num_cached, m) in pow2, (seq_len, num_cached)
