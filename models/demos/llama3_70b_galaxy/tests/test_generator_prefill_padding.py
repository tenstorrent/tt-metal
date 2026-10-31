# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host-only checks (no device opened by the tests) for Galaxy prefill padding, the prefix-caching heuristic
and the prefill warmup lengths. Llama (pad_multiple None) must match the power-of-2 functions exactly;
Qwen pads > 4096 to multiples of TtQwenModelArgs.prefill_pad_multiple (8192, pinned below) and warms every
such bucket. The Qwen value is read from the real model-args class, and the Generator call sites that read
it (getattr(self.model_args, "prefill_pad_multiple", None)) are exercised with those model args.
Note: the directory conftest (ensure_devices_tg) enumerates the chips, so run with the devices idle."""

from types import SimpleNamespace

import pytest
import torch

import models.demos.llama3_70b_galaxy.tt.generator as generator_module
from models.demos.llama3_70b_galaxy.tests.qwen_host_model_args import qwen_model_args
from models.demos.llama3_70b_galaxy.tt.generator import (
    MAX_TRACED_PREFILL_LEN,
    Generator,
    _should_skip_prefix_caching,
    get_padded_prefill_len,
    get_prefill_warmup_sequence_lengths,
    prefill_pad_multiple_for,
)
from models.demos.llama3_70b_galaxy.tt.llama_ccl import PREFILL_SUPPORT_SEQLENS
from models.demos.llama3_70b_galaxy.tt.qwen_opt_limits import QWEN_OPT_MAX_SEQ_LEN

pytestmark = pytest.mark.host_only  # no device: these tests run anywhere

# User decision (2026-10-08, keep 01b): Qwen prefill > 4096 pads to multiples of 8192 and the warmup
# compiles every such bucket. A change of TtQwenModelArgs.prefill_pad_multiple must fail here.
USER_QWEN_PAD_MULTIPLE = 8192
QWEN_WARMUP_40960 = [128, 1024, 2048, 4096, 8192, 16384, 24576, 32768, 40960]
MAX_LEN = 262144


@pytest.fixture(scope="module")
def qwen_args():
    return qwen_model_args(max_seq_len=40960)


@pytest.fixture(scope="module")
def qwen_pad_multiple(qwen_args):
    return qwen_args.prefill_pad_multiple


class _StopAtCallSite(Exception):
    pass


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


def test_qwen_pad_multiple_is_user_choice(qwen_pad_multiple):
    assert qwen_pad_multiple == USER_QWEN_PAD_MULTIPLE


def test_generator_warmup_sweep_reads_qwen_pad_multiple(qwen_args, monkeypatch, expect_error):
    # Generator._prefill_warmup_sweep (generator.py, phase 1) passes getattr(model_args, "prefill_pad_multiple")
    # to get_prefill_warmup_sequence_lengths. Stop at that call and check the arguments and the 9 buckets.
    calls = []

    def spy(max_seq_len, pad_multiple=None):
        calls.append((max_seq_len, pad_multiple))
        raise _StopAtCallSite("stopped at the call site")

    monkeypatch.setattr(generator_module, "get_prefill_warmup_sequence_lengths", spy)
    gen = Generator.__new__(Generator)  # no device: only model_args and model.args are read before the call
    gen.model_args = qwen_args
    gen.model = SimpleNamespace(args=qwen_args, switch_mode=lambda mode: None)
    with expect_error(_StopAtCallSite, "stopped at the call site"):
        gen._prefill_warmup_sweep(None, None, False, None, None)
    assert calls == [(40960, qwen_args.prefill_pad_multiple)]
    warm = get_prefill_warmup_sequence_lengths(*calls[0])
    assert len(warm) == 9 and warm == QWEN_WARMUP_40960


def test_generator_prefill_reads_qwen_pad_multiple(qwen_args, monkeypatch, expect_error):
    # Generator.prefill_forward_text passes getattr(model_args, "prefill_pad_multiple") to the prefix-caching
    # heuristic and to get_padded_prefill_len. Stop at the padding call: 40928 must pad to 40960.
    skip_calls, pad_calls = [], []

    def skip_spy(seq_len, num_cached, pad_multiple=None):
        skip_calls.append((seq_len, num_cached, pad_multiple))
        return False

    def pad_spy(seq_len, pad_multiple=None):
        pad_calls.append((seq_len, pad_multiple))
        raise _StopAtCallSite("stopped at the call site")

    monkeypatch.setattr(generator_module, "_should_skip_prefix_caching", skip_spy)
    monkeypatch.setattr(generator_module, "get_padded_prefill_len", pad_spy)
    gen = Generator.__new__(Generator)
    gen.model_args = qwen_args
    gen.model = SimpleNamespace(args=qwen_args, is_prefill_setup=True)
    gen.already_warmed_up_prefill = True
    gen.warming_up_prefill = False
    tokens = torch.zeros(1, 40928, dtype=torch.long)
    with expect_error(_StopAtCallSite, "stopped at the call site"):
        gen.prefill_forward_text(tokens, kv_cache=[None], prompt_lens=[40928], start_pos=[0])
    m = qwen_args.prefill_pad_multiple
    assert skip_calls == [(40928, 0, m)]
    assert pad_calls == [(40928, m)]
    assert get_padded_prefill_len(*pad_calls[0]) == 40960


def test_qwen_padding_exhaustive(qwen_pad_multiple):
    m = qwen_pad_multiple
    for s in range(1, MAX_LEN + 1):
        padded = get_padded_prefill_len(s, m)
        if s <= MAX_TRACED_PREFILL_LEN:
            assert padded == ref_padded_len(s), s  # traced buckets keep power-of-2 padding
        elif -(-s // m) * m > QWEN_OPT_MAX_SEQ_LEN:
            assert padded == ref_padded_len(s), s  # above the Qwen opt cap: main's power of 2
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
def test_qwen_warmup_lengths(max_seq_len, expected, qwen_pad_multiple):
    assert get_prefill_warmup_sequence_lengths(max_seq_len, qwen_pad_multiple) == expected


@pytest.mark.parametrize("max_seq_len", [4096, 8192, 20000, 40960, 65536, 131072])
def test_qwen_warmup_covers_every_bucket(max_seq_len, qwen_pad_multiple):
    # Every padded length a prompt of <= max_seq_len tokens can get is warmed (no first-request compile),
    # and each warmup length pads to itself.
    m = qwen_pad_multiple
    warm = get_prefill_warmup_sequence_lengths(max_seq_len, m)
    for length in warm:
        assert get_padded_prefill_len(length, m) == length
    buckets = {get_padded_prefill_len(s, m) for s in range(1, max_seq_len + 1)}
    assert {b for b in buckets if b <= max_seq_len} <= set(warm)
    assert set(ref_warmup_lengths(max_seq_len)) <= set(warm)  # superset of the power-of-2 list


def _sp1_pow2_bound(pad_multiple):
    """(L, max_safe): L is the smallest non-power-of-2 bucket above the traced ones (24576 for 8192).
    The heuristic keeps a cached prefix only when the padded total is >= 8x the padded new tokens, so the
    first prefix-cached (sp1) prefill with a non-power-of-2 bucket L needs a padded total of 8 * L. A prompt
    pads to 8 * L once it is longer than 8 * L - pad_multiple, so max_safe = 8 * L - pad_multiple
    (188416 for 8192), not 8 * L."""
    first = (MAX_TRACED_PREFILL_LEN // pad_multiple + 1) * pad_multiple
    smallest = next(b for b in range(first, 1 << 30, pad_multiple) if b & (b - 1))
    return smallest, 8 * smallest - pad_multiple


def test_qwen_sp1_pow2_warmup_bound_is_tight(qwen_pad_multiple):
    # generator.py _prefill_warmup_sweep phase 2 (sp1) warms powers of 2 only (comment at generator.py:437-441).
    # Before the QWEN_OPT_MAX_SEQ_LEN cap, one token past max_safe a block-aligned cached prefix left new tokens
    # that padded to L (not a power of 2). With the cap, that request (total > cap) pads exactly as main: total and
    # new tokens to powers of 2 (generator.prefill_pad_multiple_for), so its sp1 bucket is a warmed power of 2.
    m = qwen_pad_multiple
    smallest, max_safe = _sp1_pow2_bound(m)
    seq_len = max_safe + 1
    num_cached = -(-(seq_len - smallest) // 128) * 128  # SDPA_CHUNK_ALIGN-aligned prefix
    assert smallest & (smallest - 1)
    pm = prefill_pad_multiple_for(seq_len, m, num_cached)
    assert seq_len > QWEN_OPT_MAX_SEQ_LEN and pm is None
    assert get_padded_prefill_len(seq_len, m) == ref_padded_len(seq_len)
    assert _should_skip_prefix_caching(seq_len, num_cached, m) == ref_skip_prefix_caching(seq_len, num_cached)
    assert get_padded_prefill_len(seq_len - num_cached, pm) == ref_padded_len(seq_len - num_cached)


def test_qwen_sp1_pow2_warmup_guard(qwen_args, qwen_pad_multiple):
    # Guard: every Qwen max_seq_len (up to max_context_len) keeps power-of-2 sp1 buckets. This was a strict
    # xfail (found 2026-10-08: max_context_len 190190 > 188416 = _sp1_pow2_bound's max_safe, an unwarmed sp1
    # bucket of 24576 in (188416, 190190]). QWEN_OPT_MAX_SEQ_LEN closes the gap: the multiple-of-8192 padding
    # applies only while cached + padded new <= 40960, where the heuristic never keeps a prefix (a padded total
    # <= 40960 is < 8 * 24576), and beyond it every bucket is main's power of 2 (prefill_pad_multiple_for).
    _check_prefix_cached_buckets_are_powers_of_two(qwen_args.max_context_len, qwen_pad_multiple)


# 188416 = 8 * 24576 - 8192, the longest max_seq_len the sp1 power-of-2 reasoning covered before the cap.
@pytest.mark.parametrize("max_seq_len", [40960, 131072, 188416])
def test_qwen_prefix_cached_buckets_are_powers_of_two(max_seq_len, qwen_pad_multiple):
    _check_prefix_cached_buckets_are_powers_of_two(max_seq_len, qwen_pad_multiple)


def _check_prefix_cached_buckets_are_powers_of_two(max_seq_len, m):
    # The sp1 (prefix-cached) warmup keeps the power-of-2 list: the heuristic never keeps a cached
    # prefix whose new tokens pad to a non-power-of-2 bucket.
    pow2 = set(ref_warmup_lengths(max_seq_len))
    for seq_len in range(129, max_seq_len + 1, 53):
        for num_cached in range(128, seq_len, 128):
            if not _should_skip_prefix_caching(seq_len, num_cached, m):
                # the generator pads the new tokens with prefill_pad_multiple_for (main's rule above the cap)
                pm = prefill_pad_multiple_for(seq_len, m, num_cached)
                padded_new = get_padded_prefill_len(seq_len - num_cached, pm)
                assert padded_new in pow2, (seq_len, num_cached)
