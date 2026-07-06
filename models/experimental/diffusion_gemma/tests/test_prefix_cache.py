# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU unit tests for the frozen prompt-prefix KV reuse decision (APC prototype, #47466).

Device bit-exactness is proven by ``demo/prefix_cache_smoke.py`` under flock; this
file pins the *decision logic* (which reuses are bit-exact-safe and which are the
#47488 partial-prefix case) without any device.
"""

from __future__ import annotations

from models.experimental.diffusion_gemma.tt.prefix_cache import PrefixKVCache


def _aligned(real_tokens, cache_len):
    """Emulate _pad_prompt_tokens_for_prefill: real tokens then zero-pad to cache_len."""
    assert cache_len >= len(real_tokens)
    return list(real_tokens) + [0] * (cache_len - len(real_tokens))


def test_empty_resident_never_reuses():
    c = PrefixKVCache()
    plan = c.plan(_aligned([1, 2, 3], 32), prompt_len=3, cache_len=32)
    assert not plan.reuse and not plan.partial_prefix and not plan.shorter_prefix and plan.matched_len == 0


def test_exact_full_match_reuses():
    c = PrefixKVCache()
    a = _aligned([5, 6, 7, 8], 32)
    c.record(a, prompt_len=4, cache_len=32)
    plan = c.plan(a, prompt_len=4, cache_len=32)
    assert plan.reuse and plan.matched_len == 32 and not plan.partial_prefix and not plan.shorter_prefix


def test_aligned_proper_prefix_is_shorter_prefix_not_bit_exact_by_default():
    # Resident A holds 64 aligned tokens (48 real, 16 pad). B is A's first 32 real
    # tokens (32-aligned, no pad) → B_aligned == A_aligned[:32]. Mathematically (fp32)
    # reusable, but NOT bit-exact in bf16 (SDPA reduction-length) → NOT reused by default.
    c = PrefixKVCache()
    a = _aligned(list(range(1, 49)), 64)
    c.record(a, prompt_len=48, cache_len=64)
    b = list(range(1, 33))
    plan = c.plan(b, prompt_len=32, cache_len=32)
    assert plan.shorter_prefix and not plan.reuse and not plan.partial_prefix


def test_aligned_proper_prefix_reuses_only_in_approximate_tier():
    c = PrefixKVCache(allow_shorter_prefix=True)
    a = _aligned(list(range(1, 49)), 64)
    c.record(a, prompt_len=48, cache_len=64)
    b = list(range(1, 33))
    plan = c.plan(b, prompt_len=32, cache_len=32)
    assert plan.shorter_prefix and plan.reuse and plan.cache_len == 32


def test_nonaligned_proper_prefix_does_not_reuse():
    # B shares real tokens with A but B_prompt_len is not 32-aligned, so B's zero-pad
    # would claim positions holding A's real-token K/V → must NOT reuse.
    c = PrefixKVCache()
    real_a = list(range(1, 49))
    c.record(_aligned(real_a, 64), prompt_len=48, cache_len=64)
    b_real = list(range(1, 20))  # 19 real tokens (shared with A), pads to 32
    plan = c.plan(_aligned(b_real, 32), prompt_len=19, cache_len=32)
    assert not plan.reuse
    # 19 shared real tokens then B's pad(0) vs A's token 20 diverge; matched < 32.
    assert plan.matched_len == 19 and not plan.partial_prefix


def test_extending_suffix_is_partial_prefix_miss():
    # B shares a long prefix with A but extends past it (differing suffix). This is
    # the #47488 / chunked-prefill case: a genuine shared prefix, not bit-exact
    # reusable at the DG serving layer.
    c = PrefixKVCache()
    shared = list(range(1, 65))  # 64 shared real tokens
    c.record(_aligned(shared, 64), prompt_len=64, cache_len=64)
    b_real = shared + [999, 998, 997]  # extends the shared prefix
    plan = c.plan(_aligned(b_real, 96), prompt_len=67, cache_len=96)
    assert not plan.reuse and plan.partial_prefix and plan.matched_len == 64


def test_longer_than_resident_never_reuses():
    c = PrefixKVCache()
    c.record(_aligned([1, 2, 3, 4], 32), prompt_len=4, cache_len=32)
    plan = c.plan(_aligned(list(range(1, 40)), 64), prompt_len=39, cache_len=64)
    assert not plan.reuse  # cache_len 64 > resident 32


def test_record_reanchors_to_current_prompt():
    c = PrefixKVCache(allow_shorter_prefix=True)
    a = _aligned(list(range(1, 49)), 64)
    c.record(a, prompt_len=48, cache_len=64)
    # After reusing B (a shorter prefix, approximate tier), the resident re-anchors to B.
    b = list(range(1, 33))
    plan = c.plan(b, prompt_len=32, cache_len=32)
    assert plan.reuse
    c.record(b, prompt_len=32, cache_len=32)
    assert c.resident_cache_len == 32
    # A no longer reuses (resident is now the shorter B; A is longer than resident).
    plan2 = c.plan(a, prompt_len=48, cache_len=64)
    assert not plan2.reuse


def test_stats_and_prefill_time_tracking():
    c = PrefixKVCache()
    c.observe_prefill_time(10.0)
    c.observe_prefill_time(20.0)
    assert c.avg_prefill_time_s == 15.0
    a = _aligned([1, 2, 3, 4], 32)
    c.record(a, prompt_len=4, cache_len=32)
    plan = c.plan(a, prompt_len=4, cache_len=32)
    c.note_reuse(plan, prefill_time_saved_s=c.avg_prefill_time_s)
    s = c.stats()
    assert s["hits"] == 1 and s["tokens_reused"] == 32 and s["prefill_time_saved_s"] == 15.0


def test_plan_validates_cache_len_matches_aligned_length(expect_error):
    c = PrefixKVCache()
    with expect_error(ValueError):
        c.plan([1, 2, 3], prompt_len=3, cache_len=32)  # len(aligned)=3 != cache_len 32
