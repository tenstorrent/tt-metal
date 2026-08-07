# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Device-free invariants for the prefill bucketing / warm-set contract.

The bug (1.1): the sequence-pipelined prefill tail reassembles per-chunk outputs with a program whose
shape depends on the chunk COUNT, so a prompt needing more chunks than any warmed bucket first-compiles
that program under the resident decode trace. These tests pin the invariant that fixes it:

    for every servable prompt length, ``_bucket_len(plen)`` is (a) >= plen and (b) a member of the
    warmed set ``_prefill_bucket_lens()`` — i.e. no servable prompt ever hits an un-warmed program.

Plus 1.2: the advertised context equals the verified-servable context, and get_max_tokens honors it.

These are pure-Python (no ttnn device); they run in CI and are the fast gate for any change to the
bucket ladder or the advertised context.
"""

import pytest

from models.autoports.poolside_laguna_xs_2_1.tt import generator_vllm as gv


class _Stub:
    """Minimal stand-in exercising the real bucket methods without a full model/device build."""

    _prefill_bucket_lens = gv.LagunaForCausalLM._prefill_bucket_lens
    _bucket_len = gv.LagunaForCausalLM._bucket_len

    def __init__(self, max_model_len):
        self.max_model_len = max_model_len


@pytest.fixture(autouse=True)
def _no_warm_cap_override(monkeypatch):
    # The dev-only warm-cap knob must not perturb the default (serving) contract under test.
    monkeypatch.delenv("TT_LAGUNA_PREFILL_WARM_CAP", raising=False)
    # Reset the one-time-warning latch so tests are order-independent.
    if hasattr(gv.LagunaForCausalLM, "_warned_warm_cap"):
        delattr(gv.LagunaForCausalLM, "_warned_warm_cap")


def test_advertised_equals_servable():
    # 1.2: advertised is the verified-servable limit, distinct from the (larger) HF config length.
    assert gv.ADVERTISED_MAX_CONTEXT == 131072
    assert gv.HF_CONFIG_MAX_CONTEXT == 262144
    assert gv.ADVERTISED_MAX_CONTEXT <= gv.HF_CONFIG_MAX_CONTEXT


@pytest.mark.parametrize("mml", [131072, 262144, 65536, 100000, 8192, 4096])
def test_ladder_tops_at_servable(mml):
    stub = _Stub(mml)
    lens = stub._prefill_bucket_lens()
    servable = min(mml, gv.ADVERTISED_MAX_CONTEXT)
    assert lens[0] == 32  # floor is one tile, not 128
    assert 64 in lens and 128 in lens  # small-suffix buckets present
    assert lens[-1] == servable, f"ladder top {lens[-1]} != servable {servable}"
    assert lens == sorted(set(lens))  # strictly ascending, deduped


@pytest.mark.parametrize("plen,expect", [(8, 32), (32, 32), (33, 64), (64, 64), (74, 128), (120, 128), (129, 256)])
def test_small_suffix_buckets_2_1(plen, expect):
    """The dominant agentic prefill (8–74-token cached suffix) rounds to 32/64/128, not 128."""
    stub = _Stub(131072)
    assert stub._bucket_len(plen) == expect


@pytest.mark.parametrize("mml", [131072, 262144, 65536, 100000])
def test_every_servable_length_maps_into_warm_set(mml):
    """The core 1.1 invariant: no in-contract prompt length rounds to an un-warmed bucket."""
    stub = _Stub(mml)
    warm = set(stub._prefill_bucket_lens())
    servable = min(mml, gv.ADVERTISED_MAX_CONTEXT)
    # Edge cases + every power-of-two + a dense sweep of the pipelined regime (chunk-count boundaries).
    probes = {1, 2, 127, 128, 129, servable - 1, servable}
    probes |= {2**k for k in range(0, 18) if 2**k <= servable}
    probes |= {2**k + 1 for k in range(7, 18) if 2**k + 1 <= servable}
    # Multiples of the pipelined chunk (2048) — the counts the old fixed-8192 cap never warmed.
    probes |= {c for c in range(2048, servable + 1, 2048)}
    for plen in sorted(probes):
        b = stub._bucket_len(plen)
        assert b >= plen, f"bucket {b} < plen {plen} would truncate the prompt"
        assert b in warm, f"plen={plen} -> bucket {b} NOT in warmed set (would compile under trace)"


def test_high_chunk_counts_are_warmed():
    """Regression for the specific 1.1 failure: >4-chunk prompts (>=16384 at PIPE_CHUNK=2048) must be
    warmed. Under the old fixed 8192 cap only 2- and 4-chunk counts were compiled."""
    stub = _Stub(131072)
    warm = set(stub._prefill_bucket_lens())
    for L in (16384, 32768, 65536, 131072):  # 8, 16, 32, 64 chunks at chunk=2048
        assert stub._bucket_len(L) == L
        assert L in warm


def test_get_max_tokens_honors_advertised():
    cls = gv.LagunaForCausalLM
    assert cls.get_max_tokens_all_users() == gv.ADVERTISED_MAX_CONTEXT
    assert cls.get_max_tokens_all_users(max_model_len=4096) == 4096
    assert cls.get_max_tokens_all_users(max_model_len=999999) == gv.ADVERTISED_MAX_CONTEXT


def test_warm_cap_override_is_bounded_and_warns(monkeypatch, capsys):
    """The dev knob can only LOWER the ceiling and must warn that prompts above it compile under trace."""
    monkeypatch.setenv("TT_LAGUNA_PREFILL_WARM_CAP", "8192")
    if hasattr(gv.LagunaForCausalLM, "_warned_warm_cap"):
        delattr(gv.LagunaForCausalLM, "_warned_warm_cap")
    stub = _Stub(131072)
    lens = stub._prefill_bucket_lens()
    assert lens[-1] == 8192
    out = capsys.readouterr().out
    assert "WARNING" in out and "under the resident" in out
