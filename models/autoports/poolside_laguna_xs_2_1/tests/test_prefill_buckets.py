# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Device-free invariants for ragged streaming-prefill compute shapes."""

from types import SimpleNamespace

import pytest

from models.autoports.poolside_laguna_xs_2_1.tt import generator_vllm as gv


class _Stub:
    """Minimal stand-in exercising the real bucket methods without a full model/device build."""

    _STREAMING_PREFILL_ENABLED = True
    _PREFILL_STREAM_OUTER_CHUNK = 8192
    _PREFIX_CACHE_ENABLED = False
    _streaming_prefill_active = gv.LagunaForCausalLM._streaming_prefill_active
    _prefill_bucket_lens = gv.LagunaForCausalLM._prefill_bucket_lens
    _bucket_len = gv.LagunaForCausalLM._bucket_len
    _prefill_stream_outer_chunk = gv.LagunaForCausalLM._prefill_stream_outer_chunk
    _prefill_plan_for_range = gv.LagunaForCausalLM._prefill_plan_for_range
    _prefill_bucket_for_range = gv.LagunaForCausalLM._prefill_bucket_for_range
    _prefill_page_table_width = gv.LagunaForCausalLM._prefill_page_table_width

    def __init__(self, max_model_len):
        self.max_model_len = max_model_len
        self.D = 2
        self.model = SimpleNamespace(layers=[])


@pytest.fixture(autouse=True)
def _no_warm_cap_override(monkeypatch):
    # The dev-only warm-cap knob must not perturb the default (serving) contract under test.
    monkeypatch.delenv("TT_LAGUNA_PREFILL_WARM_CAP", raising=False)
    monkeypatch.delenv("TT_LAGUNA_MULTI_SEQ_POOL", raising=False)
    # Reset the one-time-warning latch so tests are order-independent.
    if hasattr(gv.LagunaForCausalLM, "_warned_warm_cap"):
        delattr(gv.LagunaForCausalLM, "_warned_warm_cap")


def test_advertised_equals_servable():
    # 1.2: advertised is the verified-servable limit, distinct from the (larger) HF config length.
    assert gv.ADVERTISED_MAX_CONTEXT == 131072
    assert gv.HF_CONFIG_MAX_CONTEXT == 262144
    assert gv.ADVERTISED_MAX_CONTEXT <= gv.HF_CONFIG_MAX_CONTEXT


@pytest.mark.parametrize("mml", [131072, 262144, 65536, 100000, 8192, 4096])
def test_streaming_ladder_tops_at_one_outer_chunk(mml):
    stub = _Stub(mml)
    lens = stub._prefill_bucket_lens()
    servable = min(mml, gv.ADVERTISED_MAX_CONTEXT)
    outer = min(servable, stub._PREFILL_STREAM_OUTER_CHUNK)
    assert lens[0] == 32  # floor is one tile, not 128
    assert 64 in lens and 128 in lens  # small-suffix buckets present
    assert lens[-1] == outer, f"ladder top {lens[-1]} != stream outer chunk {outer}"
    assert lens == sorted(set(lens))  # strictly ascending, deduped


@pytest.mark.parametrize("plen,expect", [(8, 32), (32, 32), (33, 64), (64, 64), (74, 128), (120, 128), (129, 256)])
def test_small_suffix_buckets_2_1(plen, expect):
    """The dominant agentic prefill (8–74-token cached suffix) rounds to 32/64/128, not 128."""
    stub = _Stub(131072)
    assert stub._bucket_len(plen) == expect


@pytest.mark.parametrize("mml", [131072, 262144, 65536, 100000])
def test_every_servable_length_plans_only_warmed_chunks(mml):
    """Long requests reuse warmed 8192 chunks and one warmed tail shape."""
    stub = _Stub(mml)
    warm = set(stub._prefill_bucket_lens())
    servable = min(mml, gv.ADVERTISED_MAX_CONTEXT)
    # Edge cases + every power-of-two + a dense sweep of the pipelined regime (chunk-count boundaries).
    probes = {1, 2, 127, 128, 129, servable - 1, servable}
    probes |= {2**k for k in range(0, 18) if 2**k <= servable}
    probes |= {2**k + 1 for k in range(7, 18) if 2**k + 1 <= servable}
    # Multiples of the old pipelined chunk exercise every former chunk-count cliff.
    probes |= {c for c in range(2048, servable + 1, 2048)}
    for plen in sorted(probes):
        plan = stub._prefill_plan_for_range(plen, 0, 64)
        assert sum(chunk.real_len for chunk in plan) == plen
        assert all(chunk.bucket_len in warm for chunk in plan)
        assert all(chunk.bucket_len >= chunk.real_len for chunk in plan)


def test_high_chunk_counts_reuse_one_outer_program():
    stub = _Stub(131072)
    for length in (16384, 32768, 65536, 131072):
        plan = stub._prefill_plan_for_range(length, 0, 64)
        assert len(plan) == length // 8192
        assert {(chunk.real_len, chunk.bucket_len) for chunk in plan} == {(8192, 8192)}


def test_d2_long_stream_uses_canonical_tail_but_short_cold_request_keeps_ladder():
    stub = _Stub(131072)

    short = stub._prefill_plan_for_range(65, 0, 64)
    cliff = stub._prefill_plan_for_range(16400, 0, 64)
    early_continuation = stub._prefill_plan_for_range(65, 2048, 64)
    continuation = stub._prefill_plan_for_range(16, 16384, 64)

    assert [(chunk.real_len, chunk.bucket_len) for chunk in short] == [(65, 128)]
    assert [(chunk.real_len, chunk.bucket_len) for chunk in cliff] == [
        (8192, 8192),
        (8192, 8192),
        (16, 8192),
    ]
    assert sum(chunk.bucket_len for chunk in cliff) == 24576
    assert [(chunk.real_len, chunk.bucket_len) for chunk in early_continuation] == [(65, 8192)]
    assert [(chunk.real_len, chunk.bucket_len) for chunk in continuation] == [(16, 8192)]


def test_streaming_default_is_d2_only_and_d1_retains_safe_monolithic_geometry():
    d2 = _Stub(131072)
    d1 = _Stub(131072)
    d1.D = 1

    assert d2._streaming_prefill_active() is True
    assert d1._streaming_prefill_active() is False
    assert len(d2._prefill_plan_for_range(16384, 0, 64)) == 2
    d1_plan = d1._prefill_plan_for_range(16384, 0, 64)
    assert [(chunk.relative_start, chunk.real_len, chunk.bucket_len) for chunk in d1_plan] == [(0, 16384, 16384)]
    assert d2._prefill_page_table_width(64) == 2048
    assert d1._prefill_page_table_width(64) == 4096


def test_get_max_tokens_honors_advertised():
    cls = gv.LagunaForCausalLM
    assert cls.get_max_tokens_all_users() == gv.ADVERTISED_MAX_CONTEXT
    assert cls.get_max_tokens_all_users(max_model_len=4096) == 4096
    assert cls.get_max_tokens_all_users(max_model_len=999999) == gv.ADVERTISED_MAX_CONTEXT


def test_opt_in_multi_sequence_pool_returns_exact_two_user_budget(monkeypatch):
    monkeypatch.setenv("TT_LAGUNA_MULTI_SEQ_POOL", "1")

    assert (
        gv.LagunaForCausalLM.get_max_tokens_all_users(
            num_devices=2,
            max_model_len=65536,
            max_num_seqs=2,
        )
        == 131072
    )
    assert (
        gv.LagunaForCausalLM.get_max_tokens_all_users(
            num_devices=2,
            max_model_len=4096,
            max_num_seqs=2,
        )
        == 8192
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"num_devices": 1, "max_model_len": 65536, "max_num_seqs": 2}, "num_devices=2"),
        ({"num_devices": 2, "max_model_len": 65536, "max_num_seqs": 1}, "max_num_seqs=2"),
        ({"num_devices": 2, "max_num_seqs": 2}, "max_model_len <= 65536"),
        ({"num_devices": 2, "max_model_len": 65537, "max_num_seqs": 2}, "max_model_len <= 65536"),
    ],
)
def test_opt_in_multi_sequence_pool_fails_closed(monkeypatch, kwargs, match, expect_error):
    monkeypatch.setenv("TT_LAGUNA_MULTI_SEQ_POOL", "1")

    with expect_error(ValueError, match):
        gv.LagunaForCausalLM.get_max_tokens_all_users(**kwargs)


def test_warm_cap_override_is_bounded_and_warns(monkeypatch, capsys, expect_error):
    """The dev knob fails closed when it omits a required stream shape."""
    monkeypatch.setenv("TT_LAGUNA_PREFILL_WARM_CAP", "4096")
    if hasattr(gv.LagunaForCausalLM, "_warned_warm_cap"):
        delattr(gv.LagunaForCausalLM, "_warned_warm_cap")
    stub = _Stub(131072)
    lens = stub._prefill_bucket_lens()
    assert lens[-1] == 4096
    out = capsys.readouterr().out
    assert "WARNING" in out and "rejected before device execution" in out
    with expect_error(ValueError, "largest bucket 4096 must equal outer chunk 8192"):
        stub._prefill_plan_for_range(8192, 0, 64)
