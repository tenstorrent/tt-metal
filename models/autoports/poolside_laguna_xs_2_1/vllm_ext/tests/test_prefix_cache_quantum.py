# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
from laguna_vllm_ext import prefix_cache_quantum as quantum


class _Stats:
    def __init__(self):
        self.hits = 0
        self.preempted_hits = 0


class _Blocks:
    def __init__(self, groups):
        self.blocks = tuple(groups)


class _FullAttentionSpec:
    def __init__(self, block_size):
        self.block_size = block_size


class _OtherSpec:
    def __init__(self, block_size):
        self.block_size = block_size


def _manager_class(
    *,
    raw_tokens,
    block_size=64,
    scheduler_block_size=None,
    groups=1,
    spec_type=_FullAttentionSpec,
    log_stats=True,
    enable_caching=True,
    use_eagle=False,
    eagle_group=False,
    allocation_succeeds=True,
):
    class FakeManager:
        calls = {"get": 0, "allocate": [], "cache": []}

        def __init__(self):
            self.log_stats = log_stats
            self.prefix_cache_stats = _Stats() if log_stats else None
            self.use_eagle = use_eagle
            self.coordinator = SimpleNamespace(
                scheduler_block_size=scheduler_block_size or block_size,
                cached=[],
                cache_blocks=lambda request, tokens: self.coordinator.cached.append(
                    (request.request_id, tokens)
                ),
            )
            self.enable_caching = enable_caching
            self.max_model_len = 131_072
            self.kv_cache_config = SimpleNamespace(
                kv_cache_groups=[
                    SimpleNamespace(
                        kv_cache_spec=spec_type(block_size),
                        is_eagle_group=eagle_group,
                    )
                    for _ in range(groups)
                ]
            )
            self.empty_kv_cache_blocks = _Blocks(tuple(() for _ in range(groups)))

        def create_kv_cache_blocks(self, block_groups):
            if not any(block_groups):
                return self.empty_kv_cache_blocks
            return _Blocks(block_groups)

        def get_computed_blocks(self, request):
            type(self).calls["get"] += 1
            count = raw_tokens // block_size
            blocks = _Blocks(tuple(list(range(count)) for _ in range(groups)))
            if self.log_stats:
                field = "preempted_hits" if request.num_preemptions else "hits"
                setattr(self.prefix_cache_stats, field, getattr(self.prefix_cache_stats, field) + raw_tokens)
            return blocks, raw_tokens

        def allocate_slots(
            self,
            request,
            num_new_tokens,
            num_new_computed_tokens=0,
            new_computed_blocks=None,
            num_lookahead_tokens=0,
            num_external_computed_tokens=0,
            delay_cache_blocks=False,
            num_encoder_tokens=0,
            full_sequence_must_fit=False,
            reserved_blocks=0,
            has_scheduled_reqs=True,
        ):
            type(self).calls["allocate"].append(delay_cache_blocks)
            if self.enable_caching and not delay_cache_blocks:
                self.coordinator.cache_blocks(
                    request,
                    min(request.num_computed_tokens + num_new_computed_tokens + num_new_tokens, request.num_tokens),
                )
            if not allocation_succeeds:
                return None
            return _Blocks(([object()],))

        def cache_blocks(self, request, num_computed_tokens):
            type(self).calls["cache"].append((request.request_id, num_computed_tokens))

    return FakeManager


def _request(*, prompt_tokens=32_768, preemptions=0):
    return SimpleNamespace(
        request_id="req",
        num_prompt_tokens=prompt_tokens,
        num_preemptions=preemptions,
        num_computed_tokens=0,
        num_tokens=prompt_tokens,
    )


@pytest.fixture(autouse=True)
def _enabled(monkeypatch):
    monkeypatch.setenv("TT_LAGUNA_PREFIX_CACHE", "1")
    monkeypatch.setenv("TT_LAGUNA_PREFILL_FAST_CHUNK", "8192")
    monkeypatch.setattr(quantum, "_full_attention_spec_type", lambda: _FullAttentionSpec)


@pytest.mark.parametrize(
    ("raw_tokens", "accepted_tokens", "accepted_blocks"),
    ((2048, 0, 0), (24_576, 24_576, 384), (32_704, 24_576, 384), (65_472, 57_344, 896)),
)
def test_cache_hits_are_truncated_before_ownership(raw_tokens, accepted_tokens, accepted_blocks):
    manager_class = _manager_class(raw_tokens=raw_tokens)
    assert quantum._patch_kv_cache_manager(manager_class)
    manager = manager_class()

    blocks, tokens = manager.get_computed_blocks(_request())

    assert tokens == accepted_tokens
    assert len(blocks.blocks[0]) == accepted_blocks
    assert manager.prefix_cache_stats.hits == accepted_tokens


def test_preempted_metric_is_adjusted_in_its_own_bucket():
    manager_class = _manager_class(raw_tokens=32_704)
    quantum._patch_kv_cache_manager(manager_class)
    manager = manager_class()

    manager.get_computed_blocks(_request(preemptions=1))

    assert manager.prefix_cache_stats.hits == 0
    assert manager.prefix_cache_stats.preempted_hits == 24_576


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"groups": 2}, "exactly one uniform"),
        ({"spec_type": _OtherSpec}, "FullAttentionSpec"),
        ({"block_size": 128, "scheduler_block_size": 64}, "KV group block size"),
        ({"scheduler_block_size": 128}, "scheduler block size"),
        ({"use_eagle": True}, "EAGLE/MTP"),
        ({"eagle_group": True}, "EAGLE KV group"),
        ({"enable_caching": False}, "requires KV caching"),
    ),
)
def test_unqualified_manager_geometry_fails_at_construction(kwargs, message):
    manager_class = _manager_class(raw_tokens=32_704, **kwargs)
    quantum._patch_kv_cache_manager(manager_class)

    with pytest.raises(RuntimeError, match=message):
        manager_class()


@pytest.mark.parametrize("prompt_tokens", (1, 2048, 4096, 8191))
def test_partial_canonical_chunks_never_enter_prefix_map(prompt_tokens):
    manager_class = _manager_class(raw_tokens=0)
    quantum._patch_kv_cache_manager(manager_class)
    manager = manager_class()
    request = _request(prompt_tokens=prompt_tokens)

    manager.allocate_slots(request, prompt_tokens)
    manager.cache_blocks(request, 64)

    assert manager.coordinator.cached == []
    assert manager_class.calls["allocate"] == [True]


def test_only_complete_prompt_checkpoints_enter_prefix_map():
    manager_class = _manager_class(raw_tokens=0)
    quantum._patch_kv_cache_manager(manager_class)
    manager = manager_class()
    request = _request(prompt_tokens=20_000)

    manager.allocate_slots(request, 20_000)
    manager.cache_blocks(request, 19_000)

    assert manager.coordinator.cached == [("req", 16_384)]
    assert manager_class.calls["cache"] == [("req", 16_384)]
    assert manager_class.calls["allocate"] == [True]


def test_generated_decode_tokens_never_extend_canonical_prompt_cache():
    manager_class = _manager_class(raw_tokens=0)
    quantum._patch_kv_cache_manager(manager_class)
    manager = manager_class()
    request = _request(prompt_tokens=16_384)
    request.num_tokens = 20_000
    request.num_computed_tokens = 19_999

    manager.allocate_slots(request, 1)
    manager.cache_blocks(request, 20_000)

    assert manager.coordinator.cached == [("req", 16_384)]
    assert manager_class.calls["cache"] == [("req", 16_384)]


def test_short_then_long_then_long_admits_only_the_canonical_checkpoint():
    manager_class = _manager_class(raw_tokens=32_704)
    quantum._patch_kv_cache_manager(manager_class)
    manager = manager_class()

    short = _request(prompt_tokens=2048)
    manager.allocate_slots(short, 2048)
    assert manager.coordinator.cached == []

    long_request = _request(prompt_tokens=32_768)
    manager.allocate_slots(long_request, 32_768)
    assert manager.coordinator.cached == [("req", 32_768)]

    blocks, tokens = manager.get_computed_blocks(long_request)
    assert tokens == 24_576
    assert len(blocks.blocks[0]) == 384


@pytest.mark.parametrize(
    ("argument", "message"),
    (
        ({"num_external_computed_tokens": 64}, "external KV"),
        ({"num_lookahead_tokens": 1}, "lookahead"),
        ({"num_encoder_tokens": 1}, "decoder-only"),
    ),
)
def test_unqualified_allocation_paths_fail_closed(argument, message):
    manager_class = _manager_class(raw_tokens=0)
    quantum._patch_kv_cache_manager(manager_class)
    manager = manager_class()

    with pytest.raises(RuntimeError, match=message):
        manager.allocate_slots(_request(), 64, **argument)


def test_failed_or_caller_delayed_allocation_never_enters_prefix_map():
    failed_class = _manager_class(raw_tokens=0, allocation_succeeds=False)
    quantum._patch_kv_cache_manager(failed_class)
    failed = failed_class()
    assert failed.allocate_slots(_request(), 32_768) is None
    assert failed.coordinator.cached == []

    delayed_class = _manager_class(raw_tokens=0)
    quantum._patch_kv_cache_manager(delayed_class)
    delayed = delayed_class()
    delayed.allocate_slots(_request(), 32_768, delay_cache_blocks=True)
    assert delayed.coordinator.cached == []


def test_explicit_disable_is_a_complete_noop(monkeypatch):
    monkeypatch.setenv("TT_LAGUNA_PREFIX_CACHE", "0")
    manager_class = _manager_class(raw_tokens=32_704)
    quantum._patch_kv_cache_manager(manager_class)
    manager = manager_class()

    blocks, tokens = manager.get_computed_blocks(_request(prompt_tokens=2048))
    manager.cache_blocks(_request(prompt_tokens=2048), 64)

    assert tokens == 32_704
    assert len(blocks.blocks[0]) == 511
    assert manager_class.calls["cache"] == [("req", 64)]


def test_patch_is_idempotent():
    manager_class = _manager_class(raw_tokens=0)
    assert quantum._patch_kv_cache_manager(manager_class)
    assert not quantum._patch_kv_cache_manager(manager_class)


@pytest.mark.parametrize("bad", ("0", "-1", "4096", "6000", "16384", "not-an-int"))
def test_invalid_quantum_fails_closed(monkeypatch, bad):
    monkeypatch.setenv("TT_LAGUNA_PREFILL_FAST_CHUNK", bad)
    with pytest.raises(RuntimeError, match="quantum|TT_LAGUNA_PREFILL_FAST_CHUNK"):
        quantum.canonical_prefix_quantum()


def _vllm_config(**overrides):
    values = {
        "cache_config": SimpleNamespace(enable_prefix_caching=True, block_size=64),
        "scheduler_config": SimpleNamespace(
            enable_chunked_prefill=False,
            max_num_seqs=1,
        ),
        "speculative_config": None,
        "kv_transfer_config": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_qualified_vllm_scheduler_config_is_accepted():
    quantum.validate_prefix_cache_vllm_config(_vllm_config())


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        (
            {"cache_config": SimpleNamespace(enable_prefix_caching=False, block_size=64)},
            "not enabled",
        ),
        (
            {"cache_config": SimpleNamespace(enable_prefix_caching=True, block_size=128)},
            "block size",
        ),
        (
            {"scheduler_config": SimpleNamespace(enable_chunked_prefill=True, max_num_seqs=1)},
            "chunked prefill",
        ),
        (
            {"scheduler_config": SimpleNamespace(enable_chunked_prefill=False, max_num_seqs=2)},
            "max_num_seqs",
        ),
        ({"speculative_config": object()}, "speculative"),
        ({"kv_transfer_config": object()}, "KV-transfer"),
    ),
)
def test_unqualified_vllm_scheduler_config_fails_closed(overrides, message):
    with pytest.raises(RuntimeError, match=message):
        quantum.validate_prefix_cache_vllm_config(_vllm_config(**overrides))


def test_model_internal_spec_decode_fails_closed(monkeypatch):
    monkeypatch.setenv("TT_LAGUNA_SPEC_DECODE", "1")

    with pytest.raises(RuntimeError, match="TT_LAGUNA_SPEC_DECODE"):
        quantum.validate_prefix_cache_vllm_config(_vllm_config())


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("TT_LAGUNA_PREFILL_FAST", "0"),
        ("TT_LAGUNA_PREFILL_SDPA_CHUNK", "4096"),
        ("TT_LAGUNA_HYBRID_KV", "1"),
    ),
)
def test_direct_vllm_invocation_rejects_incompatible_model_env(
    monkeypatch, name, value
):
    monkeypatch.setenv(name, value)

    with pytest.raises(RuntimeError, match=name):
        quantum.validate_prefix_cache_vllm_config(_vllm_config())
