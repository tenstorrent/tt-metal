# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from tests.model_behavior.adapters import ADAPTERS
from tests.model_behavior.adapters.profiles import PROFILES, select_sku
from tests.model_behavior.adapters.transformers import TransformersAdapter, sampled_logprob, vocabulary_groups
from tests.model_behavior.driver import Request, RequestDriver, RequestState, Sample, Sampling
from tests.model_behavior.test_logprobs import assert_logprobs_align
from tests.model_behavior.unit.test_driver import RecordingAdapter


@pytest.mark.parametrize("backend", ADAPTERS)
def test_behavior_jobs_keep_existing_ci_skus_tiers_and_workflow_selectors(backend):
    root = Path(__file__).parents[3]
    model = "llama3.3-70b-galaxy" if backend == "galaxy-llama70b" else backend
    entries = yaml.safe_load((root / "tests/pipeline_reorg/models_e2e_tests.yaml").read_text())
    entries = [entry for entry in entries if entry["model"] == model]
    original = next(entry for entry in entries if " e2e tests" in entry["name"])
    behavior = next(entry for entry in entries if "request behavior" in entry["name"])
    if backend == "galaxy-llama70b":
        skus, hf_model = ("wh_galaxy_perf",), "meta-llama/Llama-3.3-70B-Instruct"
    else:
        skus, hf_model = PROFILES[backend].skus, PROFILES[backend].hf_model
    assert set(skus) == set(original["skus"]) == set(behavior["skus"])
    assert hf_model in original["cmd"] and hf_model in behavior["cmd"]
    assert f"--model-behavior-backend={backend}" in behavior["cmd"]
    expected_tiers = {sku: config["tier"] for sku, config in original["skus"].items()}
    assert {sku: config["tier"] for sku, config in behavior["skus"].items()} == expected_tiers
    for tier in (1, 2, 3):
        workflow = yaml.load(
            (root / f".github/workflows/models-t{tier}-e2e-tests.yaml").read_text(), Loader=yaml.BaseLoader
        )
        choices = workflow["on"]["workflow_dispatch"]["inputs"]["model"]["options"]
        assert (model in choices) == (tier in expected_tiers.values())


def test_architecture_and_sku_mismatches_fail_before_model_creation():
    assert select_sku("llama3.1-8b", "wormhole_b0", 32, "wh_llmbox_perf") == "wh_llmbox_perf"
    assert select_sku("gpt-oss-120b", "wormhole_b0", 32) == "wh_galaxy_perf"
    for backend, arch, count, sku in [
        ("qwen3.6-27b", "wormhole_b0", 32, "bh_quietbox_2"),
        ("gemma-4-26b-a4b", "blackhole", 32, "bh_galaxy"),
        ("llama3.1-8b", "wormhole_b0", 1, "wh_llmbox_perf"),
        ("llama3.1-8b", "wormhole_b0", 32, None),
    ]:
        with pytest.raises(ValueError):  # allow-pytest.raises: host-only isolated suite
            select_sku(backend, arch, count, sku)


def test_vocab_oracle_preserves_tp_and_dp_domains():
    assert vocabulary_groups((1, 8), 0, 1) == [list(range(8))]
    assert vocabulary_groups((8, 4), 0, 1) == [[0, 4, 8, 12, 16, 20, 24, 28]]
    assert vocabulary_groups((2, 4), 1, 2) == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert vocabulary_groups((2, 4), 1, 1) == [[0, 1, 2, 3]]


def test_topk_logprob_lookup_uses_token_id_not_token_rank():
    logprobs = (torch.tensor([[-0.25, -2.0], [-0.5, -1.0]]), torch.tensor([[7, 9], [9, 7]]))
    assert sampled_logprob(logprobs, 0, 9) == -2.0
    assert sampled_logprob(logprobs, 1, 9) == -0.5
    with pytest.raises(ValueError, match="absent"):  # allow-pytest.raises: host-only isolated suite
        sampled_logprob(logprobs, 1, 12)


def test_prefill_oracle_uses_last_prompt_tile_row_not_admission_slot():
    adapter = TransformersAdapter.__new__(TransformersAdapter)
    adapter.capacity = 32
    adapter._prefill_reference_row = 6
    ids = torch.arange(32) + 100
    refs = -torch.arange(32, dtype=torch.float32)
    state = RequestState(Request("last", "x", Sampling(enable_log_probs=True)), 31, (1,) * 7)
    result = adapter._map_samples([state], torch.tensor([106]), torch.tensor([-6.0]), [(ids, refs)], prefill=True)
    assert result == {31: Sample(106, -6.0, -6.0)}
    with pytest.raises(ValueError, match="differs"):  # allow-pytest.raises: host-only isolated suite
        adapter._map_samples([state], torch.tensor([131]), torch.tensor([-6.0]), [(ids, refs)], prefill=True)


def test_decode_oracle_selects_physical_rows_in_later_sampling_domain():
    adapter = TransformersAdapter.__new__(TransformersAdapter)
    adapter.capacity = 128
    ids = torch.arange(128) + 100
    refs = -torch.arange(128, dtype=torch.float32)
    state = RequestState(Request("tail", "x", Sampling(enable_log_probs=True)), 127, (1,))
    result = adapter._map_samples([state], ids, refs, [(ids, refs)], prefill=False)
    assert result == {127: Sample(227, -127.0, -127.0)}


def test_decode_only_logprobs_do_not_excuse_missing_decode_observations():
    adapter = RecordingAdapter()
    adapter.logprob_phases = ("decode",)
    driver = RequestDriver(adapter)
    state = RequestState(Request("a", "x", Sampling(enable_log_probs=True), max_tokens=3), 0, (1,))
    driver._record([state], {0: 2}, phase="prefill")
    with pytest.raises(  # allow-pytest.raises: host-only isolated suite
        ValueError, match="Missing sampled-token logprob"
    ):
        driver._record([state], {0: 3}, phase="decode")
    driver._record([state], {0: Sample(3, -1.0, -1.0)}, phase="decode")
    state.request = Request("a", "x", Sampling(enable_log_probs=True), max_tokens=2)
    assert_logprobs_align(state, start_index=1)
    with pytest.raises(AssertionError, match="Missing"):  # allow-pytest.raises: host-only isolated suite
        assert_logprobs_align(state)


def test_resumed_prefill_observation_keeps_requested_slot_and_local_row():
    adapter = TransformersAdapter.__new__(TransformersAdapter)
    adapter.prefill_events = []
    adapter.generator = SimpleNamespace(
        prefill_forward_single_user_text=lambda **kwargs: 17,
        _easy_trace_prefill=lambda **kwargs: 17,
    )
    adapter._prefill_call = lambda *args: adapter.generator.prefill_forward_single_user_text(
        user_id=0, num_cached_tokens=1024
    )
    state = RequestState(Request("tail", "x"), 31, (1,) * 1500)
    adapter._checked_prefill_call([state], [1500], [1024], None)
    event = adapter.prefill_events[0]
    assert event["slot"] == 31
    assert event["observed"] == [dict(method="prefill_forward_single_user_text", slot=31, local_slot=0, start=1024)]


def test_compacted_logprobs_cannot_pass_when_only_first_row_is_enabled():
    adapter = TransformersAdapter.__new__(TransformersAdapter)
    adapter.capacity = 32
    state = RequestState(Request("head", "x", Sampling(enable_log_probs=True)), 0, (1,))
    ids = torch.arange(32)
    with pytest.raises(ValueError, match="Logprob rows"):  # allow-pytest.raises: host-only isolated suite
        adapter._map_samples([state], ids, torch.tensor([-1.0]), [(ids, -torch.ones(32))], prefill=False)


@pytest.mark.parametrize("domains,short_length,blocks", [(1, 1024, 1024), (1, 2048, 2048), (4, 1024, 1024)])
def test_pages_never_alias_between_requests_in_the_same_device_cache(domains, short_length, blocks):
    from tests.model_behavior.adapters.transformers import allocate_pages

    capacity = 32 * domains
    long_slots = (0, (capacity - 1) // 2, capacity // 2, capacity - 1)
    lengths = [8192 if slot in long_slots else short_length for slot in range(capacity)]
    table = allocate_pages(lengths, 64, blocks, domains)
    assert table.shape == (capacity, 128)
    for group in table.reshape(domains, 32, 128):
        owned = group[group != 0]
        assert owned.numel() == owned.unique().numel()
        assert int(owned.min()) == 1 and int(owned.max()) < blocks
    for slot, length in enumerate(lengths):
        assert (table[slot, : length // 64] > 0).all()
        assert (table[slot, length // 64 :] == 0).all()
    with pytest.raises(ValueError, match="exceed"):  # allow-pytest.raises: host-only isolated suite
        allocate_pages(lengths, 64, 128, domains)


def test_gemma_releases_only_completed_requests_prefill_scratch(monkeypatch):
    from tests.model_behavior.adapters.paged import PagedAdapter

    adapter = TransformersAdapter.__new__(TransformersAdapter)
    adapter.backend = "gemma-4-26b-a4b"
    adapter.profile = PROFILES[adapter.backend]
    adapter.supports_chunked_prefill = True
    adapter.page_table = torch.tensor([[1, 2], [129, 130]])
    tails = {2: "other request's unfinished prefix", 130: "completed prefix"}
    persistent_pool = object()
    attention = SimpleNamespace(
        _release_sliding_prefill_tail=lambda *, req_key: tails.pop(req_key),
        _tail_pool=persistent_pool,
    )
    adapter.generator = SimpleNamespace(model=[SimpleNamespace(layers=[SimpleNamespace(self_attn=attention)])])
    state = RequestState(Request("tail", "x", prefill_chunk_ends=(128,)), 1, (1,) * 256)

    def complete_prefill(self, admitted):
        assert admitted == [state]
        # Tails must survive through all intermediate chunks and final sampling.
        assert 130 in tails and 2 in tails
        return {1: 17}

    monkeypatch.setattr(PagedAdapter, "prefill", complete_prefill)
    assert adapter.prefill([state]) == {1: 17}
    assert tails == {2: "other request's unfinished prefix"}
    assert attention._tail_pool is persistent_pool
