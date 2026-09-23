# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from tests.model_behavior.adapters.galaxy_llama70b import GalaxyLlamaAdapter
from tests.model_behavior.driver import Request, RequestState, Sampling


class RecordingGenerator:
    """Record the actual generator boundary without loading a model or TTNN."""

    def __init__(self):
        self.trace_ids_decode = {}

    def prefill_forward_text(self, tokens, **kwargs):
        self.prefill_call = (tokens.clone(), kwargs)
        # Prefill returns REQUEST order, even when physical slots are reversed.
        return torch.tensor([101, 202])

    def decode_forward(self, tokens, positions, **kwargs):
        self.decode_call = (tokens.clone(), positions.clone(), kwargs)
        # Decode returns SLOT order, including inactive rows.
        return torch.arange(32) + 100, None


def make_adapter():
    # Bypass weight loading only; exercise the production adapter's packing and
    # output mapping with real torch tensors and a recording generator surface.
    adapter = GalaxyLlamaAdapter.__new__(GalaxyLlamaAdapter)
    adapter.generator = RecordingGenerator()
    adapter.sampling_params_type = SimpleNamespace
    adapter.enable_trace = False
    adapter.kv_cache = object()
    adapter.page_table = torch.arange(1, 513, dtype=torch.int32).reshape(32, 16)
    adapter.slot_token_capacity = [1024] * 32
    return adapter


def test_prefill_keeps_request_order_separate_from_slot_order():
    adapter = make_adapter()
    states = [
        RequestState(Request("tail", "a", Sampling(temperature=0.7, seed=17)), 31, (7, 8, 9)),
        RequestState(Request("head", "b", Sampling(repetition_penalty=1.5)), 0, (4, 5)),
    ]
    assert adapter.prefill(states) == {31: 101, 0: 202}
    tokens, kwargs = adapter.generator.prefill_call
    assert tokens.tolist() == [[7, 8, 9], [4, 5, 0]]
    assert kwargs["empty_slots"] == [31, 0]
    assert kwargs["prompt_lens"] == [3, 2]
    assert torch.equal(kwargs["page_table"], adapter.page_table[[31, 0]])
    assert kwargs["sampling_params"].temperature == [0.7, 0.0]
    assert kwargs["sampling_params"].seed == [17, None]
    assert kwargs["sampling_params"].repetition_penalty == [1.0, 1.5]


def test_decode_rebuilds_survivor_history_at_its_physical_slot():
    adapter = make_adapter()
    active = [
        RequestState(Request("survivor", "a", Sampling(temperature=0.8, seed=101)), 0, (7, 8), [10, 11, 12]),
        RequestState(Request("new", "b", Sampling(presence_penalty=1.0)), 31, (4, 5, 6), [20]),
    ]
    assert adapter.decode(active, reset_batch=True) == {0: 100, 31: 131}
    tokens, positions, kwargs = adapter.generator.decode_call
    assert tokens[0].item() == 12 and tokens[31].item() == 20
    assert positions[0].item() == 4 and positions[31].item() == 3
    assert positions[1:31].tolist() == [-1] * 30
    assert kwargs["page_table"] is adapter.page_table
    assert kwargs["reset_batch"] is True
    assert kwargs["output_tokens"][0].tolist() == [10, 11, 12]
    assert kwargs["output_tokens"][31].tolist() == [20, -1, -1]
    assert (kwargs["output_tokens"][1:31] == -1).all()
    assert kwargs["prompt_tokens"][0].tolist() == [7, 8, -1]
    assert kwargs["prompt_tokens"][31].tolist() == [4, 5, 6]
    assert kwargs["sampling_params"].seed == [101] + [None] * 31
    assert kwargs["sampling_params"].presence_penalty == [0.0] * 31 + [1.0]

    adapter.decode(active, reset_batch=False)
    _, _, kwargs = adapter.generator.decode_call
    assert kwargs["reset_batch"] is False
    assert kwargs["output_tokens"] is None
    assert kwargs["prompt_tokens"] is None


def test_traced_mode_does_not_silently_accept_eager_decode():
    adapter = make_adapter()
    adapter.enable_trace = True
    state = RequestState(Request("a", "x"), 0, (1,), [2])
    with pytest.raises(RuntimeError, match="without creating"):  # allow-pytest.raises: host-only isolated suite
        adapter.decode([state], reset_batch=True)


def test_unseeded_reuse_cannot_pass_by_leaving_a_seeded_sampler_active():
    adapter = make_adapter()
    adapter.enable_trace = True
    adapter.generator.trace_ids_decode = {True: 1}
    adapter.generator.model = SimpleNamespace(
        sampling=SimpleNamespace(
            seed_manager=SimpleNamespace(has_active_request_seed=lambda: True),
            _trace_states={"warmup": {"id": 2}},
        )
    )
    state = RequestState(Request("replacement", "x"), 31, (1,), [2])
    with pytest.raises(RuntimeError, match="previous request's seed"):  # allow-pytest.raises: host-only isolated suite
        adapter.decode([state], reset_batch=True)


def test_compacted_logprob_rows_are_rejected():
    adapter = make_adapter()
    states = [
        RequestState(Request(str(i), "x", Sampling(seed=i, enable_log_probs=i != 1)), slot, (1,))
        for i, slot in enumerate((31, 0, 15))
    ]
    references = [(torch.tensor([token]), torch.tensor([-1.0])) for token in (101, 202, 303)]
    with pytest.raises(ValueError, match="Logprob rows"):  # allow-pytest.raises: host-only isolated suite
        adapter._map_samples(
            states, torch.tensor([101, 202, 303]), torch.tensor([-1.0, -1.0]), references, prefill=True
        )


def test_sampled_logprob_cannot_be_attached_to_a_different_token():
    adapter = make_adapter()
    state = RequestState(Request("a", "x", Sampling(seed=0, enable_log_probs=True)), 31, (1,))
    with pytest.raises(ValueError, match="not the sampled token"):  # allow-pytest.raises: host-only isolated suite
        adapter._map_samples(
            [state],
            torch.tensor([202]),
            torch.tensor([-1.0]),
            [(torch.tensor([101]), torch.tensor([-1.0]))],
            prefill=True,
        )


def test_chunked_prefill_cannot_silently_fall_back_to_full_prefill():
    adapter = make_adapter()
    adapter.prefill_events = []
    adapter.generator.prefill_forward_single_user_text = lambda **kwargs: torch.tensor([1])
    adapter.generator._easy_trace_prefill = lambda **kwargs: torch.tensor([1])

    def fallback(tokens, **kwargs):
        return adapter.generator.prefill_forward_single_user_text(user_id=31, num_cached_tokens=0)

    adapter.generator.prefill_forward_text = fallback
    state = RequestState(Request("a", "x"), 31, tuple(range(2048)))
    with pytest.raises(AssertionError, match="did not resume"):  # allow-pytest.raises: host-only isolated suite
        adapter._checked_prefill_call([state], [2048], [1024], None)
