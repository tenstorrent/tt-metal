# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from models.autoports.mistralai_mistral_small_24b_instruct_2501.tt import generator as generator_module
from models.autoports.mistralai_mistral_small_24b_instruct_2501.tt import generator_vllm
from models.autoports.mistralai_mistral_small_24b_instruct_2501.tt.generator import MistralSmall24BGenerator
from models.autoports.mistralai_mistral_small_24b_instruct_2501.tt.generator_vllm import (
    MAX_CONTEXT_LEN,
    TTMistralSmall24BForCausalLM,
    _canonical_sampling_args,
)


def _mock_traced_generator(copy_calls):
    generator = object.__new__(MistralSmall24BGenerator)
    generator.batch = 2
    generator._trace_model_id = 7
    generator._slots_prefilled_since_decode = set()
    generator._normalise_page_table = lambda table, active: table
    generator._validate_page_coverage = lambda table, positions: None
    generator.set_sampling_params = lambda **kwargs: None
    generator._ensure_decode_traces = lambda *args, **kwargs: None
    generator._merge_authoritative_reset_inputs = lambda tokens, positions, slot_remap: (tokens, positions)
    generator._copy_trace_state = lambda **kwargs: copy_calls.append(kwargs)
    generator._replay_split_sampling = lambda: "device-token"
    return generator


def test_prefill_threads_active_rows_to_prevent_padded_kv_writes():
    calls = []
    hidden = object()

    class FakeModel:
        config = SimpleNamespace(prefill_chunk_size=32)

        def prefill_forward(self, tokens, **kwargs):
            calls.append((tokens, kwargs))
            return hidden

    generator = object.__new__(MistralSmall24BGenerator)
    generator.batch = 32
    generator.model = FakeModel()
    generator._tokens_device = lambda value: value
    tokens = torch.arange(15).reshape(3, 5)

    result, logical, padded = generator._run_initial_prefill(
        tokens,
        page_device="pages",
        kv_cache="cache",
        prompt_lens=[5, 4, 3],
    )

    assert result is hidden
    assert (logical, padded) == (5, 32)
    assert calls[0][0].shape == (32, 32)
    assert calls[0][1]["active_batch"] == 3


def test_trace_warmup_uses_temporary_cache_then_restores_serving_state(monkeypatch):
    copies = []
    decode_calls = []
    deallocations = []
    scratch_key = object()
    scratch_value = object()
    serving_key = SimpleNamespace(shape=(7, 8, 32, 128))

    class FakeMesh:
        def set_program_cache_misses_allowed(self, value):
            pass

    class FakeModel:
        sampler = SimpleNamespace(load_device_buffers=lambda: None)

        def allocate_kv_cache(self, *, num_blocks, shared_across_layers):
            assert num_blocks == 7
            assert shared_across_layers is True
            return [(scratch_key, scratch_value)]

        def decode_forward(self, token, current_pos, rotary_pos, **kwargs):
            decode_calls.append(kwargs)
            return "logits"

        def sample_split(self, logits, **kwargs):
            return "sampled"

    trace_ids = iter((11, 12))
    monkeypatch.setattr(generator_module.ttnn, "begin_trace_capture", lambda *args, **kwargs: next(trace_ids))
    monkeypatch.setattr(generator_module.ttnn, "end_trace_capture", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        generator_module.ttnn,
        "deallocate",
        lambda tensor, force=False: deallocations.append((tensor, force)),
    )

    generator = object.__new__(MistralSmall24BGenerator)
    generator.model = FakeModel()
    generator.mesh_device = FakeMesh()
    generator._trace_token = object()
    generator._trace_current_pos = object()
    generator._trace_rotary_pos = object()
    generator._trace_page_table = object()
    generator._sampling_k = object()
    generator._sampling_p = object()
    generator._sampling_temp = object()
    scratch_page = object()
    generator._page_table_device = lambda page_table: scratch_page
    generator._copy_trace_state = lambda **kwargs: copies.append(kwargs)
    generator._synchronize = lambda: None
    generator.trace_stats = {"decode_warmups": 0, "captures": 0}

    tokens = torch.tensor([7, 0], dtype=torch.int32)
    positions = torch.tensor([185, -1], dtype=torch.int32)
    pages = torch.tensor([[1], [-1]], dtype=torch.int32)
    generator._capture_decode_traces(
        [(serving_key, object())],
        pages,
        active_batch=2,
        tokens=tokens,
        positions=positions,
    )

    assert len(decode_calls) == 2  # scratch warmup plus serving-cache capture
    assert decode_calls[0]["kv_cache"] == [(scratch_key, scratch_value)]
    assert decode_calls[0]["page_table"] is scratch_page
    assert decode_calls[1]["kv_cache"][0][0] is serving_key
    assert decode_calls[1]["page_table"] is generator._trace_page_table
    assert torch.equal(copies[0]["tokens"], tokens)
    assert torch.equal(copies[0]["positions"], positions)
    assert len(copies) == 3
    assert all(torch.equal(call["tokens"], tokens) for call in copies)
    assert all(torch.equal(call["positions"], positions) for call in copies)
    assert deallocations == [
        (scratch_key, True),
        (scratch_value, True),
        (scratch_page, True),
    ]


def test_async_overlap_ignores_stale_host_token_and_position():
    copies = []
    generator = _mock_traced_generator(copies)

    result = generator.decode_forward(
        torch.tensor([[99], [98]]),
        torch.tensor([12, 13]),
        page_table=torch.tensor([[0], [1]], dtype=torch.int32),
        kv_cache=object(),
        sampling_mode="device",
        enable_trace=True,
        reset_batch=False,
    )

    assert result == "device-token"
    assert len(copies) == 1
    assert copies[0]["tokens"] is None
    assert copies[0]["positions"] is None
    assert torch.equal(copies[0]["page_host"], torch.tensor([[0], [1]], dtype=torch.int32))


def test_scheduler_reset_refreshes_token_position_and_page_table():
    copies = []
    generator = _mock_traced_generator(copies)
    tokens = torch.tensor([[7], [8]])
    positions = torch.tensor([20, 21])
    page_table = torch.tensor([[1], [0]], dtype=torch.int32)

    generator.decode_forward(
        tokens,
        positions,
        page_table=page_table,
        kv_cache=object(),
        sampling_mode="device",
        enable_trace=True,
        reset_batch=True,
    )

    assert len(copies) == 1
    assert torch.equal(copies[0]["tokens"], tokens)
    assert torch.equal(copies[0]["positions"], positions)
    assert torch.equal(copies[0]["page_host"], page_table)


def test_fresh_prefill_forces_reset_when_scheduler_flag_is_false():
    copies = []
    merges = []
    generator = _mock_traced_generator(copies)
    generator._slots_prefilled_since_decode = {1}
    generator._merge_authoritative_reset_inputs = lambda tokens, positions, slot_remap: merges.append(slot_remap) or (
        tokens,
        positions,
    )

    generator.decode_forward(
        torch.tensor([[7], [8]]),
        torch.tensor([20, 0]),
        page_table=torch.tensor([[1], [0]], dtype=torch.int32),
        kv_cache=object(),
        sampling_mode="device",
        enable_trace=True,
        reset_batch=False,
        slot_remap=torch.tensor([0, 1]),
    )

    assert len(merges) == 1
    assert copies[0]["tokens"] is not None
    assert copies[0]["positions"] is not None


def test_pooled_page_table_preserves_scheduler_padding_without_hidden_blocks():
    generator = object.__new__(MistralSmall24BGenerator)
    generator.batch = 3
    generator.blocks_per_slot = 4
    generator.model = SimpleNamespace(config=SimpleNamespace(num_blocks=8, pooled_kv_cache=True))
    caller = torch.tensor([[4, 0], [7, 0], [2, 0]], dtype=torch.int32)

    actual = generator._normalise_page_table(caller, active_batch=3)

    assert torch.equal(actual[:, :2], caller)
    assert torch.equal(actual[:, 2:], torch.full((3, 2), -1, dtype=torch.int32))


def test_active_pooled_cache_pages_cannot_alias_between_requests(expect_error):
    generator = object.__new__(MistralSmall24BGenerator)

    with expect_error(ValueError, "alias physical KV page 5"):
        generator._validate_page_coverage(
            torch.tensor([[5, 0], [5, 0]], dtype=torch.int32),
            torch.tensor([0, 0]),
        )

    # Repeated scheduler padding outside the used prefix remains valid.
    generator._validate_page_coverage(
        torch.tensor([[5, 0], [6, 0]], dtype=torch.int32),
        torch.tensor([0, 0]),
    )


def test_adapter_delegates_device_decode_and_passes_exact_vllm_cache():
    calls = []
    cache = object()

    class FakeGenerator:
        batch = 2
        mesh_device = object()
        model = SimpleNamespace()

        def decode_forward(self, *args, **kwargs):
            calls.append((args, kwargs))
            return "sampled-device-token"

    adapter = TTMistralSmall24BForCausalLM(FakeGenerator())
    adapter._vllm_kv_cache = cache
    sampling = SimpleNamespace(top_k=[1, 1], top_p=[0.0, 0.0], temperature=[0.0, 0.0])

    result = adapter.decode_forward(
        tokens=torch.tensor([[3], [4]]),
        start_pos=torch.tensor([8, 9]),
        page_table=torch.tensor([[0], [1]], dtype=torch.int32),
        kv_cache=cache,
        enable_trace=True,
        read_from_device=False,
        sampling_params=sampling,
        reset_batch=False,
    )

    assert result == "sampled-device-token"
    assert calls[0][1]["kv_cache"] is cache
    assert calls[0][1]["sampling_mode"] == "device"
    assert calls[0][1]["enable_trace"] is True
    assert calls[0][1]["reset_batch"] is False


def test_adapter_capabilities_and_full_context_contract():
    assert TTMistralSmall24BForCausalLM.model_capabilities == {
        "supports_prefix_caching": False,
        "supports_async_decode": True,
        "supports_sample_on_device": True,
        "max_device_top_k": 32,
        "supports_device_penalties": False,
        "supports_device_seeds": False,
    }
    assert (
        TTMistralSmall24BForCausalLM.get_max_tokens_all_users(
            num_devices=4,
            tt_data_parallel=1,
            max_model_len=MAX_CONTEXT_LEN,
            max_num_seqs=32,
        )
        == MAX_CONTEXT_LEN
    )


def test_async_token_read_uses_nonblocking_ttnn_transfer(monkeypatch):
    calls = []
    device_token = object()
    host_token = object()
    event = object()
    mesh = object()

    monkeypatch.setattr(
        generator_vllm.ttnn,
        "from_device",
        lambda tensor, *, blocking: calls.append((tensor, blocking)) or host_token,
    )
    monkeypatch.setattr(generator_vllm.ttnn, "record_event", lambda device, cq_id: event)

    adapter = TTMistralSmall24BForCausalLM.__new__(TTMistralSmall24BForCausalLM)
    adapter.mesh_device = mesh

    assert adapter.read_decode_output(device_token, async_read=True) == (host_token, [event])
    assert calls == [(device_token, False)]


def test_explicit_host_sampling_output_bypasses_device_transfer(monkeypatch):
    monkeypatch.setattr(
        generator_vllm.ttnn,
        "from_device",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected device transfer")),
    )
    adapter = TTMistralSmall24BForCausalLM.__new__(TTMistralSmall24BForCausalLM)
    host_logits = torch.randn(2, 16)

    assert adapter.read_decode_output(host_logits, async_read=True) == (host_logits, [])


def test_explicit_host_sampling_decode_restores_vllm_sequence_axis(monkeypatch):
    monkeypatch.setenv(generator_vllm.HOST_SAMPLING_COMPAT_ENV, "1")
    cache = object()

    class FakeGenerator:
        batch = 2
        mesh_device = object()
        model = SimpleNamespace()

        def decode_forward(self, *args, **kwargs):
            assert kwargs["sampling_mode"] == "host"
            return torch.randn(2, 16)

    adapter = TTMistralSmall24BForCausalLM(FakeGenerator())
    adapter._vllm_kv_cache = cache
    logits = adapter.decode_forward(
        tokens=torch.tensor([[3], [4]]),
        start_pos=torch.tensor([8, 9]),
        page_table=torch.tensor([[0], [1]], dtype=torch.int32),
        kv_cache=cache,
        sampling_params=None,
    )

    assert logits.shape == (2, 1, 16)


def test_host_sampling_forwards_scheduler_state_transition(monkeypatch):
    monkeypatch.setenv(generator_vllm.HOST_SAMPLING_COMPAT_ENV, "1")
    cache = object()
    captured = {}

    class FakeGenerator:
        batch = 2
        mesh_device = object()
        model = SimpleNamespace()

        def decode_forward(self, *args, **kwargs):
            captured.update(kwargs)
            return torch.zeros((2, 8))

    adapter = TTMistralSmall24BForCausalLM(FakeGenerator())
    adapter._vllm_kv_cache = cache
    remap = torch.tensor([1, 0], dtype=torch.int32)
    adapter.decode_forward(
        tokens=torch.tensor([[3], [4]], dtype=torch.int32),
        start_pos=torch.tensor([5, 6], dtype=torch.int32),
        page_table=torch.tensor([[0], [1]], dtype=torch.int32),
        kv_cache=cache,
        sampling_params=None,
        reset_batch=True,
        slot_remap=remap,
    )

    assert captured["reset_batch"] is True
    assert captured["slot_remap"] is remap


def test_vllm_greedy_rows_use_canonical_split_sampler_encoding():
    params = SimpleNamespace(
        top_k=torch.tensor([-1, 1]),
        top_p=torch.tensor([1.0, 0.5]),
        temperature=torch.tensor([0.0, 2.0]),
    )

    assert _canonical_sampling_args(params, 2) == {
        "top_k": [1, 1],
        "top_p": [0.0, 0.5],
        "temperature": [1.0, 2.0],
    }


def test_layout_reset_keeps_async_ahead_state_but_reloads_fresh_prefill():
    host_tokens = torch.tensor([[10], [20]], dtype=torch.int32)
    host_positions = torch.tensor([5, 0], dtype=torch.int32)
    device_tokens = torch.tensor([11, 99], dtype=torch.int32)
    device_positions = torch.tensor([6, 7], dtype=torch.int32)

    tokens, positions = MistralSmall24BGenerator._merge_reset_state(
        host_tokens,
        host_positions,
        device_tokens,
        device_positions,
        torch.tensor([0, 1], dtype=torch.int32),
        {1},
    )

    assert torch.equal(tokens, torch.tensor([[11], [20]], dtype=torch.int32))
    assert torch.equal(positions, torch.tensor([6, 0], dtype=torch.int64))


def test_layout_reset_treats_prefilled_slots_as_remap_destinations():
    tokens, positions = MistralSmall24BGenerator._merge_reset_state(
        torch.tensor([[10], [20]], dtype=torch.int32),
        torch.tensor([5, 0], dtype=torch.int32),
        torch.tensor([99, 11], dtype=torch.int32),
        torch.tensor([7, 6], dtype=torch.int32),
        torch.tensor([1, 0], dtype=torch.int32),
        {1},
    )

    assert torch.equal(tokens, torch.tensor([[11], [20]], dtype=torch.int32))
    assert torch.equal(positions, torch.tensor([6, 0], dtype=torch.int64))


def test_prefill_boundary_releases_live_decode_trace():
    generator = object.__new__(MistralSmall24BGenerator)
    generator._trace_model_id = 9
    calls = []
    generator._synchronize = lambda: calls.append("sync")
    generator._release_decode_traces = lambda: calls.append("release")

    generator.prepare_for_prefill()

    assert calls == ["sync", "release"]


def test_host_prefill_marks_scheduler_slots_as_fresh(monkeypatch):
    monkeypatch.setenv(generator_vllm.HOST_SAMPLING_COMPAT_ENV, "1")
    cache = object()
    marked = []
    calls = []

    class FakeGenerator:
        batch = 2
        mesh_device = object()
        model = SimpleNamespace()

        def prepare_for_prefill(self):
            calls.append("prepare")

        def note_prefilled_slots(self, slots):
            calls.append("mark")
            marked.extend(slots)

        def prefill_forward(self, *args, **kwargs):
            calls.append("prefill")
            return torch.randn(1, 1, 16)

    adapter = TTMistralSmall24BForCausalLM(FakeGenerator())
    adapter._vllm_kv_cache = cache
    adapter.prefill_forward(
        tokens=torch.tensor([[3, 4]]),
        page_table=torch.tensor([[0]], dtype=torch.int32),
        kv_cache=cache,
        prompt_lens=[2],
        sampling_params=None,
        empty_slots=[1],
    )

    assert marked == [1]
    assert calls == ["prepare", "mark", "prefill"]
