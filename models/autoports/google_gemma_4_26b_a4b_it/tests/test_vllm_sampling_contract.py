# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only checks of the plugin's optional sampling compatibility boundary."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm_tt_plugin.async_decode import TTAsyncDecodeController, TTDecodeSubmission
from vllm_tt_plugin.input_batch import InputBatch, TTLaneInputBatch
from vllm_tt_plugin.model_input import TTModelInput, TTSamplingParams
from vllm_tt_plugin.model_runner import TTModelRunner

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tt.generator import Gemma4Generator, _Gemma4SamplingGenerator
from models.autoports.google_gemma_4_26b_a4b_it.tt.generator_vllm import AutoportGemma4ForCausalLM
from models.common.sampling.generator import SamplingGenerator, SamplingParams, _hash_request_seed_to_device_seed
from models.common.sampling.tt_penalties import TTPenalties
from vllm.sampling_params import SamplingParams as VllmSamplingParams
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.sample.sampler import Sampler
from vllm.v1.worker.gpu_input_batch import CachedRequestState


def _input(*, decode=False, logprobs=None):
    rows = 3 if decode else 2
    table = torch.arange(12, dtype=torch.int32).reshape(3, 4)
    params = TTSamplingParams(
        temperature=torch.zeros(rows),
        top_k=torch.ones(rows, dtype=torch.int32),
        top_p=torch.ones(rows),
        presence_penalty=torch.zeros(rows),
        frequency_penalty=torch.zeros(rows),
        repetition_penalty=torch.ones(rows),
        seed=torch.zeros(rows, dtype=torch.int64),
        num_logprobs=torch.full((rows,), -2 if logprobs is None else logprobs, dtype=torch.int32),
        enable_log_probs=torch.full((rows,), logprobs is not None, dtype=torch.bool),
    )
    return TTModelInput(
        input_tokens=torch.ones((rows, 1 if decode else 4), dtype=torch.int32),
        input_positions=torch.full((rows,), 4 if decode else 0, dtype=torch.int32),
        prompt_lens=None if decode else [4, 3],
        block_tables=table,
        block_tables_per_group=[table, table.flip(1)],
        block_tables_per_layer=[table, table.flip(1)],
        unpadded_batch_size=[2],
        tt_sampling_params=params,
        multi_modal_kwargs={},
        perform_device_sampling=False,
        grammar_bitmask=[None],
        logitsprocs_list=[None],
        bad_words_token_ids_list=[{}],
        allowed_token_ids_mask_list=[None],
        generators_list=[{}],
        max_num_logprobs=[logprobs],
        prefill_empty_slots=None if decode else [2, 0],
    )


def _runner():
    runner = TTModelRunner.__new__(TTModelRunner)
    runner.num_devices = 4
    runner.tt_data_parallel_size = 1
    runner.tt_per_lane_max_num_seqs = 3
    runner.sample_on_device_mode = "all"
    runner.supports_topk_logprobs = False
    runner.model_config = SimpleNamespace(logits_processors=[])
    runner.input_batch = SimpleNamespace(
        no_allowed_token_ids=True,
        max_num_logprobs=None,
        sampling=SimpleNamespace(bad_words_token_ids={}, has_active_logitsprocs=lambda: False),
    )
    runner.trace_mode = "all"
    runner.request_specific_rope = False
    runner.kv_caches = object()
    runner.vocab_size = 16
    return runner


@pytest.mark.parametrize("decode", [False, True])
@pytest.mark.parametrize("logprobs", [None, 0, 1, 3, 5, 10])
def test_tp4_logprobs_select_host_sampling_in_both_phases(decode, logprobs):
    runner = _runner()
    runner.input_batch.max_num_logprobs = logprobs
    assert runner.check_perform_device_sampling(decode, False) is (logprobs is None)


@pytest.mark.parametrize("reason", ["allowed_tokens", "bad_words", "logits_processor", "structured_output"])
def test_host_only_parameters_select_host_sampling(reason):
    runner = _runner()
    if reason == "allowed_tokens":
        runner.input_batch.no_allowed_token_ids = False
    elif reason == "bad_words":
        runner.input_batch.sampling.bad_words_token_ids = {0: [[3]]}
    elif reason == "logits_processor":
        runner.input_batch.sampling.has_active_logitsprocs = lambda: True
    assert not runner.check_perform_device_sampling(True, reason == "structured_output")


@pytest.mark.parametrize("decode", [False, True])
@pytest.mark.parametrize("flag", [None, "0", "true"])
def test_plugin_omits_sampling_params_and_default_adapter_refuses_host_mode(monkeypatch, decode, flag, expect_error):
    if flag is None:
        monkeypatch.delenv("GEMMA4_AUTOPORT_ALLOW_HOST_SAMPLING", raising=False)
    else:
        monkeypatch.setenv("GEMMA4_AUTOPORT_ALLOW_HOST_SAMPLING", flag)
    adapter = AutoportGemma4ForCausalLM.__new__(AutoportGemma4ForCausalLM)
    fake_generator = Mock()
    fake_generator.mesh = None
    AutoportGemma4ForCausalLM.__init__(adapter, fake_generator, 3)
    runner = _runner()
    model_input = _input(decode=decode, logprobs=0)
    calls = []

    def forward(**kwargs):
        calls.append(kwargs)
        method = adapter.decode_forward if decode else adapter.prefill_forward
        return method(**kwargs)

    runner.model = SimpleNamespace(prefill_forward=forward, decode_forward=forward)
    with expect_error(ValueError, "Host sampling"):
        if decode:
            TTAsyncDecodeController(runner).submit_decode(model_input, read_from_device=False)
        else:
            runner.submit_prefill(model_input, [2])
    assert len(calls) == 1
    assert "sampling_params" not in calls[0]
    assert calls[0]["page_tables_per_layer"] is model_input.block_tables_per_layer
    if decode:
        assert "reset_batch" not in calls[0]
    else:
        assert calls[0]["empty_slots"] == [2, 0]
    fake_generator.configure_sampling.assert_not_called()
    fake_generator.prefill_forward.assert_not_called()
    fake_generator.decode_forward.assert_not_called()


@pytest.mark.parametrize("decode", [False, True])
@pytest.mark.parametrize("logprobs", [None, 0, 3])
def test_plugin_host_sampler_consumes_last_token_logits_and_returns_logprobs(decode, logprobs):
    runner = _runner()
    runner.host_sampler = Sampler()
    model_input = _input(decode=decode, logprobs=logprobs)
    rows = model_input.input_tokens.shape[0]
    logits = torch.zeros(rows, 1, runner.vocab_size)
    logits[0, 0, 3] = 8
    logits[1, 0, 7] = 7
    if decode:
        logits[2, 0, 15] = 9  # Padded slot must not enter active output.
    expected_logprobs = logits[:2, 0].log_softmax(-1)
    # These checks exercise sampling math and metadata on CPU without JIT.
    with torch.compiler.set_stance("force_eager"):
        tokens, probabilities = runner._get_output_tokens(
            tt_out=logits,
            tt_log_probs=None,
            sampling_params=model_input.tt_sampling_params,
            model_input=model_input,
            batch_size_per_dp=[2],
            perform_device_sampling=False,
            is_decode=decode,
        )
    assert tokens[0].shape == (2, 1)
    assert tokens[0].flatten().tolist() == [3, 7]
    if logprobs is None:
        assert probabilities == [None]
    else:
        result = probabilities[0]
        assert result.logprobs.shape == (2, logprobs + 1)
        assert result.logprob_token_ids[:, 0].tolist() == [3, 7]
        torch.testing.assert_close(result.logprobs[:, 0], expected_logprobs[[0, 1], [3, 7]])


def test_decode_finalizer_accepts_materialized_host_logits_without_device_processing():
    runner = _runner()
    runner.model = SimpleNamespace(
        process_decode_output_host=Mock(side_effect=AssertionError("unexpected device read"))
    )
    logits = torch.randn(3, 1, runner.vocab_size)
    submission = TTDecodeSubmission(
        tt_out=logits,
        read_events=None,
        batch_size_per_dp=[2],
        sampling_params=_input(decode=True, logprobs=0).tt_sampling_params,
        perform_device_sampling=False,
    )
    result = TTAsyncDecodeController(runner).finalize_decode(submission)
    assert result.tt_out is logits
    assert result.tt_log_probs is None
    runner.model.process_decode_output_host.assert_not_called()


def test_lane_host_logits_preserve_decode_slots_and_scatter_prefill_rows():
    batch = TTLaneInputBatch.__new__(TTLaneInputBatch)
    prefill = torch.arange(32, dtype=torch.float32).reshape(2, 1, 16)
    output = batch._host_logits(prefill, [2, 0], False, 3)
    torch.testing.assert_close(output[2], prefill[0, 0])
    torch.testing.assert_close(output[0], prefill[1, 0])
    assert torch.count_nonzero(output[1]) == 0
    decode = torch.arange(48, dtype=torch.float32).reshape(3, 1, 16)
    torch.testing.assert_close(batch._host_logits(decode, [2, 0], True, 3), decode[:, 0])


def _compat_adapter(monkeypatch):
    monkeypatch.setenv("GEMMA4_AUTOPORT_ALLOW_HOST_SAMPLING", "1")
    generator = Mock()
    generator.mesh = None
    generator.prefill_trace_enabled = False
    generator.prefill_prepared = None
    generator.model.layer_indices = (0, 1)
    generator.counters = {"token_readbacks": 0, "full_logits_readbacks": 0}
    return AutoportGemma4ForCausalLM(generator, 3), generator


@pytest.mark.parametrize("decode", [False, True])
def test_opt_in_host_submission_returns_all_logits_and_preserves_row_order(monkeypatch, decode):
    adapter, generator = _compat_adapter(monkeypatch)
    runner = _runner()
    runner.model = adapter
    model_input = _input(decode=decode, logprobs=3)
    raw_logits = object()
    generator.prefill_forward.return_value = raw_logits
    generator.decode_forward.return_value = raw_logits
    shape = (1, 1, 32, 16) if decode else (2, 1, 16)
    host_logits = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32).reshape(shape)

    def read(logits):
        assert logits is raw_logits
        generator.counters["full_logits_readbacks"] += 1
        return host_logits

    generator._read_logits.side_effect = read
    if decode:
        controller = TTAsyncDecodeController(runner)
        submission = controller.submit_decode(model_input, read_from_device=False, async_read=True)
        assert submission.read_events == []
        output = controller.finalize_decode(submission).tt_out
        kwargs = generator.decode_forward.call_args.kwargs
        assert kwargs["return_logits"] is True
        assert kwargs["device_feedback"] is False
        assert torch.equal(generator.decode_forward.call_args.args[0], model_input.input_tokens)
        assert torch.equal(generator.decode_forward.call_args.args[1], model_input.input_positions)
    else:
        output = runner.submit_prefill(model_input, [2])
        kwargs = generator.prefill_forward.call_args.kwargs
        assert kwargs["prompt_lens"] == [4, 3]
        assert "slots" not in kwargs
        assert not kwargs.get("return_all_logits", False)
    rows = model_input.input_tokens.shape[0]
    assert output.shape == (rows, 1, 16)
    torch.testing.assert_close(output[:, 0], host_logits.reshape(-1, 16)[:rows])
    assert kwargs["kv_cache"] is runner.kv_caches
    assert torch.equal(kwargs["page_table"][1], model_input.block_tables_per_layer[1])
    assert kwargs["page_table"][1].data_ptr() == model_input.block_tables_per_layer[1].data_ptr()
    assert generator.counters == {"token_readbacks": 0, "full_logits_readbacks": 1}
    generator.configure_sampling.assert_not_called()
    generator.sample_prefill.assert_not_called()
    generator.sampler.sample.assert_not_called()
    assert adapter._sampling_signature is None
    assert adapter.read_decode_output(output) is output
    assert adapter.process_decode_output_host(output, is_tokens=False) is output


def test_device_host_device_transition_reconfigures_without_changing_device_default(monkeypatch):
    adapter, generator = _compat_adapter(monkeypatch)
    model_input = _input(decode=True)
    device_output = object()
    generator.decode_forward.return_value = device_output
    generator._read_logits.return_value = torch.zeros(1, 1, 32, 16)
    params = SimpleNamespace(temperature=0, top_k=1)
    common = dict(
        tokens=model_input.input_tokens,
        start_pos=model_input.input_positions,
        page_table=model_input.block_tables,
        page_tables_per_layer=model_input.block_tables_per_layer,
        kv_cache=object(),
        read_from_device=False,
        reset_batch=False,
    )
    assert adapter.decode_forward(**common, sampling_params=params) is device_output
    assert adapter.decode_forward(**common, sampling_params=params) is device_output
    generator._read_logits.assert_not_called()
    assert generator.configure_sampling.call_count == 1
    assert generator.decode_forward.call_args.kwargs["device_feedback"] is True
    host = adapter.decode_forward(**common)
    assert host.shape == (3, 1, 16)
    assert adapter._sampling_signature is None
    assert adapter.decode_forward(**common, sampling_params=params) is device_output
    assert generator.configure_sampling.call_count == 2
    assert generator._release_trace.call_count == 3
    calls = generator.decode_forward.call_args_list
    assert [call.kwargs["device_feedback"] for call in calls] == [False, True, False, False]
    assert [call.kwargs.get("return_logits", False) for call in calls] == [False, False, True, False]
    assert generator._read_logits.call_count == 1


@pytest.mark.parametrize("return_logits", [False, True])
def test_generator_capture_and_replay_only_sample_in_token_mode(monkeypatch, return_logits):
    gen = Gemma4Generator.__new__(Gemma4Generator)
    gen.prefill_prepared = None
    gen.mesh = object()
    gen.trace_debug = False
    gen.sampled_mode = True
    gen.host_sampling = False
    gen.tokens = torch.zeros(1, 1, 1, 32, dtype=torch.int32)
    gen.positions = torch.full((1, 32), 5, dtype=torch.int32)
    gen.cache_positions = torch.tensor([5, 7, -1], dtype=torch.int32)
    gen.table = object()
    gen.cache = object()
    gen.batch = 3
    gen.active_slots = (0, 1)
    gen.sampler = Mock()
    gen.sampler.tt_sampling.seeds_tt_tensor = torch.full((32,), 42, dtype=torch.int32)
    gen.sampler.sample.return_value = gen.tokens
    gen.model = Mock()
    logits = torch.randn(1, 1, 32, 16)
    gen.model.decode_forward.return_value = logits
    gen._sampler_logits = lambda value: value
    gen._format_tokens = Mock()
    gen._read_logits = Mock(side_effect=AssertionError("Low-level replay must not read logits"))
    gen.counters = {}
    monkeypatch.setattr(ttnn, "clone", lambda value: value.clone())
    monkeypatch.setattr(ttnn, "copy", lambda source, target: target.copy_(source))

    def plus_one(value, skip_negative_entries=False):
        value.add_((value >= 0).to(value.dtype) if skip_negative_entries else 1)

    increments = Mock(side_effect=plus_one)
    monkeypatch.setattr(ttnn, "plus_one", increments)
    monkeypatch.setattr(ttnn, "synchronize_device", Mock())
    monkeypatch.setattr(ttnn, "begin_trace_capture", Mock(return_value=17))
    monkeypatch.setattr(ttnn, "end_trace_capture", Mock())
    execute_trace = Mock()
    monkeypatch.setattr(ttnn, "execute_trace", execute_trace)
    gen._capture(return_logits=return_logits)
    assert gen._trace_returns_logits is return_logits
    torch.testing.assert_close(gen.positions, torch.full((1, 32), 5, dtype=torch.int32))
    torch.testing.assert_close(gen.cache_positions, torch.tensor([5, 7, -1], dtype=torch.int32))
    torch.testing.assert_close(gen.sampler.tt_sampling.seeds_tt_tensor, torch.full((32,), 42, dtype=torch.int32))
    assert gen.sampler.precompile.call_count == int(not return_logits)
    assert gen.sampler.capture_trace.call_count == int(not return_logits)
    seed_increments = sum(call.args[0] is gen.sampler.tt_sampling.seeds_tt_tensor for call in increments.call_args_list)
    assert seed_increments == (0 if return_logits else 2)
    assert gen._replay() is (logits if return_logits else gen.tokens)
    assert gen.sampler.sample.call_count == int(not return_logits)
    assert gen.counters["model_replays"] == 1
    assert gen.counters.get("sampling_replays", 0) == int(not return_logits)
    gen._read_logits.assert_not_called()
    execute_trace.assert_called_once_with(gen.mesh, 17, cq_id=0, blocking=False)


@pytest.mark.parametrize("host_sampling", [False, True])
def test_decode_formats_public_tokens_eagerly_only_for_host_sampling(host_sampling):
    gen = Gemma4Generator.__new__(Gemma4Generator)
    gen.trace_id = 17
    gen._trace_returns_logits = False
    gen.host_sampling = host_sampling
    gen.batch = 2
    gen.active_slots = (0, 1)
    gen.cache = object()
    gen.table = torch.arange(8, dtype=torch.int32).reshape(2, 4)
    gen.table_host = gen.table.clone()
    gen.public_tokens_view = object()
    gen._replay = Mock()
    gen._format_tokens = Mock()
    output = gen.decode_forward(
        None,
        torch.tensor([33, 65]),
        page_table=gen.table,
        kv_cache=gen.cache,
        device_feedback=True,
    )
    assert output is gen.public_tokens_view
    gen._replay.assert_called_once_with()
    assert gen._format_tokens.call_count == int(host_sampling)


def test_sampler_trace_formats_public_output_after_canonical_sampling_and_penalties(monkeypatch):
    sampler = _Gemma4SamplingGenerator.__new__(_Gemma4SamplingGenerator)
    sampler.mesh_device = object()
    sampler.cq_id = 0
    sampler._penalties_active = True
    sampler._log_probs_active = False
    sampler.seed_manager = SimpleNamespace(has_active_request_seed=lambda: False)
    key, slot = (True, False, False), {"id": None, "input": None, "output": None}
    sampler._trace_slot = lambda *args: (key, slot)
    sampler._trace_states = {key: slot}
    sampler._active_trace_bucket = object()
    tokens = torch.zeros(1, 1, 1, 32, dtype=torch.int32)
    public = torch.zeros(1, 1, 1, 3, dtype=torch.int32)
    sampler.bind_public_tokens(tokens, public)
    logits, log_probs = object(), object()
    events = []

    def sample(value, *, tt_out_tok):
        assert value is logits and tt_out_tok is tokens
        events.append("sample")
        tokens.flatten()[:3] = torch.tensor([7, 11, 13])
        return tokens, log_probs

    sampler.tt_sampling = Mock(side_effect=sample, force_argmax_sampling=False)
    sampler.tt_penalties = SimpleNamespace(
        apply=Mock(side_effect=lambda value: events.append("apply_penalties") or value),
        update_output_tokens=Mock(side_effect=lambda value: events.append("count_token")),
    )

    def copy_tokens(value, *, starts, ends, steps, output_tensor):
        assert value is tokens and output_tensor is public
        assert starts == (0, 0, 0, 0) and ends == (1, 1, 1, 3) and steps == (1, 1, 1, 1)
        events.append("public_copy")
        output_tensor.copy_(value[..., :3])

    monkeypatch.setattr(ttnn, "slice", copy_tokens)
    monkeypatch.setattr(ttnn, "begin_trace_capture", lambda *args, **kwargs: events.append("begin") or 17)
    monkeypatch.setattr(ttnn, "end_trace_capture", lambda *args, **kwargs: events.append("end"))
    monkeypatch.setattr(ttnn, "synchronize_device", Mock())
    monkeypatch.setattr("models.common.sampling.generator._mark_trace_buffers_corruptible", Mock())
    output = sampler.capture_trace(logits, tt_out_tok=tokens, skip_precompile=True)
    assert events == ["begin", "apply_penalties", "sample", "count_token", "public_copy", "end"]
    assert output[0] is tokens and output[1] is log_probs
    torch.testing.assert_close(public.flatten(), torch.tensor([7, 11, 13], dtype=torch.int32))
    sampler.tt_penalties.update_output_tokens.assert_called_once_with(tokens)
    events.clear()
    execute = Mock(side_effect=lambda *args, **kwargs: events.append("replay"))
    monkeypatch.setattr(ttnn, "execute_trace", execute)
    assert sampler.sample(logits, tt_out_tok=tokens, enable_trace=True) is output
    assert events == ["replay"]
    execute.assert_called_once_with(sampler.mesh_device, 17, cq_id=0, blocking=False)


@pytest.mark.parametrize("output_binding", ["none", "other", "bound"])
def test_sampler_prefill_formats_only_explicit_bound_output(monkeypatch, output_binding):
    sampler = _Gemma4SamplingGenerator.__new__(_Gemma4SamplingGenerator)
    tokens, public, logits, result = object(), object(), object(), object()
    sampler.bind_public_tokens(tokens, public)
    canonical = Mock(return_value=result)
    monkeypatch.setattr(SamplingGenerator, "_run_sampling", canonical)
    sampler.format_public_tokens = Mock()
    output_tensor = {"none": None, "other": object(), "bound": tokens}[output_binding]
    assert sampler._run_sampling(logits, penalties_on=False, tt_out_tok=output_tensor, count_tokens=False) is result
    canonical.assert_called_once_with(logits, penalties_on=False, tt_out_tok=output_tensor, count_tokens=False)
    assert sampler.format_public_tokens.call_count == int(output_binding == "bound")


@pytest.mark.parametrize("tokens,feedback", [(None, False), (torch.zeros(1, 1), True)])
def test_generator_logits_mode_requires_scheduler_feedback(tokens, feedback, expect_error):
    gen = Gemma4Generator.__new__(Gemma4Generator)
    with expect_error(ValueError, "explicit scheduler tokens and positions"):
        gen.decode_forward(
            tokens,
            torch.tensor([5]),
            page_table=object(),
            kv_cache=object(),
            return_logits=True,
            device_feedback=feedback,
        )


def test_generator_logits_decode_refreshes_scheduler_state_and_recaptures_on_mode_change(monkeypatch):
    gen = Gemma4Generator.__new__(Gemma4Generator)
    gen.mesh = object()
    gen.trace_id = 17
    gen._trace_returns_logits = False
    gen.cache = object()
    gen.batch = 3
    gen.active_slots = (0, 1)
    gen.table = torch.arange(12).reshape(3, 4)
    gen.table_host = gen.table.clone()
    gen.trace_logits = object()
    gen.public_tokens_view = object()
    gen.host_sampling = False
    gen.sampler = Mock()
    gen.counters = {}
    gen._format_tokens = Mock()
    gen._read_logits = Mock(side_effect=AssertionError("Low-level decode must return device logits"))

    def bind(*, positions, **kwargs):
        gen.tokens = torch.zeros(1, 1, 1, 32, dtype=torch.int32)
        gen.positions = torch.full((1, 32), -1, dtype=torch.int32)
        gen.positions[0, :3] = positions
        gen.cache_positions = positions.clone()

    gen._bind = Mock(side_effect=bind)

    def capture(*, return_logits=False):
        gen._trace_returns_logits = return_logits

    gen._capture = Mock(side_effect=capture)
    gen._copy = Mock(side_effect=lambda source, target, counter: target.copy_(source))
    monkeypatch.setattr(ttnn, "execute_trace", Mock())
    first_tokens = torch.tensor([[4], [5], [0]])
    first_positions = torch.tensor([31, 63, -1])
    common = dict(page_table=gen.table, kv_cache=gen.cache)
    assert gen.decode_forward(first_tokens, first_positions, **common, return_logits=True) is gen.trace_logits
    torch.testing.assert_close(gen.tokens.flatten()[:3], first_tokens.flatten().int())
    second_tokens = torch.tensor([[11], [13], [0]])
    second_positions = torch.tensor([32, 64, -1])
    assert gen.decode_forward(second_tokens, second_positions, **common, return_logits=True) is gen.trace_logits
    assert gen._bind.call_count == 1
    gen._capture.assert_called_once_with(return_logits=True)
    torch.testing.assert_close(gen.tokens.flatten()[:3], second_tokens.flatten().int())
    torch.testing.assert_close(gen.positions[0, :3], second_positions.int())
    torch.testing.assert_close(gen.cache_positions, second_positions)
    gen.sampler.sample.assert_not_called()
    gen._format_tokens.assert_not_called()
    assert gen.decode_forward(second_tokens, second_positions, **common) is gen.public_tokens_view
    assert gen._bind.call_count == 2
    assert gen._capture.call_args.kwargs == {"return_logits": False}
    assert gen.sampler.sample.call_count == 1
    gen._format_tokens.assert_not_called()
    gen._read_logits.assert_not_called()


def test_decode_trims_wire_padding_but_preserves_interior_slots(monkeypatch):
    adapter, _ = _compat_adapter(monkeypatch)
    tokens = torch.arange(32).reshape(32, 1)
    positions = torch.full((32,), -1, dtype=torch.int32)
    positions[0], positions[2] = 33, 65
    table = torch.arange(128).reshape(32, 4)
    ids, pos, tables = adapter._decode_inputs(tokens, positions, table, [table, table], reset_batch=True)
    assert ids.shape == (3, 1)
    assert pos.tolist() == [33, -1, 65]
    assert all(t.shape == (3, 4) for t in tables)
    assert ids.data_ptr() == tokens.data_ptr()
    # Steady decode retains its established active extent despite stale host state.
    ids, pos, tables = adapter._decode_inputs(
        tokens, torch.full_like(positions, -1), table, [table, table], reset_batch=False
    )
    assert ids.shape == (3, 1)
    assert adapter._decode_batch == 3


@pytest.mark.parametrize("padding_id", [0, 19])
def test_prefill_penalty_mask_excludes_padding_and_stale_tokens(monkeypatch, padding_id):
    adapter, generator = _compat_adapter(monkeypatch)
    tokens = torch.tensor([[2, 3, 3, padding_id, padding_id], [2, 4, 0, 5, 6]], dtype=torch.int32)
    original_tokens = tokens.clone()
    penalties = TTPenalties.__new__(TTPenalties)
    penalties._total_batch = 32
    penalties.vocab_size = 32
    penalties.prompt_mask = object()
    penalties._shard_dims_mask = None
    penalties._copy_int_host_to_device = Mock()
    sampler = SimpleNamespace(_penalties_active=True, tt_penalties=penalties)

    def configure(params, *, prompt_tokens, _reuse_trace=False):
        assert not _reuse_trace
        SamplingGenerator.reset_prompt_tokens(sampler, prompt_tokens)

    generator.configure_sampling.side_effect = configure
    generator.sample_prefill.return_value = torch.tensor([7, 8])
    monkeypatch.setattr(ttnn, "get_device_tensors", lambda value: [value])
    monkeypatch.setattr(ttnn, "to_torch", lambda value: value)
    table = torch.arange(8).reshape(2, 4)
    adapter.prefill_forward(
        tokens,
        table,
        object(),
        [3, 5],
        sampling_params=SimpleNamespace(repetition_penalty=1.2),
        page_tables_per_layer=[table, table],
    )
    expected_history = torch.tensor([[2, 3, 3, -1, -1], [2, 4, 0, 5, 6]], dtype=torch.int32)
    torch.testing.assert_close(generator.configure_sampling.call_args.kwargs["prompt_tokens"], expected_history)
    actual_mask = penalties._copy_int_host_to_device.call_args.args[1]
    expected_mask = torch.zeros(32, 32, dtype=torch.int32)
    expected_mask[0, [2, 3]] = 1
    expected_mask[1, [0, 2, 4, 5, 6]] = 1
    torch.testing.assert_close(actual_mask, expected_mask)
    assert generator.prefill_forward.call_args.args[0] is tokens
    torch.testing.assert_close(tokens, original_tokens)
    generator._read_logits.assert_not_called()


def test_plugin_decode_prompt_history_masks_stale_tail_in_request_order():
    batch = InputBatch.__new__(InputBatch)
    batch._req_ids = ["A", "B"]
    batch.num_prompt_tokens = torch.tensor([3, 5]).numpy()
    tokens = torch.tensor([[2, 3, 3, 19, 19], [2, 4, 0, 5, 6]], dtype=torch.int32)
    batch.token_ids_cpu = tokens.numpy()
    history = batch.make_prompt_token_ids_tensor([1, 0])
    expected = torch.tensor([[2, 4, 0, 5, 6], [2, 3, 3, -1, -1]], dtype=torch.int32)
    torch.testing.assert_close(history, expected)
    assert batch.token_ids_cpu[0, 3:].tolist() == [19, 19]


def _adapter_with_cpu_seed_state(monkeypatch):
    gen = Gemma4Generator.__new__(Gemma4Generator)
    gen.mesh = None
    gen.host_sampling = False
    gen.trace_debug = False
    gen._trace_returns_logits = False
    gen._release_trace = Mock()
    gen.sampler = SimpleNamespace(
        tt_sampling=SimpleNamespace(seeds_tt_tensor=torch.zeros(32, dtype=torch.int32)),
        reset_sampling_params=Mock(),
        reset_prompt_tokens=Mock(),
        reset_output_state=Mock(),
    )
    gen._copy = lambda value, target, counter: target.copy_(value)
    gen.model = SimpleNamespace(layer_indices=(0, 1), decode_forward=Mock(return_value=object()))
    gen._sampler_logits = lambda value: value

    def decode(tokens, start_pos, **kwargs):
        gen.tokens = tokens
        gen.positions = start_pos.clone()
        gen.cache_positions = start_pos.clone()
        gen.table = kwargs["page_table"]
        gen.cache = kwargs["kv_cache"]
        gen.batch = len(start_pos)
        gen.active_slots = tuple(range(gen.batch))
        return gen._forward()

    def increment(value, skip_negative_entries=False):
        value.add_((value >= 0).to(value.dtype) if skip_negative_entries else 1)

    gen.decode_forward = Mock(side_effect=decode)
    monkeypatch.setattr(ttnn, "plus_one", increment)
    return AutoportGemma4ForCausalLM(gen, 2), gen


def test_sampling_reconfiguration_preserves_surviving_request_seed_progress(monkeypatch):
    adapter, gen = _adapter_with_cpu_seed_state(monkeypatch)
    table = torch.arange(8).reshape(2, 4)
    common = dict(
        tokens=torch.tensor([[8], [9]]),
        page_table=table,
        page_tables_per_layer=[table, table],
        kv_cache=object(),
        read_from_device=False,
        prompt_tokens=torch.tensor([[2, 3], [2, 4]]),
    )
    params = SamplingParams(
        temperature=[0.5, 0.5], top_k=[32, 32], top_p=[1.0, 1.0], seed=[6, 99], repetition_penalty=[1.5, 1.5]
    )
    finished = SamplingParams(
        temperature=[0.5, 0.0], top_k=[32, 1], top_p=[1.0, 1.0], seed=[6, None], repetition_penalty=[1.5, 1.0]
    )
    seeds = []
    for step, (sampling_params, positions) in enumerate([(params, [3, 3]), (params, [4, 4]), (finished, [5, -1])], 1):
        history = torch.full((2, 3), -1, dtype=torch.int32)
        history[0, :step] = torch.tensor([8, 10, 12])[:step]
        if step < 3:
            history[1, :step] = torch.tensor([9, 11])[:step]
        adapter.decode_forward(
            **common,
            start_pos=torch.tensor(positions),
            sampling_params=sampling_params,
            output_tokens=history,
            reset_batch=step in (1, 3),
        )
        seeds.append(int(gen.sampler.tt_sampling.seeds_tt_tensor[0]))
    base = _hash_request_seed_to_device_seed(6, 0)
    assert seeds == [base + 1, base + 2, base + 3]
    assert gen.decode_forward.call_args_list[1].kwargs["device_feedback"] is True


def test_identical_sampling_parameters_refresh_remapped_history_on_reset(monkeypatch):
    adapter, gen = _adapter_with_cpu_seed_state(monkeypatch)
    table = torch.arange(8).reshape(2, 4)
    params = SamplingParams(
        temperature=[0.5, 0.5], top_k=[32, 32], top_p=[1.0, 1.0], seed=[6, 6], repetition_penalty=[1.5, 1.5]
    )
    prompt = torch.tensor([[2, 3], [2, 4]])
    history = torch.tensor([[8, -1], [9, 11]])
    for remap in (None, torch.tensor([1, 0])):
        expected_prompt = prompt if remap is None else prompt[remap]
        expected_history = history if remap is None else history[remap]
        adapter.decode_forward(
            tokens=torch.tensor([[8], [11]]),
            start_pos=torch.tensor([3, 4]),
            page_table=table,
            page_tables_per_layer=[table, table],
            kv_cache=object(),
            sampling_params=params,
            prompt_tokens=expected_prompt,
            output_tokens=expected_history,
            slot_remap=remap,
            reset_batch=True,
            read_from_device=False,
        )
    torch.testing.assert_close(gen.sampler.reset_prompt_tokens.call_args.args[0], expected_prompt)
    torch.testing.assert_close(gen.sampler.reset_output_state.call_args.args[0], expected_history)


@pytest.mark.parametrize("opt_in", [False, True])
@pytest.mark.parametrize("reset", [False, True])
@pytest.mark.parametrize("resume_prefill", [False, True])
def test_plugin_produces_padded_counts_without_copying_penalty_history(opt_in, reset, resume_prefill):
    runner = _runner()
    runner.model = SimpleNamespace(needs_sampling_output_counts=opt_in)
    runner.model_config.is_multimodal_model = False
    runner.max_num_blocks_per_req = 4
    runner._layer_to_group_idx = None
    runner._req_state_slot = {"A": 0, "B": 1}
    runner._decode_layout_changed_since_last_decode = reset
    runner.requests = {}
    batch = InputBatch(
        max_num_reqs=3,
        max_model_len=32,
        max_num_batched_tokens=64,
        vocab_size=32,
        block_sizes=[8],
        kernel_block_sizes=[8],
    )
    for row, (req_id, outputs) in enumerate((("A", [8]), ("B", [9, 10, 11]))):
        request = CachedRequestState(
            req_id=req_id,
            prompt_token_ids=[2, row + 3],
            mm_features=[],
            sampling_params=VllmSamplingParams(temperature=0.5, seed=6),
            generator=None,
            block_ids=([row + 1],),
            num_computed_tokens=0 if resume_prefill else 2,
            output_token_ids=outputs,
        )
        runner.requests[req_id] = request
        batch.add_request(request)
    batch.refresh_logitsprocs()
    batch.make_prompt_token_ids_tensor = Mock(side_effect=AssertionError("unexpected prompt history copy"))
    batch.make_output_token_ids_tensor = Mock(side_effect=AssertionError("unexpected output history copy"))
    runner.input_batch = batch
    scheduler = SchedulerOutput.make_empty()
    scheduler.num_scheduled_tokens = {"A": 3, "B": 5} if resume_prefill else {"A": 1, "B": 1}
    scheduler.total_num_scheduled_tokens = sum(scheduler.num_scheduled_tokens.values())
    scheduler.scheduled_cached_reqs = CachedRequestData(
        req_ids=["A", "B"],
        resumed_req_ids={"A", "B"} if resume_prefill else set(),
        new_token_ids=[],
        all_token_ids={},
        new_block_ids=[None, None],
        num_computed_tokens=[2, 2],
        num_output_tokens=[1, 3],
    )
    inputs = runner._prepare_model_inputs(scheduler, None)
    assert inputs.reset_batch is (reset and not resume_prefill)
    assert inputs.prompt_tokens is None and inputs.output_tokens is None
    if opt_in:
        assert inputs.output_token_counts.dtype == torch.int32
        assert inputs.output_token_counts.tolist() == ([1, 3] if resume_prefill else [1, 3, 0])
    else:
        assert inputs.output_token_counts is None


@pytest.mark.parametrize("opt_in", [False, True])
@pytest.mark.parametrize("device", [False, True])
@pytest.mark.parametrize("provide_counts", [False, True])
@pytest.mark.parametrize("decode", [False, True])
def test_plugin_forwards_counts_only_for_opt_in_device_sampling(opt_in, device, provide_counts, decode):
    runner = _runner()
    forward = Mock()
    runner.model = SimpleNamespace(needs_sampling_output_counts=opt_in, decode_forward=forward, prefill_forward=forward)
    counts = torch.tensor([1, 3, 0] if decode else [1, 3], dtype=torch.int32) if provide_counts else None
    inputs = replace(_input(decode=decode), perform_device_sampling=device, output_token_counts=counts)
    if decode:
        TTAsyncDecodeController(runner).submit_decode(inputs, read_from_device=False)
    else:
        runner.submit_prefill(inputs, [2])
    kwargs = forward.call_args.kwargs
    if opt_in and device and provide_counts:
        assert kwargs["output_token_counts"] is counts
    else:
        assert "output_token_counts" not in kwargs


def test_seed_counts_restore_without_penalties_and_stable_decode_does_not_reconfigure(monkeypatch):
    adapter, gen = _adapter_with_cpu_seed_state(monkeypatch)
    gen._copy = Mock(side_effect=gen._copy)
    table = torch.arange(8).reshape(2, 4)
    kwargs = dict(
        tokens=torch.tensor([[8], [9]]),
        start_pos=torch.tensor([6, 4]),
        page_table=table,
        page_tables_per_layer=[table, table],
        kv_cache=object(),
        sampling_params=SamplingParams(temperature=[0.5, 0.5], top_k=[32, 32], top_p=[1.0, 1.0], seed=[6, 99]),
        read_from_device=False,
    )
    adapter.decode_forward(**kwargs, output_token_counts=torch.tensor([4, 2]), reset_batch=True)
    base = _hash_request_seed_to_device_seed(6, 0)
    assert gen.sampler.tt_sampling.seeds_tt_tensor[0] == base + 4
    assert gen._copy.call_count == 1
    # Counts may be stale during overlapped stable decode. Device state owns the
    # next draw until a trusted reset, so these host values must not be uploaded.
    adapter.decode_forward(**kwargs, output_token_counts=torch.tensor([1, 1]), reset_batch=False)
    assert gen.sampler.tt_sampling.seeds_tt_tensor[0] == base + 5
    assert gen._copy.call_count == 1
    assert gen.sampler.reset_sampling_params.call_count == 1
    gen.sampler.reset_prompt_tokens.assert_not_called()
    assert gen.sampler.reset_output_state.call_count == 1


@pytest.mark.parametrize("counts", [[1], [[1, 2]], [1.5, 2.0], [-1, 2], [True, False]])
def test_adapter_rejects_invalid_output_counts_before_sampler_reset(monkeypatch, counts, expect_error):
    adapter, gen = _adapter_with_cpu_seed_state(monkeypatch)
    with expect_error(ValueError, "output_token_counts"):
        adapter.decode_forward(
            torch.tensor([[8], [9]]),
            torch.tensor([6, 4]),
            torch.arange(8).reshape(2, 4),
            object(),
            sampling_params=SamplingParams(temperature=0.5, top_k=32, top_p=1.0, seed=6),
            output_token_counts=counts,
            read_from_device=False,
        )
    gen.sampler.reset_sampling_params.assert_not_called()


@pytest.mark.parametrize("temperature", [0.0, 0.5])
def test_adapter_page_growth_restores_state_without_releasing_trace(monkeypatch, temperature):
    adapter, gen = _adapter_with_cpu_seed_state(monkeypatch)
    params = SamplingParams(temperature=[temperature] * 2, top_k=[32] * 2, top_p=[1.0] * 2, seed=[6, 99])
    gen.configure_sampling(params)
    gen._release_trace.reset_mock()
    gen.sampler.reset_sampling_params.reset_mock()
    gen.trace_id = 17
    gen.batch = 2
    gen.active_slots = (0, 1)
    gen.cache = object()
    original = torch.arange(8, dtype=torch.int32).reshape(2, 4)
    gen.table = (original.clone(), original.clone())
    gen.table_host = (original.clone(), original.clone())
    gen.positions = torch.full((1, 32), -1, dtype=torch.int32)
    gen.cache_positions = torch.zeros(2, dtype=torch.int32)
    gen.tokens = torch.zeros(1, 1, 1, 32, dtype=torch.int32)
    gen.public_tokens_view = object()
    gen._copy = Mock(side_effect=gen._copy)
    gen._bind = Mock(side_effect=AssertionError("unchanged trace bindings"))
    gen._capture = Mock(side_effect=AssertionError("unexpected recapture"))
    gen._replay = Mock()
    gen._format_tokens = Mock()
    gen.decode_forward = Gemma4Generator.decode_forward.__get__(gen)
    adapter._sampling_on_host = False
    adapter._sampling_signature = repr(params)
    updated = original.clone()
    updated[0, 2] = 17
    adapter.decode_forward(
        torch.tensor([[8], [9]]),
        torch.tensor([64, 61]),
        updated,
        gen.cache,
        sampling_params=params,
        page_tables_per_layer=[updated, updated],
        output_token_counts=torch.tensor([63, 60]),
        reset_batch=True,
        read_from_device=False,
    )
    assert gen.trace_id == 17
    gen._release_trace.assert_not_called()
    gen.sampler.reset_sampling_params.assert_not_called()
    gen._replay.assert_called_once()
    torch.testing.assert_close(gen.table[0], updated)
    seed_uploads = [call for call in gen._copy.call_args_list if call.args[2] == "request_seed_refreshes"]
    assert len(seed_uploads) == int(temperature != 0)


def test_resumed_prefill_restores_seed_and_actual_penalty_masks_before_sampling(monkeypatch):
    adapter, gen = _adapter_with_cpu_seed_state(monkeypatch)
    penalties = TTPenalties.__new__(TTPenalties)
    penalties._total_batch = 32
    penalties.vocab_size = 32
    penalties._shard_dims_mask = penalties._shard_dims_gathered = None
    penalties._op_kwargs = {}
    for name in ("prompt_mask", "output_mask", "output_counts", "output_counts_gathered"):
        setattr(penalties, name, torch.zeros(32, 32, dtype=torch.int32))
    penalties._copy_int_host_to_device = lambda target, source, dims: target.copy_(source)
    gen.sampler.reset_prompt_tokens.side_effect = penalties.reset_prompt_tokens
    gen.sampler.reset_output_state.side_effect = penalties.reset_output_tokens
    monkeypatch.setattr(ttnn, "mul", lambda source, scalar, *, output_tensor: output_tensor.copy_(source * scalar))
    monkeypatch.setattr(ttnn, "reshape", lambda value, shape: value.reshape(shape))
    monkeypatch.setattr(ttnn, "get_device_tensors", lambda value: [value])
    monkeypatch.setattr(ttnn, "to_torch", lambda value: value)
    gen.model.config = SimpleNamespace(vocab_size=64)
    logits = torch.zeros(2, 16)
    gen.prefill_forward = Mock(return_value=logits)
    gen.sampler.sample = Mock(return_value=torch.tensor([7, 9]))
    runner = _runner()
    runner.model = adapter
    tokens = torch.tensor([[2, 3, 8, 8, 19], [2, 4, 5, 19, 19]], dtype=torch.int32)
    original = tokens.clone()
    params = _input().tt_sampling_params
    params = replace(
        params,
        seed=torch.tensor([6, 99]),
        temperature=torch.tensor([0.5, 0.5]),
        repetition_penalty=torch.tensor([1.5, 1.5]),
        presence_penalty=torch.ones(2),
        frequency_penalty=torch.ones(2),
    )
    inputs = replace(
        _input(),
        input_tokens=tokens,
        prompt_lens=[4, 3],
        output_token_counts=torch.tensor([2, 0]),
        tt_sampling_params=params,
        perform_device_sampling=True,
    )
    output = runner.submit_prefill(inputs, [2])
    assert output.flatten().tolist() == [7, 9]
    assert gen.sampler.tt_sampling.seeds_tt_tensor[0] == _hash_request_seed_to_device_seed(6, 0) + 2
    assert gen.sampler.tt_sampling.seeds_tt_tensor[1] == _hash_request_seed_to_device_seed(99, 0)
    expected_prompt = torch.zeros(32, 32, dtype=torch.int32)
    expected_prompt[0, [2, 3]] = 1
    expected_prompt[1, [2, 4, 5]] = 1
    torch.testing.assert_close(penalties.prompt_mask, expected_prompt)
    expected_output = torch.zeros(32, 32, dtype=torch.int32)
    expected_output[0, 8] = 2
    torch.testing.assert_close(penalties.output_counts, expected_output)
    torch.testing.assert_close(penalties.output_mask, (expected_output > 0).int())
    assert gen.prefill_forward.call_args.args[0] is tokens
    assert gen.prefill_forward.call_args.kwargs["prompt_lens"] == [4, 3]
    torch.testing.assert_close(tokens, original)


def test_prefill_rejects_retained_count_beyond_logical_prefix(monkeypatch, expect_error):
    adapter, gen = _compat_adapter(monkeypatch)
    with expect_error(ValueError, "cannot exceed"):
        adapter.prefill_forward(
            torch.tensor([[2, 3, 4, 19], [2, 3, 19, 19]]),
            torch.arange(8).reshape(2, 4),
            object(),
            [3, 2],
            sampling_params=SamplingParams(temperature=0.5, top_k=32, top_p=1.0, seed=6),
            output_token_counts=torch.tensor([4, 0]),
        )
    gen.configure_sampling.assert_not_called()
    gen.prefill_forward.assert_not_called()
