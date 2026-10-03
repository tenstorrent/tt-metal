# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host contracts; no checkpoint load, TT tensor allocation, or mesh access."""

import json
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import ttnn
from models.demos.gpt_oss.tt.model import Model as GptOssModel
from models.demos.gpt_oss_120b_qb2.tt.generator import Generator, TraceEvidence
from models.demos.gpt_oss_120b_qb2.tt.generator_vllm import TTGptOssForCausalLM
from models.demos.gpt_oss_120b_qb2.tt.model import decode_trace_buckets
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.generator import Generator as SharedGenerator


def host_logits_inner():
    model = SimpleNamespace(
        n_layers=1,
        vocab_size=128,
        mesh_config=SimpleNamespace(get_config=lambda mode: SimpleNamespace(tp=1)),
        concat_device_output=lambda output: output,
    )
    model.process_output_decode = MethodType(GptOssModel.process_output_decode, model)
    inner = SimpleNamespace(model=[model], model_args=[SimpleNamespace(max_batch_size=32)], data_parallel=1)
    inner.process_decode_output_host = MethodType(SharedGenerator.process_decode_output_host, inner)
    return inner


@pytest.mark.parametrize("widths", [(1, 32), (32, 1), (4, 8)])
def test_host_logits_preserve_tensor_rows_without_mutating_model_args(widths):
    generator = object.__new__(Generator)
    generator._inner = host_logits_inner()
    generator.model_args = SimpleNamespace(max_batch_size=32)
    for width in widths:
        output = torch.arange(width * 128).reshape(1, 1, width, 128)
        logits, _ = generator.process_decode_output_host([(output, None)])
        assert logits.shape == (width, 1, 128)
        assert logits[-1, 0, -1] == width * 128 - 1
        assert generator.model_args.max_batch_size == 32


def test_teardown_calls_canonical_idempotent_release_once():
    generator = object.__new__(Generator)
    inner = SimpleNamespace(release_persistent_capture=Mock())
    generator._inner = inner
    generator._torn_down = False
    generator.teardown()
    generator.teardown()
    inner.release_persistent_capture.assert_called_once_with()


@pytest.mark.parametrize("width", [1, 4, 8, 32])
@pytest.mark.parametrize("read_from_device", [False, True])
def test_synchronous_host_decode_preserves_full_vocabulary(width, read_from_device):
    generator = object.__new__(Generator)
    generator.model_args = SimpleNamespace(max_batch_size=32, max_context_len=131072)
    generator._inner = host_logits_inner()
    generator.model = generator._inner.model[0]
    device_output = object()
    generator._inner.decode_forward = Mock(return_value=device_output)
    generator._inner.read_decode_output = Mock(
        return_value=[(torch.arange(width * 128).reshape(1, 1, width, 128), None)]
    )
    generator.trace_evidence = TraceEvidence()
    generator._record_decode_staging = Mock()
    output = generator.decode_forward(
        tokens=torch.zeros(width, 1, dtype=torch.int64),
        start_pos=torch.zeros(width, dtype=torch.int64),
        page_table=torch.zeros(width, 1, dtype=torch.int32),
        kv_cache=[object()],
        enable_trace=False,
        sampling_mode="host",
        reload_inputs=True,
        reload_page_table=False,
        reload_sampling_params=False,
        reset_sampling_state=False,
        read_from_device=read_from_device,
    )
    assert generator._inner.decode_forward.call_args.kwargs["reload_inputs"] is True
    assert generator._inner.decode_forward.call_args.kwargs["read_from_device"] is False
    if read_from_device:
        generator._inner.read_decode_output.assert_called_once_with(device_output)
        assert output.shape == (width, 128)
        assert output[-1, -1] == width * 128 - 1
    else:
        generator._inner.read_decode_output.assert_not_called()
        assert output is device_output


def test_ring_ownership_moves_with_slots_and_released_slot_resumes_cold():
    adapter = object.__new__(TTGptOssForCausalLM)
    adapter.max_batch_size = 4
    adapter.max_model_len = 131072
    adapter._sliding_layers = [0]
    adapter._ring_of_slot = [0, 1, 2, 3]
    adapter._slot_prefill_end = [8192, 4096, None, 1024]
    adapter._ring_tables_dirty = False
    adapter._apply_ring_slot_remap([1, 0, 3, 2])
    assert adapter._ring_of_slot == [1, 0, 3, 2]
    assert adapter._slot_prefill_end == [4096, 8192, 1024, None]
    assert adapter._ring_tables_dirty
    assert adapter._resume_plan(8192, 1) == (8192, 8192, False)
    adapter.release_request(1)
    assert adapter._resume_plan(8192, 1) == (5632, 8192, True)


def test_selected_buckets_exclude_historically_corrupt_sixteen_row_trace():
    assert decode_trace_buckets(32) == (1, 4, 8, 32)
    # Requests 9 through 16 therefore select the 32-row device graph.
    for active in (9, 15, 16, 17, 31, 32):
        assert next(width for width in decode_trace_buckets(32) if width >= active) == 32


def test_shared_generator_resumes_at_model_alignment_without_changing_other_models():
    from models.tt_transformers.tt.generator import Generator as SharedGenerator

    defaults = dict(SharedGenerator.model_capabilities)
    model = SimpleNamespace(mesh_device=None, kv_cache=[])
    generator = Generator(model, SimpleNamespace(tokenizer=None), cache_owner="vllm")
    cached = [0, 64, 511, 512, 8191, 8192]
    cache_tensor = SimpleNamespace(shape=(4640, 2, 64, 64))
    cache = [[[cache_tensor, cache_tensor]]]
    aligned = generator._inner._align_resume_offsets(cached, [position + 513 for position in cached], cache)
    assert aligned == [0, 0, 0, 512, 7680, 8192]
    assert SharedGenerator.model_capabilities == defaults


@pytest.mark.parametrize("device_sampling", [False, True])
def test_decode_warmup_uses_current_interface_and_only_traces_device_routes(device_sampling, monkeypatch):
    monkeypatch.delenv("TT_LEAN_DECODE_WARMUP", raising=False)
    model = SimpleNamespace(mesh_device=None, kv_cache=[], n_layers=1)
    generator = Generator(model, SimpleNamespace(tokenizer=None), cache_owner="vllm")
    generator._inner.decode_forward = Mock()
    for trace in (False, True):
        generator._inner.decode_forward.reset_mock()
        generator.warmup_model_decode(
            kv_cache=[object()],
            enable_trace=trace,
            max_batch_size=4,
            num_blocks=16,
            can_sample_on_device=device_sampling,
            skip_trace_precompile=trace,
        )
        calls = generator._inner.decode_forward.call_args_list
        assert len(calls) == (5 if device_sampling else 0) + int(not trace)
        assert sum(call.kwargs["sampling_params"] is None for call in calls) == int(not trace)
        for call in calls:
            assert call.kwargs["tokens"].shape == (4, 1)
            assert call.kwargs["start_pos"].shape == (4,)
            assert call.kwargs["page_table"].shape == (4, 16)
            assert call.kwargs["enable_trace"] is trace
            assert call.kwargs["read_from_device"] is False


def test_host_prefill_trims_each_layer_table_using_its_cache_block_axis():
    generator = object.__new__(Generator)
    generator.model_args = SimpleNamespace(max_batch_size=32, max_context_len=131072)
    generator.model = SimpleNamespace(n_layers=2)
    generator._inner = SimpleNamespace(mode=None)
    generator._prepare_prefill_variants = Mock(return_value=set())
    generator._record_compiled_prefill_variants = Mock()
    generator._prefill_one = Mock(return_value=torch.zeros(1, 128))
    # Individual caches are [K, V], each [physical_pages, kv_heads, block, head].
    caches = [[SimpleNamespace(shape=(64, 2, block, 64))] * 2 for block in (64, 128)]
    tables = [torch.arange(16).reshape(2, 8), torch.arange(32, 48).reshape(2, 8)]
    output = generator.prefill_forward(
        torch.zeros(2, 129, dtype=torch.int64),
        page_table=tables[0],
        kv_cache=caches,
        prompt_lens=[65, 129],
        page_tables_per_layer=tables,
    )
    assert output.shape == (2, 1, 128)
    for row, (first_width, second_width) in enumerate([(2, 1), (3, 2)]):
        passed = generator._prefill_one.call_args_list[row].kwargs["page_tables_per_layer"]
        assert torch.equal(passed[0], tables[0][row : row + 1, :first_width])
        assert torch.equal(passed[1], tables[1][row : row + 1, :second_width])


@pytest.mark.parametrize("tokens", [256, 512, 640, 769])
def test_ring_override_cannot_discard_prefix_resume_history(monkeypatch, tokens):
    from models.demos.gpt_oss_120b_qb2.tt import sliding_ring

    monkeypatch.setenv(sliding_ring.ENV_SLIDING_RING_TOKENS, str(tokens))
    with pytest.raises(ValueError, match="prefix resumes"):  # allow-pytest.raises: host-only, no device conftest
        sliding_ring._ring_tokens()


def test_missing_precision_manifest_fails_instead_of_changing_policy(monkeypatch, tmp_path):
    from models.demos.gpt_oss_120b_qb2.tt import precision

    monkeypatch.delenv("GPT_OSS_120B_PRECISION_CONFIG", raising=False)
    monkeypatch.setattr(precision, "DEFAULT_PRECISION_CONFIG_PATH", tmp_path / "missing.json")
    with pytest.raises(FileNotFoundError):  # allow-pytest.raises: host-only, no device conftest
        precision.load_precision_config()


@pytest.mark.parametrize("last_tile, expected_fill", [(192, 224), (-1, 1024)])
def test_single_user_prefill_limits_cache_fill_to_valid_last_tile(last_tile, expected_fill):
    from models.demos.gpt_oss_120b_qb2.tt.model import Model

    model = object.__new__(Model)
    model.norm = SimpleNamespace(decode_mode=True)
    model._run_decoder_stack = Mock(return_value=object())
    Model._forward_layers_and_head(
        model,
        hidden_states=SimpleNamespace(shape=(1, 1, 1024, 2880)),
        is_decode=False,
        batch_size=1,
        get_last_token=last_tile,
    )
    assert model._run_decoder_stack.call_args.kwargs["fill_seq_lens"] == [expected_fill]


def test_batched_host_prefill_projects_hidden_states_once_then_single_user_keeps_logits(monkeypatch):
    from models.demos.gpt_oss_120b_qb2.tt.model import Model, _GPTOSSModel

    model = object.__new__(Model)
    model.norm = SimpleNamespace(decode_mode=True)
    model._prefill_rope_slices = {64: [None, None]}
    weight = torch.arange(28, dtype=torch.float32).reshape(4, 7) / 28

    def project(hidden):
        return torch.nn.functional.layer_norm(hidden, (4,)) @ weight

    model._apply_norm_and_lm_head = Mock(side_effect=project)
    model._run_decoder_stack = lambda hidden_states, skip_lm_head=False, **kwargs: (
        hidden_states if skip_lm_head else project(hidden_states)
    )
    monkeypatch.setattr(ttnn, "reshape", torch.reshape)
    monkeypatch.setattr(
        _GPTOSSModel,
        "process_logits_after_prefill_trace",
        lambda self, output, last: output[..., (last // 32) * 32 : (last // 32 + 1) * 32, :],
    )
    for batch in (4, 1):
        hidden = torch.arange(batch * 64 * 4, dtype=torch.float32).reshape(1, 1, batch * 64, 4)
        output = model._forward_layers_and_head(hidden_states=hidden, is_decode=False, batch_size=batch)
        output = output.reshape(batch, 1, 64, -1)
        for row in range(batch):
            actual = model.process_logits_after_prefill_trace(output[row : row + 1], 47)
            expected = project(hidden.reshape(batch, 1, 64, 4)[row : row + 1, :, 32:64])
            torch.testing.assert_close(actual, expected)
            assert actual.shape == (1, 1, 32, 7)
    # Four batched rows need projection; the following single-user result is already logits.
    assert model._apply_norm_and_lm_head.call_count == 4


def test_serving_report_serializes_fabric_without_mutating_capabilities(monkeypatch, tmp_path):
    adapter = object.__new__(TTGptOssForCausalLM)
    adapter.model = SimpleNamespace(
        n_layers=36,
        kv_cache_owner="vllm",
        precision_runtime_evidence=lambda: {},
        precision_config=SimpleNamespace(
            decoder_policy_for_layer=lambda index: SimpleNamespace(kv_cache_dtype=ttnn.bfloat8_b)
        ),
    )
    adapter.generator = SimpleNamespace(capability_report=lambda: {})
    adapter.mesh_device = SimpleNamespace(shape=(1, 4))
    adapter.max_model_len = 131072
    adapter.max_batch_size = 32
    adapter._sliding_layers = list(range(0, 36, 2))
    adapter._ring_block_base = 4640
    adapter._cache_tensor_indices = []
    adapter._cache_shapes = []
    adapter.serving_counters = {}
    fabric = adapter.model_capabilities["fabric_config"]["config"]
    monkeypatch.setenv("GPT_OSS_120B_RESULTS", str(tmp_path))
    adapter._write_serving_capability()
    report = json.loads((tmp_path / "vllm_serving_capability.json").read_text())
    assert report["model_capabilities"]["fabric_config"]["config"] == str(fabric)
    assert adapter.model_capabilities["fabric_config"]["config"] is fabric
    assert report["resident_layers"] == 36


@pytest.mark.parametrize("logical_len,padded_len", [(3, 128), (4, 128), (128, 128), (129, 1024), (511, 1024)])
@pytest.mark.parametrize("return_all_logits", [False, True])
def test_host_prefill_matches_shared_padding_and_preserves_logical_outputs(logical_len, padded_len, return_all_logits):
    generator = object.__new__(Generator)
    page_table = torch.zeros(1, 8, dtype=torch.int32)
    token_ids = torch.arange(1, logical_len + 1).reshape(1, logical_len)
    all_logits = torch.arange(padded_len * 128).reshape(1, 1, padded_len, 128)

    def forward(embedded, *, get_last_token, **kwargs):
        return all_logits if get_last_token == -1 else all_logits[:, :, get_last_token : get_last_token + 32]

    prepare = Mock(side_effect=lambda tokens, **kwargs: (tokens, None, None, page_table))
    generator.model = SimpleNamespace(prepare_inputs_prefill=prepare, ttnn_prefill_forward=Mock(side_effect=forward))
    generator._gather_prefill_logits = lambda output: output
    generator.trace_evidence = TraceEvidence()
    actual = generator._prefill_one(token_ids, page_table, [], return_all_logits=return_all_logits)

    prepared = prepare.call_args.args[0]
    assert prepared.shape == (1, padded_len)
    torch.testing.assert_close(prepared[:, :logical_len], token_ids, rtol=0, atol=0)
    assert torch.count_nonzero(prepared[:, logical_len:]) == 0
    expected = all_logits[0, 0, :logical_len] if return_all_logits else all_logits[0, 0, logical_len - 1 : logical_len]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def decode_contract_generator():
    """Use the production shared decode body, with device operations replaced."""
    generator = object.__new__(Generator)
    generator.model_args = SimpleNamespace(max_batch_size=32, max_context_len=131072)
    generator.trace_evidence = TraceEvidence()
    generator._decode_started = False
    generator._prepared_device_sampling_params = None
    model = SimpleNamespace(switch_mode=Mock())
    inner = SimpleNamespace(
        model=[model],
        data_parallel=1,
        mode=Mode.DECODE,
        _decode_forward_trace_text=Mock(return_value=object()),
        _decode_forward_no_trace_text=Mock(return_value=object()),
        sample_decode_on_device=Mock(side_effect=lambda logits, **kwargs: logits),
        _apply_sampling_slot_remap=Mock(),
        read_decode_output=Mock(side_effect=lambda output: output),
        process_decode_output_host=Mock(side_effect=lambda output, **kwargs: (output, None)),
    )
    inner.decode_forward = MethodType(SharedGenerator.decode_forward, inner)
    generator._inner = inner
    generator._sampling_has_active_request_seed = Mock(return_value=False)
    generator._replay_prepared_sampling = Mock(side_effect=lambda logits, **kwargs: logits)
    return generator


def contract_decode(generator, **overrides):
    args = dict(
        tokens=torch.tensor([[7], [9]]),
        start_pos=torch.tensor([31, 63]),
        page_table=torch.tensor([[3, 4], [5, 6]], dtype=torch.int32),
        kv_cache=[],
        read_from_device=False,
        reload_inputs=False,
        reload_page_table=False,
        reload_sampling_params=False,
        reset_sampling_state=False,
    )
    args.update(overrides)
    return generator.decode_forward(**args)


@pytest.mark.parametrize(
    "reload_inputs,reload_page_table", [(False, False), (False, True), (True, False), (True, True)]
)
@pytest.mark.parametrize("reload_sampling_params,reset_sampling_state", [(False, False), (True, False), (True, True)])
def test_decode_commands_reach_shared_model_and_sampler(
    reload_inputs, reload_page_table, reload_sampling_params, reset_sampling_state
):
    generator = decode_contract_generator()
    prompt = torch.tensor([[1, 2], [3, 4]])
    output = torch.tensor([[8], [10]])
    contract_decode(
        generator,
        reload_inputs=reload_inputs,
        reload_page_table=reload_page_table,
        reload_sampling_params=reload_sampling_params,
        reset_sampling_state=reset_sampling_state,
        prompt_tokens=prompt,
        output_tokens=output,
        slot_remap=[0, 1],
    )
    model_call = generator._inner._decode_forward_trace_text.call_args.kwargs
    assert model_call["reload_inputs"] is reload_inputs
    assert model_call["reload_page_table"] is reload_page_table
    sampler_call = generator._inner.sample_decode_on_device.call_args.kwargs
    assert sampler_call["reload_sampling_params"] is reload_sampling_params
    assert sampler_call["reset_sampling_state"] is reset_sampling_state
    assert sampler_call["reload_inputs"] is reload_inputs
    assert sampler_call["prompt_tokens"] is prompt
    assert sampler_call["output_tokens"] is output
    assert sampler_call["slot_remap"] == [0, 1]
    assert generator.trace_evidence.token_input_host_refreshes == int(reload_inputs)
    assert generator.trace_evidence.page_table_only_refreshes == int(reload_page_table and not reload_inputs)
    assert generator.trace_evidence.sampling_state_host_refreshes == int(reload_sampling_params)


@pytest.mark.parametrize("mode", ["device", "host"])
def test_eager_decode_requires_authoritative_inputs(mode):
    generator = decode_contract_generator()
    with pytest.raises(ValueError, match="reload_inputs=True"):  # allow-pytest.raises: host-only, no device conftest
        contract_decode(generator, enable_trace=False, sampling_mode=mode)
    contract_decode(generator, enable_trace=False, sampling_mode=mode, reload_inputs=True)
    generator._inner._decode_forward_no_trace_text.assert_called_once()
    generator._inner._decode_forward_trace_text.assert_not_called()


@pytest.mark.parametrize("page_only", [False, True])
def test_fixed_sampling_replay_preserves_parameter_and_history_state(page_only):
    generator = decode_contract_generator()
    contract_decode(generator, reload_inputs=True, reload_sampling_params=True, reset_sampling_state=True)
    generator._inner.sample_decode_on_device.reset_mock()
    contract_decode(generator, reload_page_table=page_only, reuse_sampling_state=True, slot_remap=[0, 1])
    generator._inner.sample_decode_on_device.assert_not_called()
    generator._replay_prepared_sampling.assert_called_once()
    assert generator._inner._decode_forward_trace_text.call_args.kwargs["reload_page_table"] is page_only
    assert generator.trace_evidence.sampling_state_host_refreshes == 1


@pytest.mark.parametrize("command", ["reload_inputs", "reload_sampling_params", "reset_sampling_state"])
def test_fixed_sampling_replay_rejects_state_changes(command):
    generator = decode_contract_generator()
    contract_decode(generator, reload_inputs=True, reload_sampling_params=True)
    with pytest.raises(ValueError, match="change sampling state"):  # allow-pytest.raises: host-only, no device conftest
        contract_decode(generator, reuse_sampling_state=True, **{command: True})


def test_fixed_sampling_replay_rejects_remap_or_active_request_seed():
    generator = decode_contract_generator()
    contract_decode(generator, reload_inputs=True, reload_sampling_params=True)
    with pytest.raises(ValueError, match="unchanged slot layout"):  # allow-pytest.raises: host-only, no device conftest
        contract_decode(generator, reuse_sampling_state=True, slot_remap=[1, 0])
    generator._sampling_has_active_request_seed.return_value = True
    with pytest.raises(RuntimeError, match="explicit request seeds"):  # allow-pytest.raises: CPU-only
        contract_decode(generator, reuse_sampling_state=True)


def test_decode_warmup_uses_shared_commands_without_request_history_reset():
    generator = decode_contract_generator()
    generator._inner._create_decode_warmup_inputs = Mock(
        return_value=(torch.zeros(2, 1), torch.zeros(2), torch.zeros(2, 2, dtype=torch.int32))
    )
    params = object()
    generator._inner._create_sampling_params = Mock(return_value=[None, params])
    generator.warmup_model_decode(
        kv_cache=[], enable_trace=True, max_batch_size=2, num_blocks=2, can_sample_on_device=True
    )
    generator._inner._decode_forward_trace_text.assert_called_once()
    kwargs = generator._inner.sample_decode_on_device.call_args.kwargs
    assert kwargs["reload_inputs"] is True
    assert kwargs["reload_sampling_params"] is True
    assert kwargs["reset_sampling_state"] is False


@pytest.mark.parametrize("reload_inputs", [False, True])
def test_page_only_reload_preserves_async_tokens_and_positions(reload_inputs, monkeypatch):
    buffers = [torch.tensor([[71]]), torch.tensor([91]), torch.tensor([92]), torch.tensor([[1, 2]])]
    inner = SimpleNamespace(
        data_parallel=1,
        model=[SimpleNamespace(prepare_decode_inputs_host=lambda token, pos, page: [token, pos, pos + 1, page])],
        model_args=[SimpleNamespace(mesh_device=object())],
        trace_ids_decode={True: {0: 123}},
        trace_inputs_decode={True: [buffers]},
        trace_output_decode={True: object()},
    )

    def copy_inputs(*, host_tensors, device_tensors):
        for host, device in zip(host_tensors, device_tensors):
            device.copy_(host)

    monkeypatch.setitem(SharedGenerator._decode_forward_trace_text.__globals__, "copy_host_to_device", copy_inputs)
    monkeypatch.setattr(ttnn, "copy_host_to_device_tensor", lambda host, device: device.copy_(host))
    monkeypatch.setattr(ttnn, "execute_trace", Mock())
    SharedGenerator._decode_forward_trace_text(
        inner,
        [torch.tensor([[7]])],
        [torch.tensor([11])],
        page_table=[torch.tensor([[3, 4]])],
        on_device_sampling=True,
        reload_inputs=reload_inputs,
        reload_page_table=True,
    )
    assert buffers[0].item() == (7 if reload_inputs else 71)
    assert buffers[1].item() == (11 if reload_inputs else 91)
    assert buffers[2].item() == (12 if reload_inputs else 92)
    assert buffers[3].tolist() == [[3, 4]]
    ttnn.execute_trace.assert_called_once()


@pytest.mark.parametrize("ring_dirty", [False, True])
@pytest.mark.parametrize("recapture", [False, True])
@pytest.mark.parametrize("reload_page_table", [False, True])
@pytest.mark.parametrize("reload_sampling_params,reset_sampling_state", [(False, False), (True, False), (True, True)])
def test_vllm_adapter_forwards_explicit_state_commands(
    reload_page_table, reload_sampling_params, reset_sampling_state, ring_dirty, recapture
):
    from collections import defaultdict
    from contextlib import contextmanager

    generator = decode_contract_generator()
    adapter = object.__new__(TTGptOssForCausalLM)
    adapter._require_generator = lambda: generator
    adapter._transition_sampling_lifecycle = Mock()
    adapter._device_trace_recapture_requires_reset = recapture
    adapter._decode_bucket = Mock(return_value=2)
    adapter._active_decode_bucket = 2
    adapter._slice_page_tables = lambda tables, bucket: tables
    adapter._apply_ring_slot_remap = Mock()
    adapter._sampling_state_reusable = Mock(return_value=False)
    adapter._last_sampling_key = None
    adapter._ring_tables_dirty = ring_dirty
    adapter.serving_counters = defaultdict(int)

    @contextmanager
    def route(*args, **kwargs):
        # Production routing clears this marker before yielding.
        adapter._ring_tables_dirty = False
        yield

    adapter._route_page_tables = route
    adapter.decode_forward(
        tokens=torch.tensor([[7], [9]]),
        start_pos=torch.tensor([31, 63]),
        page_table=torch.tensor([[3, 4], [5, 6]], dtype=torch.int32),
        kv_cache=[],
        sampling_params=SimpleNamespace(temperature=1.0, top_k=10, top_p=0.9),
        read_from_device=False,
        reload_inputs=reset_sampling_state,
        reload_page_table=reload_page_table,
        reload_sampling_params=reload_sampling_params,
        reset_sampling_state=reset_sampling_state,
    )
    trace = generator._inner._decode_forward_trace_text.call_args.kwargs
    sampler = generator._inner.sample_decode_on_device.call_args.kwargs
    assert trace["reload_inputs"] is (reset_sampling_state or recapture)
    assert trace["reload_page_table"] is (reload_page_table or ring_dirty)
    assert sampler["reload_sampling_params"] is (reload_sampling_params or recapture)
    assert sampler["reset_sampling_state"] is (reset_sampling_state or recapture)
    assert adapter._device_trace_recapture_requires_reset is False


@pytest.mark.parametrize("teacher_forced", [False, True])
def test_standalone_generation_reloads_teacher_tokens_but_preserves_free_run(teacher_forced):
    generator = decode_contract_generator()
    generator.reset = Mock()
    generator._kv_cache = []
    generator._require_private_page_table = Mock(return_value=torch.tensor([[3, 4]], dtype=torch.int32))
    generator._device_prefill_sample = Mock(return_value=5)
    generator._eos_token_ids = Mock(return_value=set())
    generator._inner._decode_forward_trace_text.return_value = torch.tensor([7])
    generator._write_runtime_evidence = Mock()
    generator._trace_handles_before_measurement = {}
    generator._warmed_before_measurement = False
    output = generator.generate([1, 2], 3, next_input=(lambda step, token: 20 + step) if teacher_forced else None)
    assert output == [5, 7, 7]
    calls = generator._inner._decode_forward_trace_text.call_args_list
    assert [call.kwargs["reload_inputs"] for call in calls] == [True, teacher_forced]
    assert [int(call.kwargs["tokens"][0][0, 0]) for call in calls] == ([20, 21] if teacher_forced else [5, 5])
    assert [
        call.kwargs["reset_sampling_state"] for call in generator._inner.sample_decode_on_device.call_args_list
    ] == [
        True,
        teacher_forced,
    ]


def test_split_token_out_initializes_once_then_uses_current_deferred_contract():
    generator = decode_contract_generator()
    generator.reset = Mock()
    generator._kv_cache = []
    generator._require_private_page_table = Mock(return_value=torch.tensor([[3, 4]], dtype=torch.int32))
    generator._device_prefill_sample = Mock(return_value=5)
    generator.read_decode_output = Mock(return_value=torch.tensor([7]))
    metrics = generator.run_device_token_out(
        [1, 2], 3, sampling_params=SimpleNamespace(seed=None, temperature=0.0, top_k=1, top_p=1.0)
    )
    assert metrics["output_tokens"] == 3
    assert metrics["final_token"] == 7
    assert generator._inner.sample_decode_on_device.call_count == 1
    generator._replay_prepared_sampling.assert_called_once()
    assert [call.kwargs["reload_inputs"] for call in generator._inner._decode_forward_trace_text.call_args_list] == [
        True,
        False,
    ]
