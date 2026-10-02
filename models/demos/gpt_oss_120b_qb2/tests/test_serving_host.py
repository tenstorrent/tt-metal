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
@pytest.mark.parametrize("force_host_tokens", [False, True])
def test_synchronous_host_decode_preserves_full_vocabulary(width, read_from_device, force_host_tokens):
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
        force_host_tokens=force_host_tokens,
        read_from_device=read_from_device,
    )
    if force_host_tokens:
        assert generator._inner._slots_prefilled_since_decode == set(range(width))
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
