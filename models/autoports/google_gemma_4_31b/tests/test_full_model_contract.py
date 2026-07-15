# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
from pathlib import Path

import torch

import ttnn
from models.autoports.google_gemma_4_31b.tests.run_full_model_qualitative import (
    _aligned_logits_comparison,
    _block_match_summary,
    _stable_logits_summary,
)
from models.autoports.google_gemma_4_31b.tt.generator import Gemma4Generator, Gemma4GreedyTP4Sampler, build_generator
from models.autoports.google_gemma_4_31b.tt.model import (
    ROPE_POSITION_INACTIVE_SENTINEL,
    DecodeTraceState,
    Gemma4FullModel,
    Gemma4FullModelConfig,
    _kv_cache_identity,
    _pad_rope_positions,
    _sequence_tile_ranges,
)
from models.autoports.google_gemma_4_31b.tt.multichip_decoder import (
    DEFAULT_MULTICHIP_OPTIMIZATION_POLICY,
    MultichipDecoder,
)
from models.common.modules.sampling.sampling_1d import Sampling1D
from models.common.readiness_check.contract import Generator


def test_generator_declares_standard_readiness_contract():
    assert issubclass(Gemma4Generator, Generator)
    signature = inspect.signature(Gemma4Generator.generate)
    assert signature.parameters["enable_trace"].kind is inspect.Parameter.KEYWORD_ONLY
    assert "next_input" in signature.parameters
    assert "host_sampling_compat" in signature.parameters
    assert callable(build_generator)
    assert {"model_dir", "mesh_device"} <= set(inspect.signature(build_generator).parameters)


def test_aligned_lm_head_diagnostic_metrics_are_tie_stable_and_detect_block_permutation():
    legacy = torch.tensor([4.0, 4.0, 2.0, 1.0], dtype=torch.bfloat16)
    optimized = torch.tensor([4.0, 4.0, 2.5, 1.0], dtype=torch.bfloat16)
    summary = _stable_logits_summary(legacy, top_k=4)
    assert summary["argmax"] == 0
    assert summary["exact_max_ids"] == [0, 1]
    assert summary["top1_top2_margin"] == 0.0
    comparison = _aligned_logits_comparison(legacy, optimized)
    assert -1.0 <= comparison["pcc"] <= 1.0
    assert comparison["exact_bf16_fraction"] == 0.75
    assert comparison["max_abs_delta"] == 0.5

    reference_blocks = torch.tensor([1.0, 2.0, 10.0, 20.0], dtype=torch.bfloat16)
    swapped_blocks = torch.tensor([10.0, 20.0, 1.0, 2.0], dtype=torch.bfloat16)
    matches = _block_match_summary(reference_blocks, swapped_blocks, block_size=2)
    assert [entry["best_legacy_block"] for entry in matches] == [1, 0]
    assert all(entry["best_mse"] == 0.0 for entry in matches)


def test_full_model_evidence_harness_exposes_focused_repeat_modes():
    source_path = Path(__file__).with_name("run_full_model_qualitative.py")
    source = source_path.read_text(encoding="utf-8")
    assert "--aligned-ab-only" in source
    assert "--benchmark-only" in source
    assert "benchmark_warmups" in source
    assert "statistics.median" in source
    assert 'GEMMA4_31B_LM_HEAD_DRAM_SHARDED", "1"' in source


def test_full_model_preserves_tp4_decoder_and_context_defaults():
    config = Gemma4FullModelConfig()
    assert config.max_seq_len == 262_144
    assert config.max_batch_size == 1
    assert config.lm_head_weight_dtype == ttnn.bfloat16
    assert config.logits_dtype == ttnn.bfloat16
    assert config.lm_head_math_fidelity == ttnn.MathFidelity.HiFi2
    assert config.lm_head_dram_sharded is True
    assert config.lm_head_num_cores == 4
    assert config.lm_head_in0_block_w == 2
    assert config.lm_head_split_size == 8192
    source = inspect.getsource(Gemma4FullModel.__init__)
    assert "MultichipDecoder.from_state_dict" in source
    assert "models.demos.gemma4.tt.model" not in source

    terminal_source = inspect.getsource(Gemma4FullModel._terminal)
    sharded_projection_source = inspect.getsource(Gemma4FullModel._project_sharded_lm_head_tile)
    assert "MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig" in source
    assert "self.lm_head_weights" in sharded_projection_source
    assert "ttnn.sharded_to_interleaved" in sharded_projection_source
    assert "ttnn.concat" in sharded_projection_source
    assert "_sequence_tile_ranges(logical_rows)" in terminal_source
    assert "ttnn.concat(tile_logits, dim=-2" in terminal_source

    policy = DEFAULT_MULTICHIP_OPTIMIZATION_POLICY
    assert policy.attention_weight_dtype == ttnn.bfloat8_b
    assert policy.mlp_gate_up_weight_dtype == ttnn.bfloat4_b
    assert policy.mlp_down_weight_dtype == ttnn.bfloat4_b
    assert policy.attention_math_fidelity == ttnn.MathFidelity.LoFi
    assert policy.mlp_gate_up_math_fidelity == ttnn.MathFidelity.LoFi
    assert policy.mlp_down_math_fidelity == ttnn.MathFidelity.LoFi
    assert MultichipDecoder.mesh_profile["activation_contract"].startswith("replicated BF16")
    assert "BFP8" in MultichipDecoder.mesh_profile["kv_cache"]


def test_sharded_lm_head_tiles_arbitrary_logical_prefill_rows(expect_error):
    expected = {
        1: ((0, 1),),
        31: ((0, 31),),
        32: ((0, 32),),
        33: ((0, 32), (32, 33)),
        63: ((0, 32), (32, 63)),
        64: ((0, 32), (32, 64)),
        65: ((0, 32), (32, 64), (64, 65)),
        149: ((0, 32), (32, 64), (64, 96), (96, 128), (128, 149)),
        249: (
            (0, 32),
            (32, 64),
            (64, 96),
            (96, 128),
            (128, 160),
            (160, 192),
            (192, 224),
            (224, 249),
        ),
    }
    for logical_rows, ranges in expected.items():
        assert _sequence_tile_ranges(logical_rows) == ranges
        assert ranges[0][0] == 0
        assert ranges[-1][1] == logical_rows
        assert all(1 <= end - start <= ttnn.TILE_SIZE for start, end in ranges)
        assert all(left[1] == right[0] for left, right in zip(ranges, ranges[1:]))
    with expect_error(ValueError, "positive"):
        _sequence_tile_ranges(0)

    terminal_source = inspect.getsource(Gemma4FullModel._terminal)
    assert terminal_source.index("self.final_norm.forward(hidden)") < terminal_source.index(
        "for start, end in _sequence_tile_ranges(logical_rows)"
    )
    assert terminal_source.count("self.final_norm.forward(hidden)") == 1


def test_non_aligned_and_inactive_position_normalization():
    positions = _pad_rope_positions(torch.tensor([33, -1, 1057], dtype=torch.int32))
    assert tuple(positions.shape) == (1, 32)
    assert positions[0, :3].tolist() == [33, ROPE_POSITION_INACTIVE_SENTINEL, 1057]
    assert torch.all(positions[0, 3:] == ROPE_POSITION_INACTIVE_SENTINEL)
    lookup = torch.bitwise_and(positions, (1 << 31) - 1)
    assert lookup[0, :3].tolist() == [33, 0, 1057]
    assert lookup[0, 3:].count_nonzero().item() == 0
    source = inspect.getsource(Gemma4FullModel.prefill_forward)
    assert "prompt_len" in source
    assert "% 32" not in source

    # TTNN full-range slices alias their input.  Short (<=32-token) prompts
    # must transfer ownership of the original hidden tensor to _terminal;
    # deallocating it before RMSNorm reproduces "Tensor is not allocated".
    host_prefill_source = inspect.getsource(Gemma4FullModel.prefill_forward)
    device_prefill_source = inspect.getsource(Gemma4FullModel.prefill_forward_device_logits)
    for prefill_source in (host_prefill_source, device_prefill_source):
        assert "tile_start == 0 and tile_end == hidden.shape[-2]" in prefill_source
        assert "last_hidden = hidden" in prefill_source or "last_tile = hidden" in prefill_source
    assert "for user_id, requested_len in enumerate(prompt_lens)" in device_prefill_source
    assert "ttnn.concat(row_logits, dim=-2" in device_prefill_source
    assert "device-logit prefill currently requires one prompt row" not in device_prefill_source


def test_low_level_api_exposes_explicit_external_state():
    for method_name, required in {
        "prefill_forward": {"tokens", "page_table", "kv_cache", "prompt_lens"},
        "decode_forward": {"tokens", "start_pos", "page_table", "kv_cache"},
        "prepare_token_out_decode": {
            "first_input_tokens",
            "start_positions",
            "page_table",
            "kv_cache",
            "page_table_generations",
            "prompt_lengths",
            "active_batch_size",
        },
        "decode_next_token_traced": {"page_table", "kv_cache", "page_table_generations"},
    }.items():
        signature = inspect.signature(getattr(Gemma4Generator, method_name))
        assert required <= set(signature.parameters)


def test_split_sampling_is_canonical_and_trace_owned():
    init_source = inspect.getsource(Gemma4Generator.__init__)
    assert "Gemma4GreedyTP4Sampler" in init_source
    assert "self.greedy_tp4_sampler" in init_source
    assert "Sampling1D.from_config" in init_source
    assert "max_top_k=32" in init_source
    assert "pad_to_power_of_2=False" in init_source
    assert "sampling_cluster_axis=1" in init_source
    assert "num_gather_links=2" in init_source
    assert "use_broadcast_all_gather=True" in init_source
    assert "gather_values_dtype=ttnn.float32" in init_source
    assert "min_candidate_gather_width" not in init_source
    assert "use_composite_all_gather" not in init_source
    assert "SamplingGenerator" not in init_source
    assert Gemma4Generator._sample_eager.__annotations__.get("return") is None
    assert Sampling1D.__module__ == "models.common.modules.sampling.sampling_1d"

    eager_source = inspect.getsource(Gemma4Generator._sample_eager)
    assert "_get_eager_sampler" in eager_source
    assert "int(logits.shape[-2])" in eager_source
    assert "int(tt_out_tok.shape[-1])" in eager_source
    assert "batch_size=batch_size" in eager_source
    assert "decode traces are live" in eager_source
    assert "_is_semantic_greedy" in eager_source
    assert "self.greedy_tp4_sampler.decode_forward" in eager_source

    topk_source = inspect.getsource(Sampling1D._sample_topk)
    assert topk_source.index("ttnn.typecast(topk_indices") < topk_source.index("ttnn.deallocate(topk_indices)")

    gather_source = inspect.getsource(Sampling1D._topk_multi_device)
    assert "gather_values_dtype" in gather_source
    assert "dtype=ttnn.bfloat16" in gather_source
    assert "ROW_MAJOR_LAYOUT" not in gather_source

    ccl_source = inspect.getsource(Sampling1D._perform_all_gather)
    assert "use_broadcast_all_gather" in ccl_source
    assert "use_broadcast=True" in ccl_source
    assert "get_and_cycle_ag_semaphore_handles" in ccl_source
    assert "get_and_cycle_barrier_semaphore_handle" in ccl_source

    resolver_source = inspect.getsource(__import__(Sampling1D.__module__, fromlist=["_resolve_sampling1d_config"]))
    assert "requires FP32 candidate values" in resolver_source

    capture_source = inspect.getsource(Gemma4Generator._capture_sampling_trace)
    assert "begin_trace_capture" in capture_source
    assert "tt_out_tok=tt_out_tok" in capture_source
    assert "use_greedy_tp4" in capture_source
    assert "self.greedy_tp4_sampler.decode_forward" in capture_source
    steady_source = inspect.getsource(Gemma4Generator.decode_next_token_traced)
    assert "to_torch" not in steady_source
    assert "get_device_tensors" not in steady_source
    assert "execute_decode_trace" in steady_source
    assert "_execute_sampling_trace" in steady_source

    hardware_source = Path(__file__).with_name("test_full_model.py").read_text(encoding="utf-8")
    reduced_source = hardware_source.split("def test_reduced_full_model_prefill_split_greedy_and_trace", 1)[1]
    reduced_source = reduced_source.split("\n@pytest.mark", 1)[0]
    assert "topk_value == host_argmax_value" in reduced_source
    assert "force_argmax=True" not in reduced_source


def test_custom_greedy_sampler_has_deterministic_tie_and_aligned_fixed_slot_writes():
    sampler_source = inspect.getsource(Gemma4GreedyTP4Sampler.decode_forward)
    assert "Topology.Linear" in sampler_source
    assert "ttnn.generic_op" in sampler_source
    assert "ttnn.all_gather" in sampler_source
    assert "output_tensor=self.gathered_pairs" in sampler_source
    assert "ttnn.to_torch" not in sampler_source
    assert "def teardown(self)" in inspect.getsource(Gemma4GreedyTP4Sampler)

    generator_source = inspect.getsource(Gemma4GreedyTP4Sampler._tile_winner_program)
    assert "output_pair_page_bytes = 16" in generator_source
    assert "active_batch_size" in generator_source

    kernel_dir = Path(__file__).parents[1] / "tt/kernels"
    tile_source = (kernel_dir / "gemma4_argmax_tile_local_winner.cpp").read_text(encoding="utf-8")
    reduce_source = (kernel_dir / "gemma4_argmax_pair_reduce.cpp").read_text(encoding="utf-8")
    for source in (tile_source, reduce_source):
        assert "candidate_index < best_index" in source
        assert "active_batch_size" in source
    assert "NOC_DRAM_WRITE_ALIGNMENT_BYTES" in reduce_source
    assert "active_batch_size * sizeof(uint32_t)" in reduce_source
    assert "noc_async_write(scratch_addr, output_accessor.get_noc_addr(0), output_write_bytes)" in reduce_source
    assert "batch * sizeof(uint32_t)" not in reduce_source

    hardware_source = Path(__file__).with_name("test_full_model.py").read_text(encoding="utf-8")
    assert "host[0, 0, 7, 177] = 100.0" in hardware_source
    assert "host[0, 0, 7, 192] = 100.0" in hardware_source
    assert "== [0, 32_767]" in hardware_source


def test_eager_sampler_is_keyed_to_actual_batch_without_changing_canonical_trace_sampler(monkeypatch, expect_error):
    generator = Gemma4Generator.__new__(Gemma4Generator)
    generator.max_batch_size = 2
    generator.mesh_device = object()
    generator.sampler = object()
    generator.force_argmax_sampler = object()
    generator._eager_samplers = {}
    generator._sampling_trace_id = None

    class TraceState:
        trace_id = None

    class Model:
        vocab_size = 262_144
        trace_state = TraceState()

    generator.model = Model()
    configs = []

    class FakeSampler:
        def load_device_buffers(self):
            return None

    def make_sampler(config):
        configs.append(config)
        return FakeSampler()

    monkeypatch.setattr(Sampling1D, "from_config", make_sampler)
    batch_one = generator._get_eager_sampler(1, force_argmax=False)
    assert configs[-1].max_batch_size == 1
    assert configs[-1].allow_force_argmax is False
    assert configs[-1].sampling_cluster_axis == 1
    assert configs[-1].num_gather_links == 2
    assert configs[-1].use_broadcast_all_gather is True
    assert configs[-1].gather_values_dtype == ttnn.float32
    assert generator._get_eager_sampler(1, force_argmax=False) is batch_one
    assert len(configs) == 1
    assert generator._get_eager_sampler(2, force_argmax=False) is generator.sampler
    assert generator._get_eager_sampler(2, force_argmax=True) is generator.force_argmax_sampler

    generator.model.trace_state.trace_id = 7
    with expect_error(RuntimeError, "before decode trace capture"):
        generator._get_eager_sampler(1, force_argmax=True)

    class Shaped:
        def __init__(self, shape):
            self.shape = shape

    with expect_error(RuntimeError, "traces are live"):
        generator._sample_eager(Shaped((1, 1, 2, 64)), tt_out_tok=Shaped((1, 1, 1, 2)))

    generator.model.trace_state.trace_id = None
    with expect_error(ValueError, "same batch size"):
        generator._sample_eager(Shaped((1, 1, 1, 64)), tt_out_tok=Shaped((1, 1, 1, 2)))

    get_source = inspect.getsource(Gemma4Generator._get_eager_sampler)
    assert "Sampling1D.from_config" in get_source
    assert "max_batch_size=batch_size" in get_source
    assert "return self.force_argmax_sampler if force_argmax else self.sampler" in get_source


def test_sampler_token_feedback_uses_output_width_for_fixed_slots():
    new_buffer_source = inspect.getsource(Gemma4Generator._new_token_buffer)
    assert "(1, 1, 1, batch_size)" in new_buffer_source

    embed_source = inspect.getsource(Gemma4FullModel.embed_tokens)
    assert "reshape(1, 1, 1, tokens.shape[0])" in embed_source
    assert 'int(tokens.shape[-1]) if mode == "decode"' in embed_source

    decode_source = inspect.getsource(Gemma4FullModel.decode_forward)
    initialize_source = inspect.getsource(Gemma4FullModel.initialize_trace_state)
    refresh_source = inspect.getsource(Gemma4FullModel.write_trace_tokens_from_host)
    for source in (decode_source, initialize_source):
        assert "reshape(1, 1, 1, batch_size)" in source
    assert "reshape(1, 1, 1, state.batch_size)" in refresh_source

    capture_source = inspect.getsource(Gemma4Generator._capture_sampling_trace)
    assert "int(tt_out_tok.shape[-1])" in capture_source
    assert "int(logits.shape[-2])" in capture_source
    assert "token output [1,1,1,B]" in capture_source


def test_decode_trace_advances_positions_and_refreshes_changed_tables_only():
    decode_source = inspect.getsource(Gemma4FullModel.decode_forward_device_state)
    assert "plus_one(cache_position, skip_negative_entries=True)" in decode_source
    assert "plus_one(rope_position, skip_negative_entries=True)" in decode_source
    assert "ttnn.add" not in decode_source
    assert "ttnn.copy" not in decode_source

    hidden_source = inspect.getsource(Gemma4FullModel.decode_hidden_device)
    assert "ttnn.bitwise_and(rope_position, rope_position_lookup_mask)" in hidden_source
    assert "current_position=rope_lookup_position" in hidden_source

    capture_source = inspect.getsource(Gemma4FullModel.capture_decode_trace)
    assert "warmup_logits.deallocate(True)" in capture_source

    refresh_source = inspect.getsource(Gemma4FullModel.refresh_trace_page_tables)
    assert "source is target" in refresh_source
    assert "page_table_generations" in refresh_source
    assert "ttnn.copy" in refresh_source


def test_eager_decode_rejects_a_false_traced_contract(expect_error):
    generator = Gemma4Generator.__new__(Gemma4Generator)
    generator.page_tables = object()
    generator.kv_cache = object()

    class StubModel:
        calls = 0

        def decode_forward(self, *args, **kwargs):
            self.calls += 1
            return "eager"

    generator.model = StubModel()
    with expect_error(ValueError, "prepare_token_out_decode"):
        generator.decode_forward(
            torch.zeros((1, 1)), torch.zeros((1,)), page_table=None, kv_cache=None, enable_trace=True
        )
    assert generator.model.calls == 0
    assert (
        generator.decode_forward(
            torch.zeros((1, 1)), torch.zeros((1,)), page_table=None, kv_cache=None, enable_trace=False
        )
        == "eager"
    )
    assert generator.model.calls == 1


def test_cache_handles_are_both_external_or_both_internal(expect_error):
    generator = Gemma4Generator.__new__(Gemma4Generator)
    generator.page_tables = object()
    generator.kv_cache = object()
    assert generator._resolve_cache_pair(None, None) == (generator.page_tables, generator.kv_cache, False)
    external_page_table, external_cache = object(), object()
    assert generator._resolve_cache_pair(external_page_table, external_cache) == (
        external_page_table,
        external_cache,
        True,
    )
    with expect_error(ValueError, "both"):
        generator._resolve_cache_pair(external_page_table, None)
    with expect_error(ValueError, "both"):
        generator._resolve_cache_pair(None, external_cache)

    prefill_source = inspect.getsource(Gemma4Generator.prefill_forward)
    assert "kv_cache is None and isinstance(page_table, ttnn.Tensor)" in prefill_source
    assert prefill_source.index("page_table = None") < prefill_source.index("self._resolve_cache_pair")


def test_kv_cache_identity_tracks_allocations_not_list_wrappers():
    key, value = object(), object()
    assert _kv_cache_identity([[key, value]]) == _kv_cache_identity([(key, value)])
    assert _kv_cache_identity([[key, value]]) != _kv_cache_identity([[key, object()]])


def test_external_page_table_refresh_requires_explicit_complete_generation(monkeypatch, expect_error):
    source, target = object(), object()
    model = Gemma4FullModel.__new__(Gemma4FullModel)
    model.trace_state = DecodeTraceState(
        page_tables=[target],
        page_table_identities=[id(source)],
        page_table_generations=[7],
    )
    copies = []
    monkeypatch.setattr(ttnn, "copy", lambda src, dst: copies.append((src, dst)))
    with expect_error(ValueError, "explicit"):
        model.refresh_trace_page_tables([source])
    with expect_error(ValueError, "layer count"):
        model.refresh_trace_page_tables([source], generations=[])
    model.refresh_trace_page_tables([source], generations=[7])
    assert copies == []
    model.refresh_trace_page_tables([source], generations=[8])
    assert copies == [(source, target)]
    model.refresh_trace_page_tables([source], generations=[8])
    assert copies == [(source, target)]


def test_context_window_and_fixed_slot_state_are_validated_before_device_work(expect_error):
    generator = Gemma4Generator.__new__(Gemma4Generator)
    generator.cache_context = 8
    generator.max_batch_size = 4
    generator._validate_generation_window(8, 1)
    generator._validate_generation_window(5, 4)
    with expect_error(ValueError, "context"):
        generator._validate_generation_window(8, 2)

    prompt_lengths, active = generator._normalize_decode_slots(
        torch.tensor([5, -1, 3]), prompt_lengths=[5, 0, 2], active_batch_size=2
    )
    assert prompt_lengths == (5, 0, 2)
    assert active == 2
    with expect_error(ValueError, "active_batch_size"):
        generator._normalize_decode_slots(torch.tensor([5, -1]), prompt_lengths=[5, 0], active_batch_size=2)
    with expect_error(ValueError, "position -1"):
        generator._normalize_decode_slots(torch.tensor([5, -2]), prompt_lengths=None, active_batch_size=None)

    prepare_source = inspect.getsource(Gemma4Generator.prepare_token_out_decode)
    assert "self.max_batch_size - tokens.shape[0]" in prepare_source
    assert "torch.full((padding,), -1" in prepare_source
    assert "prompt_lengths=normalized_prompt_lengths" in prepare_source
    assert "active_batch_size=active_batch_size" in prepare_source


def test_optimized_token_out_has_no_host_logits_boundary():
    generate_source = inspect.getsource(Gemma4Generator.generate)
    assert "host_sampling_compat" in generate_source
    assert "return_device_logits=True" in generate_source
    assert "prepare_token_out_decode" in generate_source
    assert "torch.argmax" in generate_source  # Explicit compatibility branch only.

    for method in (
        Gemma4Generator.prepare_token_out_decode,
        Gemma4Generator.decode_next_token_traced,
        Gemma4FullModel.execute_decode_trace,
    ):
        source = inspect.getsource(method)
        assert "torch.argmax" not in source
        assert "logits_to_torch" not in source


def test_eager_device_logits_release_model_owned_decode_inputs():
    source = inspect.getsource(Gemma4FullModel.decode_forward)
    return_index = source.index("if return_device_logits:")
    for tensor_name in ("token_input", "rope_position", "rope_position_lookup_mask", "cache_position"):
        assert source.index(f"{tensor_name}.deallocate(True)") < return_index


def test_generator_capabilities_do_not_overclaim_async_or_prefix_cache():
    capabilities = Gemma4Generator.model_capabilities
    assert capabilities["supports_prefix_caching"] is False
    assert capabilities["supports_async_decode"] is False
    assert capabilities["supports_mixed_prompt_lengths"] is True
    assert capabilities["supports_inactive_rows"] is True
    assert capabilities["supports_on_device_sampling"] is True


def test_generator_derives_non_default_batch_from_model_config():
    signature = inspect.signature(Gemma4Generator.__init__)
    assert signature.parameters["max_batch_size"].default is None
    source = inspect.getsource(Gemma4Generator.__init__)
    assert "model_config is None and model is not None" in source
    assert "self.model_config.max_batch_size if max_batch_size is None" in source
    build_source = inspect.getsource(build_generator)
    assert 'kwargs.get("model") is not None' in build_source
    assert 'model_config = kwargs["model"].config' in build_source


def test_new_requests_release_both_traces_before_prefill_allocations():
    release_source = inspect.getsource(Gemma4Generator._release_all_decode_traces)
    assert "release_trace" in release_source
    assert "model.release_decode_trace" in release_source
    reset_source = inspect.getsource(Gemma4Generator.reset)
    prefill_source = inspect.getsource(Gemma4Generator.prefill_forward)
    assert "_release_all_decode_traces" in reset_source
    assert "_release_all_decode_traces" in prefill_source
